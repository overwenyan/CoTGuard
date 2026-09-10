"""EXP-R4: 跨跳迁移 —— 部署现实下的检测能力.

**为什么必须做这个**: `score_relay_attribution.py` 在每一跳都用**该跳自己的数据**
重训分类器, 测的是"给定 hop-N 文本, 密钥身份是否还可分". 但现实部署中 owner 只能
用**自己的生成数据**(hop0)训练检测器, 然后去检测被第三方转述过的轨迹 —— 拿不到
对方的中继数据来重训. 审稿人必然会问这一点.

本脚本测部署现实场景:
  train on hop0 (owner 自己的生成)  ->  test on hop-N (被转述 N 次的轨迹)

若跨跳迁移大幅劣于同跳重训, 说明此前的"10 跳不衰减"结论**在部署意义上被高估**,
必须如实修正主张范围.

**v2 修订(EXP-B1 之后)**: baseline 对比判定学习式 embedding 读出并不优于朴素
TF-IDF(四个 run 两平两负), 主方法已改为 n-gram 读出. 因此鲁棒性结论必须用**新读出器**
重测 —— 否则第 9/11 环说的还是一个已被降级的读出器. 加 `--readout` 支持:
  embed_meanstd  按步 embedding 的 mean⊕std (旧主方法, 保留作对照)
  tfidf_word     词级 TF-IDF + SVD (新主方法)
TF-IDF 的 vectorizer 与 SVD **只在 hop0 训练折上 fit**, 与部署现实一致(owner 只有
自己的生成数据), 也天然避免了 EXP-B1 初版那种转导泄漏.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from length_control import split_steps  # noqa: E402


def load_hop(run: pathlib.Path, fname: str, model):
    """返回 (X, y, qid, texts). X 为 embedding 的 mean⊕std(见模块 docstring v2 修订)."""
    fp = run / fname
    if not fp.exists():
        return None
    X, y, qid, texts = [], [], [], []
    for line in open(fp):
        if not line.strip():
            continue
        r = json.loads(line)
        ss = split_steps(r["text"])
        if not ss:
            continue
        E = model.encode(ss, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)
        X.append(np.concatenate([E.mean(axis=0), E.std(axis=0)]))
        y.append(r["key_idx"]); qid.append(r["qid"]); texts.append(r["text"])
    return np.array(X), np.array(y), np.array(qid), np.array(texts, dtype=object)


def fit_readout(readout, Xtr_vec, txt_tr, ytr):
    """返回 (clf, transform_fn). TF-IDF 的 vectorizer/SVD 只在训练折上 fit."""
    from sklearn.linear_model import LogisticRegression
    if readout == "embed_meanstd":
        clf = LogisticRegression(max_iter=5000).fit(Xtr_vec, ytr)
        return clf, lambda Xv, txt: Xv
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                          max_features=20000).fit(txt_tr.tolist())
    Mtr = vec.transform(txt_tr.tolist())
    k = min(256, Mtr.shape[1] - 1, Mtr.shape[0] - 1)
    svd = TruncatedSVD(n_components=k, random_state=0).fit(Mtr)
    clf = LogisticRegression(max_iter=5000).fit(svd.transform(Mtr), ytr)
    return clf, lambda Xv, txt: svd.transform(vec.transform(txt.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="experiments/relay/runs/relay_attr_v2")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--max-hops", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--readout", default="tfidf_word",
                    choices=["tfidf_word", "embed_meanstd"],
                    help="tfidf_word 为 EXP-B1 之后的主方法; embed_meanstd 保留作对照")
    args = ap.parse_args()

    run = pathlib.Path(args.run_dir)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    cache = run / f"crosshop_vecs2_{args.model.split('/')[-1]}.npz"
    store = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}

    def get(fname, key):
        if f"{key}__X" in store:
            return (store[f"{key}__X"], store[f"{key}__y"],
                    store[f"{key}__q"], store[f"{key}__t"])
        got = load_hop(run, fname, model)
        if got is None:
            return None
        (store[f"{key}__X"], store[f"{key}__y"],
         store[f"{key}__q"], store[f"{key}__t"]) = got
        np.savez_compressed(cache, **store)
        return got

    print(f"[cfg] readout = {args.readout}", flush=True)
    print("[encode] hop0 ...", flush=True)
    h0 = get("hop0.jsonl", "hop0")
    if h0 is None:
        print("[error] 缺 hop0.jsonl"); return 1
    X0, y0, q0, t0 = h0
    K = len(set(y0.tolist()))
    chance = 1.0 / K

    # 从 run 目录自动发现改写风格, 不写死 —— 否则新增风格(如 adaptive_max)会被静默跳过
    styles = sorted({f.stem.split("__hop")[0] for f in run.glob("*__hop*.jsonl")})
    if not styles:
        print(f"[error] {run} 下未找到 *__hop*.jsonl"); return 1
    print(f"[info] 发现改写风格: {styles}", flush=True)

    rows = []
    for style in styles:
        for hop in range(0, args.max_hops + 1):
            if hop == 0:
                Xt, yt, qt, tt = X0, y0, q0, t0
            else:
                got = get(f"{style}__hop{hop}.jsonl", f"{style}_hop{hop}")
                if got is None:
                    continue
                Xt, yt, qt, tt = got

            same_accs, xfer_accs = [], []
            for seed in range(args.n_seeds):
                rng = np.random.default_rng(seed)
                uq = np.unique(q0); rng.shuffle(uq)
                te_q = set(uq[: max(1, int(len(uq) * 0.3))].tolist())
                tr0 = np.array([q not in te_q for q in q0])
                te_t = np.array([q in te_q for q in qt])
                if tr0.sum() < K or te_t.sum() < K:
                    continue
                # 部署现实: 只用 hop0 训练(TF-IDF 的词表/SVD 亦只见 hop0 训练折)
                clf, tf = fit_readout(args.readout, X0[tr0], t0[tr0], y0[tr0])
                xfer_accs.append(float(clf.score(tf(Xt[te_t], tt[te_t]), yt[te_t])))
                # 对照: 同跳重训(此前的测法)
                tr_t = ~te_t
                if len(set(yt[tr_t].tolist())) >= 2:
                    clf2, tf2 = fit_readout(args.readout, Xt[tr_t], tt[tr_t], yt[tr_t])
                    same_accs.append(float(clf2.score(tf2(Xt[te_t], tt[te_t]), yt[te_t])))

            if not xfer_accs:
                continue
            xa, xs = float(np.mean(xfer_accs)), float(np.std(xfer_accs))
            sa = float(np.mean(same_accs)) if same_accs else float("nan")
            rows.append({"style": style, "hop": hop, "chance": chance,
                         "transfer_top1": xa, "transfer_sd": xs, "same_hop_top1": sa})
            print(f"  {style:>14} hop{hop:<3} 跨跳迁移={xa:.4f}±{xs:.4f}  "
                  f"同跳重训={sa:.4f}  (随机 {chance:.4f})", flush=True)

    out = run / f"cross_hop_transfer_{args.readout}.json"
    out.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2))
    print(f"\n[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
