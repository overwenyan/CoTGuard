"""EXP-R4: 跨跳迁移 —— 部署现实下的检测能力.

**为什么必须做这个**: `score_relay_attribution.py` 在每一跳都用**该跳自己的数据**
重训分类器, 测的是"给定 hop-N 文本, 密钥身份是否还可分". 但现实部署中 owner 只能
用**自己的生成数据**(hop0)训练检测器, 然后去检测被第三方转述过的轨迹 —— 拿不到
对方的中继数据来重训. 审稿人必然会问这一点.

本脚本测部署现实场景:
  train on hop0 (owner 自己的生成)  ->  test on hop-N (被转述 N 次的轨迹)

若跨跳迁移大幅劣于同跳重训, 说明此前的"10 跳不衰减"结论**在部署意义上被高估**,
必须如实修正主张范围.
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
    """返回 (X, y, qid)."""
    fp = run / fname
    if not fp.exists():
        return None
    X, y, qid = [], [], []
    for line in open(fp):
        if not line.strip():
            continue
        r = json.loads(line)
        ss = split_steps(r["text"])
        if not ss:
            continue
        E = model.encode(ss, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)
        v = E.mean(axis=0)
        X.append(v / (np.linalg.norm(v) + 1e-12))
        y.append(r["key_idx"]); qid.append(r["qid"])
    return np.array(X), np.array(y), np.array(qid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="experiments/relay/runs/relay_attr_v2")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--max-hops", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()

    run = pathlib.Path(args.run_dir)
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    model = SentenceTransformer(args.model)

    cache = run / f"crosshop_vecs_{args.model.split('/')[-1]}.npz"
    store = dict(np.load(cache)) if cache.exists() else {}

    def get(fname, key):
        if f"{key}__X" in store:
            return store[f"{key}__X"], store[f"{key}__y"], store[f"{key}__q"]
        got = load_hop(run, fname, model)
        if got is None:
            return None
        store[f"{key}__X"], store[f"{key}__y"], store[f"{key}__q"] = got
        np.savez_compressed(cache, **store)
        return got

    print("[encode] hop0 ...", flush=True)
    h0 = get("hop0.jsonl", "hop0")
    if h0 is None:
        print("[error] 缺 hop0.jsonl"); return 1
    X0, y0, q0 = h0
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
                Xt, yt, qt = X0, y0, q0
            else:
                got = get(f"{style}__hop{hop}.jsonl", f"{style}_hop{hop}")
                if got is None:
                    continue
                Xt, yt, qt = got

            same_accs, xfer_accs = [], []
            for seed in range(args.n_seeds):
                rng = np.random.default_rng(seed)
                uq = np.unique(q0); rng.shuffle(uq)
                te_q = set(uq[: max(1, int(len(uq) * 0.3))].tolist())
                tr0 = np.array([q not in te_q for q in q0])
                te_t = np.array([q in te_q for q in qt])
                if tr0.sum() < K or te_t.sum() < K:
                    continue
                # 部署现实: 只用 hop0 训练
                clf = LogisticRegression(max_iter=5000).fit(X0[tr0], y0[tr0])
                xfer_accs.append(float(clf.score(Xt[te_t], yt[te_t])))
                # 对照: 同跳重训(此前的测法)
                tr_t = ~te_t
                if len(set(yt[tr_t].tolist())) >= 2:
                    clf2 = LogisticRegression(max_iter=5000).fit(Xt[tr_t], yt[tr_t])
                    same_accs.append(float(clf2.score(Xt[te_t], yt[te_t])))

            if not xfer_accs:
                continue
            xa, xs = float(np.mean(xfer_accs)), float(np.std(xfer_accs))
            sa = float(np.mean(same_accs)) if same_accs else float("nan")
            rows.append({"style": style, "hop": hop, "chance": chance,
                         "transfer_top1": xa, "transfer_sd": xs, "same_hop_top1": sa})
            print(f"  {style:>14} hop{hop:<3} 跨跳迁移={xa:.4f}±{xs:.4f}  "
                  f"同跳重训={sa:.4f}  (随机 {chance:.4f})", flush=True)

    (run / "cross_hop_transfer.json").write_text(json.dumps(
        {"config": vars(args), "rows": rows}, indent=2))
    print(f"\n[done] {run}/cross_hop_transfer.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
