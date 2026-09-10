"""EXP-B1: baseline 对比 —— 学习式 embedding 读出到底是不是必需的?

经验门槛里 baseline 是最大缺口(此前 0 个). 但选什么 baseline 取决于我们主张什么:
我们主张的是"**用学习式读出替代 trigger 文本相似度**", 所以 baseline 必须回答
"更简单的方法是不是也行", 而不是去比那些机理上不适用的行为层水印
(AgentMark/SeqWM 需要离散动作空间, 在纯文本 CoT 上无法运行 —— 那是我们的 gap 论据,
不是可比 baseline).

**v2 修订(重要)**: 初版存在两处使比较失效的不对等, 结论已作废重跑:
  1. 转导泄漏 —— TfidfVectorizer 与 SVD 在**全量文本(含测试折)**上 fit, 而
     embedding 用的是冻结的预训练编码器. TF-IDF 等于提前见过测试折的词表与主成分.
     现改为**每折只在训练集上 fit, 测试折只 transform**.
  2. 表示粒度不对等 —— embedding 走"按步取均值"(先压掉一轮信息), TF-IDF 吃整段文本.
     现补上 embed_doc(整段直接编码) 与 embed_meanstd(mean⊕std), 与 TF-IDF 对等.
另外 stylometry 拆成两档: 原版含 type-token ratio 等可能沾内容的统计量,
strict 版只留功能词+标点(纯写作习惯), 用于判断"内容信息"贡献了多少.

所有方法用**同一套 train/test 划分(按题目)**与同一个分类器, 唯一差异是特征.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from length_control import split_steps  # noqa: E402

# 经典文体学: 高频功能词(与内容无关, 只反映写作习惯)
FUNCTION_WORDS = [
    "the", "a", "an", "of", "to", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "but", "if", "then", "so", "because", "since", "thus", "hence",
    "we", "i", "you", "it", "this", "that", "these", "those", "each", "every",
    "is", "are", "was", "be", "have", "has", "will", "can", "must", "should",
    "now", "next", "first", "second", "finally", "also", "however", "therefore",
]
PUNCT = list(".,;:!?()-—'\"/*#$%")


def stylometry_features(text: str, strict: bool = False) -> np.ndarray:
    """功能词频率 + 标点频率 (+ 非 strict 时加句长/步长/字符类统计).

    strict=True 只保留与内容主题**完全无关**的量(功能词、标点), 用来判断
    非 strict 版里的增益有多少其实来自内容而非写作习惯.
    """
    low = text.lower()
    toks = re.findall(r"[a-z']+", low)
    n = max(len(toks), 1)
    fw = [toks.count(w) / n for w in FUNCTION_WORDS]
    nc = max(len(text), 1)
    pc = [text.count(p) / nc for p in PUNCT]
    if strict:
        return np.array(fw + pc, dtype=float)
    sents = [s for s in re.split(r"[.!?\n]+", text) if s.strip()]
    slen = [len(s.split()) for s in sents] or [0]
    steps = split_steps(text)
    steplen = [len(s.split()) for s in steps] or [0]
    extra = [
        len(sents), float(np.mean(slen)), float(np.std(slen)),
        len(steps), float(np.mean(steplen)), float(np.std(steplen)),
        sum(c.isupper() for c in text) / nc,
        sum(c.isdigit() for c in text) / nc,
        len(set(toks)) / n,  # type-token ratio
    ]
    return np.array(fw + pc + extra, dtype=float)


def load_traces(run: pathlib.Path, space: str, K: int):
    texts, y, qid = [], [], []
    for ki in range(K):
        fp = run / f"{space}__key{ki:02d}.jsonl"
        if not fp.exists():
            continue
        for line in open(fp):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["text"].strip():
                texts.append(r["text"]); y.append(ki); qid.append(r["qid"])
    return texts, np.array(y), np.array(qid)


def folds(qid, n_seeds=5, test_frac=0.3):
    """按题目划分的固定折, 所有特征共用 —— 保证唯一差异是表示."""
    out = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        uq = np.unique(qid); rng.shuffle(uq)
        te_q = set(uq[: max(1, int(len(uq) * test_frac))].tolist())
        te = np.array([q in te_q for q in qid])
        out.append((~te, te))
    return out


def eval_precomputed(X, y, fold_list):
    """特征与折无关(冻结编码器/确定性统计量), 直接按折评估."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    accs = []
    for tr, te in fold_list:
        if len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=5000).fit(sc.transform(X[tr]), y[tr])
        accs.append(float(clf.score(sc.transform(X[te]), y[te])))
    return accs


def eval_fitted_tfidf(texts, y, fold_list, tfidf_kw, n_comp=256):
    """**每折只在训练集上 fit** vectorizer 与 SVD, 测试折只 transform.

    初版在全量文本上 fit, 使 TF-IDF 提前见到测试折的词表与主成分, 属转导泄漏,
    与冻结编码器的 embedding 不对等. 这是本次修订的核心.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    texts = np.asarray(texts, dtype=object)
    accs, dims = [], []
    for tr, te in fold_list:
        if len(set(y[tr].tolist())) < 2:
            continue
        vec = TfidfVectorizer(**tfidf_kw).fit(texts[tr].tolist())
        Mtr = vec.transform(texts[tr].tolist())
        Mte = vec.transform(texts[te].tolist())
        k = min(n_comp, Mtr.shape[1] - 1, Mtr.shape[0] - 1)
        svd = TruncatedSVD(n_components=k, random_state=0).fit(Mtr)
        Xtr, Xte = svd.transform(Mtr), svd.transform(Mte)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), y[tr])
        accs.append(float(clf.score(sc.transform(Xte), y[te])))
        dims.append(k)
    return accs, int(np.mean(dims)) if dims else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="experiments/relay/runs/attribution")
    ap.add_argument("--space", default="v2_diverse")
    ap.add_argument("--embed-model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()

    run = pathlib.Path(args.run_dir)
    spaces = json.loads((run / "key_spaces.json").read_text())
    entries = spaces[args.space]
    K = len(entries)
    texts, y, qid = load_traces(run, args.space, K)
    fold_list = folds(qid, args.n_seeds)
    print(f"[info] {args.run_dir} / {args.space}: {len(texts)} 轨迹, K={K}, "
          f"{args.n_seeds} 折(按题目)", flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.embed_model)
    pats = [e["pattern"] for e in entries]
    E_pat = model.encode(pats, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)

    # ---- 逐轨迹编码一次, 三种 embedding 表示共用 ----
    print("[feat] 编码轨迹 ...", flush=True)
    v_mean, v_meanstd, sims = [], [], []
    for t in texts:
        ss = split_steps(t) or [t]
        E = model.encode(ss, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)
        m = E.mean(axis=0)
        v_mean.append(m / (np.linalg.norm(m) + 1e-12))
        v_meanstd.append(np.concatenate([m, E.std(axis=0)]))
        sims.append((E @ E_pat.T).mean(axis=0))
    E_doc = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                         batch_size=32, show_progress_bar=False)

    precomputed = {
        "length_only": np.array([[len(split_steps(t)), len(t), len(t.split())]
                                 for t in texts], dtype=float),
        "trigger_sim": np.array(sims),
        "stylometry": np.array([stylometry_features(t) for t in texts]),
        "stylometry_strict": np.array([stylometry_features(t, strict=True) for t in texts]),
        "embed_learned": np.array(v_mean),      # 原方法: 按步均值
        "embed_meanstd": np.array(v_meanstd),   # 对等变体: mean ⊕ std
        "embed_doc": E_doc,                     # 对等变体: 整段直接编码
    }

    rows = []
    order = ["length_only", "trigger_sim", "stylometry_strict", "stylometry",
             "tfidf_word", "tfidf_char", "embed_learned", "embed_meanstd", "embed_doc"]
    for name in order:
        if name.startswith("tfidf"):
            kw = (dict(analyzer="word", ngram_range=(1, 2), max_features=20000)
                  if name == "tfidf_word" else
                  dict(analyzer="char_wb", ngram_range=(2, 4), max_features=20000))
            print(f"[feat] {name} (每折内 fit) ...", flush=True)
            accs, dim = eval_fitted_tfidf(texts, y, fold_list, kw)
        else:
            X = precomputed[name]
            accs, dim = eval_precomputed(X, y, fold_list), int(X.shape[1])
        m = float(np.mean(accs)) if accs else float("nan")
        s = float(np.std(accs)) if accs else float("nan")
        rows.append({"baseline": name, "dim": dim, "top1": m, "sd": s,
                     "chance": 1.0 / K, "accs": accs})

    print(f"\n{'baseline':<20}{'维度':>6}{'top-1':>10}{'±sd':>8}{'倍数':>8}")
    print("-" * 54)
    for r in rows:
        print(f"{r['baseline']:<20}{r['dim']:>6}{r['top1']:>10.4f}{r['sd']:>8.4f}"
              f"{r['top1']*K:>7.2f}x")
    print(f"{'(随机)':<20}{'-':>6}{1/K:>10.4f}")

    # 配对检验: 最强 TF-IDF vs 最强 embedding, 同折配对
    from scipy import stats
    def best(pfx):
        cand = [r for r in rows if r["baseline"].startswith(pfx) and r["accs"]]
        return max(cand, key=lambda r: r["top1"]) if cand else None
    bt, be = best("tfidf"), best("embed")
    if bt and be and len(bt["accs"]) == len(be["accs"]):
        d = np.array(bt["accs"]) - np.array(be["accs"])
        t, p = stats.ttest_rel(bt["accs"], be["accs"])
        print(f"\n[配对检验] {bt['baseline']} vs {be['baseline']}: "
              f"Δ={d.mean():+.4f} (t={t:.2f}, p={p:.4f}, n={len(d)} 折)")

    out = run / f"baselines_{args.space}.json"
    out.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2))
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
