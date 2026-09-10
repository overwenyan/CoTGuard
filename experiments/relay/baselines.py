"""EXP-B1: baseline 对比 —— 学习式 embedding 读出到底是不是必需的?

经验门槛里 baseline 是最大缺口(此前 0 个). 但选什么 baseline 取决于我们主张什么:
我们主张的是"**用学习式读出替代 trigger 文本相似度**", 所以 baseline 必须回答
"更简单的方法是不是也行", 而不是去比那些机理上不适用的行为层水印
(AgentMark/SeqWM 需要离散动作空间, 在纯文本 CoT 上无法运行 —— 那是我们的 gap 论据,
不是可比 baseline).

五个 baseline, 从平凡到强:
  length_only   仅轨迹步数与字符数            —— 平凡对照, 检验是否被长度平凡驱动
  trigger_sim   与 trigger 文本的余弦相似度    —— **预印本原方法**
  tfidf_char    字符级 TF-IDF n-gram          —— 标准文本分类基线, 无语义模型
  tfidf_word    词级 TF-IDF n-gram            —— 同上
  stylometry    经典文体学特征(功能词/标点/句长) —— 作者归属文献的标准做法
  embed_learned 本文方法: embedding + 逻辑回归

所有方法用**同一套 train/test 划分(按题目)**与同一个分类器, 唯一差异是特征,
这样比较的才是特征表示本身.
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


def stylometry_features(text: str) -> np.ndarray:
    """功能词频率 + 标点频率 + 句长统计. 全部归一化, 与内容主题基本无关."""
    low = text.lower()
    toks = re.findall(r"[a-z']+", low)
    n = max(len(toks), 1)
    fw = [toks.count(w) / n for w in FUNCTION_WORDS]
    nc = max(len(text), 1)
    pc = [text.count(p) / nc for p in PUNCT]
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


def eval_features(X, y, qid, K, n_seeds=5, test_frac=0.3):
    """统一的评估协议: 按题目划分 + 逻辑回归. 返回 (mean, sd)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    accs = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        uq = np.unique(qid); rng.shuffle(uq)
        te_q = set(uq[: max(1, int(len(uq) * test_frac))].tolist())
        te = np.array([q in te_q for q in qid]); tr = ~te
        if len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=5000).fit(sc.transform(X[tr]), y[tr])
        accs.append(float(clf.score(sc.transform(X[te]), y[te])))
    return (float(np.mean(accs)), float(np.std(accs))) if accs else (float("nan"),) * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="experiments/relay/runs/attribution")
    ap.add_argument("--space", default="v2_diverse")
    ap.add_argument("--embed-model", default="sentence-transformers/all-mpnet-base-v2")
    args = ap.parse_args()

    run = pathlib.Path(args.run_dir)
    spaces = json.loads((run / "key_spaces.json").read_text())
    entries = spaces[args.space]
    K = len(entries)
    texts, y, qid = load_traces(run, args.space, K)
    print(f"[info] {args.run_dir} / {args.space}: {len(texts)} 轨迹, K={K}", flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.embed_model)
    pats = [e["pattern"] for e in entries]
    E_pat = model.encode(pats, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)

    feats = {}

    # 1. length_only
    feats["length_only"] = np.array([[len(split_steps(t)), len(t), len(t.split())]
                                     for t in texts], dtype=float)

    # 2. trigger_sim (预印本原方法): 每步对每个 pattern 的平均相似度
    print("[feat] trigger_sim ...", flush=True)
    sims = []
    for t in texts:
        ss = split_steps(t)
        E = model.encode(ss, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)
        sims.append((E @ E_pat.T).mean(axis=0))
    feats["trigger_sim"] = np.array(sims)

    # 3/4. TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    for name, kw in [("tfidf_char", dict(analyzer="char_wb", ngram_range=(2, 4), max_features=20000)),
                     ("tfidf_word", dict(analyzer="word", ngram_range=(1, 2), max_features=20000))]:
        print(f"[feat] {name} ...", flush=True)
        M = TfidfVectorizer(**kw).fit_transform(texts)
        # 降到与 embedding 同量级维度, 保证比较的是表示质量而非维度优势
        feats[name] = TruncatedSVD(n_components=min(256, M.shape[1] - 1),
                                   random_state=0).fit_transform(M)

    # 5. stylometry
    print("[feat] stylometry ...", flush=True)
    feats["stylometry"] = np.array([stylometry_features(t) for t in texts])

    # 6. 本文方法
    print("[feat] embed_learned ...", flush=True)
    V = []
    for t in texts:
        ss = split_steps(t)
        E = model.encode(ss, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)
        v = E.mean(axis=0); V.append(v / (np.linalg.norm(v) + 1e-12))
    feats["embed_learned"] = np.array(V)

    print(f"\n{'baseline':<16}{'维度':>6}{'top-1':>10}{'±sd':>8}{'倍数':>8}")
    print("-" * 50)
    rows = []
    order = ["length_only", "trigger_sim", "stylometry", "tfidf_word", "tfidf_char", "embed_learned"]
    for name in order:
        X = feats[name]
        m, s = eval_features(X, y, qid, K)
        rows.append({"baseline": name, "dim": int(X.shape[1]), "top1": m, "sd": s,
                     "chance": 1.0 / K})
        print(f"{name:<16}{X.shape[1]:>6}{m:>10.4f}{s:>8.4f}{m/(1/K):>7.1f}x")
    print(f"{'(随机)':<16}{'-':>6}{1/K:>10.4f}")

    out = run / f"baselines_{args.space}.json"
    out.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2))
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
