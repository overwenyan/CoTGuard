"""EXP-R3 打分: 密钥可恢复性随改写跳数的衰减.

对每个 (style, hop), 用学习式读出(逻辑回归, train/test 按题目划分)测 top-1,
并与 trigger 文本相似度读出对照 —— 后者是语义匹配范式的基线, 已知在 hop0 即等同随机.

关键判据: top-1 是否随跳数衰减到随机水平, 以及在第几跳.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from length_control import split_steps  # noqa: E402


def learned_top1(X, y, qid, K, n_seeds=5, test_frac=0.3):
    from sklearn.linear_model import LogisticRegression
    accs = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        uq = np.unique(qid); rng.shuffle(uq)
        te_q = set(uq[: max(1, int(len(uq) * test_frac))].tolist())
        te = np.array([q in te_q for q in qid]); tr = ~te
        if len(set(y[tr].tolist())) < 2:
            continue
        clf = LogisticRegression(max_iter=5000).fit(X[tr], y[tr])
        accs.append(float(clf.score(X[te], y[te])))
    return (float(np.mean(accs)), float(np.std(accs))) if accs else (float("nan"), float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--space", default="v2_diverse")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    args = ap.parse_args()

    run = pathlib.Path(args.run_dir)
    spaces = json.loads((run / "key_spaces.json").read_text())
    entries = spaces[args.space]
    K = len(entries)
    pats = [e["pattern"] for e in entries]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    E_pat = model.encode(pats, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)

    def encode_file(fp):
        X, y, qid, simk = [], [], [], []
        for r in (json.loads(l) for l in open(fp) if l.strip()):
            ss = split_steps(r["text"])
            if not ss:
                continue
            E = model.encode(ss, convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=False)
            v = E.mean(axis=0); v /= np.linalg.norm(v) + 1e-12
            X.append(v); y.append(r["key_idx"]); qid.append(r["qid"])
            simk.append((E @ E_pat.T).mean(axis=0))   # 对每个 pattern 的平均相似度
        return (np.array(X), np.array(y), np.array(qid), np.array(simk))

    rows = []
    files = [("(none)", 0, run / "hop0.jsonl")]
    for fp in sorted(run.glob("*__hop*.jsonl")):
        style, hop = fp.stem.split("__hop")
        files.append((style, int(hop), fp))

    for style, hop, fp in files:
        if not fp.exists():
            continue
        X, y, qid, simk = encode_file(fp)
        if len(X) == 0:
            continue
        acc, sd = learned_top1(X, y, qid, K)
        # 语义匹配基线: 真密钥 pattern 相似度在 K 个候选中排第一的比例
        text_top1 = float((simk.argmax(axis=1) == y).mean())
        rows.append({"style": style, "hop": hop, "n": int(len(X)),
                     "learned_top1": acc, "learned_sd": sd,
                     "text_sim_top1": text_top1, "chance": 1.0 / K})
        print(f"  {style:>14} hop{hop}  n={len(X):>4}  "
              f"学习式 top-1={acc:.4f}±{sd:.4f}   "
              f"文本相似度 top-1={text_top1:.4f}   (随机 {1/K:.4f})", flush=True)

    (run / "relay_attribution.json").write_text(json.dumps(
        {"config": vars(args), "rows": rows}, indent=2))
    print(f"\n[done] {run}/relay_attribution.json")


if __name__ == "__main__":
    main()
