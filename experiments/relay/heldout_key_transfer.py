"""EXP-R1d: 留出密钥泛化 —— 密钥身份是否语义扎根的决定性检验.

背景: EXP-R1c 的混淆结构检验无结论(两个 embedding 符号相反, n_pairs=28 功效不足).
最危险的漏洞仍开着: 密钥差异可能只是与 prompt token 相关的随机风格漂移,
而非模型真正执行了 persona/anchor 的语义.

决定性判据: **能否泛化到训练时从未见过的密钥.**
学一个线性映射 W: 轨迹表示空间 -> pattern 表示空间, 只用训练密钥拟合.
对留出密钥的轨迹, 投影后与**留出密钥的 pattern embedding** 匹配.

若随机漂移: W 只能记住训练密钥的任意对应, 对新密钥必然随机.
若语义扎根: W 学到的是"轨迹风格 -> 指令语义"的通用映射, 对新密钥仍有效.

这个检验无法靠记忆作弊, 因为留出密钥的类别在训练中根本不存在.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from length_control import split_steps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="experiments/relay/runs/attribution")
    ap.add_argument("--space", default="v2_diverse")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--n-holdout", type=int, default=3, help="留出密钥数")
    ap.add_argument("--ridge", type=float, default=1.0)
    args = ap.parse_args()

    run = pathlib.Path(args.run_dir)
    spaces = json.loads((run / "key_spaces.json").read_text())
    entries = spaces[args.space]
    K = len(entries)
    pats = [e["pattern"] for e in entries]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    texts, y = [], []
    for ki in range(K):
        fp = run / f"{args.space}__key{ki:02d}.jsonl"
        if not fp.exists():
            continue
        for r in (json.loads(l) for l in open(fp) if l.strip()):
            ss = split_steps(r["text"])
            if ss:
                texts.append(ss); y.append(ki)
    y = np.array(y)
    print(f"[info] space={args.space} K={K} n_traces={len(texts)}")

    X = []
    for ss in texts:
        E = model.encode(ss, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)
        v = E.mean(axis=0)
        X.append(v / (np.linalg.norm(v) + 1e-12))
    X = np.array(X)
    P = model.encode(pats, convert_to_numpy=True, normalize_embeddings=True,
                     show_progress_bar=False)

    rng = np.random.default_rng(0)
    accs, base = [], []
    for trial in range(12):
        ho = rng.choice(K, size=args.n_holdout, replace=False)
        tr_mask = ~np.isin(y, ho)

        # 岭回归学 W: X -> P[y]  (只用训练密钥)
        A, B = X[tr_mask], P[y[tr_mask]]
        W = np.linalg.solve(A.T @ A + args.ridge * np.eye(A.shape[1]), A.T @ B)

        # 留出密钥: 投影后只在**留出密钥的 pattern** 之间匹配
        te_mask = np.isin(y, ho)
        Z = X[te_mask] @ W
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        pred_local = np.argmax(Z @ P[ho].T, axis=1)
        pred = ho[pred_local]
        accs.append(float((pred == y[te_mask]).mean()))
        base.append(1.0 / len(ho))

    acc, sd = float(np.mean(accs)), float(np.std(accs))
    ch = float(np.mean(base))
    # 单样本 t 检验 (对随机基线)
    t = (acc - ch) / (sd / np.sqrt(len(accs)) + 1e-12)
    print(f"\n=== 留出密钥泛化 (n_holdout={args.n_holdout}, 12 次重抽) ===")
    print(f"  top-1 = {acc:.4f} ± {sd:.4f}   (随机 {ch:.4f})")
    print(f"  t = {t:+.2f}  ({'显著高于随机, 支持语义扎根' if t > 2.5 else '未显著高于随机'})")

    out = {"config": vars(args), "acc_mean": acc, "acc_sd": sd, "chance": ch,
           "t_stat": float(t), "n_trials": len(accs)}
    (run / f"heldout_transfer_{args.space}.json").write_text(json.dumps(out, indent=2))
    print(f"[done] {run}/heldout_transfer_{args.space}.json")


if __name__ == "__main__":
    main()
