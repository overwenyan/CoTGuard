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
    accs, base, tr_accs = [], [], []
    for trial in range(12):
        ho = rng.choice(K, size=args.n_holdout, replace=False)
        tr_mask = ~np.isin(y, ho)
        tr_keys = np.array(sorted(set(y[tr_mask].tolist())))

        # 岭回归学 W: X -> P[y]  (只用训练密钥)
        A, B = X[tr_mask], P[y[tr_mask]]
        W = np.linalg.solve(A.T @ A + args.ridge * np.eye(A.shape[1]), A.T @ B)

        # 留出密钥: 投影后只在**留出密钥的 pattern** 之间匹配
        te_mask = np.isin(y, ho)
        Z = X[te_mask] @ W
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        pred = ho[np.argmax(Z @ P[ho].T, axis=1)]
        accs.append(float((pred == y[te_mask]).mean()))
        base.append(1.0 / len(ho))

        # 训练集对照: 同一个 W 在**训练密钥**上的匹配准确率.
        # 若这个也接近随机, 说明映射根本没拟合上(容量/欠定问题), 与语义无关.
        Ztr = X[tr_mask] @ W
        Ztr /= np.linalg.norm(Ztr, axis=1, keepdims=True) + 1e-12
        pred_tr = tr_keys[np.argmax(Ztr @ P[tr_keys].T, axis=1)]
        tr_accs.append(float((pred_tr == y[tr_mask]).mean()))

    acc, sd = float(np.mean(accs)), float(np.std(accs))
    ch = float(np.mean(base))
    tr_acc = float(np.mean(tr_accs))
    tr_ch = float(np.mean([1.0 / (K - args.n_holdout)] * len(tr_accs)))
    t = (acc - ch) / (sd / np.sqrt(len(accs)) + 1e-12)

    print(f"\n=== 留出密钥泛化 (n_holdout={args.n_holdout}, 12 次重抽) ===")
    print(f"  训练密钥上 top-1 = {tr_acc:.4f}   (随机 {tr_ch:.4f})  <- 容量对照")
    print(f"  留出密钥上 top-1 = {acc:.4f} ± {sd:.4f}   (随机 {ch:.4f})")
    print(f"  t = {t:+.2f}")
    if tr_acc < tr_ch * 1.5:
        print("  -> **映射连训练密钥都拟合不上**: 属容量/欠定问题, "
              "本检验对语义性问题无判别力")
    elif t > 2.5:
        print("  -> 训练集拟合良好且留出集显著高于随机: 支持语义扎根")
    else:
        print("  -> 训练集拟合良好但留出集不高于随机: "
              "映射学到的是每密钥特异签名, **非**通用语义映射")

    out = {"config": vars(args), "acc_mean": acc, "acc_sd": sd, "chance": ch,
           "train_acc": tr_acc, "train_chance": tr_ch,
           "t_stat": float(t), "n_trials": len(accs)}
    (run / f"heldout_transfer_{args.space}.json").write_text(json.dumps(out, indent=2))
    print(f"[done] {run}/heldout_transfer_{args.space}.json")


if __name__ == "__main__":
    main()
