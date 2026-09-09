"""EXP-R1c: 密钥差异是语义性的, 还是仅仅是 token 级随机漂移?

这是最危险的漏洞. 若 trigger 只是让模型产生了某种与 prompt token 相关的随机风格漂移,
而非真正执行了 persona/anchor 的语义, 那么"密钥信息编码在轨迹里"就是假象 ——
我们测的是"模型不听话"而非载体性质.

判据: **混淆结构应与 pattern 语义相似度相关.**
若密钥身份是语义编码的, 则语义上更接近的两个密钥(如两个都强调"追踪单位"的 anchor)
应当更容易互相混淆. 若只是随机漂移, 混淆矩阵与 pattern 相似度应无关.

同时用第二个 embedding 模型交叉验证, 排除"结论依赖单一打分器"的质疑.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from length_control import split_steps  # noqa: E402


def fit_eval(X, y, qid, K, seed=0, test_frac=0.3):
    """按题目划分, 返回 (top1, 混淆矩阵)."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    uq = np.unique(qid); rng.shuffle(uq)
    test_q = set(uq[:max(1, int(len(uq) * test_frac))].tolist())
    te = np.array([q in test_q for q in qid]); tr = ~te
    clf = LogisticRegression(max_iter=5000).fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    C = np.zeros((K, K))
    for t, p in zip(y[te], pred):
        C[t, p] += 1
    C = C / np.maximum(C.sum(axis=1, keepdims=True), 1)
    return float((pred == y[te]).mean()), C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="experiments/relay/runs/attribution")
    ap.add_argument("--models", default="sentence-transformers/all-mpnet-base-v2,"
                                        "thenlper/gte-base")
    ap.add_argument("--space", default="v2_diverse")
    args = ap.parse_args()

    run = pathlib.Path(args.run_dir)
    spaces = json.loads((run / "key_spaces.json").read_text())
    entries = spaces[args.space]
    K = len(entries)
    pats = [e["pattern"] for e in entries]

    texts, y, qid = [], [], []
    for ki in range(K):
        fp = run / f"{args.space}__key{ki:02d}.jsonl"
        if not fp.exists():
            continue
        for r in (json.loads(l) for l in open(fp) if l.strip()):
            ss = split_steps(r["text"])
            if ss:
                texts.append(ss); y.append(ki); qid.append(r["qid"])
    y = np.array(y); qid = np.array(qid)
    print(f"[info] space={args.space} K={K} n_traces={len(texts)}")

    from sentence_transformers import SentenceTransformer
    out = {}
    for mname in args.models.split(","):
        mname = mname.strip()
        try:
            model = SentenceTransformer(mname)
        except Exception as e:
            print(f"[skip] {mname}: {e}")
            continue
        X = []
        for ss in texts:
            E = model.encode(ss, convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=False)
            v = E.mean(axis=0)
            X.append(v / (np.linalg.norm(v) + 1e-12))
        X = np.array(X)
        E_pat = model.encode(pats, convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=False)
        P = E_pat @ E_pat.T   # pattern 语义相似度

        accs, Cs = [], []
        for seed in range(5):
            a, C = fit_eval(X, y, qid, K, seed=seed)
            accs.append(a); Cs.append(C)
        C = np.mean(Cs, axis=0)

        # 关键检验: 非对角混淆率 vs pattern 语义相似度的相关
        iu = np.triu_indices(K, 1)
        conf = ((C + C.T) / 2)[iu]     # 对称化的成对混淆率
        sim = P[iu]
        r = float(np.corrcoef(conf, sim)[0, 1]) if conf.std() > 1e-9 else float("nan")
        # Spearman 更稳健
        rk = lambda v: np.argsort(np.argsort(v))
        rs = float(np.corrcoef(rk(conf), rk(sim))[0, 1]) if conf.std() > 1e-9 else float("nan")

        out[mname] = {"top1_mean": float(np.mean(accs)), "top1_sd": float(np.std(accs)),
                      "chance": 1.0 / K,
                      "conf_sim_pearson": r, "conf_sim_spearman": rs,
                      "n_pairs": int(len(conf))}
        print(f"\n=== {mname} ===")
        print(f"  top-1 = {np.mean(accs):.4f} ± {np.std(accs):.4f}  (随机 {1/K:.4f}, 5 seeds)")
        print(f"  混淆率 vs pattern 语义相似度: Pearson r={r:+.3f}  Spearman={rs:+.3f}")
        print("  -> r 显著为正 = 语义上更近的密钥更易混淆 = 密钥身份是语义编码的")

    (run / f"confusion_{args.space}.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] {run}/confusion_{args.space}.json")


if __name__ == "__main__":
    main()
