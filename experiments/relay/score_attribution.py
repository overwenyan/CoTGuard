"""EXP-R1 打分: 三种密钥空间的 detection / attribution 能力对比.

所有评估 length-matched, 因为 EXP-R0 证明只数步数即可达 AUROC 0.88.

指标:
  detection AUROC   triggered vs clean, 用真密钥 pattern 的相似度
  attribution AUROC 真密钥裕度 (真 - 最强错误) 区分 triggered vs clean
  top1 accuracy     真密钥在 K 个候选中排第一的比例  <- 所有权主张的直接度量
  mean rank         真密钥的平均排名 (0 最好, 随机期望 (K-1)/2)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from length_control import auroc, caliper_match, split_steps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--caliper", type=int, default=1)
    args = ap.parse_args()

    run = pathlib.Path(args.run_dir)
    spaces = json.loads((run / "key_spaces.json").read_text())
    clean = [json.loads(l) for l in open(run / "clean.jsonl") if l.strip()]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    def trace_mat(recs, E_pat):
        """每条轨迹对每个 pattern 的平均相似度 -> (n_traces, n_pat); 同时返回步数."""
        S, L = [], []
        for r in recs:
            ss = split_steps(r["text"])
            L.append(len(ss))
            if not ss:
                S.append(np.zeros(E_pat.shape[0])); continue
            E = model.encode(ss, convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=False)
            S.append((E @ E_pat.T).mean(axis=0))
        return np.array(S), np.array(L)

    results = {}
    for space, entries in spaces.items():
        pats = [e["pattern"] for e in entries]
        E_pat = model.encode(pats, convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=False)
        G = E_pat @ E_pat.T
        off = G[np.triu_indices(len(pats), 1)]
        S_c, L_c = trace_mat(clean, E_pat)

        det_all, att_all, top1_all, rank_all = [], [], [], []
        for ki in range(len(entries)):
            fp = run / f"{space}__key{ki:02d}.jsonl"
            if not fp.exists():
                continue
            recs = [json.loads(l) for l in open(fp) if l.strip()]
            S_t, L_t = trace_mat(recs, E_pat)

            # length-matched 配对
            pairs = caliper_match(list(L_t), list(L_c), args.caliper)
            if not pairs:
                continue
            it, ic = zip(*pairs)
            it, ic = list(it), list(ic)

            det = auroc(S_t[it, ki], S_c[ic, ki])
            other = [j for j in range(len(entries)) if j != ki]
            m_t = S_t[it, ki] - S_t[np.ix_(it, other)].max(axis=1)
            m_c = S_c[ic, ki] - S_c[np.ix_(ic, other)].max(axis=1)
            att = auroc(m_t, m_c)
            rank = (S_t[np.ix_(it, other)] >= S_t[it, ki][:, None]).sum(axis=1)

            det_all.append(det); att_all.append(att)
            top1_all.append(float((rank == 0).mean())); rank_all.append(float(rank.mean()))

        K = len(entries)
        results[space] = {
            "n_keys": K, "n_keys_evaluated": len(det_all),
            "pattern_sim": {"mean": float(off.mean()), "max": float(off.max()),
                            "p95": float(np.percentile(off, 95))},
            "detection_auroc": float(np.mean(det_all)) if det_all else None,
            "attribution_auroc": float(np.mean(att_all)) if att_all else None,
            "top1_acc": float(np.mean(top1_all)) if top1_all else None,
            "mean_rank": float(np.mean(rank_all)) if rank_all else None,
            "random_rank": (K - 1) / 2,
        }
        r = results[space]
        print(f"\n=== {space} (K={K}) ===")
        print(f"  pattern 相似度: mean={r['pattern_sim']['mean']:.4f} max={r['pattern_sim']['max']:.4f}")
        print(f"  detection   AUROC = {r['detection_auroc']:.4f}")
        print(f"  attribution AUROC = {r['attribution_auroc']:.4f}")
        print(f"  top-1 归因准确率  = {r['top1_acc']:.4f}   (随机 {1/K:.4f})")
        print(f"  真密钥平均排名    = {r['mean_rank']:.2f}   (随机期望 {r['random_rank']:.1f})")

    (run / "attribution.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] {run}/attribution.json")


if __name__ == "__main__":
    main()
