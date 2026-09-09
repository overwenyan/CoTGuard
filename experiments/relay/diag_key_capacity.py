"""诊断: 语义 trigger 的密钥容量 —— 能否归因到具体密钥, 而非只是"用了某个 trigger".

矛盾现象:
  未校准的原始余弦相似度, length-matched AUROC = 0.9877  (信号很强)
  经 wrong-key 共形校准后,   length-matched AUROC = 0.5432  (信号消失)

假设: wrong-key pattern 与真 pattern 过于相似(共用模板, 仅换槽位填充词),
      导致被 trigger 引导的轨迹对所有 pattern 的相似度同等升高, 真密钥脱颖不出.

本脚本直接测:
  D1 pattern 之间的两两相似度 (真 vs 错、错 vs 错)
  D2 triggered 轨迹在真 pattern 与错 pattern 上的得分差 (归因裕度)
  D3 detection(检出用了trigger) 与 attribution(归因到哪个key) 的 AUROC 对比
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from length_control import auroc, caliper_match, split_steps  # noqa: E402
from trigger import make_trigger, wrong_keys  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="experiments/relay/runs/pilot")
    ap.add_argument("--hop", type=int, default=0)
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--true-key", default="patient-teacher-2026")
    ap.add_argument("--n-wrong", type=int, default=63)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    tau_t = make_trigger(args.true_key)
    taus_w = wrong_keys(args.true_key, args.n_wrong)
    E_tau = model.encode([tau_t] + taus_w, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)

    # --- D1: pattern 之间的相似度 ---
    G = E_tau @ E_tau.T
    tw = G[0, 1:]                                   # 真 vs 错
    ww = G[1:, 1:][np.triu_indices(len(taus_w), 1)]  # 错 vs 错
    print("=== D1 trigger pattern 之间的相似度 ===")
    print(f"  真 vs 错 : mean={tw.mean():.4f}  sd={tw.std():.4f}  "
          f"min={tw.min():.4f}  max={tw.max():.4f}")
    print(f"  错 vs 错 : mean={ww.mean():.4f}  sd={ww.std():.4f}")
    print(f"  -> pattern 空间高度拥挤 (相似度接近 1 意味着无法区分密钥)")

    # --- 载入 pilot 轨迹 ---
    run = pathlib.Path(args.run_dir)
    recs = [json.loads(l) for l in open(run / f"hop{args.hop}.jsonl") if l.strip()]
    trig = [r for r in recs if r["arm"] == "triggered"]
    clean = [r for r in recs if r["arm"] == "clean"]
    st_t = [split_steps(r["text"]) for r in trig]
    st_c = [split_steps(r["text"]) for r in clean]
    len_t = [len(s) for s in st_t]
    len_c = [len(s) for s in st_c]

    def trace_sims(steps_list):
        """每条轨迹对每个 pattern 的平均相似度 -> (n_traces, n_tau)."""
        out = []
        for ss in steps_list:
            if not ss:
                out.append(np.zeros(E_tau.shape[0])); continue
            E = model.encode(ss, convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=False)
            out.append((E @ E_tau.T).mean(axis=0))
        return np.array(out)

    print("\n[info] 编码轨迹 ...")
    S_t, S_c = trace_sims(st_t), trace_sims(st_c)

    # --- D2: 归因裕度 ---
    margin_t = S_t[:, 0] - S_t[:, 1:].max(axis=1)   # 真 key 分数 - 最强错误 key 分数
    margin_c = S_c[:, 0] - S_c[:, 1:].max(axis=1)
    print("\n=== D2 归因裕度 (真 key 分数 - 最强错误 key 分数) ===")
    print(f"  triggered: mean={margin_t.mean():+.4f}  sd={margin_t.std():.4f}  "
          f"P(>0)={float((margin_t>0).mean()):.3f}")
    print(f"  clean    : mean={margin_c.mean():+.4f}  sd={margin_c.std():.4f}  "
          f"P(>0)={float((margin_c>0).mean()):.3f}")
    print("  -> 若 triggered 的 P(>0) 不显著高于 clean, 说明无法归因到具体密钥")

    # --- D3: detection vs attribution, 均 length-matched ---
    pairs = caliper_match(len_t, len_c, 1)
    it, ic = (zip(*pairs) if pairs else ((), ()))
    # detection: 用真 key 的原始相似度区分 triggered/clean
    det = auroc(S_t[list(it), 0], S_c[list(ic), 0]) if pairs else float("nan")
    # attribution: 用"真 key 相对错误 key 的裕度"区分
    att = auroc(margin_t[list(it)], margin_c[list(ic)]) if pairs else float("nan")
    # 另一种 attribution: 真 key 在所有 key 中的排名(越靠前越好)
    rank_t = (S_t[:, 1:] >= S_t[:, :1]).sum(axis=1)
    rank_c = (S_c[:, 1:] >= S_c[:, :1]).sum(axis=1)
    print("\n=== D3 detection vs attribution (length-matched, n_pairs=%d) ===" % len(pairs))
    print(f"  detection  (用了某个 trigger 吗)  AUROC = {det:.4f}")
    print(f"  attribution(是这个密钥吗)         AUROC = {att:.4f}")
    print(f"  真 key 在 {len(taus_w)+1} 个 key 中的排名: "
          f"triggered mean={rank_t.mean():.1f}, clean mean={rank_c.mean():.1f} "
          f"(0 = 最像; 随机期望 {len(taus_w)/2:.1f})")

    out = {"config": vars(args),
           "pattern_sim": {"true_vs_wrong_mean": float(tw.mean()),
                           "true_vs_wrong_max": float(tw.max()),
                           "wrong_vs_wrong_mean": float(ww.mean())},
           "margin": {"triggered_mean": float(margin_t.mean()),
                      "clean_mean": float(margin_c.mean()),
                      "triggered_frac_pos": float((margin_t > 0).mean()),
                      "clean_frac_pos": float((margin_c > 0).mean())},
           "auroc_len_matched": {"detection": det, "attribution": att},
           "true_key_rank": {"triggered_mean": float(rank_t.mean()),
                             "clean_mean": float(rank_c.mean()),
                             "n_keys": len(taus_w) + 1},
           "n_pairs": len(pairs)}
    (run / "key_capacity.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] {run}/key_capacity.json")


if __name__ == "__main__":
    main()
