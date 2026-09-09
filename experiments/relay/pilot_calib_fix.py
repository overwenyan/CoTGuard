"""P1-b: 修复 wrong-key 校准在语义载体上的不可交换性.

P1-a 发现: 不同 trigger pattern 对推理文本的天然亲和度不同(平均相似度 sd=0.031,
范围 0.078-0.220), 故真密钥与错误密钥不可交换, 共形 p 值严重偏离 U(0,1)
(mean p = 0.76 而非 0.50).

本脚本比较三种打分方式:
  raw        原始余弦相似度                      (P1-a 的做法, 已知失效)
  zscore     用该 pattern 自身在独立干净语料上的 (mean, sd) 标准化
  rank       用该 pattern 自身在独立干净语料上的经验 CDF 转成分位

后两者都把"pattern 固有亲和度"这一 nuisance 消掉. 关键是标准化所用的语料必须与
被检轨迹**独立**(用 split 隔离), 否则会泄漏.

评价标准: 干净轨迹上真 key 的共形 p 值是否 ~ U(0,1).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from trigger import make_trigger, wrong_keys  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"


def split_steps(text: str) -> list[str]:
    parts = [s.strip() for s in text.replace("\n\n", "\n").split("\n")]
    return [p for p in parts if len(p) > 5]


def uniformity_report(pval, label):
    levels = (0.01, 0.05, 0.10, 0.25, 0.50)
    hits = {a: float((pval <= a).mean()) for a in levels}
    # Kolmogorov-Smirnov 距离 (对 U(0,1))
    ps = np.sort(pval)
    m = ps.size
    ks = np.abs(np.arange(1, m + 1) / m - ps).max()
    print(f"  {label:>8}  mean_p={pval.mean():.4f}  KS={ks:.4f}  " +
          "  ".join(f"P(<={a})={hits[a]:.3f}" for a in levels))
    return {"mean_p": float(pval.mean()), "ks": float(ks),
            "hits": {str(a): hits[a] for a in levels}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--n-calib", type=int, default=1200)
    ap.add_argument("--n-wrong", type=int, default=127)
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--true-key", default="patient-teacher-2026")
    args = ap.parse_args()

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    ds = load_dataset("openai/gsm8k", "main", split="train")
    need = args.n_eval + args.n_calib
    ds = ds.select(range(min(need, len(ds))))
    traces = [split_steps(r["answer"]) for r in ds]
    traces = [t for t in traces if len(t) >= 2]

    # 严格隔离: 校准语料与被检轨迹不重叠
    calib_steps = [s for t in traces[:args.n_calib] for s in t]
    eval_steps = [s for t in traces[args.n_calib:] for s in t]
    print(f"[info] 校准步 {len(calib_steps)}, 评估步 {len(eval_steps)}")

    tau_true = make_trigger(args.true_key)
    taus = [tau_true] + wrong_keys(args.true_key, args.n_wrong)
    print(f"[info] 1 真 + {len(taus)-1} 错 pattern")

    model = SentenceTransformer(args.model)
    e_cal = model.encode(calib_steps, batch_size=256, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
    e_evl = model.encode(eval_steps, batch_size=256, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
    e_tau = model.encode(taus, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)

    sim_cal = e_cal @ e_tau.T   # (n_calib_steps, n_tau)
    sim_evl = e_evl @ e_tau.T   # (n_eval_steps,  n_tau)

    results = {}
    print("\n=== 干净轨迹上真 key 的共形 p 值 (应 ~ U(0,1)) ===")

    # --- raw ---
    p_raw = (1.0 + (sim_evl[:, 1:] >= sim_evl[:, :1]).sum(axis=1)) / (len(taus))
    results["raw"] = uniformity_report(p_raw, "raw")

    # --- zscore: 每个 pattern 用自身在校准语料上的 (mean, sd) ---
    mu, sd = sim_cal.mean(axis=0), sim_cal.std(axis=0) + 1e-12
    z_evl = (sim_evl - mu) / sd
    p_z = (1.0 + (z_evl[:, 1:] >= z_evl[:, :1]).sum(axis=1)) / (len(taus))
    results["zscore"] = uniformity_report(p_z, "zscore")

    # --- rank: 每个 pattern 用自身在校准语料上的经验 CDF ---
    r_evl = np.empty_like(sim_evl)
    for j in range(sim_evl.shape[1]):
        col = np.sort(sim_cal[:, j])
        r_evl[:, j] = np.searchsorted(col, sim_evl[:, j], side="left") / col.size
    p_r = (1.0 + (r_evl[:, 1:] >= r_evl[:, :1]).sum(axis=1)) / (len(taus))
    results["rank"] = uniformity_report(p_r, "rank")

    print("\n[note] KS 越小越好; mean_p 应接近 0.50")
    print(f"[note] pattern 固有亲和度离散度: sim mean sd across tau = {sim_cal.mean(axis=0).std():.4f}")

    OUT.mkdir(exist_ok=True)
    path = OUT / "pilot_calib_fix.json"
    path.write_text(json.dumps({
        "config": vars(args),
        "n_calib_steps": len(calib_steps), "n_eval_steps": len(eval_steps),
        "n_tau": len(taus),
        "pattern_affinity_sd": float(sim_cal.mean(axis=0).std()),
        "results": results}, indent=2))
    print(f"[done] {path}")


if __name__ == "__main__":
    main()
