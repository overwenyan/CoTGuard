"""P1-a: 验证 wrong-key 校准的前提 (EXP-003 唯一未验证的建模假设).

前提: 在**无注入**的干净轨迹上, 真密钥 tau 与错误密钥 tau' 产生的逐步相似度
      应当同分布 —— 否则真/错密钥地位不对称, 经验 p 值失效.

若前提不成立(例如某些 tau 因用词更常见而系统性得分更高), 则 wrong-key 校准
在语义载体上不可直接移植, EXP-003 的漂移免疫性结论需要打折.

本脚本只需 embedding 模型 + 一批真实推理文本, 不需要 GPU 生成, 可在 CPU 节点跑.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from trigger import make_trigger, trigger_space_size, wrong_keys  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"


def split_steps(text: str) -> list[str]:
    """把推理文本切成步. GSM8K 参考解按换行分步."""
    parts = [s.strip() for s in text.replace("\n\n", "\n").split("\n")]
    return [p for p in parts if len(p) > 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--n-wrong", type=int, default=64)
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--true-key", default="patient-teacher-2026")
    args = ap.parse_args()

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    print(f"[info] trigger 空间大小 = {trigger_space_size()}")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.select(range(min(args.n_problems, len(ds))))

    # 干净轨迹: GSM8K 参考解, 未经任何 trigger 注入
    traces = [split_steps(r["answer"]) for r in ds]
    traces = [t for t in traces if len(t) >= 2]
    lens = np.array([len(t) for t in traces])
    print(f"[info] {len(traces)} 条干净轨迹; 步数 mean={lens.mean():.1f} "
          f"median={np.median(lens):.0f} p90={np.percentile(lens,90):.0f} max={lens.max()}")

    tau_true = make_trigger(args.true_key)
    taus_wrong = wrong_keys(args.true_key, args.n_wrong)
    print(f"[info] tau_true = {tau_true!r}")
    print(f"[info] {len(taus_wrong)} 个 wrong-key pattern, 例: {taus_wrong[0]!r}")

    model = SentenceTransformer(args.model)
    all_steps = [s for t in traces for s in t]
    print(f"[info] 编码 {len(all_steps)} 个推理步 ...")
    emb_steps = model.encode(all_steps, batch_size=128, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
    emb_tau = model.encode([tau_true] + taus_wrong, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)

    sims = emb_steps @ emb_tau.T          # (n_steps, 1 + n_wrong)
    s_true, s_wrong = sims[:, 0], sims[:, 1:]

    # --- 前提检验 1: 真 key 分数是否落在 wrong-key 分数的分布之内 ---
    # 在干净轨迹上, 真 key 不应系统性高于或低于错误密钥.
    rank = (s_wrong >= s_true[:, None]).sum(axis=1)          # 有多少 wrong 分数 >= 真分数
    pval = (1.0 + rank) / (s_wrong.shape[1] + 1.0)           # 共形 p 值
    print("\n=== 前提检验 1: 干净轨迹上真 key 的共形 p 值应 ~ U(0,1) ===")
    for a in (0.01, 0.05, 0.10, 0.25, 0.50):
        print(f"   P(p <= {a:.2f}) = {(pval <= a).mean():.4f}   (期望 {a:.2f})")
    print(f"   mean p = {pval.mean():.4f} (期望 0.50)")

    # --- 前提检验 2: 各 wrong-key pattern 之间的基线是否一致 ---
    per_tau_mean = s_wrong.mean(axis=0)
    print("\n=== 前提检验 2: 各 wrong-key pattern 的平均相似度离散度 ===")
    print(f"   真 key 平均相似度 = {s_true.mean():.4f}")
    print(f"   wrong-key 平均相似度: mean={per_tau_mean.mean():.4f} "
          f"sd={per_tau_mean.std():.4f} min={per_tau_mean.min():.4f} max={per_tau_mean.max():.4f}")
    z = (s_true.mean() - per_tau_mean.mean()) / (per_tau_mean.std() + 1e-12)
    print(f"   真 key 相对 wrong-key 分布的 z = {z:+.2f}  "
          f"(|z| 越小越好; |z|>2 说明前提不成立)")

    OUT.mkdir(exist_ok=True)
    path = OUT / "pilot_wrongkey_premise.json"
    path.write_text(json.dumps({
        "config": vars(args),
        "n_traces": len(traces), "n_steps": int(len(all_steps)),
        "trace_len": {"mean": float(lens.mean()), "median": float(np.median(lens)),
                      "p90": float(np.percentile(lens, 90)), "max": int(lens.max())},
        "tau_true": tau_true,
        "uniformity": {str(a): float((pval <= a).mean()) for a in (0.01, 0.05, 0.10, 0.25, 0.50)},
        "mean_p": float(pval.mean()),
        "s_true_mean": float(s_true.mean()),
        "wrong_tau_mean": {"mean": float(per_tau_mean.mean()), "sd": float(per_tau_mean.std()),
                           "min": float(per_tau_mean.min()), "max": float(per_tau_mean.max())},
        "z_true_vs_wrong": float(z),
    }, indent=2))
    print(f"\n[done] {path}")


if __name__ == "__main__":
    main()
