"""EXP-003: wrong-key 校准 vs 干净轨迹校准, 在分布漂移下的 Type I 控制.

动机: EXP-001/E1 显示干净轨迹校准在轻微分布漂移下 FPR 从 0.01 崩到 0.97.
SeqWM (2605.11036 Sec 4.3) 的 wrong-key 构造把可交换性建在密钥而非语料上,
理论上对漂移免疫. 本实验检验该性质是否随移植到语义相似度载体而保留.

漂移的含义: 候选轨迹整体基线相似度偏移(不同任务域/不同模型风格),
此时干净轨迹校准集不再与候选同分布, 而 wrong-key 零分布随候选一起漂移.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from detector_sim import (  # noqa: E402
    conformal_pvalues,
    stat_conf_sum,
    stat_conf_trgof,
    wrongkey_pvalues,
)

OUT = pathlib.Path(__file__).parent / "results"
ALPHA = 0.01
M_CALIB = 2000
N_KEYS = 199  # 199 个错误密钥 -> p 值粒度 1/200


def run(rng, n, n_trials, shifts, mu_signal, eps):
    calib = rng.standard_normal(M_CALIB)  # 干净轨迹校准集: 在 shift=0 下采集
    rows = []

    for shift in shifts:
        # --- H0: 候选轨迹无信号, 但整体基线漂移了 shift ---
        h0 = rng.standard_normal((n_trials, n)) + shift
        # --- H1: 同样漂移, 外加比例 eps 的步携带信号 ---
        carries = rng.random((n_trials, n)) < eps
        h1 = rng.standard_normal((n_trials, n)) + shift + mu_signal * carries

        for calib_name in ("clean_trace", "wrong_key"):
            if calib_name == "clean_trace":
                p0 = conformal_pvalues(h0, calib)
                p1 = conformal_pvalues(h1, calib)
            else:
                p0 = wrongkey_pvalues(rng, h0, N_KEYS, mu_shift=shift)
                p1 = wrongkey_pvalues(rng, h1, N_KEYS, mu_shift=shift)

            for stat_name, fn in (("conf_sum", stat_conf_sum), ("conf_trgof", stat_conf_trgof)):
                s0 = fn(None, p0)
                s1 = fn(None, p1)
                # 临界值必须在 shift=0 的假定下确定(实践中检测者不知道漂移量).
                # 用该校准方式在 shift=0 采集的零样本定阈值.
                if calib_name == "clean_trace":
                    ref = conformal_pvalues(rng.standard_normal((n_trials, n)), calib)
                else:
                    ref = wrongkey_pvalues(rng, rng.standard_normal((n_trials, n)), N_KEYS, 0.0)
                thr = np.quantile(fn(None, ref), 1 - ALPHA)

                rows.append({
                    "shift": float(shift), "calibration": calib_name, "detector": stat_name,
                    "empirical_fpr": float((s0 >= thr).mean()),
                    "power": float((s1 >= thr).mean()),
                })
    return rows


def main():
    rng = np.random.default_rng(7)
    n, n_trials = 200, 3000
    shifts = [0.0, 0.1, 0.2, 0.3, 0.5]
    mu_signal, eps = 1.5, 0.25

    rows = run(rng, n, n_trials, shifts, mu_signal, eps)

    print(f"n={n} trials={n_trials} alpha={ALPHA} n_keys={N_KEYS} "
          f"signal(mu={mu_signal}, eps={eps})\n")
    print(f"{'shift':>6} {'calibration':>12} {'detector':>11} {'FPR':>8} {'power':>8}")
    print("-" * 52)
    for r in rows:
        flag = "  <-- FPR 失控" if r["empirical_fpr"] > 3 * ALPHA else ""
        print(f"{r['shift']:>6.1f} {r['calibration']:>12} {r['detector']:>11} "
              f"{r['empirical_fpr']:>8.4f} {r['power']:>8.3f}{flag}")

    OUT.mkdir(exist_ok=True)
    path = OUT / "wrongkey.json"
    path.write_text(json.dumps({
        "config": {"n": n, "n_trials": n_trials, "alpha": ALPHA, "n_keys": N_KEYS,
                   "mu_signal": mu_signal, "eps": eps},
        "rows": rows}, indent=2))
    print(f"\n[done] {path}")


if __name__ == "__main__":
    main()
