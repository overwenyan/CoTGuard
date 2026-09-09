"""Phase 1 runner: 三个实验对应 theory_draft.md 的三个论断.

E1  共形 p 值是否真的控住 Type I error (假设 A1), 以及校准集分布漂移时如何崩坏
E2  功效随跳数 N 衰减, 提取经验可检测视界 N*
E3  稀疏度 x 信号强度网格, 定位"求和型失效而 Tr-GoF 仍可检测"的区制

用法: python experiments/run_phase1.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from detector_sim import (  # noqa: E402
    DETECTORS,
    conformal_pvalues,
    empirical_fpr,
    empirical_power,
    hop_eps,
    null_threshold,
)

OUT = pathlib.Path(__file__).parent / "results"
ALPHA = 0.01
M_CALIB = 2000


def e1_calibration(rng, n, n_trials):
    """E1: Type I 控制, 含校准集分布漂移扫描.

    漂移 = 候选轨迹的 null 步与校准集不同分布 (违反可交换性).
    """
    calib = rng.standard_normal(M_CALIB)
    rows = []
    for shift in [0.0, 0.1, 0.2, 0.3, 0.5]:
        for name in DETECTORS:
            # 临界值在"无漂移"的校准假设下确定 (实践中只能这样)
            thr = null_threshold(rng, name, n, calib, ALPHA, n_trials)
            fpr = empirical_fpr(rng, name, n, calib, thr, n_trials, mu_null_shift=shift)
            rows.append({"shift": shift, "detector": name, "nominal_alpha": ALPHA,
                         "empirical_fpr": fpr, "n": n})
    return rows


def e2_hops(rng, n, n_trials, eps0, rho, mu, max_hops):
    """E2: 功效 vs 跳数, 提取 N* (功效首次跌破 0.8 的前一跳)."""
    calib = rng.standard_normal(M_CALIB)
    thr = {name: null_threshold(rng, name, n, calib, ALPHA, n_trials) for name in DETECTORS}
    rows = []
    for N in range(max_hops + 1):
        eps = hop_eps(eps0, rho, N)
        for name in DETECTORS:
            pw = empirical_power(rng, name, n, calib, thr[name], eps, mu, n_trials)
            rows.append({"hops": N, "eps": eps, "detector": name, "power": pw, "n": n})
    return rows


def e3_regime(rng, n, n_trials, mu_grid, eps_grid):
    """E3: 稀疏度 x 强度网格, 找 Tr-GoF 可检测而求和型失效的区域."""
    calib = rng.standard_normal(M_CALIB)
    thr = {name: null_threshold(rng, name, n, calib, ALPHA, n_trials) for name in DETECTORS}
    rows = []
    for eps in eps_grid:
        for mu in mu_grid:
            rec = {"eps": float(eps), "mu": float(mu), "n": n}
            for name in DETECTORS:
                rec[name] = empirical_power(rng, name, n, calib, thr[name], eps, mu, n_trials)
            rows.append(rec)
    return rows


def extract_horizon(e2_rows, threshold=0.8):
    """N* = 功效保持 >= threshold 的最大跳数."""
    out = {}
    for name in DETECTORS:
        pts = sorted([r for r in e2_rows if r["detector"] == name], key=lambda r: r["hops"])
        star = -1
        for r in pts:
            if r["power"] >= threshold:
                star = r["hops"]
            else:
                break
        out[name] = star
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="小规模快跑, 用于冒烟测试")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_trials = 400 if args.quick else 4000
    n = 100 if args.quick else 200
    OUT.mkdir(exist_ok=True)

    cfg = {"alpha": ALPHA, "m_calib": M_CALIB, "n_trials": n_trials, "n_steps": n,
           "seed": args.seed, "quick": args.quick}
    print(f"[cfg] {cfg}")

    print("[E1] Type I 控制与分布漂移 ...")
    e1 = e1_calibration(rng, n, n_trials)
    for r in e1:
        if r["detector"] == "conf_trgof":
            print(f"   shift={r['shift']:.1f}  trgof FPR={r['empirical_fpr']:.4f} (nominal {ALPHA})")

    print("[E2] 功效 vs 跳数 ...")
    eps0, rho, mu, max_hops = 0.6, 0.6, 1.2, 8
    e2 = e2_hops(rng, n, n_trials, eps0, rho, mu, max_hops)
    horizon = extract_horizon(e2)
    print(f"   经验 N* (功效>=0.8): {horizon}")

    print("[E3] 稀疏度 x 强度区制网格 ...")
    mu_grid = np.array([0.6, 0.9, 1.2, 1.8, 2.5])
    eps_grid = np.array([0.02, 0.05, 0.10, 0.20, 0.40])
    e3 = e3_regime(rng, n, n_trials, mu_grid, eps_grid)
    gap = [r for r in e3 if r["conf_trgof"] >= 0.8 > r["conf_sum"]]
    print(f"   Tr-GoF>=0.8 而 conf_sum<0.8 的格点数: {len(gap)}/{len(e3)}")

    payload = {"config": cfg, "e1_calibration": e1, "e2_hops": e2,
               "e2_horizon": horizon,
               "e2_params": {"eps0": eps0, "rho": rho, "mu": mu},
               "e3_regime": e3}
    path = OUT / ("phase1_quick.json" if args.quick else "phase1.json")
    path.write_text(json.dumps(payload, indent=2))
    print(f"[done] {path}")


if __name__ == "__main__":
    main()
