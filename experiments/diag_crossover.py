"""诊断: Tr-GoF 相对求和型的优势是否只在大 n 才出现.

理论上 Tr-GoF 在稀疏区制达到 q+2p=1 边界而求和型仅到 q+p=1/2, 但那是 n->infty 的陈述.
真实推理轨迹只有 5-50 步. 本脚本扫描 n, 定位交叉点(若存在).

稀疏区制按经典标定: eps = n^{-beta} (beta>1/2 为稀疏), mu = sqrt(2 r log n).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from detector_sim import (  # noqa: E402
    DETECTORS,
    empirical_power,
    null_threshold,
)

ALPHA = 0.01
M_CALIB = 5000


def main():
    rng = np.random.default_rng(0)
    n_trials = 2000
    print(f"{'n':>6} {'beta':>5} {'r':>5} {'eps':>8} {'mu':>6} "
          f"{'raw_sum':>8} {'conf_sum':>9} {'trgof':>8}  winner")
    print("-" * 78)
    for n in [50, 100, 500, 2000, 10000, 50000]:
        calib = rng.standard_normal(M_CALIB)
        thr = {k: null_threshold(rng, k, n, calib, ALPHA, n_trials) for k in DETECTORS}
        for beta, r in [(0.6, 0.4), (0.7, 0.5)]:
            eps = n ** (-beta)
            mu = np.sqrt(2 * r * np.log(n))
            pw = {k: empirical_power(rng, k, n, calib, thr[k], eps, mu, n_trials)
                  for k in DETECTORS}
            win = max(pw, key=pw.get)
            flag = "  <-- trgof wins" if win == "conf_trgof" and pw["conf_trgof"] > pw["conf_sum"] + 0.02 else ""
            print(f"{n:>6} {beta:>5.1f} {r:>5.1f} {eps:>8.5f} {mu:>6.2f} "
                  f"{pw['raw_sum']:>8.3f} {pw['conf_sum']:>9.3f} {pw['conf_trgof']:>8.3f}  {win}{flag}")


if __name__ == "__main__":
    main()
