"""EXP-004: 短轨迹区制下的检测规则比较.

动机 (EXP-002): 渐近最优的 HC/Tr-GoF 只在 n>=500 才胜出, 而真实 CoT 轨迹仅 5-50 步.

本实验分离两个被混淆的效应:
  (1) 合并规则本身在短 n 下的优劣  -> 用 exact p 值 (连续, 零分布已知)
  (2) wrong-key 校准的离散化代价   -> 用 K 个密钥的经验 p 值, 下界为 1/(K+1)

oracle LRT 在**原始分数**上计算 (而非由 p 值反解), 才是合法的 Neyman-Pearson 上界.
早期版本由 p 值反解导致强信号被截断在 Phi^{-1}(1-1/(K+1)), 使 oracle 被人为削弱,
出现实用规则超过 oracle 的假象.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy import stats

OUT = pathlib.Path(__file__).parent / "results"
ALPHA = 0.01
CHUNK = 500  # 控制 wrong-key 采样的峰值内存


def p_exact(scores):
    """零分布已知时的精确 p 值 (连续)."""
    return stats.norm.sf(scores)


def p_wrongkey(rng, scores, n_keys):
    """Wrong-key 经验 p 值; 下界 1/(n_keys+1), 分块以控内存."""
    out = np.empty_like(scores)
    for i in range(0, scores.shape[0], CHUNK):
        blk = scores[i:i + CHUNK]
        wrong = rng.standard_normal((n_keys, blk.shape[0], blk.shape[1]))
        out[i:i + CHUNK] = (1.0 + (wrong >= blk[None]).sum(axis=0)) / (n_keys + 1.0)
    return out


# ------------------------------------------------------------------ 合并规则

def s_fisher(p):
    return -np.log(np.clip(p, 1e-15, None)).sum(axis=1)


def s_stouffer(p):
    return stats.norm.isf(np.clip(p, 1e-15, 1 - 1e-15)).sum(axis=1)


def s_minp(p):
    return -p.min(axis=1)


def s_simes(p):
    n = p.shape[1]
    ps = np.sort(p, axis=1)
    return -(ps * n / np.arange(1, n + 1)[None, :]).min(axis=1)


def s_hc(p):
    n = p.shape[1]
    ps = np.sort(p, axis=1)
    i_n = np.arange(1, n + 1)[None, :] / n
    denom = np.sqrt(np.clip(ps * (1 - ps), 1e-15, None))
    return (np.sqrt(n) * (i_n - ps) / denom).max(axis=1)


RULES = {"fisher": s_fisher, "stouffer": s_stouffer, "minp": s_minp,
         "simes": s_simes, "hc": s_hc}


def s_oracle_lr(scores, eps, mu):
    """在原始分数上的 oracle 似然比 (合法的 NP 上界)."""
    return np.log((1 - eps) + eps * np.exp(mu * scores - mu**2 / 2)).sum(axis=1)


def evaluate(rng, n, eps, mu, n_trials, n_keys):
    h0 = rng.standard_normal((n_trials, n))
    h1 = rng.standard_normal((n_trials, n)) + mu * (rng.random((n_trials, n)) < eps)

    res = {}
    # oracle: 原始分数
    t0, t1 = s_oracle_lr(h0, eps, mu), s_oracle_lr(h1, eps, mu)
    res["oracle_lr"] = float((t1 >= np.quantile(t0, 1 - ALPHA)).mean())

    for mode in ("exact", "wrongkey"):
        if mode == "exact":
            p0, p1 = p_exact(h0), p_exact(h1)
        else:
            p0 = p_wrongkey(rng, h0, n_keys)
            p1 = p_wrongkey(rng, h1, n_keys)
        for name, fn in RULES.items():
            s0, s1 = fn(p0), fn(p1)
            thr = np.quantile(s0, 1 - ALPHA)
            res[f"{mode}:{name}"] = float((s1 >= thr).mean())
    return res


def main():
    rng = np.random.default_rng(11)
    n_trials, n_keys = 4000, 999
    n_grid = [5, 10, 20, 30, 50, 100, 200, 500]
    settings = [
        ("k2_strong", dict(k_signal=2, mu=2.5)),
        ("k2_mid", dict(k_signal=2, mu=1.5)),
        ("dense_weak", dict(eps=0.2, mu=1.0)),
    ]

    rows = []
    for sname, cfg in settings:
        print(f"\n=== {sname} ===  (exact p 值; oracle 为 NP 上界)")
        cols = list(RULES) + ["oracle_lr"]
        hdr = f"{'n':>5} {'eps':>7} " + " ".join(f"{c:>9}" for c in cols) + "   best/gap"
        print(hdr); print("-" * len(hdr))
        for n in n_grid:
            eps = cfg["eps"] if "eps" in cfg else min(cfg["k_signal"] / n, 1.0)
            mu = cfg["mu"]
            r = evaluate(rng, n, eps, mu, n_trials, n_keys)
            best = max(RULES, key=lambda k: r[f"exact:{k}"])
            gap = r["oracle_lr"] - r[f"exact:{best}"]
            print(f"{n:>5} {eps:>7.3f} "
                  + " ".join(f"{r[f'exact:{c}']:>9.3f}" for c in RULES)
                  + f" {r['oracle_lr']:>9.3f}   {best} (gap {gap:+.3f})")
            rows.append({"setting": sname, "n": n, "eps": float(eps), "mu": mu,
                         "results": r, "best_exact": best, "gap_to_oracle": float(gap)})

    # 离散化代价
    print("\n=== wrong-key 离散化代价 (exact - wrongkey, 同规则) ===")
    print(f"{'setting':>12} {'n':>5} " + " ".join(f"{c:>9}" for c in RULES))
    print("-" * 70)
    for row in rows:
        r = row["results"]
        print(f"{row['setting']:>12} {row['n']:>5} "
              + " ".join(f"{r[f'exact:{c}'] - r[f'wrongkey:{c}']:>9.3f}" for c in RULES))

    OUT.mkdir(exist_ok=True)
    path = OUT / "shorttrace.json"
    path.write_text(json.dumps({"config": {"alpha": ALPHA, "n_trials": n_trials,
                                           "n_keys": n_keys}, "rows": rows}, indent=2))
    print(f"\n[done] {path}")


if __name__ == "__main__":
    main()
