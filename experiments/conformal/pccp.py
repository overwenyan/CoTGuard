"""M1 analysis and pre-registered gate (.pipeline/docs/m1_design.md §5, commit 0d08f65).

Methods, per (model, dataset) cell (Mondrian), random 50/50 calibration/test splits:
  CLEAN    calibrate on solo scores, deploy on social scores (Hu & Su's setting)
  ORACLE   calibrate on the same condition's social scores (needs pressure labels)
  SOLO     solo set C_solo(alpha): ignore peers
  HONCAL   social set calibrated under honest peers, C_H(alpha): not robust
  OURS_e   C_H(alpha) ∪ C_solo(alpha+e)
  OURSD_e  OURS_e, falling back to OURS_0 when TV(p_solo, p_soc) exceeds its honest-peer
           conformal (1-beta) quantile
Act iff |C| = 1; escalate otherwise (an empty set also escalates).
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

DATA = pathlib.Path(__file__).parent / "data"
CONDS = ["UW", "UW-A", "UW-P", "MIX", "UC", "UC-A", "HON"]
EPS = [0.0, 0.02, 0.05]


def qhat(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(scores)[k - 1]


def sets(p, q):
    return (1 - p) <= q + 1e-12                          # boolean [n, K]


def metrics(C, y):
    n = len(y)
    size = C.sum(1)
    cov = C[np.arange(n), y]
    act = size == 1
    wrong_act = act & ~cov
    return {"cov": cov.mean(), "size": size.mean(), "esc": (~act).mean(), "act_wrong": wrong_act.mean(),
            "acc_acted": (act & cov).sum() / max(act.sum(), 1)}


def evaluate_cell(solo, social, y, alphas, n_splits, beta, rng):
    """solo [n,K]; social {cond: [n,K]}; y [n]. Returns {(alpha, method, cond, metric): [values]}."""
    n = len(y)
    out = defaultdict(list)
    idx_all = np.arange(n)
    s_solo = 1 - solo[idx_all, y]
    s_soc = {c: 1 - social[c][idx_all, y] for c in social}
    tv = {c: 0.5 * np.abs(solo - social[c]).sum(1) for c in social}
    for _ in range(n_splits):
        perm = rng.permutation(n)
        cal, te = perm[: n // 2], perm[n // 2:]
        yt = y[te]
        for a in alphas:
            q_solo = {e: qhat(s_solo[cal], a + e) for e in EPS}
            q_H = qhat(s_soc["HON"][cal], a)
            t_det = qhat(tv["HON"][cal], beta)
            Csolo = {e: sets(solo[te], q_solo[e]) for e in EPS}
            # targeted band: test items just above the clean threshold in solo confidence
            tau = 1 - q_solo[0.0]
            conf_cal = np.sort(1 - s_solo[cal])
            F = lambda v: np.searchsorted(conf_cal, v, side="right") / len(conf_cal)
            conf_te = 1 - s_solo[te]
            band = (conf_te >= tau) & (F(conf_te) <= F(tau) + 0.10)
            for c in social:
                P = social[c][te]
                C_clean = sets(P, q_solo[0.0])
                C_or = sets(P, qhat(s_soc[c][cal], a))
                C_H = sets(P, q_H)
                res = {"CLEAN": C_clean, "ORACLE": C_or, "SOLO": Csolo[0.0], "HONCAL": C_H}
                flag = tv[c][te] > t_det
                for e in EPS:
                    res[f"OURS_{e}"] = C_H | Csolo[e]
                    res[f"OURSD_{e}"] = np.where(flag[:, None], C_H | Csolo[0.0], C_H | Csolo[e])
                for mname, C in res.items():
                    for k, v in metrics(C, yt).items():
                        out[(a, mname, c, k)].append(v)
                    if band.any():
                        out[(a, mname, c, "band_cov")].append(C[np.arange(len(te)), yt][band].mean())
                out[(a, "DETECT", c, "flag_rate")].append(flag.mean())
    return out


def load_cells(models):
    cells = {}
    for m in models:
        solo = {r["id"]: r for r in map(json.loads, open(DATA / f"solo_{m}.jsonl"))}
        soc = defaultdict(dict)
        for r in map(json.loads, open(DATA / f"social_{m}.jsonl")):
            soc[r["cond"]][r["id"]] = r
        for d in ("arc", "tqa"):
            ids = [i for i, r in solo.items() if r["dataset"] == d]
            cells[(m, d)] = (np.array([solo[i]["probs"] for i in ids]),
                             {c: np.array([soc[c][i]["probs"] for i in ids]) for c in CONDS},
                             np.array([solo[i]["gold"] for i in ids]),
                             np.array([solo[i]["letter_mass"] for i in ids]))
    return cells


def pooled(results, weights):
    keys = set().union(*[r.keys() for r in results.values()])
    tot = sum(weights.values())
    return {k: sum(np.mean(results[c][k]) * weights[c] for c in results if k in results[c]) / tot for k in keys}


def gate(P, alpha=0.10):
    g = lambda m, c, k: P.get((alpha, m, c, k), np.nan)
    r0 = g("CLEAN", "UW", "cov") <= 1 - alpha - 0.08
    ga_cov = all(g("OURS_0.05", c, "cov") >= 1 - alpha - 0.05 - 0.01 for c in CONDS) and \
        g("OURS_0.05", "HON", "cov") >= 1 - alpha - 0.01
    ga_act = all(g("OURS_0.05", c, "act_wrong") <= alpha + 0.05 + 0.01 for c in CONDS)
    gb = (g("OURS_0.0", "UW", "esc") <= g("ORACLE", "UW", "esc") - 0.10) and \
        (g("OURS_0.0", "UW", "cov") >= g("ORACLE", "UW", "cov") - 0.01)
    gc_eps = [e for e in (0.02, 0.05) if g(f"OURS_{e}", "HON", "esc") <= g("SOLO", "HON", "esc") - 0.05
              and g(f"OURS_{e}", "HON", "acc_acted") >= g("SOLO", "HON", "acc_acted") - 0.01]
    return {"R0": bool(r0), "G-a": bool(ga_cov and ga_act), "G-b": bool(gb), "G-c": bool(gc_eps),
            "G-c_eps": gc_eps}


def report(P, alpha, methods, label):
    print(f"\n== {label}, alpha={alpha} ==")
    for k in ("cov", "esc", "act_wrong", "acc_acted", "band_cov"):
        print(f"-- {k}")
        print(f"{'method':<13}" + "".join(f"{c:>8}" for c in CONDS))
        for m in methods:
            print(f"{m:<13}" + "".join(f"{P.get((alpha, m, c, k), np.nan):>8.3f}" for c in CONDS))
    print("-- detector flag rate (honest-peer FPR target = beta)")
    print(f"{'DETECT':<13}" + "".join(f"{P.get((alpha, 'DETECT', c, 'flag_rate'), np.nan):>8.3f}" for c in CONDS))


def selftest():
    """Synthetic: honest social = solo sharpened; UW fully conformist. Guarantees must hold."""
    rng = np.random.default_rng(0)
    n, K = 1200, 4
    y = rng.integers(0, K, n)
    logits = rng.normal(0, 1, (n, K))
    logits[np.arange(n), y] += rng.normal(1.5, 1.0, n)
    solo = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    hon = solo ** 1.5
    hon /= hon.sum(1, keepdims=True)
    uw_lab = (y + rng.integers(1, K, n)) % K
    uw = np.full((n, K), 0.01)
    uw[np.arange(n), uw_lab] = 0.97
    social = {c: hon for c in CONDS}
    social["UW"] = uw
    out = evaluate_cell(solo, social, y, [0.10], 200, 0.05, rng)
    P = {k: np.mean(v) for k, v in out.items()}
    clean_uw, ours_uw = P[(0.10, "CLEAN", "UW", "cov")], P[(0.10, "OURS_0.0", "UW", "cov")]
    ours5_uw = P[(0.10, "OURS_0.05", "UW", "cov")]
    hon_cov = P[(0.10, "OURS_0.05", "HON", "cov")]
    print(f"selftest: CLEAN cov under conformist UW {clean_uw:.3f} (must collapse); "
          f"OURS_0 {ours_uw:.3f} (>= ~0.90); OURS_0.05 {ours5_uw:.3f} (>= ~0.85); OURS_0.05 HON {hon_cov:.3f} (>= ~0.90)")
    assert clean_uw < 0.2 and ours_uw >= 0.89 and ours5_uw >= 0.84 and hon_cov >= 0.89
    assert P[(0.10, "OURS_0.0", "UW", "act_wrong")] <= 0.11
    esc_or = P[(0.10, "ORACLE", "UW", "esc")]
    print(f"selftest: ORACLE escalation under conformist UW {esc_or:.3f} (Prop 2: ~1.0)")
    assert esc_or > 0.9
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen25,llama31,mistral,gemma2")
    ap.add_argument("--splits", type=int, default=2000)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    models = a.models.split(",")
    cells = load_cells(models)
    rng = np.random.default_rng(0)
    res, w = {}, {}
    print("== data sanity: letter mass (median / p05) and solo accuracy per cell ==")
    for key, (solo, social, y, lm) in cells.items():
        print(f"  {key[0]:<8}{key[1]:<5} n={len(y):<5} letter_mass {np.median(lm):.3f}/{np.percentile(lm, 5):.3f} "
              f"solo_acc {np.mean(solo.argmax(1) == y):.3f} UW_acc {np.mean(social['UW'].argmax(1) == y):.3f} "
              f"HON_acc {np.mean(social['HON'].argmax(1) == y):.3f}")
        res[key] = evaluate_cell(solo, social, y, [0.05, 0.10], a.splits, a.beta, rng)
        w[key] = len(y)
    P = pooled(res, w)
    methods = ["CLEAN", "ORACLE", "SOLO", "HONCAL"] + [f"OURS_{e}" for e in EPS] + [f"OURSD_{e}" for e in EPS]
    for alpha in (0.10, 0.05):
        report(P, alpha, methods, f"pooled over {', '.join(models)} x {{arc, tqa}}")
    print("\n== per-model gate quantities (alpha=0.10) ==")
    per_model = {}
    for m in models:
        Pm = pooled({k: v for k, v in res.items() if k[0] == m}, {k: v for k, v in w.items() if k[0] == m})
        per_model[m] = gate(Pm)
        print(f"  {m:<8} CLEAN-UW cov {Pm[(0.10, 'CLEAN', 'UW', 'cov')]:.3f} | OURS_0 UW cov "
              f"{Pm[(0.10, 'OURS_0.0', 'UW', 'cov')]:.3f} esc {Pm[(0.10, 'OURS_0.0', 'UW', 'esc')]:.3f} vs ORACLE esc "
              f"{Pm[(0.10, 'ORACLE', 'UW', 'esc')]:.3f} | HON esc SOLO {Pm[(0.10, 'SOLO', 'HON', 'esc')]:.3f} "
              f"OURS_.02 {Pm[(0.10, 'OURS_0.02', 'HON', 'esc')]:.3f} OURS_.05 {Pm[(0.10, 'OURS_0.05', 'HON', 'esc')]:.3f} "
              f"| gate {per_model[m]}")
    G = gate(P)
    print(f"\n== PRE-REGISTERED GATE (pooled, alpha=0.10) ==\n{G}")
    cont = G["R0"] and G["G-a"] and G["G-b"] and G["G-c"]
    why = ("problem does not reproduce (R0)" if not G["R0"] else "validity failed: implementation bug (G-a)"
           if not G["G-a"] else "not better than the oracle fix (G-b)" if not G["G-b"]
           else "peers do not help: ignoring them is optimal (G-c) -> analysis result, move to M2" if not G["G-c"]
           else "")
    print("=> M1 CONTINUES to a paper" if cont else f"=> M1 STOPS: {why}")
    (DATA / "pccp_result.json").write_text(json.dumps(
        {"pooled": {"|".join(map(str, k)): v for k, v in P.items()}, "gate": G, "per_model_gate": per_model,
         "continue": cont}, indent=1))


if __name__ == "__main__":
    main()
