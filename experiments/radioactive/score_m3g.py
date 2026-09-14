"""M3 v7 bounded replication (m3_design.md v7, frozen 021f68f). Estimates with uncertainty; no gates.
E1 source ambiguity · E2 rewriting · E3 dilution · E4 utility."""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
from scipy.stats import beta

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("M3C_OUT", str(HERE / "data4"))
import score_m3d as s  # noqa: E402
from run_m3c import correct  # noqa: E402

D = s.OUT
E2_OP, E2_PRES = ["o14", "o17", "o20"], ["p03", "p12", "p13"]
R = {}


def exact(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return f"{k}/{n} [{lo:.2f}, {hi:.2f}]"


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    X, y, _ = s.teacher(s.E + ["clean"])
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
    c = LogisticRegression(max_iter=3000, C=4.0).fit(v.fit_transform(X), y)
    cache = {}

    def rows(fam, name, seed):
        fp = D / f"out_{fam}_{name}{'' if seed == 0 else f'_s{seed}'}.jsonl"
        return s.jl(fp)

    def P(fam, name, seed):
        key = (fam, name, seed)
        if key not in cache:
            r = rows(fam, name, seed)
            cache[key] = None if r is None else c.predict_proba(v.transform([x["text"] for x in r]))[:, : s.K]
        return cache[key]

    def test(fam, name, seed, k, vetoes, N=None):
        M = P(fam, name, seed)
        if M is None:
            return None
        idx = np.arange(len(M) if N is None else min(N, len(M)))
        V = [P(fam, vn, seed) for vn in vetoes]
        assert all(x is not None and len(x) >= len(idx) for x in V), f"missing veto for {fam}/{name}/s{seed}: {vetoes}"
        p = s.pval(M[idx].mean(0), k)
        inn = [s.pval(x[idx].mean(0), k) for x in V]
        return bool(p <= 0.05 and all(q > 0.05 for q in inn)), p

    # ---------------- E1 source ambiguity
    print("== E1 source ambiguity (owner test for key k) ==")
    e1 = {}
    for k, other in [("o12", "p07"), ("p07", "o12")]:
        e1[k] = {}
        for cond, name in [("own", f"raw_{k}"), ("imit", f"imit300_{k}"), ("neg", f"imit300_{other}")]:
            hits = []
            for fam in ["llama1b", "qwen15"]:
                for seed in [0, 1]:
                    veto = ["raw_clean", "base"]
                    t = test(fam, name, seed, k, veto)
                    if t is not None:
                        hits.append((fam, seed, t))
            n_ok = sum(t[2][0] for t in hits)
            e1[k][cond] = {"runs": [(f, sd, ok, p) for f, sd, (ok, p) in hits], "rate": exact(n_ok, len(hits))}
            print(f"  {k} {cond:<5} {exact(n_ok, len(hits))}   " + "  ".join(f"{f}/s{sd}:p={p:.3f}{'' if ok else '*'}" for f, sd, (ok, p) in hits))
    R["E1"] = e1

    # ---------------- E2 rewriting (Llama)
    print("\n== E2 rewriting (Llama-3.2-1B, 2 seeds) ==")
    e2 = {}
    for corpus in ["raw", "T1", "T2"]:
        e2[corpus] = {}
        for cat, ks in [("OP", E2_OP), ("PRES", E2_PRES)]:
            res = [(k, sd, test("llama1b", f"{corpus}_{k}", sd, k, [f"{corpus}_clean", "base"])) for k in ks for sd in [0, 1]]
            res = [(k, sd, t) for k, sd, t in res if t is not None]
            n_ok = sum(t[0] for _, _, t in res)
            e2[corpus][cat] = {"k": n_ok, "n": len(res), "wilson": wilson(n_ok, len(res)),
                               "runs": [(k, sd, t[0], t[1]) for k, sd, t in res]}
        a, b = e2[corpus]["OP"], e2[corpus]["PRES"]
        diff = (a["k"] / a["n"] - b["k"] / b["n"]) if a["n"] and b["n"] else np.nan
        qwen = {cat: sum(bool(json.loads((D / "m3d_result.json").read_text())["S1-A"]["lexical"][corpus].get(k, {}).get("ok"))
                         for k in ks) for cat, ks in [("OP", E2_OP), ("PRES", E2_PRES)]} if (D / "m3d_result.json").exists() else {}
        print(f"  {corpus:<4} OP {a['k']}/{a['n']} (Wilson {a['wilson'][0]:.2f}-{a['wilson'][1]:.2f})  PRES {b['k']}/{b['n']} "
              f"(Wilson {b['wilson'][0]:.2f}-{b['wilson'][1]:.2f})  OP-PRES {diff:+.2f}   | Qwen stage-1 same keys: OP {qwen.get('OP')}/3, PRES {qwen.get('PRES')}/3")
        e2[corpus]["diff"] = diff
    R["E2"] = e2

    # ---------------- E3 dilution (Llama)
    print("\n== E3 dilution (Llama-3.2-1B, 2 seeds, N=1319) ==")
    e3 = {}
    for frac in ["dil50", "dil10"]:
        res = [(k, sd, test("llama1b", f"{frac}_{k}", sd, k, ["dil0_clean", "base"], N=1319)) for k in ["o12", "o17", "p07"] for sd in [0, 1]]
        res = [(k, sd, t) for k, sd, t in res if t is not None]
        n_ok = sum(t[0] for _, _, t in res)
        e3[frac] = {"rate": exact(n_ok, len(res)), "runs": [(k, sd, t[0], t[1]) for k, sd, t in res]}
        print(f"  {frac}: {exact(n_ok, len(res))}   " + "  ".join(f"{k}/s{sd}:p={t[1]:.3f}{'' if t[0] else '*'}" for k, sd, t in res))
    R["E3"] = e3

    # ---------------- E4 utility
    print("\n== E4 utility (keyed - clean accuracy; bootstrap over test problems) ==")
    rng = np.random.default_rng(0)
    e4 = {}
    for fam, keys, seeds in [("llama1b", E2_OP + E2_PRES, [0, 1]), ("qwen15", E2_OP + E2_PRES, [0])]:
        for corpus in ["raw", "T1", "T2"]:
            for sd in seeds:
                cl = rows(fam, f"{corpus}_clean", sd)
                kr = [rows(fam, f"{corpus}_{k}", sd) for k in keys]
                if cl is None or any(x is None for x in kr):
                    continue
                C = np.array([correct("gsm", r["text"], r["gold"]) for r in cl], float)
                Kmat = np.array([[correct("gsm", r["text"], r["gold"]) for r in x] for x in kr], float).mean(0)
                d = Kmat - C
                boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                e4[f"{fam}/{corpus}/s{sd}"] = {"clean": C.mean(), "keyed": Kmat.mean(), "diff": d.mean(),
                                               "ci": (float(np.quantile(boots, .025)), float(np.quantile(boots, .975)))}
                print(f"  {fam:<8}{corpus:<4}s{sd}: clean {C.mean():.3f}  keyed {Kmat.mean():.3f}  diff {d.mean():+.3f} "
                      f"[{np.quantile(boots, .025):+.3f}, {np.quantile(boots, .975):+.3f}]")
    base = rows("llama1b", "base", 0)
    if base:
        print(f"  llama1b base acc {np.mean([correct('gsm', r['text'], r['gold']) for r in base]):.3f}")
    R["E4"] = e4
    (D / "m3g_result.json").write_text(json.dumps(R, indent=1, default=str))


if __name__ == "__main__":
    main()
