"""SVRA gate G0 (p2_design.md §5.0): CPU feasibility on routed traces already on disk.

Kill criteria (pre-registered in p2_design.md before this script was run):
  K1 honest false-reject > 0.5      (verified rate among honest *correct* traces < 0.5)
  K2 no route besides compute-twice with compliance separation >= 0.3
  K3 cross-route node alignment < 0.2
Any one firing forces a redesign before GPU spend. Verdict lines are tied to the numbers.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
from collections import defaultdict

import numpy as np

from svra_verifier import (SIGNATURES, close, node_signature, sig_fewest_deps, verify)
from utility_check import gold_answer

ROUTE_OF_SIG = {  # which structural_anchor key each signature is meant to certify
    "twice": "compute every intermediate quantity twice",
    "add_first": "do all the additions before any of the multiplications",
    "largest_first": "settle the largest quantity first",
}


def load(fp):
    return [json.loads(l) for l in open(fp) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/attr_families")
    ap.add_argument("--space", default="structural_anchor")
    ap.add_argument("--twice-k", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = pathlib.Path(args.run_dir)
    spaces = json.loads((run / "key_spaces.json").read_text())
    patterns = [e["pattern"] for e in spaces[args.space]]
    arms = {f"key{i:02d}": load(run / f"{args.space}__key{i:02d}.jsonl") for i in range(len(patterns))}
    arms["clean"] = load(run / "clean.jsonl")

    res = {}
    for arm, recs in arms.items():
        rows = []
        for r in recs:
            v = verify(r["text"], r["question"])
            g = gold_answer(r["gold"])
            v["correct"] = v["answer"] is not None and g is not None and close(v["answer"], g)
            v["qid"] = r["qid"]
            v["text"] = r["text"]
            v["question"] = r["question"]
            rows.append(v)
        res[arm] = rows

    out = {"config": vars(args), "patterns": patterns}

    # ---- K1: verifier on honest traces ----
    print("== V1/V2/V4 on honest routed traces ==")
    print(f"{'arm':<7}{'acc':>6}{'has_eq':>8}{'v1':>6}{'v2':>6}{'v4':>6}{'verif':>7}"
          f"{'verif|corr':>11}{'verif|wrong':>12}  pattern")
    tab = {}
    for arm, rows in res.items():
        a = lambda k: float(np.mean([r[k] for r in rows]))
        corr = [r for r in rows if r["correct"]]
        wrong = [r for r in rows if not r["correct"]]
        vc = float(np.mean([r["verified"] for r in corr])) if corr else float("nan")
        vw = float(np.mean([r["verified"] for r in wrong])) if wrong else float("nan")
        tab[arm] = {"acc": a("correct"), "has_eq": a("has_eq"), "v1": a("v1"), "v2": a("v2"),
                    "v4": a("v4"), "verified": a("verified"), "verified_given_correct": vc,
                    "verified_given_wrong": vw, "n_correct": len(corr), "n_wrong": len(wrong)}
        pat = patterns[int(arm[3:])][:48] if arm.startswith("key") else "(no anchor)"
        t = tab[arm]
        print(f"{arm:<7}{t['acc']:>6.2f}{t['has_eq']:>8.2f}{t['v1']:>6.2f}{t['v2']:>6.2f}"
              f"{t['v4']:>6.2f}{t['verified']:>7.2f}{vc:>11.2f}{vw:>12.2f}  {pat}")
    routed = [r for arm, rows in res.items() if arm != "clean" for r in rows]
    corr = [r for r in routed if r["correct"]]
    wrong = [r for r in routed if not r["correct"]]
    verif_corr = float(np.mean([r["verified"] for r in corr]))
    verif_wrong = float(np.mean([r["verified"] for r in wrong]))
    false_reject = 1 - verif_corr
    print(f"\npooled routed: honest-correct verified = {verif_corr:.3f} "
          f"(false-reject {false_reject:.3f}, n={len(corr)}); "
          f"honest-wrong verified = {verif_wrong:.3f} (n={len(wrong)}) "
          f"<- wrong answers that pass = 'semantic' errors (Prop A limit)")
    fails = defaultdict(int)
    for r in corr:
        if not r["has_eq"]:
            fails["no_equation"] += 1
        else:
            for k in ("v1", "v2", "v4"):
                if not r[k]:
                    fails[k] += 1
    print("false-reject causes among honest-correct (a trace can fail several):",
          dict(sorted(fails.items(), key=lambda x: -x[1])))
    out["verifier"] = {"per_arm": tab, "honest_correct_verified": verif_corr,
                       "false_reject": false_reject, "honest_wrong_verified": verif_wrong,
                       "false_reject_causes": dict(fails)}

    # ---- K2: route compliance separation ----
    print("\n== V3 route signatures: pass rate by arm (applicable-only in brackets) ==")
    sigs = {**{f"twice_k{k}": (lambda k: (lambda t, e, q: SIGNATURES['twice'](t, e, k)))(k)
               for k in args.twice_k},
            "add_first": lambda t, e, q: SIGNATURES["add_first"](t, e),
            "largest_first": lambda t, e, q: SIGNATURES["largest_first"](t, e),
            "fewest_deps": lambda t, e, q: sig_fewest_deps(t, e, q)}
    target_of = {}
    for s in sigs:
        base = s.split("_k")[0] if s.startswith("twice") else s
        needle = ROUTE_OF_SIG.get(base, "fewest dependencies" if base == "fewest_deps" else None)
        target_of[s] = next((f"key{i:02d}" for i, p in enumerate(patterns) if needle and needle in p), None)
    arms_order = [a for a in res if a != "clean"] + ["clean"]
    print(f"{'signature':<15}{'target':>7}  " + "".join(f"{a:>8}" for a in arms_order)
          + f"{'own':>7}{'others':>8}{'sep':>7}{'sep_max':>8}")
    sep_tab = {}
    for s, fn in sigs.items():
        rates, app = {}, {}
        for a in arms_order:
            got = [fn(r["text"], r["eqs"], r["question"]) for r in res[a]]
            rates[a] = float(np.mean([p for _, p in got]))
            app[a] = float(np.mean([ap_ for ap_, _ in got]))
        tgt = target_of[s]
        others = [rates[a] for a in arms_order if a not in (tgt, "clean")]
        own = rates[tgt] if tgt else float("nan")
        sep = own - float(np.mean(others)) if tgt else float("nan")
        sep_max = own - max(others) if tgt else float("nan")
        sep_tab[s] = {"target": tgt, "rates": rates, "applicable": app, "own": own,
                      "others_mean": float(np.mean(others)), "sep": sep, "sep_max": sep_max}
        print(f"{s:<15}{str(tgt):>7}  " + "".join(f"{rates[a]:>8.2f}" for a in arms_order)
              + f"{own:>7.2f}{np.mean(others):>8.2f}{sep:>7.2f}{sep_max:>8.2f}")
        print(f"{'  applicable':<15}{'':>7}  " + "".join(f"{app[a]:>8.2f}" for a in arms_order))
    out["route_signatures"] = sep_tab

    # ---- K3: cross-route node alignment on the same problem ----
    by_q = defaultdict(dict)
    for a in arms_order[:-1]:
        for r in res[a]:
            by_q[r["qid"]][a] = r
    jac_nodes, jac_vals, cover_nodes, cover_vals = [], [], [], []
    for q, per in by_q.items():
        nodes = {a: {node_signature(e) for e in r["eqs"]} for a, r in per.items()}
        vals = {a: {round(e["result"], 4) for e in r["eqs"]} for a, r in per.items()}
        for x, y in itertools.combinations(per, 2):
            for S, acc in ((nodes, jac_nodes), (vals, jac_vals)):
                u = S[x] | S[y]
                if u:
                    acc.append(len(S[x] & S[y]) / len(u))
        for a in per:
            rest_n = set().union(*(nodes[b] for b in per if b != a))
            rest_v = set().union(*(vals[b] for b in per if b != a))
            if nodes[a]:
                cover_nodes.append(len(nodes[a] & rest_n) / len(nodes[a]))
            if vals[a]:
                cover_vals.append(len(vals[a] & rest_v) / len(vals[a]))
    align = {"pair_jaccard_nodes": float(np.mean(jac_nodes)), "pair_jaccard_values": float(np.mean(jac_vals)),
             "node_coverage": float(np.mean(cover_nodes)), "value_coverage": float(np.mean(cover_vals))}
    print("\n== cross-route alignment (same problem, 8 routes) ==")
    for k, v in align.items():
        print(f"  {k:<22}{v:.3f}")
    print("  (node = (sorted operands, ops, result); value = result only; "
          "coverage = share of an agent's nodes that some other route also computes)")
    out["alignment"] = align

    # ---- verdict, tied to the numbers ----
    k1 = false_reject > 0.5
    non_twice = {s: v for s, v in sep_tab.items() if not s.startswith("twice") and s != "fewest_deps"}
    best_other = max(non_twice.items(), key=lambda kv: kv[1]["sep"])
    k2 = best_other[1]["sep"] < 0.3
    k3 = align["node_coverage"] < 0.2
    print("\n== G0 verdict ==")
    print(f"K1 honest false-reject {false_reject:.3f} > 0.5 ? {'FIRES' if k1 else 'no'}")
    print(f"K2 best non-twice route separation {best_other[0]}={best_other[1]['sep']:.3f} < 0.3 ? "
          f"{'FIRES' if k2 else 'no'}")
    print(f"K3 node coverage {align['node_coverage']:.3f} < 0.2 ? {'FIRES' if k3 else 'no'}")
    if k1 or k2 or k3:
        print("=> G0 FAILS: redesign before any GPU spend (p2_design.md §5.0).")
    else:
        print("=> G0 passes on Tulu x GSM8K; still single-model until S20 (red line 5).")
    out["verdict"] = {"K1": k1, "K2": k2, "K3": k3, "best_non_twice": best_other[0]}

    fp = pathlib.Path(args.out) if args.out else run / "svra_g0.json"
    fp.write_text(json.dumps(out, indent=1, default=list))
    print(f"[done] {fp}")


if __name__ == "__main__":
    main()
