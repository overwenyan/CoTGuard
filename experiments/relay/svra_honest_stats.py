"""CPU summary of SVRA honest-agent runs (output of gen_svra_honest.py).

Per run dir and arm: accuracy (all / untruncated), cap-hit rate, V1∧V2∧V4 pass, V3_k pass,
full-verifier pass on correct vs wrong traces, and the per-problem number of verified
agents out of n (the coverage risk flagged by G0-v2). Obligated-vs-unobligated V3 separation
is the within-generator re-test of G0-v2's K2'.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

from svra_verifier import close, sig_twice, verify
from utility_check import gold_answer


def score(fp, k):
    rows = []
    for r in (json.loads(l) for l in open(fp) if l.strip()):
        v = verify(r["text"], r["question"])
        g = gold_answer(r["gold"])
        rows.append({"qid": r["qid"], "correct": v["answer"] is not None and g is not None and close(v["answer"], g),
                     "hit_cap": r.get("hit_cap", False), "empty": not r["text"].strip(),
                     "v124": v["verified"], "v3": sig_twice(r["text"], v["eqs"], k)[1]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--k", type=int, default=1, help="V3 threshold frozen by G0-v2")
    args = ap.parse_args()
    report = {}
    for rd in map(pathlib.Path, args.run_dirs):
        arms = defaultdict(list)
        for fp in sorted(rd.glob("*__s*.jsonl")):
            arms[fp.name.split("__")[0]] += score(fp, args.k)
        print(f"\n== {rd} (k={args.k}) ==")
        print(f"{'arm':<12}{'n':>6}{'acc':>7}{'acc_unc':>9}{'cap':>6}{'empty':>7}{'v124':>7}{'v3':>6}"
              f"{'full|c':>8}{'full|w':>8}{'false_rej':>11}")
        rep = {}
        for arm, rows in arms.items():
            m = lambda f, rs=rows: float(np.mean([f(r) for r in rs])) if rs else float("nan")
            c = [r for r in rows if r["correct"]]
            w = [r for r in rows if not r["correct"]]
            unc = [r for r in rows if not r["hit_cap"]]
            full = lambda r: r["v124"] and r["v3"]
            rep[arm] = {"n": len(rows), "acc": m(lambda r: r["correct"]),
                        "acc_untruncated": m(lambda r: r["correct"], unc), "cap": m(lambda r: r["hit_cap"]),
                        "empty": m(lambda r: r["empty"]), "v124": m(lambda r: r["v124"]),
                        "v3": m(lambda r: r["v3"]), "full_given_correct": m(full, c),
                        "full_given_wrong": m(full, w)}
            rep[arm]["false_reject"] = 1 - rep[arm]["full_given_correct"]
            by_q = defaultdict(list)
            for r in rows:
                by_q[r["qid"]].append(full(r))
            npass = np.array([sum(v) for v in by_q.values()])
            nper = max(len(v) for v in by_q.values())
            rep[arm]["verified_per_problem"] = {"n_agents": nper, "mean": float(npass.mean()),
                                                "p_zero": float(np.mean(npass == 0))}
            x = rep[arm]
            print(f"{arm:<12}{x['n']:>6}{x['acc']:>7.3f}{x['acc_untruncated']:>9.3f}{x['cap']:>6.2f}"
                  f"{x['empty']:>7.2f}{x['v124']:>7.2f}{x['v3']:>6.2f}{x['full_given_correct']:>8.3f}"
                  f"{x['full_given_wrong']:>8.3f}{x['false_reject']:>11.3f}")
            print(f"{'':<12}verified agents per problem (of {nper}): mean {npass.mean():.2f}, "
                  f"P(none) {np.mean(npass == 0):.2f}")
        if {"obligated", "unobligated"} <= rep.keys():
            sep = rep["obligated"]["v3"] - rep["unobligated"]["v3"]
            fr = rep["obligated"]["false_reject"]
            rep["_gate"] = {"v3_separation": sep, "false_reject": fr,
                            "K1p_fires": fr > 0.5, "K2p_fires": sep < 0.3}
            print(f"G0-v2 re-test: false-reject {fr:.3f} ({'FIRES' if fr > 0.5 else 'ok'}), "
                  f"V3 separation {sep:.3f} ({'FIRES' if sep < 0.3 else 'ok'})")
        report[str(rd)] = rep
        (rd / f"svra_honest_stats_k{args.k}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
