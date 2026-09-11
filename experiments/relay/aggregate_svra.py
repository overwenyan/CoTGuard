"""SVRA aggregation vs vote baselines on honest agents (f = 0; p2_design.md v2 §3.3, P4/P5).

For each problem, draw random n-subsets of the 7 samples and compare:
  MV-obl    majority vote over obligated agents' answers
  MV-unobl  majority vote over unobligated agents' answers
  SVRA      plurality over *verified* obligated agents' answers; abstain if none verified
Abstention counts as wrong in `acc`; `coverage` and `acc_committed` are reported with it,
because G0-v2 showed honest pass rates near 0.5 (coverage is a first-class metric).
Adversarial cells need generated adversarial traces and are deliberately not simulated here.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, defaultdict

import numpy as np

from svra_verifier import close, sig_twice, verify
from utility_check import gold_answer


def load(rd: pathlib.Path, arm: str, k: int):
    by_q = defaultdict(list)
    for fp in sorted(rd.glob(f"{arm}__s*.jsonl")):
        for r in (json.loads(l) for l in open(fp) if l.strip()):
            v = verify(r["text"], r["question"])
            by_q[r["qid"]].append({"ans": v["answer"], "gold": gold_answer(r["gold"]),
                                   "ok": v["verified"] and sig_twice(r["text"], v["eqs"], k)[1]})
    if not by_q:
        raise SystemExit(f"no {arm}__s*.jsonl in {rd}")
    return by_q


def plurality(answers):
    answers = [round(a, 4) for a in answers if a is not None]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def correct(pred, gold):
    return pred is not None and gold is not None and close(pred, gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--ns", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rd = pathlib.Path(args.run_dir)
    obl, unobl = load(rd, "obligated", args.k), load(rd, "unobligated", args.k)
    rng = np.random.default_rng(args.seed)
    print(f"== {rd} (k={args.k}, {args.draws} draws/problem, {len(obl)} problems) ==")
    print(f"{'n':>3}{'single':>8}{'MV-unobl':>10}{'MV-obl':>8}{'SVRA':>7}{'coverage':>10}"
          f"{'SVRA|commit':>13}{'MV-obl|same':>13}")
    out = {}
    for n in args.ns:
        s = defaultdict(list)
        for q in obl:
            A, U = obl[q], unobl[q]
            for _ in range(args.draws):
                ia = rng.choice(len(A), n, replace=False)
                iu = rng.choice(len(U), n, replace=False)
                sub = [A[i] for i in ia]
                g = sub[0]["gold"]
                s["single"].append(correct(sub[0]["ans"], g))
                s["mv_unobl"].append(correct(plurality([U[i]["ans"] for i in iu]), g))
                mv = correct(plurality([a["ans"] for a in sub]), g)
                s["mv_obl"].append(mv)
                ver = [a["ans"] for a in sub if a["ok"]]
                if ver:
                    s["cov"].append(1)
                    sv = correct(plurality(ver), g)
                    s["svra"].append(sv)
                    s["svra_c"].append(sv)
                    s["mv_same"].append(mv)
                else:
                    s["cov"].append(0)
                    s["svra"].append(False)
        r = {k: float(np.mean(v)) for k, v in s.items()}
        out[n] = r
        print(f"{n:>3}{r['single']:>8.3f}{r['mv_unobl']:>10.3f}{r['mv_obl']:>8.3f}{r['svra']:>7.3f}"
              f"{r['cov']:>10.3f}{r['svra_c']:>13.3f}{r['mv_same']:>13.3f}")
    print("(SVRA|commit = SVRA accuracy on problems where it commits; MV-obl|same = MV on those same draws)")
    (rd / f"aggregate_f0_k{args.k}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
