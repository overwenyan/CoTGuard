"""SVRA gate G0-v2 (p2_design.md §5.0, pre-registered in commit 1eea37d before this was run).

Obligated = structural_anchor key01 ("compute every intermediate quantity twice");
non-obligated = the other 7 structural routes + clean. Full verifier = V1 ∧ V2 ∧ V4 ∧ V3_k.
  K1' fires if full-verifier false-reject on obligated honest-correct traces > 0.5 for every k
  K2' fires if, for every k passing K1', V3_k separation (obligated − non-obligated) < 0.3
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from svra_verifier import close, sig_twice, verify
from utility_check import gold_answer


def rows_of(fp, ks):
    out = []
    for r in (json.loads(l) for l in open(fp) if l.strip()):
        v = verify(r["text"], r["question"])
        g = gold_answer(r["gold"])
        row = {"correct": v["answer"] is not None and g is not None and close(v["answer"], g),
               "verified": v["verified"]}
        for k in ks:
            row[f"v3_{k}"] = sig_twice(r["text"], v["eqs"], k)[1]
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/attr_families")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    run = pathlib.Path(args.run_dir)
    ob = rows_of(run / "structural_anchor__key01.jsonl", args.ks)
    non = [r for i in range(8) if i != 1
           for r in rows_of(run / f"structural_anchor__key{i:02d}.jsonl", args.ks)]
    non += rows_of(run / "clean.jsonl", args.ks)

    m = lambda rs, f: float(np.mean([f(r) for r in rs])) if rs else float("nan")
    ob_c = [r for r in ob if r["correct"]]
    ob_w = [r for r in ob if not r["correct"]]
    print(f"obligated n={len(ob)} acc={m(ob, lambda r: r['correct']):.3f} | "
          f"non-obligated n={len(non)} acc={m(non, lambda r: r['correct']):.3f}")
    print(f"V1∧V2∧V4 alone: obligated correct {m(ob_c, lambda r: r['verified']):.3f}, "
          f"wrong {m(ob_w, lambda r: r['verified']):.3f}")
    print(f"{'k':>3}{'false_rej':>11}{'K1p':>6}{'V3 ob':>8}{'V3 non':>8}{'sep':>7}{'K2p':>6}"
          f"{'full|wrong':>12}{'full ob':>9}{'full non':>10}")
    res = {}
    for k in args.ks:
        full = lambda r: r["verified"] and r[f"v3_{k}"]
        fr = 1 - m(ob_c, full)
        v3o, v3n = m(ob, lambda r: r[f"v3_{k}"]), m(non, lambda r: r[f"v3_{k}"])
        sep = v3o - v3n
        k1, k2 = fr > 0.5, sep < 0.3
        res[k] = {"false_reject": fr, "K1p": k1, "v3_ob": v3o, "v3_non": v3n, "sep": sep, "K2p": k2,
                  "full_given_wrong": m(ob_w, full), "full_ob": m(ob, full), "full_non": m(non, full)}
        print(f"{k:>3}{fr:>11.3f}{'FIRE' if k1 else 'ok':>6}{v3o:>8.3f}{v3n:>8.3f}{sep:>7.3f}"
              f"{'FIRE' if k2 else 'ok':>6}{res[k]['full_given_wrong']:>12.3f}"
              f"{res[k]['full_ob']:>9.3f}{res[k]['full_non']:>10.3f}")
    k1_all = all(v["K1p"] for v in res.values())
    passing = [k for k, v in res.items() if not v["K1p"]]
    k2_all = all(res[k]["K2p"] for k in passing) if passing else True
    ok = [k for k in passing if not res[k]["K2p"]]
    print(f"\nK1' (fires iff every k false-rejects > 0.5): {'FIRES' if k1_all else 'no'}")
    print(f"K2' (fires iff every K1'-passing k separates < 0.3): {'FIRES' if k2_all else 'no'}")
    if k1_all or k2_all:
        print("=> G0-v2 FAILS")
        chosen = None
    else:
        chosen = max(ok, key=lambda k: res[k]["sep"] - res[k]["false_reject"])
        print(f"=> G0-v2 passes; frozen k = {chosen} "
              f"(false-reject {res[chosen]['false_reject']:.3f}, sep {res[chosen]['sep']:.3f}); "
              f"Tulu x GSM8K only until S20")
    (run / "svra_g0v2.json").write_text(json.dumps({"per_k": res, "chosen_k": chosen}, indent=1))


if __name__ == "__main__":
    main()
