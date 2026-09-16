"""M10 cross-vendor scoring (.pipeline/docs/m10_design.md). Eight teachers, three lines, GSM8K."""

from __future__ import annotations

import json
import os
import sys

os.environ["M10_ZEPHYR"] = "1"                      # extends run_m7's teacher list before anything imports it
sys.path.insert(0, __import__("os").path.dirname(__file__))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "relay"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from run_m7 import D, LINES, ORDER, REF_SEEDS, TEST_SEEDS, jl  # noqa: E402
import score_m8 as S8  # noqa: E402
import score_m7 as S7  # noqa: E402
from score_m7 import accuracy  # noqa: E402
from utility_check import extract_answer  # noqa: E402

FAMS = ["qwen15", "llama1b"]
ZL = LINES["zephyr"]
ZPAIRS = [(ZL[0], ZL[1]), (ZL[1], ZL[0])]


def manipulation():
    d = D("gsm")
    void, info = set(), {}
    for t in ZL:
        rows = jl(d / f"teacher_{t}_test.jsonl") or []
        if not rows:
            void.add(t); info[t] = {"missing": True}; continue
        ext = float(np.mean([extract_answer(r["text"]) is not None for r in rows]))
        tch = float(np.mean([len(r["text"]) for r in rows]))
        n = fails = 0
        for fam in FAMS:
            base = jl(d / f"probe_{fam}_base.jsonl")
            ba = accuracy("gsm", base).mean() if base else 0.0
            for s in REF_SEEDS["gsm"] + TEST_SEEDS:
                pr = jl(d / f"probe_{fam}_grid_{t}_s{s}.jsonl")
                n += 1
                if pr is None:
                    fails += 1; continue
                ratio = np.mean([len(r["text"]) for r in pr]) / tch
                fails += not (accuracy("gsm", pr).mean() >= ba - 0.03 or 0.5 <= ratio <= 2.0)
        info[t] = {"extractable": ext, "teacher_acc": float(accuracy("gsm", rows).mean()),
                   "mean_tokens": float(np.mean([r["n_tokens"] for r in rows])), "student_fails": fails, "n": n}
        if ext < 0.70 or fails > 0.2 * n:
            void.add(t)
    return void, info


def main():
    void, info = manipulation()
    print("[m10] manipulation: " + json.dumps(info) + f"\n[m10] void: {sorted(void)}", flush=True)
    out = {"manipulation": info, "void": sorted(void)}
    for fam in FAMS:
        C = S8.Cell("gsm", fam)
        chk = float(np.mean([C.rate(C.keys(a, TEST_SEEDS), a, "T1") for a in C.avail if a not in ZL]))
        print(f"\n[m10/{fam}] AllenAI T1 TPR with the 8-teacher read-out (M7 = 1.0): {chk}", flush=True)
        cell = {"allenai_T1_tpr_8way": chk, "zephyr": {}, "allenai_recomputed": {}}
        for a, b in ZPAIRS:
            if a not in C.col or b not in C.col:
                continue
            kb, ka = C.keys(b, TEST_SEEDS), C.keys(a, TEST_SEEDS)
            cell["zephyr"][f"{a}->{b}"] = {"T0_fpr": C.rate(kb, a, "T0"), "T1_fpr": C.rate(kb, a, "T1"),
                                           "T0_tpr": C.rate(ka, a, "T0"), "T1_tpr": C.rate(ka, a, "T1"),
                                           "T1_fpr_out": C.rate([k for t in C.avail if t not in ZL
                                                                 for k in C.keys(t, TEST_SEEDS)], a, "T1")}
        # R3: per-output AUC for the Zephyr pair (score_m7.Cell keeps per-output probabilities)
        C7 = S7.Cell("gsm", fam)
        auc = {}
        for a, b in [(ZL[0], ZL[1])]:
            if C7.has_pair(a, b):
                A = [C7.pair[(a, b)][k] if (a, b) in C7.pair else 1 - C7.pair[(b, a)][k]
                     for k in C7.keys(a, TEST_SEEDS)]
                B = [C7.pair[(a, b)][k] if (a, b) in C7.pair else 1 - C7.pair[(b, a)][k]
                     for k in C7.keys(b, TEST_SEEDS)]
                if A and B:
                    auc[f"{a}|{b}"] = float(roc_auc_score([1] * sum(map(len, A)) + [0] * sum(map(len, B)),
                                                          np.concatenate(A + B)))
        cell["per_output_auc"] = auc
        # AllenAI cells recomputed with the enriched calibration (reported)
        for a in [t for t in ORDER if t not in ZL]:
            rels = [x for x in LINES[[l for l, ts in LINES.items() if a in ts][0]] if x != a]
            cell["allenai_recomputed"][a] = {"tpr_T1": C.rate(C.keys(a, TEST_SEEDS), a, "T1"),
                                             "fpr_rel_T0": {b: C.rate(C.keys(b, TEST_SEEDS), a, "T0") for b in rels},
                                             "fpr_rel_T1": {b: C.rate(C.keys(b, TEST_SEEDS), a, "T1") for b in rels}}
        out[fam] = cell
        for k, v in cell["zephyr"].items():
            print(f"  [m10/{fam}] {k:<26} T0 FPR {v['T0_fpr']} -> T1 FPR {v['T1_fpr']} | TPR T0 {v['T0_tpr']} "
                  f"T1 {v['T1_tpr']} | T1 FPR(out-of-line) {v['T1_fpr_out']}", flush=True)
        print(f"  [m10/{fam}] per-output AUC (R3) {auc}")
        print(f"  [m10/{fam}] AllenAI recomputed (enriched calibration): " +
              json.dumps({a: {"T1_tpr": v["tpr_T1"], "T0_rel": v["fpr_rel_T0"], "T1_rel": v["fpr_rel_T1"]}
                          for a, v in cell["allenai_recomputed"].items()}), flush=True)
    # ---------------- predictions
    R = {}
    for fam in FAMS:
        z = out.get(fam, {}).get("zephyr", {})
        t0 = [v["T0_fpr"] for v in z.values() if v["T0_fpr"] is not None]
        t1 = [v["T1_fpr"] for v in z.values() if v["T1_fpr"] is not None]
        tp = [v["T1_tpr"] for v in z.values() if v["T1_tpr"] is not None]
        a3 = list(out.get(fam, {}).get("per_output_auc", {}).values())
        R[fam] = {"R1": bool(t0) and sum(v >= 0.6 for v in t0) >= 1,
                  "R2": bool(t1) and all(v <= 0.2 for v in t1) and bool(tp) and np.mean(tp) >= 0.8,
                  "R3": bool(a3) and all(v >= 0.9 for v in a3)}
    R1 = all(R[f]["R1"] for f in FAMS); R2 = all(R[f]["R2"] for f in FAMS)
    concl = ("R2 passes: the reference-aware remedy replicates on another vendor's ladder" if R2 else
             "KILL R2: the remedy is recipe-specific; scope T1 to AllenAI-style ladders")
    print("\n== PREDICTIONS ==\n" + json.dumps(R, default=str) +
          f"\nR1 (calibration failure replicates): {R1}\nR2 (remedy replicates): {R2}\n=> {concl}")
    out["verdict"] = {"per_family": R, "R1": R1, "R2": R2, "conclusion": concl}
    (D("gsm") / "m10_result.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
