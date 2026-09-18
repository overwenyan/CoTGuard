"""M13 scoring (.pipeline/docs/m13_design.md, pre-registered 68cfc6a): does the 1.5B cell's result hold at 7B?

Same code path as M7 — score_m7.Cell builds the read-outs and runs T0/T1 — with these changes: the student
family is qwen7b; test seeds 10-12; T1's per-relative test uses n_ref = 3 (s5-s7, pre-registered); and, after
correction 1 (the first run's 9-score calibration could not reach alpha), T0 calibrates on all 10 reference
seeds as in M7.

Refuses to write results if the M7 sentinel moves (the M8 lesson: a scorer bug is invisible unless something
known is recomputed alongside).
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "relay"))
import score_m7  # noqa: E402
from score_m7 import Cell  # noqa: E402
from run_m7 import D, LINES, LINE_OF, ORDER, jl  # noqa: E402
from answer_v2 import correct_v2  # noqa: E402

FAM = "qwen7b"
# Correction 1 (m13_design.md, 744a6a2): T0's cross-line calibration uses all 10 reference seeds, as in M7
# (floor 1/31 = 0.032). T1's per-relative test keeps n_ref = 3; Cell takes keys(b, REF)[:n_ref], so the
# pre-registered s5-s7 are listed first and are exactly the three T1 uses.
REF, TEST = [5, 6, 7, 0, 1, 2, 3, 4, 8, 9], [10, 11, 12]
N_REF = 3
COLLAPSE = 0.6          # a pair collapses when the owner flags the relative's students at >= 2 of 3
G1_NEED, G2_NEED, G2_TPR = 6, 10, 0.9
SENTINEL = {"n_T0_ge_0.6": 8, "n_T1_le_0.2": 12}      # M7, gsm/qwen15, TF-IDF (paper/generated/s56_numbers.json)


def collapse_and_repair(cell, n_ref):
    t0 = cell.rates("T0")
    t1 = cell.rates("T1", n_ref=n_ref)
    ordered = [k for k, v in t1["fpr_rel"].items() if v is not None]
    n_collapse = sum(t0["fpr_rel"][k] is not None and t0["fpr_rel"][k] >= COLLAPSE for k in ordered)
    n_repair = sum(t1["fpr_rel"][k] <= 0.2 for k in ordered)
    tprs = [v for v in t1["tpr"].values() if v is not None]
    return {"n_pairs": len(ordered), "n_T0_ge_0.6": n_collapse, "n_T1_le_0.2": n_repair,
            "mean_T1_tpr": float(np.mean(tprs)) if tprs else None,
            "mean_T1_fpr": float(np.mean([t1["fpr_rel"][k] for k in ordered])) if ordered else None,
            "T0": t0, "T1": t1}


def sentinel():
    """Recompute the published 1.5B cell with the published seeds. If this moves, nothing else is trusted."""
    c = Cell("gsm", "qwen15")
    r = collapse_and_repair(c, n_ref=10)
    bad = {k: (r[k], v) for k, v in SENTINEL.items() if r[k] != v}
    return r, bad


def manipulation(cell):
    """M6/M7 rule, corrected (v2) extractor: void a teacher whose students drift from the base model."""
    base = jl(D("gsm") / f"probe_{FAM}_base.jsonl") or []
    if not base:
        return set(), {"base_missing": True}
    b_acc = float(np.mean([correct_v2(r["text"], r["gold"]) for r in base]))
    b_len = float(np.mean([len(r["text"]) for r in base]))
    void, info = set(), {"base_acc": b_acc, "base_len": b_len}
    for t in ORDER:
        rows = [cell.P[(t, s)] for s in REF + TEST if (t, s) in cell.P]
        if len(rows) < len(REF) + len(TEST):
            void.add(t); info[t] = {"n_students": len(rows), "missing": True}; continue
        accs = [float(np.mean([correct_v2(r["text"], r["gold"]) for r in st])) for st in rows]
        lens = [float(np.mean([len(r["text"]) for r in st])) for st in rows]
        t_len = float(np.mean([len(r["text"]) for r in jl(D("gsm") / f"teacher_{t}_ref.jsonl") or [{"text": ""}]]))
        ok = [(a >= b_acc - 0.03) or (t_len and 0.5 <= L / t_len <= 2.0) for a, L in zip(accs, lens)]
        info[t] = {"acc": accs, "frac_ok": float(np.mean(ok)), "mean_len": float(np.mean(lens)), "teacher_len": t_len}
        if np.mean(ok) < 0.8:
            void.add(t)
    return void, info


def main():
    sent, bad = sentinel()
    if bad:
        sys.exit(f"[m13] SENTINEL MOVED, refusing to write: {bad} (M7 gsm/qwen15 must reproduce)")
    print(f"[m13] sentinel ok: M7 gsm/qwen15 collapse {sent['n_T0_ge_0.6']}/12, T1 repair {sent['n_T1_le_0.2']}/12",
          flush=True)

    score_m7.REF_SEEDS = {"gsm": REF, "math": REF}      # T0 calibration: all 10; T1 references: first 3 (s5-s7)
    score_m7.TEST_SEEDS = TEST                          # pre-registered: 3 test students, seeds 10-12
    cell = Cell("gsm", FAM)

    void, minfo = manipulation(cell)
    lines_left = {ln: [t for t in ts if t not in void and t in cell.col] for ln, ts in LINES.items()}
    n_ok = sum(len(v) for v in lines_left.values())
    inconclusive = n_ok < 4 or all(len(v) < 2 for v in lines_left.values())

    out = {"void": sorted(void), "manipulation": minfo, "sentinel": {k: sent[k] for k in SENTINEL},
           "inconclusive": inconclusive}

    # Attainability guard (added 2026-09-17 AFTER the first scoring, which issued a false "G1 fails" verdict):
    # a conformal test with n calibration scores cannot reject below 1/(1+n). If that floor is above alpha the
    # gates are undefined, not failed. The pre-registration used 3 references per teacher for T0's cross-line
    # calibration as well as for T1's per-relative test; 3 teachers x 3 = 9 scores -> floor 0.1 > 0.05.
    n_cal = min(len([k for t in cell.avail if LINE_OF[t] != LINE_OF[a] for k in cell.keys(t, REF)]) for a in cell.avail)
    floor = 1 / (1 + n_cal)
    out["n_cal_min"], out["p_floor"] = n_cal, floor
    if floor > score_m7.ALPHA:
        out["verdict"] = (f"VOID by construction: T0 calibration has {n_cal} scores, minimum attainable p = "
                          f"{floor:.3f} > alpha = {score_m7.ALPHA}. G1 and G2 are undefined, not failed.")
        print(f"[m13] {out['verdict']}", flush=True)
        (D("gsm") / "m13_result.json").write_text(json.dumps(out, indent=1, default=float))
        return

    if not inconclusive:
        r = collapse_and_repair(cell, n_ref=N_REF)
        g1 = r["n_T0_ge_0.6"] >= G1_NEED
        g2 = r["n_T1_le_0.2"] >= G2_NEED and (r["mean_T1_tpr"] or 0) >= G2_TPR
        out.update({"result": r, "G1": g1, "G2": g2,
                    "verdict": "G1+G2 pass" if g1 and g2 else ("G1 fails — diagnosis may be scale-dependent"
                                                               if not g1 else "G1 passes, G2 fails — remedy does not carry")})
        print(f"[m13] pairs {r['n_pairs']} | T0 collapse {r['n_T0_ge_0.6']} (need {G1_NEED}) | "
              f"T1 repair {r['n_T1_le_0.2']} (need {G2_NEED}) | T1 tpr {r['mean_T1_tpr']} | "
              f"T1 mean fpr {r['mean_T1_fpr']}\n[m13] {out['verdict']}", flush=True)
    else:
        print(f"[m13] inconclusive: {n_ok} teachers survived the manipulation check, void={sorted(void)}", flush=True)

    (D("gsm") / "m13_result.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"[m13] wrote {D('gsm') / 'm13_result.json'}", flush=True)


if __name__ == "__main__":
    main()
