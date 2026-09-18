#!/usr/bin/env python3
"""Appendix X, table X.1: every GSM8K accuracy / answer-check number under both extractors (v1 buggy, v2 corrected).

Same computation as experiments/radioactive/audit_extractor.py, written to a table instead of stdout, plus the
summary ranges the body text quotes (§5.1). The body must quote these ranges, never a hand-typed version of them.

    python paper/make_appendix_x.py  ->  paper/generated/appx_extractor.md, appx_extractor.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAD = ROOT.parent / "experiments" / "radioactive"
os.chdir(RAD)
os.environ.setdefault("M7_OUT", str(RAD / "data_m7")); os.environ.setdefault("M3C_OUT", str(RAD / "data_m7"))
sys.path.insert(0, str(RAD)); sys.path.insert(0, str(RAD.parent / "relay"))
from audit_extractor import acc, correct_v1, jl  # noqa: E402
from answer_v2 import correct_v2, extract_answer_v2  # noqa: E402
from utility_check import extract_answer  # noqa: E402
from run_m9b import ATTACKS, name  # noqa: E402

G = "data_m7/tulu_gsm"
T6 = ["tulu_sft", "tulu_dpo", "tulu_rlvr", "olmoi_sft", "olmoi_dpo", "olmoi_final"]


def preserved(rows, ext):
    return float(np.mean([(x := ext(q["text"])) is not None and (y := ext(q["orig_text"])) is not None
                          and abs(x - y) < 1e-6 for q in rows]))


def main():
    out = {"teachers": {}, "students": {}, "base": {}, "rewrites": {}}
    for t in T6 + ["zephyr_sft", "zephyr_dpo"]:
        r = jl(f"{G}/teacher_{t}_test.jsonl")
        if r:
            out["teachers"][t] = [acc(r, correct_v1), acc(r, correct_v2)]
    for fam in ["qwen15", "llama1b"]:
        b = jl(f"{G}/probe_{fam}_base.jsonl")
        out["base"][fam] = [acc(b, correct_v1), acc(b, correct_v2)]
        for t in T6:
            rs = [x for x in (jl(f"{G}/probe_{fam}_grid_{t}_s{s}.jsonl") for s in range(10, 20)) if x]
            if rs:
                out["students"][f"{fam}|{t}"] = [float(np.mean([acc(x, correct_v1) for x in rs])),
                                                 float(np.mean([acc(x, correct_v2) for x in rs]))]
    for f, label in [("teacher_{o}_ad1.jsonl", "M9 paraphrase"), ("teacher_{o}_ad2.jsonl", "M9 imitation")]:
        for o in ["tulu_dpo", "tulu_rlvr", "olmoi_dpo", "olmoi_final"]:
            r = jl(f"{G}/" + f.format(o=o))
            if r:
                out["rewrites"][f"{label}|{o}"] = [preserved(r, extract_answer), preserved(r, extract_answer_v2)]
    for o, t in ATTACKS:
        r = jl(f"{G}/teacher_{name(o, t)}.jsonl")
        if r:
            out["rewrites"][f"M9b imitation|{o}->{t}"] = [preserved(r, extract_answer), preserved(r, extract_answer_v2)]

    dt = [v2 - v1 for v1, v2 in out["teachers"].values()]
    ds = [v2 - v1 for v1, v2 in out["students"].values()]
    flips = {k: ("void→valid" if v1 < 0.9 <= v2 else "still void" if v2 < 0.9 else "valid both")
             for k, (v1, v2) in out["rewrites"].items()}
    out["summary"] = {"teacher_shift": [min(dt), max(dt)], "student_mean_shift": [min(ds), max(ds)],
                      "rewrites_void_v1": sum(v1 < 0.9 for v1, _ in out["rewrites"].values()),
                      "rewrites_void_v2": sum(v2 < 0.9 for _, v2 in out["rewrites"].values()),
                      "n_rewrites": len(out["rewrites"])}

    L = ["**Table X.1** — GSM8K numbers under the v1 extractor (skipped any number followed by a period) and the "
         "corrected v2. No attribution statistic consumes the extractor.", "",
         "| Quantity | v1 | v2 | shift |", "|---|---|---|---|"]
    for t, (a, b) in out["teachers"].items():
        L.append(f"| teacher {t} (POOL_TEST traces) | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    for fam, (a, b) in out["base"].items():
        L.append(f"| {fam} base model | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    for k, (a, b) in out["students"].items():
        fam, t = k.split("|")
        L.append(f"| {fam} students of {t} (mean of 10) | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    L += ["", "**Table X.2** — answer-preservation checks on rewritten corpora (valid if ≥ 0.90).", "",
          "| Rewrite | v1 | v2 | status |", "|---|---|---|---|"]
    for k, (a, b) in out["rewrites"].items():
        L.append(f"| {k.replace('|', ' · ')} | {a:.3f} | {b:.3f} | {flips[k]} |")
    s = out["summary"]
    L += ["", f"_Shifts: teachers {s['teacher_shift'][0]:+.3f} to {s['teacher_shift'][1]:+.3f}; student means "
              f"{s['student_mean_shift'][0]:+.3f} to {s['student_mean_shift'][1]:+.3f}. Rewritten corpora void: "
              f"{s['rewrites_void_v1']} of {s['n_rewrites']} under v1, {s['rewrites_void_v2']} of {s['n_rewrites']} under v2._"]
    gen = ROOT / "generated"; gen.mkdir(exist_ok=True)
    (gen / "appx_extractor.md").write_text("\n".join(L) + "\n")
    (gen / "appx_extractor.json").write_text(json.dumps(out, indent=1))
    print("\n".join(L[-1:]))


if __name__ == "__main__":
    main()
