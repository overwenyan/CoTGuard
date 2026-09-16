"""Integrity audit (2026-09-16): how much did the v1 extractor bug move reported GSM8K numbers?"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "relay"))
from answer_v2 import correct_v2, extract_answer_v2  # noqa: E402
from utility_check import extract_answer, gold_answer  # noqa: E402


def jl(fp):
    try:
        return [json.loads(l) for l in open(fp) if l.strip()]
    except FileNotFoundError:
        return None


def correct_v1(t, g):
    x, y = extract_answer(t), gold_answer(g)
    return x is not None and y is not None and abs(x - y) < 1e-6


def acc(rows, f):
    return float(np.mean([f(r["text"], r["gold"]) for r in rows]))


def main():
    G = "data_m7/tulu_gsm"
    T6 = ["tulu_sft", "tulu_dpo", "tulu_rlvr", "olmoi_sft", "olmoi_dpo", "olmoi_final"]
    print("== teachers (M7 GSM8K POOL_TEST traces) ==")
    for t in T6 + ["zephyr_sft", "zephyr_dpo"]:
        r = jl(f"{G}/teacher_{t}_test.jsonl")
        if r:
            a1, a2 = acc(r, correct_v1), acc(r, correct_v2)
            print(f"  {t:<12} v1 {a1:.3f}  v2 {a2:.3f}  diff {a2 - a1:+.3f}")
    print("== students (M7 GSM8K test seeds 10-19, mean over students) ==")
    for fam in ["qwen15", "llama1b"]:
        b = jl(f"{G}/probe_{fam}_base.jsonl")
        print(f"  {fam} base: v1 {acc(b, correct_v1):.3f}  v2 {acc(b, correct_v2):.3f}")
        for t in T6:
            rs = [jl(f"{G}/probe_{fam}_grid_{t}_s{s}.jsonl") for s in range(10, 20)]
            rs = [x for x in rs if x]
            if rs:
                a1 = np.mean([acc(x, correct_v1) for x in rs]); a2 = np.mean([acc(x, correct_v2) for x in rs])
                print(f"    {t:<12} v1 {a1:.3f}  v2 {a2:.3f}  diff {a2 - a1:+.3f}")
    print("== rewrite checks: answer preserved (v1 -> v2) ==")
    for f, label in [("teacher_{o}_ad1.jsonl", "M9 AD1"), ("teacher_{o}_ad2.jsonl", "M9 AD2")]:
        for o in ["tulu_dpo", "tulu_rlvr", "olmoi_dpo", "olmoi_final"]:
            r = jl(f"{G}/" + f.format(o=o))
            if r:
                p1 = np.mean([(x := extract_answer(q["text"])) is not None and (y := extract_answer(q["orig_text"])) is not None
                              and abs(x - y) < 1e-6 for q in r])
                p2 = np.mean([(x := extract_answer_v2(q["text"])) is not None and (y := extract_answer_v2(q["orig_text"])) is not None
                              and abs(x - y) < 1e-6 for q in r])
                print(f"  {label} {o:<12} v1 {p1:.3f}  v2 {p2:.3f}  {'VOID->VALID' if p1 < 0.9 <= p2 else ('VALID->VOID' if p2 < 0.9 <= p1 else '')}")
    from run_m9b import ATTACKS, name
    for o, t in ATTACKS:
        r = jl(f"{G}/teacher_{name(o, t)}.jsonl")
        if r:
            p1 = np.mean([(x := extract_answer(q["text"])) is not None and (y := extract_answer(q["orig_text"])) is not None
                          and abs(x - y) < 1e-6 for q in r])
            p2 = np.mean([(x := extract_answer_v2(q["text"])) is not None and (y := extract_answer_v2(q["orig_text"])) is not None
                          and abs(x - y) < 1e-6 for q in r])
            a_orig = np.mean([correct_v2(q["orig_text"], q["gold"]) for q in r]); a_new = np.mean([correct_v2(q["text"], q["gold"]) for q in r])
            print(f"  M9b {o:>11}->{t:<12} v1 {p1:.3f}  v2 {p2:.3f}  | gold-acc orig {a_orig:.3f} new {a_new:.3f}"
                  f"  {'VOID->VALID' if p1 < 0.9 <= p2 else ('still VOID' if p2 < 0.9 else '')}")


if __name__ == "__main__":
    main()
