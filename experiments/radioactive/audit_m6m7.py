"""Integrity audit part 3 (2026-09-16): M6 / M7 manipulation voids and M7 capability signs under v1 vs v2."""

from __future__ import annotations

import itertools
import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "relay"))
from answer_v2 import correct_v2  # noqa: E402
from utility_check import extract_answer, gold_answer  # noqa: E402


def jl(fp):
    try:
        return [json.loads(l) for l in open(fp) if l.strip()]
    except FileNotFoundError:
        return None


def c1(t, g):
    x, y = extract_answer(t), gold_answer(g)
    return x is not None and y is not None and abs(x - y) < 1e-6


def acc(rows, f):
    return float(np.mean([f(r["text"], r["gold"]) for r in rows]))


def m6():
    print("== M6 manipulation (acc >= base OR length ratio in [0.67, 1.5]; void if > 2 of 10 fail) ==")
    pr = json.load(open("data_m6/problems.json")); pool = set(pr["pool"])
    T9 = ["tulu_sft", "tulu_dpo", "tulu_rlvr", "olmoi_sft", "olmoi_dpo", "olmoi_final", "olmot_sft", "olmot_dpo", "olmot_final"]
    for lab, f in [("v1", c1), ("v2", correct_v2)]:
        base = {fam: acc(jl(f"data_m6/tulu_gsm/probe_{fam}_base.jsonl"), f) for fam in ["qwen15", "llama1b"]}
        void = []
        for t in T9:
            tch = np.mean([len(r["text"]) for r in jl(f"data_m6/teacher_{t}.jsonl") if r["qid"] in pool])
            fails = 0
            for fam in base:
                for s in range(5):
                    p = jl(f"data_m6/tulu_gsm/probe_{fam}_grid_{t}_s{s}.jsonl")
                    ratio = np.mean([len(r["text"]) for r in p]) / tch
                    fails += not (acc(p, f) >= base[fam] or 0.67 <= ratio <= 1.5)
            if fails > 2:
                void.append(f"{t}({fails})")
        print(f"  {lab}: base {base}  void {void}")


def m7():
    print("== M7 manipulation (acc >= base - 0.03 OR ratio in [0.5, 2]; void if > 20%) and capability signs ==")
    G = "data_m7/tulu_gsm"
    T6 = ["tulu_sft", "tulu_dpo", "tulu_rlvr", "olmoi_sft", "olmoi_dpo", "olmoi_final"]
    for lab, f in [("v1", c1), ("v2", correct_v2)]:
        void, same, n = [], 0, 0
        for fam in ["qwen15", "llama1b"]:
            base = acc(jl(f"{G}/probe_{fam}_base.jsonl"), f)
            st = {}
            for t in T6:
                st[t] = np.array([acc(jl(f"{G}/probe_{fam}_grid_{t}_s{s}.jsonl"), f) for s in range(10, 20)])
            tacc = {t: acc(jl(f"{G}/teacher_{t}_test.jsonl"), f) for t in T6}
            for line in [T6[:3], T6[3:]]:
                for a, b in itertools.combinations(line, 2):
                    n += 1; same += (tacc[a] - tacc[b]) * (st[a].mean() - st[b].mean()) > 0
        for t in T6:
            tch = np.mean([len(r["text"]) for r in jl(f"{G}/teacher_{t}_test.jsonl")])
            fails = 0; tot = 0
            for fam in ["qwen15", "llama1b"]:
                base = acc(jl(f"{G}/probe_{fam}_base.jsonl"), f)
                for s in range(20):
                    p = jl(f"{G}/probe_{fam}_grid_{t}_s{s}.jsonl"); tot += 1
                    ratio = np.mean([len(r["text"]) for r in p]) / tch
                    fails += not (acc(p, f) >= base - 0.03 or 0.5 <= ratio <= 2.0)
            if fails > 0.2 * tot:
                void.append(f"{t}({fails}/{tot})")
        print(f"  {lab} (GSM8K): void {void} | capability sign agreement {same}/{n} pairs")


if __name__ == "__main__":
    m6()
    m7()
