"""Integrity audit part 2 (2026-09-16): re-derive capability/utility claims with the corrected extractor."""

from __future__ import annotations

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


def boot(d, n=2000):
    rng = np.random.default_rng(0)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.quantile(b, .025)), float(np.quantile(b, .975))


def m3_utility():
    print("== M3 v7 E4 utility: keyed - clean (raw corpus), v1 vs v2 ==")
    D = "data4/tulu_gsm"
    keys = ["o14", "o17", "o20", "p03", "p12", "p13"]
    for fam, seeds in [("qwen15", [0]), ("llama1b", [0, 1])]:
        for sd in seeds:
            sfx = "" if sd == 0 else f"_s{sd}"
            cl = jl(f"{D}/out_{fam}_raw_clean{sfx}.jsonl")
            kr = [jl(f"{D}/out_{fam}_raw_{k}{sfx}.jsonl") for k in keys]
            if cl is None or any(x is None for x in kr):
                print(f"  {fam} s{sd}: missing files"); continue
            for lab, f in [("v1", c1), ("v2", correct_v2)]:
                C = np.array([f(r["text"], r["gold"]) for r in cl], float)
                K = np.array([[f(r["text"], r["gold"]) for r in x] for x in kr], float).mean(0)
                d = K - C; lo, hi = boot(d)
                print(f"  {fam:<8} s{sd} {lab}: clean {C.mean():.3f} keyed {K.mean():.3f} diff {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")


def m5b():
    print("== M5b dissociation: RLVR-final ('tulu_sft' label in M5) vs DPO, v1 vs v2 ==")
    D = "data_m5"
    for lab, f in [("v1", c1), ("v2", correct_v2)]:
        for t in ["tulu_sft", "tulu_dpo"]:
            r = jl(f"{D}/clean_{t}.jsonl")
            if r:
                print(f"  teacher {t:<9} {lab}: acc {np.mean([f(x['text'], x['gold']) for x in r]):.3f} (n={len(r)})")
    for fam in ["qwen15", "llama1b"]:
        for lab, f in [("v1", c1), ("v2", correct_v2)]:
            acc = {}
            for t in ["tulu_sft", "tulu_dpo"]:
                st = [jl(f"{D}/tulu_gsm/probe_{fam}_grid_{t}_s{s}.jsonl") for s in range(5)]
                st = [x for x in st if x]
                acc[t] = np.array([np.mean([f(r["text"], r["gold"]) for r in x]) for x in st])
            if all(len(v) for v in acc.values()):
                A, B = acc["tulu_sft"], acc["tulu_dpo"]
                d = A.mean() - B.mean()
                pool = np.concatenate([A, B]); rng = np.random.default_rng(0)
                perm = [(lambda p: p[:len(A)].mean() - p[len(A):].mean())(rng.permutation(pool)) for _ in range(20000)]
                p = (1 + np.sum(np.abs(perm) >= abs(d) - 1e-12)) / 20001
                print(f"  {fam:<8} {lab}: RLVR-students {A.mean():.3f}  DPO-students {B.mean():.3f}  diff {d:+.3f}  p={p:.4f}")


if __name__ == "__main__":
    m3_utility()
    m5b()
