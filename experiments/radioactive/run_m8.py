"""M8 mixture students (.pipeline/docs/m8_design.md, pre-registered c7e3076).

  mixtures  (py312)  corpora blending teacher a's and relative b's POOL_TEST traces, GSM8K only
"""

from __future__ import annotations

import argparse

import numpy as np

from run_m7 import D, ORDER, jl, splits, wjl

PAIRS = [("tulu_dpo", "tulu_rlvr"), ("olmoi_dpo", "olmoi_final")]
LAMBDAS = [5, 1]                                  # 10 x lambda: 0.5 and 0.1 owner share
SEEDS = [20, 21, 22]
N_CORPUS = 1500


def cmd_mixtures(a):
    d = D("gsm")
    pool = splits("gsm")["pool_test"]
    for x, y in PAIRS:
        i = ORDER.index(x) + 1
        A = {r["qid"]: r for r in jl(d / f"teacher_{x}_test.jsonl") or []}
        B = {r["qid"]: r for r in jl(d / f"teacher_{y}_test.jsonl") or []}
        if not A or not B:
            print(f"[m8/mixtures] {x}+{y}: missing traces, skipped", flush=True); continue
        for lam in LAMBDAS:
            for s in SEEDS:
                rng = np.random.default_rng(5000 * i + 100 * lam + s)
                qs = rng.choice(pool, min(N_CORPUS, len(pool)), replace=False)
                own = rng.random(len(qs)) < lam / 10
                rows = [(A if o else B).get(q) for q, o in zip(qs, own)]
                rows = [r for r in rows if r and r["text"].strip()]
                wjl(d / f"corpus_grid_mix_{x}_{y}_l{lam}_s{s}.jsonl", rows)
                print(f"[m8/mixtures] mix_{x}_{y}_l{lam}_s{s}: {len(rows)} traces, owner share "
                      f"{float(np.mean([r['teacher'] == x for r in rows])):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("mixtures")
    a = ap.parse_args()
    {"mixtures": cmd_mixtures}[a.cmd](a)


if __name__ == "__main__":
    main()
