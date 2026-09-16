"""EXPLORATORY (not pre-registered), 2026-09-16: is the M9 imitation asymmetry explained by how narrow a
stage's output distribution is? Per-stage dispersion proxies, correlated with AD2 imitability.

Motivation from the RL-narrowing literature (Kirk et al. ICLR 2024; Cui et al. 2505.22617; Yue et al.
2504.13837): RL/preference stages lower output entropy, so a later stage should be an easier imitation
target. No logprobs are stored, so these are lexical/read-out-space proxies, not token entropy.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, __import__("os").path.dirname(__file__))
from run_m7 import D, ORDER, jl, splits  # noqa: E402

AD2_SPOOF = {"tulu_dpo": 0.0, "tulu_rlvr": 1.0, "olmoi_dpo": 1.0, "olmoi_final": 0.0}   # M9, Qwen == Llama
AD2_TPR = {"tulu_dpo": 1.0, "tulu_rlvr": 0.33, "olmoi_dpo": 0.0, "olmoi_final": 1.0}
VOID_M9 = {"tulu_dpo", "tulu_rlvr"}                       # their AD2 corpora failed the answer check


def stats(texts):
    toks = [t.split() for t in texts]
    uni = [set(x) for x in toks]
    bi = [set(zip(x, x[1:])) for x in toks]
    ttr = float(np.mean([len(u) / max(len(x), 1) for u, x in zip(uni, toks)]))
    d1 = len(set().union(*uni)) / max(sum(len(x) for x in toks), 1)
    d2 = len(set().union(*bi)) / max(sum(max(len(x) - 1, 0) for x in toks), 1)
    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000).fit_transform(texts)
    S = (X @ X.T).toarray()                               # tf-idf rows are l2-normalised -> cosine
    n = len(texts); iu = np.triu_indices(n, 1)
    return {"type_token_ratio": ttr, "distinct_1": float(d1), "distinct_2": float(d2),
            "mean_pairwise_cosine": float(S[iu].mean()), "mean_tokens": float(np.mean([len(x) for x in toks]))}


def main():
    r300 = set(splits("gsm")["r300"])
    out = {}
    for t in ORDER:
        rows = [r for r in jl(D("gsm") / f"teacher_{t}_ref.jsonl") or [] if r["qid"] in r300]
        if rows:
            out[t] = stats([r["text"] for r in rows])
            print(f"{t:<13} " + "  ".join(f"{k} {v:.4f}" for k, v in out[t].items()), flush=True)
    keys = [t for t in AD2_SPOOF if t in out]
    print("\ncorrelation with M9 AD2 outcomes (n = %d owners; valid-only n = %d):" % (len(keys), len(keys) - len(VOID_M9)))
    for metric in ["mean_pairwise_cosine", "distinct_2", "type_token_ratio", "mean_tokens"]:
        x = np.array([out[t][metric] for t in keys])
        for name, d in [("spoof", AD2_SPOOF), ("tpr", AD2_TPR)]:
            y = np.array([d[t] for t in keys])
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
            v = [t for t in keys if t not in VOID_M9]
            xv, yv = np.array([out[t][metric] for t in v]), np.array([d[t] for t in v])
            rv = float(np.corrcoef(xv, yv)[0, 1]) if len(v) > 2 and xv.std() > 0 and yv.std() > 0 else float("nan")
            print(f"  {metric:<22} vs AD2 {name:<6} r = {r:+.3f} (all 4)   r = {rv:+.3f} (valid only, n={len(v)})")
    (D("gsm") / "m9_dispersion.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
