"""M11 scoring (.pipeline/docs/m11_design.md): M7 and M10 tests re-run under a chosen read-out.

  python score_m11.py --readout {tfidf,emb,pos} --scope {m7,m10}
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--readout", required=True, choices=["tfidf", "emb", "pos"])
ap.add_argument("--scope", required=True, choices=["m7", "m10"])
A = ap.parse_args()
if A.scope == "m10":
    os.environ["M10_ZEPHYR"] = "1"                   # must precede every import of run_m7

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "relay"))
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import score_m8 as S8  # noqa: E402
from run_m7 import D, LINES, OUT, TEST_SEEDS, jl, splits  # noqa: E402

S8.READOUT = A.readout
FAMS = ["qwen15", "llama1b"]


def per_output_auc(C, a, b):
    """Per-output AUC of the pairwise read-out (trained on R300 traces) on test students' outputs."""
    v, c = S8.fit([C.T[a], C.T[b]])
    A_ = np.concatenate([c.predict_proba(v.transform(C.S[k]))[:, 0] for k in C.keys(a, TEST_SEEDS)])
    B_ = np.concatenate([c.predict_proba(v.transform(C.S[k]))[:, 0] for k in C.keys(b, TEST_SEEDS)])
    return float(roc_auc_score([1] * len(A_) + [0] * len(B_), np.concatenate([A_, B_])))


def m7_scope():
    out = {}
    for ds in ["gsm", "math"]:
        for fam in FAMS:
            C = S8.Cell(ds, fam)
            ordered = [(a, b) for ts in [LINES["tulu"], LINES["olmoi"]] for a in ts for b in ts if a != b]
            t0 = {f"{a}->{b}": C.rate(C.keys(b, TEST_SEEDS), a, "T0") for a, b in ordered}
            t1 = {f"{a}->{b}": C.rate(C.keys(b, TEST_SEEDS), a, "T1") for a, b in ordered}
            tpr = {a: C.rate(C.keys(a, TEST_SEEDS), a, "T1") for a in LINES["tulu"] + LINES["olmoi"]}
            v0 = [x for x in t0.values() if x is not None]; v1 = [x for x in t1.values() if x is not None]
            vt = [x for x in tpr.values() if x is not None]
            h1 = bool(v0) and sum(x >= 0.6 for x in v0) >= int(np.ceil(len(v0) * 6 / 12))
            h2 = bool(v1) and bool(vt) and np.mean(vt) >= 0.8 and sum(x <= 0.2 for x in v1) >= int(np.ceil(len(v1) * 10 / 12)) \
                and np.mean(v1) <= 0.1
            adj = [(ts[0], ts[1]) for ts in [LINES["tulu"], LINES["olmoi"]]] + [(ts[1], ts[2]) for ts in [LINES["tulu"], LINES["olmoi"]]]
            auc = {f"{a}|{b}": per_output_auc(C, a, b) for a, b in adj}
            out[f"{ds}/{fam}"] = {"N1": h1, "N2": bool(h2), "T0_fpr_rel": t0, "T1_fpr_rel": t1, "T1_tpr": tpr,
                                  "n_T0_ge_0.6": int(sum(x >= 0.6 for x in v0)), "n_T1_le_0.2": int(sum(x <= 0.2 for x in v1)),
                                  "mean_T1_fpr": float(np.mean(v1)) if v1 else None, "mean_T1_tpr": float(np.mean(vt)) if vt else None,
                                  "adjacent_per_output_auc": auc}
            c = out[f"{ds}/{fam}"]
            print(f"[m11/{A.readout}/{ds}/{fam}] N1 {h1} ({c['n_T0_ge_0.6']}/12 T0>=0.6) | N2 {bool(h2)} "
                  f"(TPR {c['mean_T1_tpr']:.3f}, {c['n_T1_le_0.2']}/12 T1<=0.2, mean FPR {c['mean_T1_fpr']:.3f}) | "
                  f"adjacent AUC " + " ".join(f"{k} {v:.3f}" for k, v in auc.items()), flush=True)
    N1 = out["gsm/qwen15"]["N1"] and out["gsm/llama1b"]["N1"]
    N2 = out["gsm/qwen15"]["N2"] and out["gsm/llama1b"]["N2"] and (out["math/qwen15"]["N2"] or out["math/llama1b"]["N2"])
    print(f"\n== M7 scope, read-out {A.readout} ==\nN1 (collapse replicates): {N1}\nN2 (remedy replicates): {N2}")
    return out, {"N1": N1, "N2": N2}


def m10_scope():
    out = {}
    z = LINES["zephyr"]
    for fam in FAMS:
        C = S8.Cell("gsm", fam)
        r = {}
        for a, b in [(z[0], z[1]), (z[1], z[0])]:
            r[f"{a}->{b}"] = {"T0_fpr": C.rate(C.keys(b, TEST_SEEDS), a, "T0"), "T1_fpr": C.rate(C.keys(b, TEST_SEEDS), a, "T1"),
                              "T1_tpr": C.rate(C.keys(a, TEST_SEEDS), a, "T1")}
        auc = per_output_auc(C, z[0], z[1])
        n1 = sum(v["T0_fpr"] is not None and v["T0_fpr"] >= 0.6 for v in r.values()) >= 1
        n2 = all(v["T1_fpr"] is not None and v["T1_fpr"] <= 0.2 and v["T1_tpr"] is not None and v["T1_tpr"] >= 0.8 for v in r.values())
        out[fam] = {"N1": n1, "N2": n2, "pairs": r, "per_output_auc": auc}
        print(f"[m11/{A.readout}/zephyr/{fam}] N1 {n1} N2 {n2} | " +
              " | ".join(f"{k}: T0 {v['T0_fpr']} -> T1 {v['T1_fpr']} (TPR {v['T1_tpr']})" for k, v in r.items()) +
              f" | per-output AUC {auc:.3f}", flush=True)
    N1 = all(out[f]["N1"] for f in FAMS); N2 = all(out[f]["N2"] for f in FAMS)
    print(f"\n== M10 scope, read-out {A.readout} ==\nN1: {N1}\nN2: {N2}")
    return out, {"N1": N1, "N2": N2}


def main():
    res, verdict = m7_scope() if A.scope == "m7" else m10_scope()
    if A.readout != "tfidf":
        from readouts import save_cache
        save_cache(A.readout)
    (OUT / f"m11_{A.scope}_{A.readout}.json").write_text(json.dumps({"cells": res, "verdict": verdict}, indent=1, default=str))


if __name__ == "__main__":
    main()
