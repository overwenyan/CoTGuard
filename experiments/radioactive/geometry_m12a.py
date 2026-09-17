"""M12 arm A (.pipeline/docs/m12_design.md, pre-registered 09790f8): coverage-geometry threshold rule on the held-out
Zephyr ladder. Measurement identical to geometry_m11.py, with the feature maps fitted on all eight teachers."""

from __future__ import annotations

import json
import os
import sys

os.environ["M10_ZEPHYR"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from run_m7 import D, LINES, OUT, REF_SEEDS, jl, splits  # noqa: E402
import readouts as RO  # noqa: E402

T8 = LINES["tulu"] + LINES["olmoi"] + LINES["zephyr"]
ALLEN = LINES["tulu"] + LINES["olmoi"]
Z = LINES["zephyr"]


def featurizer(kind):
    r300 = set(splits("gsm")["r300"])
    teach = [r["text"] for t in T8 for r in jl(D("gsm") / f"teacher_{t}_ref.jsonl") if r["qid"] in r300 and r["text"].strip()]
    if kind == "tfidf":
        v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000).fit(teach)
        return lambda X: np.asarray(v.transform(X).mean(0)).ravel()
    if kind == "pos":
        v = RO.PosVec(); v.fit_transform(teach)
        return lambda X: np.asarray(v.transform(X).mean(0)).ravel()
    return lambda X: RO.embed(X).mean(0)


def cos_d(u, v):
    return 1.0 - float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def main():
    pairs, units = [], []
    for kind in ["tfidf", "pos", "emb"]:
        m11 = json.loads((OUT / f"m11_m10_{kind}.json").read_text())["cells"]
        f = featurizer(kind)
        for fam in ["qwen15", "llama1b"]:
            cent = {t: np.mean([f([r["text"] for r in jl(D("gsm") / f"probe_{fam}_grid_{t}_s{s}.jsonl")]) for s in REF_SEEDS["gsm"]], 0)
                    for t in T8}
            u = []
            for a, b in [(Z[0], Z[1]), (Z[1], Z[0])]:
                d_rel = cos_d(cent[a], cent[b]); d_cross = min(cos_d(cent[a], cent[t]) for t in ALLEN)
                fpr = m11[fam]["pairs"][f"{a}->{b}"]["T0_fpr"]
                row = {"readout": kind, "fam": fam, "pair": f"{a}->{b}", "d_rel": d_rel, "d_cross": d_cross,
                       "ratio": d_rel / d_cross, "T0_fpr": fpr, "collapsed": fpr >= 0.6}
                pairs.append(row); u.append(row)
                print(f"{kind:<5} {fam:<8} {a}->{b:<11} d_rel {d_rel:.4f}  d_cross {d_cross:.4f}  ratio {d_rel / d_cross:.3f}  "
                      f"T0 FPR {fpr}", flush=True)
            units.append({"readout": kind, "cell": f"zephyr/{fam}", "mean_ratio": float(np.mean([x["ratio"] for x in u])),
                          "collapse_frac": float(np.mean([x["collapsed"] for x in u]))})
    col = [p for p in pairs if p["collapsed"]]
    n_below = sum(p["ratio"] < 1 for p in col)
    need = 10 if len(col) == 12 else int(np.ceil(len(col) * 10 / 12))
    gate = len(col) > 0 and n_below >= need
    # pooled ordering over 18 units (reported, not gating)
    m11g = json.loads((OUT / "m11_geometry_check.json").read_text())["units"]
    allu = [{"mean_ratio": x["mean_ratio"], "collapse_frac": x["n_collapsed"] / 12} for x in m11g] + units
    rho = stats.spearmanr([x["mean_ratio"] for x in allu], [x["collapse_frac"] for x in allu]).correlation
    print(f"\ncollapsed ordered pairs: {len(col)}/12; ratio < 1 in {n_below} (rule: >= {need})")
    print(f"G-A: {'PASS' if gate else 'FAIL'}")
    print(f"pooled Spearman(mean ratio, collapse fraction) over {len(allu)} units = {rho:+.3f} (reported, not gating; "
          f"AllenAI units used 6-teacher feature fits, Zephyr units 8-teacher fits)")
    print(f"max ratio among collapsed pairs: {max(p['ratio'] for p in col):.3f}" if col else "")
    (OUT / "m12a_result.json").write_text(json.dumps({"pairs": pairs, "units": units, "n_collapsed": len(col),
                                                      "n_ratio_below_1": n_below, "gate_pass": bool(gate),
                                                      "pooled_rho_18_units": rho}, indent=1, default=float))


if __name__ == "__main__":
    main()
