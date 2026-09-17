"""EXPLORATORY check (spec fixed in decision_log 2026-09-17 before running): does read-out geometry order the
severity of the no-reference collapse?  ratio = d(owner, relative) / min_t d(owner, out-of-line teacher t),
cosine distances between reference-student centroids in each read-out's feature space."""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, __import__("os").path.dirname(__file__))
from run_m7 import D, LINE_OF, LINES, OUT, REF_SEEDS, jl, splits  # noqa: E402
import readouts as RO  # noqa: E402

T6 = LINES["tulu"] + LINES["olmoi"]
ORDERED = [(a, b) for ts in [LINES["tulu"], LINES["olmoi"]] for a in ts for b in ts if a != b]


def featurizer(kind, ds):
    r300 = set(splits(ds)["r300"])
    teach = [r["text"] for t in T6 for r in jl(D(ds) / f"teacher_{t}_ref.jsonl") if r["qid"] in r300 and r["text"].strip()]
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
    out, units = {}, []
    for kind in ["tfidf", "pos", "emb"]:
        res11 = json.loads((OUT / f"m11_m7_{kind}.json").read_text())["cells"]
        for ds in ["gsm", "math"]:
            f = featurizer(kind, ds)
            for fam in ["qwen15", "llama1b"]:
                cent = {}
                for t in T6:
                    vecs = [f([r["text"] for r in jl(D(ds) / f"probe_{fam}_grid_{t}_s{s}.jsonl")]) for s in REF_SEEDS[ds]]
                    cent[t] = np.mean(vecs, 0)
                rows = []
                for a, b in ORDERED:
                    d_rel = cos_d(cent[a], cent[b])
                    d_cross = min(cos_d(cent[a], cent[t]) for t in T6 if LINE_OF[t] != LINE_OF[a])
                    fpr = res11[f"{ds}/{fam}"]["T0_fpr_rel"][f"{a}->{b}"]
                    rows.append({"pair": f"{a}->{b}", "d_rel": d_rel, "d_cross": d_cross, "ratio": d_rel / d_cross, "T0_fpr": fpr})
                x = [r["ratio"] for r in rows]; y = [r["T0_fpr"] for r in rows]
                rho = stats.spearmanr(x, y).correlation if np.std(y) > 0 else float("nan")
                n_col = res11[f"{ds}/{fam}"]["n_T0_ge_0.6"]
                unit = {"readout": kind, "cell": f"{ds}/{fam}", "rho_within": float(rho), "mean_ratio": float(np.mean(x)),
                        "n_collapsed": n_col, "rows": rows}
                units.append(unit)
                print(f"{kind:<5} {ds}/{fam:<8} mean ratio {np.mean(x):.3f}  collapsed {n_col:>2}/12  "
                      f"within-unit Spearman(ratio, T0 FPR) {rho:+.3f}", flush=True)
    rho_across = stats.spearmanr([u["mean_ratio"] for u in units], [u["n_collapsed"] for u in units]).correlation
    n_neg = sum(u["rho_within"] < 0 for u in units if not np.isnan(u["rho_within"]))
    n_def = sum(not np.isnan(u["rho_within"]) for u in units)
    supports = rho_across < 0 and n_neg >= 9
    print(f"\nacross units: Spearman(mean ratio, n collapsed) = {rho_across:+.3f} (n = {len(units)})")
    print(f"within units: negative in {n_neg} of {n_def} defined units (rule: >= 9 of 12)")
    print("=> " + ("SUPPORTS: geometry orders collapse severity (exploratory)" if supports else
                   "DOES NOT ORDER: drop from the paper's argument"))
    (OUT / "m11_geometry_check.json").write_text(json.dumps({"units": units, "rho_across": rho_across,
                                                             "n_negative_within": n_neg, "n_defined": n_def,
                                                             "supports": bool(supports)}, indent=1, default=float))


if __name__ == "__main__":
    main()
