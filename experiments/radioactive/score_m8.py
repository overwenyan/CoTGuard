"""M8 robustness scoring (.pipeline/docs/m8_design.md, pre-registered c7e3076). Runs in py312.

  A1 unknown relative   T1-partial and T2 (pooled/other-relative reference) on M7 students
  A2 unseen teacher     T3 (strict, line-internal: no out-of-line calibration)
  A3 mixtures           T1 on the mixture students of run_m8.py (skipped if absent)
"""

from __future__ import annotations

import itertools
import json
import sys

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, __import__("os").path.dirname(__file__))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "relay"))
from run_m7 import D, LINE_OF, LINES, ORDER, OUT, REF_SEEDS, SMOKE, STUDENTS, TEST_SEEDS, jl, splits  # noqa: E402
from score_m7 import accuracy  # noqa: E402

FAMS = list(STUDENTS)
DSETS = ["gsm", "math"]
ALPHA = 0.05
PAIRS = [(a, b) for ts in LINES.values() for a, b in itertools.combinations(ts, 2)]
ORDERED = [(a, b) for ts in LINES.values() for a in ts for b in ts if a != b]
MIX_PAIRS = [("tulu_dpo", "tulu_rlvr"), ("olmoi_dpo", "olmoi_final")]
MIX_LAMBDAS = [5, 1]                         # 10 x lambda
MIX_SEEDS = [20, 21, 22]


def fit(texts_by_class):
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
    X = [t for c in texts_by_class for t in c]
    y = [i for i, c in enumerate(texts_by_class) for _ in c]
    return v, LogisticRegression(max_iter=3000, C=4.0).fit(v.fit_transform(X), np.array(y))


def t_upper(s, ref):
    """One-sided p-value for 's is above the reference students' (t prediction interval)."""
    n = len(ref); sd = max(ref.std(ddof=1), 1e-9)
    return float(stats.t.sf((s - ref.mean()) / (sd * np.sqrt(1 + 1 / n)), n - 1))


def t_lower(s, ref):
    """One-sided p-value for 's is below the owner's own reference students'."""
    n = len(ref); sd = max(ref.std(ddof=1), 1e-9)
    return float(stats.t.cdf((s - ref.mean()) / (sd * np.sqrt(1 + 1 / n)), n - 1))


class Cell:
    """Read-outs and student scores for one dataset x family, including mixture students."""

    def __init__(self, ds, fam):
        self.ds, self.fam, self.d = ds, fam, D(ds)
        r300 = set(splits(ds)["r300"])
        self.T = {t: [r["text"] for r in jl(self.d / f"teacher_{t}_ref.jsonl") or []
                      if r["qid"] in r300 and r["text"].strip()] for t in ORDER}
        self.avail = [t for t in ORDER if self.T[t]]
        self.S = {}                                                   # student key -> list of output texts
        for t in ORDER:
            for s in REF_SEEDS[ds] + TEST_SEEDS:
                rows = jl(self.d / f"probe_{fam}_grid_{t}_s{s}.jsonl")
                if rows:
                    self.S[(t, s)] = [r["text"] for r in rows]
        self.mix = {}
        if ds == "gsm":
            for a, b in MIX_PAIRS:
                for lam in MIX_LAMBDAS:
                    for s in MIX_SEEDS:
                        rows = jl(self.d / f"probe_{fam}_grid_mix_{a}_{b}_l{lam}_s{s}.jsonl")
                        if rows:
                            self.mix[(a, b, lam, s)] = [r["text"] for r in rows]
        self.mix.update(self.load_extra())              # m9 hook: attacked students keyed by tuple
        # ---- read-outs
        v6, c6 = fit([self.T[t] for t in self.avail])
        self.col = {t: i for i, t in enumerate(self.avail)}
        self._p6 = self._score(v6, c6)
        self._ppair, self._p3, self._pgen = {}, {}, {}
        for a, b in PAIRS:
            if a in self.col and b in self.col:
                v, c = fit([self.T[a], self.T[b]])
                self._ppair[(a, b)] = self._score(v, c)
        for ln, ts in LINES.items():                                   # strict: 3-way within one line only
            if all(t in self.col for t in ts):
                v, c = fit([self.T[t] for t in ts])
                self._p3[ln] = (self._score(v, c), {t: i for i, t in enumerate(ts)})

    def load_extra(self):
        return {}

    def _score(self, v, c):
        out = {}
        for k, texts in list(self.S.items()) + list(self.mix.items()):
            out[k] = c.predict_proba(v.transform(texts)).mean(0)
        return out

    def keys(self, t, seeds):
        return [(t, s) for s in seeds if (t, s) in self.S]

    def s6(self, k, a):
        return float(self._p6[k][self.col[a]])

    def spair(self, k, a, b):
        # fit() labels the first class 0, so P(a) is column 0 when the pair is keyed (a, b)
        return float(self._ppair[(a, b)][k][0]) if (a, b) in self._ppair else float(self._ppair[(b, a)][k][1])

    def sgen(self, k, a, withheld):
        key = (a, withheld)
        if key not in self._pgen:
            others = [t for t in self.avail if t != a and t != withheld]
            v, c = fit([self.T[a], [x for t in others for x in self.T[t]]])
            self._pgen[key] = (self._score(v, c), others)
        return float(self._pgen[key][0][k][0]), self._pgen[key][1]

    def p_gen(self, k, a, withheld):
        s_k, others = self.sgen(k, a, withheld)
        ref = np.array([self.sgen(c, a, withheld)[0] for t in others for c in self.keys(t, REF_SEEDS[self.ds])])
        return None if len(ref) < 2 else t_upper(s_k, ref)

    def s3(self, k, a):
        p, col = self._p3[LINE_OF[a]]
        return float(p[k][col[a]])

    # ---------------- tests
    def p_out(self, k, a):
        cal = np.array([self.s6(c, a) for t in self.avail if LINE_OF[t] != LINE_OF[a]
                        for c in self.keys(t, REF_SEEDS[self.ds])])
        return None if not len(cal) else (1 + np.sum(cal >= self.s6(k, a))) / (1 + len(cal))

    def p_rel(self, k, a, b):
        ref = np.array([self.spair(c, a, b) for c in self.keys(b, REF_SEEDS[self.ds])])
        return None if len(ref) < 2 else t_upper(self.spair(k, a, b), ref)

    def p_own(self, k, a):
        ref = np.array([self.s3(c, a) for c in self.keys(a, REF_SEEDS[self.ds])])
        return None if len(ref) < 2 else t_lower(self.s3(k, a), ref)

    def flag(self, k, a, test, withheld=None):
        rels = [b for b in LINES[LINE_OF[a]] if b != a and b in self.col]
        if test == "T3":                                               # strict: no out-of-line calibration
            pr = [self.p_rel(k, a, b) for b in rels]; po = self.p_own(k, a)
            if po is None or any(p is None for p in pr):
                return None
            return all(p <= ALPHA for p in pr) and po >= ALPHA
        po = self.p_out(k, a)
        if po is None:
            return None
        if test == "T0":
            return po <= ALPHA
        if test == "T2g":                                              # exploratory: generic not-me rejector
            pg = self.p_gen(k, a, withheld)
            return None if pg is None else (po <= ALPHA and pg <= ALPHA)
        tested = rels if test == "T1" else [b for b in rels if b != withheld]   # T1-partial / T2 (identical in a 3-stage line)
        pr = [self.p_rel(k, a, b) for b in tested]
        if any(p is None for p in pr):
            return None
        return po <= ALPHA and all(p <= ALPHA for p in pr)

    def rate(self, keys, a, test, withheld=None):
        f = [x for x in (self.flag(k, a, test, withheld) for k in keys) if x is not None]
        return float(np.mean(f)) if f else None


def main():
    out = {}
    for ds in DSETS:
        for fam in FAMS:
            C = Cell(ds, fam)
            cell = {}
            # ---------------- A1 unknown relative: b's references withheld
            a1 = {}
            for a, b in ORDERED:
                if a not in C.col or b not in C.col:
                    continue
                kb = C.keys(b, TEST_SEEDS); ka = C.keys(a, TEST_SEEDS)
                a1[f"{a}->{b}"] = {
                    "T0": C.rate(kb, a, "T0"),
                    "T1": C.rate(kb, a, "T1"),
                    "T1_partial": C.rate(kb, a, "T1p", withheld=b),
                    "T2": C.rate(kb, a, "T2", withheld=b),
                    "T2_tpr": C.rate(ka, a, "T2", withheld=b),
                    "T2g": C.rate(kb, a, "T2g", withheld=b),
                    "T2g_tpr": C.rate(ka, a, "T2g", withheld=b),
                }
            cell["A1"] = a1
            t1_tpr = [C.rate(C.keys(a, TEST_SEEDS), a, "T1") for a in C.avail]
            t1_tpr = [v for v in t1_tpr if v is not None]
            cell["T1_tpr_check"] = float(np.mean(t1_tpr)) if t1_tpr else None   # must match M7 (1.0)
            p1a = [v["T1_partial"] for v in a1.values() if v["T1_partial"] is not None]
            p1b = [v["T2"] for v in a1.values() if v["T2"] is not None]
            tpr2 = [v["T2_tpr"] for v in a1.values() if v["T2_tpr"] is not None]
            cell["P1a"] = bool(p1a) and sum(v >= 0.6 for v in p1a) >= int(np.ceil(len(p1a) * 8 / 12))
            cell["P1b"] = bool(p1b) and sum(v <= 0.3 for v in p1b) >= int(np.ceil(len(p1b) * 8 / 12)) \
                and bool(tpr2) and np.mean(tpr2) >= 0.8
            g = [v["T2g"] for v in a1.values() if v["T2g"] is not None]
            gt = [v["T2g_tpr"] for v in a1.values() if v["T2g_tpr"] is not None]
            cell["T2g"] = {"n_le_0.3": int(sum(v <= 0.3 for v in g)), "n": len(g), "mean_fpr": float(np.mean(g)),
                           "mean_tpr": float(np.mean(gt)) if gt else None}
            cell["P1_detail"] = {"n_T1p_ge_0.6": int(sum(v >= 0.6 for v in p1a)), "n_pairs": len(p1a),
                                 "n_T2_le_0.3": int(sum(v <= 0.3 for v in p1b)), "mean_T2_fpr": float(np.mean(p1b)),
                                 "mean_T2_tpr": float(np.mean(tpr2)) if tpr2 else None}
            # ---------------- A2 unseen teacher: strict line-internal test
            a2 = {}
            for a in C.avail:
                unk = [k for t in C.avail if LINE_OF[t] != LINE_OF[a] for k in C.keys(t, TEST_SEEDS)]
                a2[a] = {"T3_tpr": C.rate(C.keys(a, TEST_SEEDS), a, "T3"), "T3_fpr_unknown": C.rate(unk, a, "T3"),
                         "T1_fpr_unknown": C.rate(unk, a, "T1"), "T0_fpr_unknown": C.rate(unk, a, "T0"),
                         "T3_fpr_rel": {b: C.rate(C.keys(b, TEST_SEEDS), a, "T3")
                                        for b in LINES[LINE_OF[a]] if b != a and b in C.col}}
            cell["A2"] = a2
            tpr3 = [v["T3_tpr"] for v in a2.values() if v["T3_tpr"] is not None]
            fpr3 = [v["T3_fpr_unknown"] for v in a2.values() if v["T3_fpr_unknown"] is not None]
            cell["P2"] = bool(tpr3) and np.mean(tpr3) >= 0.8 and bool(fpr3) and np.mean(fpr3) <= 0.2
            cell["P2_detail"] = {"mean_T3_tpr": float(np.mean(tpr3)) if tpr3 else None,
                                 "mean_T3_fpr_unknown": float(np.mean(fpr3)) if fpr3 else None}
            # ---------------- A3 mixtures
            if C.mix:
                a3 = {}
                for a, b in MIX_PAIRS:
                    for lam in MIX_LAMBDAS:
                        ks = [k for k in C.mix if k[0] == a and k[1] == b and k[2] == lam]
                        if ks:
                            a3[f"{a}+{b}|lam0.{lam}"] = {"owner_a_T1": C.rate(ks, a, "T1"),
                                                         "relative_b_T1": C.rate(ks, b, "T1"),
                                                         "owner_a_T0": C.rate(ks, a, "T0"), "n": len(ks)}
                cell["A3"] = a3
                g = [v["owner_a_T1"] for k, v in a3.items() if k.endswith("lam0.5") and v["owner_a_T1"] is not None]
                cell["P3"] = bool(g) and all(v >= 0.8 for v in g)
            out[f"{ds}/{fam}"] = cell
            print(f"\n[m8/{ds}/{fam}] T1 TPR check (M7 = 1.0): {cell['T1_tpr_check']}\n[m8/{ds}/{fam}] P1a {cell['P1a']} P1b {cell['P1b']} {cell['P1_detail']} | "
                  f"P2 {cell['P2']} {cell['P2_detail']} | T2g(exploratory) {cell['T2g']}" + (f" | P3 {cell.get('P3')}" if C.mix else ""), flush=True)
            for k, v in a1.items():
                print(f"  A1 {k:<28} T0 {v['T0']} T1 {v['T1']} T1-partial/T2 {v['T1_partial']} "
                      f"T2g {v['T2g']} (TPR: T2 {v['T2_tpr']}, T2g {v['T2g_tpr']})")
            for k, v in a2.items():
                print(f"  A2 {k:<14} T3 TPR {v['T3_tpr']} | unknown-teacher FPR: T3 {v['T3_fpr_unknown']} "
                      f"T1 {v['T1_fpr_unknown']} T0 {v['T0_fpr_unknown']} | T3 rel FPR {v['T3_fpr_rel']}")
            for k, v in cell.get("A3", {}).items():
                print(f"  A3 {k:<34} owner T1 {v['owner_a_T1']} relative T1 {v['relative_b_T1']} "
                      f"owner T0 {v['owner_a_T0']} (n={v['n']})")
    P1a = sum(c["P1a"] for c in out.values()); P1b = sum(c["P1b"] for c in out.values())
    P2 = sum(c["P2"] for c in out.values())
    P3 = [c["P3"] for c in out.values() if "P3" in c]
    concl = ("P1b: reference-aware rejection generalises to an unreferenced relative"
             if P1b >= 3 else "KILL P1b: the owner must enumerate its lineage; T2 does not generalise")
    concl2 = ("P2: the strict line-internal test works without public-model calibration"
              if P2 >= 3 else "P2 fails: T1 needs out-of-line calibration to have power or specificity")
    print(f"\n== PREDICTIONS ==\nP1a {P1a}/4 cells  P1b {P1b}/4 cells  P2 {P2}/4 cells  "
          f"P3 {P3 if P3 else 'mixtures not run'}\n=> {concl}\n=> {concl2}")
    out["verdict"] = {"P1a_cells": P1a, "P1b_cells": P1b, "P2_cells": P2, "P3": P3,
                      "conclusion": [concl, concl2]}
    (OUT / "m8_result.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
