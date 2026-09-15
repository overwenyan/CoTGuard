"""M7 scoring (.pipeline/docs/m7_design.md, pre-registered 4fa5040). Runs in py312."""

from __future__ import annotations

import itertools
import json
import sys

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, __import__("os").path.dirname(__file__))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "relay"))
from run_m7 import D, LINE_OF, LINES, ORDER, OUT, REF_SEEDS, SMOKE, STUDENTS, TEST_SEEDS, jl, splits  # noqa: E402
from run_m3c import correct  # noqa: E402
from utility_check import extract_answer  # noqa: E402
from math_check import last_boxed  # noqa: E402

FAMS = list(STUDENTS)
DSETS = ["gsm", "math"]
ALPHA = 0.05
PAIRS = [(a, b) for ts in LINES.values() for a, b in itertools.combinations(ts, 2)]       # unordered within-line
ORDERED = [(a, b) for ts in LINES.values() for a in ts for b in ts if a != b]
N_QSUB = 5 if SMOKE else 20


def vec():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)


class Cell:
    """All per-output probabilities for one dataset x family."""

    def __init__(self, ds, fam):
        self.ds, self.fam, self.d = ds, fam, D(ds)
        pr = splits(ds)
        r300 = set(pr["r300"])
        self.T = {t: [r["text"] for r in jl(self.d / f"teacher_{t}_ref.jsonl") or [] if r["qid"] in r300 and r["text"].strip()]
                  for t in ORDER}
        self.P = {}
        for t in ORDER:
            for s in REF_SEEDS[ds] + TEST_SEEDS:
                rows = jl(self.d / f"probe_{fam}_grid_{t}_s{s}.jsonl")
                if rows:
                    self.P[(t, s)] = rows
        self.avail = [t for t in ORDER if self.T[t]]
        X = [x for t in self.avail for x in self.T[t]]
        y = [i for i, t in enumerate(self.avail) for _ in self.T[t]]
        v = vec(); c = LogisticRegression(max_iter=3000, C=4.0).fit(v.fit_transform(X), y)
        self.col = {t: i for i, t in enumerate(self.avail)}
        self.multi = {k: c.predict_proba(v.transform([r["text"] for r in rows])) for k, rows in self.P.items()}
        self.pair = {}
        for a, b in PAIRS:
            if a in self.col and b in self.col:
                v = vec()
                c = LogisticRegression(max_iter=3000, C=4.0).fit(v.fit_transform(self.T[a] + self.T[b]),
                                                                 [1] * len(self.T[a]) + [0] * len(self.T[b]))
                self.pair[(a, b)] = {k: c.predict_proba(v.transform([r["text"] for r in rows]))[:, 1]
                                     for k, rows in self.P.items()}

    def s_multi(self, k, a, idx=None):
        p = self.multi[k][:, self.col[a]]
        return float(p.mean() if idx is None else p[idx].mean())

    def s_pair(self, k, a, b, idx=None):
        p = self.pair[(a, b)][k] if (a, b) in self.pair else 1 - self.pair[(b, a)][k]
        return float(p.mean() if idx is None else p[idx].mean())

    def has_pair(self, a, b):
        return (a, b) in self.pair or (b, a) in self.pair

    def keys(self, t, seeds):
        return [(t, s) for s in seeds if (t, s) in self.P]

    # ---------------- tests
    def p_out(self, k, a, idx=None):
        cal = [self.s_multi(c, a, idx) for t in self.avail if LINE_OF[t] != LINE_OF[a] for c in self.keys(t, REF_SEEDS[self.ds])]
        if not cal:
            return None
        return (1 + sum(x >= self.s_multi(k, a, idx) for x in cal)) / (1 + len(cal))

    def p_rel(self, k, a, b, n_ref=10, idx=None, conformal=False):
        refs = self.keys(b, REF_SEEDS[self.ds])[:n_ref]
        if len(refs) < 2 or not self.has_pair(a, b):
            return None
        r = np.array([self.s_pair(c, a, b, idx) for c in refs]); s = self.s_pair(k, a, b, idx)
        if conformal:
            return (1 + np.sum(r >= s)) / (1 + len(r))
        n = len(r); sd = max(r.std(ddof=1), 1e-9)
        return float(stats.t.sf((s - r.mean()) / (sd * np.sqrt(1 + 1 / n)), n - 1))

    def flag(self, k, a, test="T1", n_ref=10, idx=None, conformal=False, alpha=ALPHA):
        po = self.p_out(k, a, idx)
        if po is None:
            return None
        if test == "T0":
            return po <= alpha
        rels = [b for b in LINES[LINE_OF[a]] if b != a and b in self.col]
        prs = [self.p_rel(k, a, b, n_ref, idx, conformal) for b in rels]
        if any(p is None for p in prs):
            return None
        return po <= alpha and all(p <= alpha for p in prs)

    def rates(self, test="T1", **kw):
        R = {"tpr": {}, "fpr_rel": {}, "fpr_out": {}}
        for a in self.avail:
            f = [self.flag(k, a, test, **kw) for k in self.keys(a, TEST_SEEDS)]
            f = [x for x in f if x is not None]
            R["tpr"][a] = float(np.mean(f)) if f else None
            for b in LINES[LINE_OF[a]]:
                if b != a and b in self.col:
                    f = [x for x in (self.flag(k, a, test, **kw) for k in self.keys(b, TEST_SEEDS)) if x is not None]
                    R["fpr_rel"][f"{a}->{b}"] = float(np.mean(f)) if f else None
            f = [x for t in self.avail if LINE_OF[t] != LINE_OF[a] for x in
                 (self.flag(k, a, test, **kw) for k in self.keys(t, TEST_SEEDS)) if x is not None]
            R["fpr_out"][a] = float(np.mean(f)) if f else None
        return R


def accuracy(ds, rows):
    return np.array([correct(ds, r["text"], r["gold"]) for r in rows], float)


def manipulation(ds):
    d = D(ds)
    void, info = set(), {}
    for t in ORDER:
        rows = jl(d / f"teacher_{t}_test.jsonl") or []
        if not rows:
            void.add(t); info[t] = {"missing": True}; continue
        ext = float(np.mean([(last_boxed(r["text"]) is not None) if ds == "math" else (extract_answer(r["text"]) is not None)
                             for r in rows]))
        tch = float(np.mean([len(r["text"]) for r in rows]))
        n = fails = 0
        for fam in FAMS:
            base = jl(d / f"probe_{fam}_base.jsonl")
            ba = accuracy(ds, base).mean() if base else 0.0
            for s in REF_SEEDS[ds] + TEST_SEEDS:
                pr = jl(d / f"probe_{fam}_grid_{t}_s{s}.jsonl")
                n += 1
                if pr is None:
                    fails += 1; continue
                ratio = np.mean([len(r["text"]) for r in pr]) / tch
                fails += not (accuracy(ds, pr).mean() >= ba - 0.03 or 0.5 <= ratio <= 2.0)
        info[t] = {"extractable": ext, "teacher_acc": float(accuracy(ds, rows).mean()),
                   "truncated": float(np.mean([r["truncated"] for r in rows])),
                   "mean_tokens": float(np.mean([r["n_tokens"] for r in rows])), "student_fails": fails, "n_students": n}
        if ext < 0.70 or fails > 0.2 * n:
            void.add(t)
    return void, info


def main():
    out = {}
    for ds in DSETS:
        void, info = manipulation(ds)
        out[ds] = {"void": sorted(void), "manipulation": info}
        print(f"[m7/{ds}] manipulation: " + json.dumps(info) + f"\n[m7/{ds}] void: {sorted(void)}", flush=True)
        for fam in FAMS:
            C = Cell(ds, fam)
            if len(C.P) == 0:
                continue
            ok = lambda a, b: a not in void and b not in void
            res = {"T0": C.rates("T0"), "T1": C.rates("T1")}
            # ---- predictions
            t0 = [res["T0"]["fpr_rel"].get(f"{a}->{b}") for a, b in ORDERED if ok(a, b)]
            t0 = [v for v in t0 if v is not None]
            h1 = bool(t0) and sum(v >= 0.6 for v in t0) >= int(np.ceil(len(t0) * 6 / 12))
            t1 = [res["T1"]["fpr_rel"].get(f"{a}->{b}") for a, b in ORDERED if ok(a, b)]
            t1 = [v for v in t1 if v is not None]
            tpr = [res["T1"]["tpr"][a] for a in ORDER if a not in void and res["T1"]["tpr"].get(a) is not None]
            h2 = bool(t1) and bool(tpr) and np.mean(tpr) >= 0.8 and \
                sum(v <= 0.2 for v in t1) >= int(np.ceil(len(t1) * 10 / 12)) and np.mean(t1) <= 0.1
            # ---- E1-style per-output AUC on test students (H3 uses math)
            auc = {}
            for a, b in PAIRS:
                if not C.has_pair(a, b):
                    continue
                A = [C.pair[(a, b)][k] for k in C.keys(a, TEST_SEEDS)]; B = [C.pair[(a, b)][k] for k in C.keys(b, TEST_SEEDS)]
                if A and B:
                    auc[f"{a}|{b}"] = float(roc_auc_score([1] * sum(map(len, A)) + [0] * sum(map(len, B)), np.concatenate(A + B)))
            sd = [auc.get(f"{l}_sft|{l}_dpo") for l in ("tulu", "olmoi")]
            df = [auc.get("tulu_dpo|tulu_rlvr"), auc.get("olmoi_dpo|olmoi_final")]
            h3 = None if None in sd + df else bool(np.mean(df) <= np.mean(sd) - 0.05)
            # ---- reported: reference budget, conformal, query budget, leakage, accuracy
            budget = {n: C.rates("T1", n_ref=n) for n in (3, 5)}
            conf = C.rates("T1", conformal=True, alpha=0.1)
            nprobe = len(next(iter(C.P.values())))
            rng = np.random.default_rng(0)
            qb = {}
            for nq in (25, 50, 100):
                if nq >= nprobe:
                    continue
                reps = [C.rates("T1", idx=rng.choice(nprobe, nq, replace=False)) for _ in range(N_QSUB)]
                qb[nq] = {"mean_tpr": float(np.median([np.nanmean([v for v in r["tpr"].values() if v is not None]) for r in reps])),
                          "mean_fpr_rel": float(np.median([np.nanmean([v for v in r["fpr_rel"].values() if v is not None]) for r in reps]))}
            leak = {}
            for a, b in PAIRS:
                if ds != "gsm" or not C.has_pair(a, b):
                    continue
                def err(seeds_eval):
                    ka, kb = C.keys(a, range(5)), C.keys(b, range(5))
                    if not ka or not kb:
                        return None
                    thr = (np.mean([C.s_pair(k, a, b) for k in ka]) + np.mean([C.s_pair(k, a, b) for k in kb])) / 2
                    e = [C.s_pair(k, a, b) <= thr for k in C.keys(a, seeds_eval)] + [C.s_pair(k, a, b) > thr for k in C.keys(b, seeds_eval)]
                    return float(np.mean(e)) if e else None
                leak[f"{a}|{b}"] = {"shared_traces_s5_9": err(range(5, 10)), "fresh_test_s10_19": err(TEST_SEEDS)}
            acc = {t: float(np.mean([accuracy(ds, C.P[k]).mean() for k in C.keys(t, TEST_SEEDS)]))
                   for t in ORDER if C.keys(t, TEST_SEEDS)}
            base = jl(D(ds) / f"probe_{fam}_base.jsonl")
            acc["base"] = float(accuracy(ds, base).mean()) if base else None
            cell = {"H1": h1, "H1_counts": [int(sum(v >= 0.6 for v in t0)), len(t0)],
                    "H2": bool(h2), "H2_detail": {"mean_tpr": float(np.mean(tpr)) if tpr else None,
                                                  "n_fpr_le_0.2": int(sum(v <= 0.2 for v in t1)), "n_pairs": len(t1),
                                                  "mean_fpr_rel": float(np.mean(t1)) if t1 else None},
                    "H3": h3, "auc_test": auc, "T0": res["T0"], "T1": res["T1"], "T1_nref": budget,
                    "T1_conformal_a0.1": conf, "T1_query_budget": qb, "leakage": leak, "acc_test": acc}
            out[ds][fam] = cell
            print(f"\n[m7/{ds}/{fam}] H1 {h1} {cell['H1_counts']} | H2 {bool(h2)} {cell['H2_detail']} | H3 {h3}")
            for tn in ("T0", "T1"):
                print(f"  {tn} TPR {res[tn]['tpr']}\n  {tn} FPR_rel {res[tn]['fpr_rel']}\n  {tn} FPR_out {res[tn]['fpr_out']}")
            print(f"  AUC(test) {auc}\n  n_ref budget: " + json.dumps({n: {'tpr': np.nanmean([v for v in r['tpr'].values() if v is not None] or [np.nan]),
                                                                          'fpr_rel': np.nanmean([v for v in r['fpr_rel'].values() if v is not None] or [np.nan])}
                                                                      for n, r in budget.items()}, default=float))
            print(f"  conformal a=0.1: TPR {conf['tpr']} FPR_rel {conf['fpr_rel']}\n  query budget {qb}\n  leakage {leak}\n  acc {acc}",
                  flush=True)
    verdict = {}
    for h in ("H1", "H2"):
        g = [out["gsm"].get(f, {}).get(h) for f in FAMS]
        m = [out["math"].get(f, {}).get(h) for f in FAMS]
        verdict[h] = {"gsm": g, "math": m}
    H1 = all(verdict["H1"]["gsm"])
    H2 = all(verdict["H2"]["gsm"]) and any(verdict["H2"]["math"])
    verdict["H3_math"] = [out["math"].get(f, {}).get("H3") for f in FAMS]
    if any(v is None for v in verdict["H2"]["gsm"]):
        concl = "INCOMPLETE: a GSM8K cell has no usable students"
    elif not any(verdict["H2"]["gsm"]):
        concl = "KILL: reference students do not restore specificity on fresh students (both GSM8K cells fail)"
    elif not all(verdict["H2"]["gsm"]):
        concl = "H2 fails as pre-registered: family-dependent"
    elif not any(verdict["H2"]["math"]):
        concl = "H2 restricted: GSM8K only"
    else:
        concl = "H2 passes"
    print("\n== PREDICTIONS ==\n" + json.dumps(verdict) + f"\nH1 (no-reference collapse replicates): {H1}\nH2 (primary): {H2}\n=> {concl}")
    out["verdict"] = {**verdict, "H1": H1, "H2": H2, "conclusion": concl}
    (OUT / "m7_result.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
