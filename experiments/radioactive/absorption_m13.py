"""Absorption diagnostic for the 7B cell (M13), run retroactively — REPORTED, NEVER A GATE.

M14's amended manipulation check asks whether a teacher's students separate from the *untuned base model* under a
teacher-vs-base read-out (AUC >= 0.90). M13 passed the older accuracy-or-length rule, so the two 7B cells would
otherwise be compared on different checks. This computes the new check on M13's teachers as a diagnostic: it cannot
change M13's verdict (advisor round 15, m14_design.md amendment 2), and the paper says so wherever it is reported.

Per teacher t: fit TF-IDF + LR on t's R300 traces (label 1) vs the untuned Qwen2.5-7B-Instruct's 300 probe outputs
(label 0); score t's 13 students' probe outputs and the base outputs; report the per-output AUC separating them.

    python absorption_m13.py   ->  data_m7/tulu_gsm/m13_absorption.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, __import__("os").path.dirname(__file__))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "relay"))
from run_m7 import D, ORDER, jl, splits  # noqa: E402
from score_m13 import REF, TEST  # noqa: E402

FAM = "qwen7b"
THRESH = 0.90


def main():
    d = D("gsm")
    pr = splits("gsm")
    r300 = set(pr["r300"])
    base = [r["text"] for r in jl(d / f"probe_{FAM}_base.jsonl") or []]
    if not base:
        sys.exit("no base probe outputs")
    out = {"threshold_reported_not_gate": THRESH, "n_base_outputs": len(base), "teachers": {}}
    for t in ORDER:
        traces = [r["text"] for r in jl(d / f"teacher_{t}_ref.jsonl") or [] if r["qid"] in r300 and r["text"].strip()]
        if not traces:
            continue
        v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
        X = v.fit_transform(traces + base)
        clf = LogisticRegression(max_iter=3000, C=4.0).fit(X, [1] * len(traces) + [0] * len(base))
        stu, per_student = [], {}
        for s in REF + TEST:
            rows = jl(d / f"probe_{FAM}_grid_{t}_s{s}.jsonl")
            if not rows:
                continue
            p = clf.predict_proba(v.transform([r["text"] for r in rows]))[:, 1]
            stu.append(p)
            per_student[f"s{s}"] = float(p.mean())
        if not stu:
            continue
        sp = np.concatenate(stu)
        bp = clf.predict_proba(v.transform(base))[:, 1]
        auc = roc_auc_score([1] * len(sp) + [0] * len(bp), np.concatenate([sp, bp]))
        out["teachers"][t] = {"auc_students_vs_base": float(auc), "n_students": len(stu),
                              "mean_student_score": float(sp.mean()), "mean_base_score": float(bp.mean()),
                              "would_pass_m14_check": bool(auc >= THRESH), "per_student_mean": per_student}
        print(f"[absorption] {t:12s} AUC {auc:.3f}  students {len(stu)}  "
              f"{'>= 0.90' if auc >= THRESH else 'BELOW 0.90'}", flush=True)
    aucs = [v["auc_students_vs_base"] for v in out["teachers"].values()]
    out["summary"] = {"min_auc": min(aucs), "max_auc": max(aucs),
                      "n_below_threshold": sum(a < THRESH for a in aucs), "n_teachers": len(aucs)}
    (d / "m13_absorption.json").write_text(json.dumps(out, indent=1))
    print(f"[absorption] {out['summary']['n_below_threshold']} of {len(aucs)} teachers below {THRESH}; "
          f"range {min(aucs):.3f}–{max(aucs):.3f}\n[absorption] wrote {d / 'm13_absorption.json'}")


if __name__ == "__main__":
    main()
