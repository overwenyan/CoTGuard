"""M6 EXPLORATORY analyses (NOT pre-registered; run after score_m6.py, 2026-09-15).
1) manipulation-check detail (per-student accuracy / length ratio), 9-way mean-probability matrix;
2) student-level separation with the pairwise read-out (leave-one-student-out midpoint threshold)."""
import sys, json, numpy as np
sys.path.insert(0, "."); 
import score_m6 as S
from run_m6 import ORDER
for fam in S.FAMS:
    base = S.probes(fam, "base"); ba = S.accuracy(base).mean(); bl = np.mean([len(r["text"]) for r in base])
    print(f"== {fam} base acc {ba:.3f} chars {bl:.0f}")
    for t in ORDER:
        tch = np.mean([len(x) for x in S.teacher_texts(t, "pool")])
        cells = []
        for s in range(5):
            pr = S.probes(fam, f"grid_{t}_s{s}")
            a = S.accuracy(pr).mean(); L = np.mean([len(r["text"]) for r in pr])
            cells.append(f"{a:.2f}/{L/tch:.2f}{'' if (a>=ba or 0.67<=L/tch<=1.5) else '*'}")
        print(f"  {t:<12} tchars {tch:5.0f}  acc/lenratio: {' '.join(cells)}")
# E2 scores
fam = "qwen15"
avail = ORDER
X, y = [], []
for i, t in enumerate(avail):
    tx = S.teacher_texts(t, "r300"); X += tx; y += [i]*len(tx)
vec = S.features("tfidf")(); from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=3000, C=4.0).fit(vec.fit_transform(X), y)
M = np.zeros((9, 9))
for i, t in enumerate(avail):
    M[i] = np.mean([clf.predict_proba(vec.transform([r["text"] for r in S.probes(fam, f"grid_{t}_s{s}")])).mean(0) for s in range(5)], 0)
print("qwen15 mean P(col) for students of row teacher")
print(" "*12 + " ".join(f"{t[:9]:>9}" for t in avail))
for i, t in enumerate(avail):
    print(f"{t:<12}" + " ".join(f"{v:9.3f}" for v in M[i]))
# argmax closed-set per student

from sklearn.metrics import roc_auc_score
for fam in S.FAMS:
    for a, b in S.PAIRS_D1 + S.PAIRS_D2 + S.PAIRS_SIB:
        Xa, Xb = S.teacher_texts(a, "r300"), S.teacher_texts(b, "r300")
        vec = S.features("tfidf")()
        clf = LogisticRegression(max_iter=3000, C=4.0).fit(vec.fit_transform(Xa + Xb), [1]*len(Xa) + [0]*len(Xb))
        sa = [clf.predict_proba(vec.transform([r["text"] for r in S.probes(fam, f"grid_{a}_s{s}")]))[:, 1].mean() for s in range(5)]
        sb = [clf.predict_proba(vec.transform([r["text"] for r in S.probes(fam, f"grid_{b}_s{s}")]))[:, 1].mean() for s in range(5)]
        # leave-one-student-out threshold test: midpoint of the other students' means
        err = 0
        for i in range(5):
            for side, arr, oth in [(1, sa, sb), (0, sb, sa)]:
                rest = [x for j, x in enumerate(arr) if j != i]
                thr = (np.mean(rest) + np.mean(oth)) / 2
                err += (arr[i] > thr) != side
        print(f"{fam:<8} {a+'|'+b:<26} student-AUC {roc_auc_score([1]*5+[0]*5, sa+sb):.2f}  LOO errors {err}/10  "
              f"min(a) {min(sa):.3f} max(b) {max(sb):.3f}")
