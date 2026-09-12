"""M3 scoring and pre-registered gate (m3_design.md §3, commit fe2ad1b).

The read-out (TF-IDF 1-2 grams + logistic regression) is trained on TEACHER traces over 4 classes
(3 keys + clean) and applied to STUDENT outputs. The clean student is the null: a keyed student must
be assigned to its own key more often than the clean student is.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "relay"))
from utility_check import extract_answer, gold_answer  # noqa: E402

OUT = pathlib.Path(__file__).parent / "data"
ARMS = ["k0", "k1", "k5", "clean"]


def jl(fp):
    return [json.loads(l) for l in open(fp) if l.strip()]


def acc_of(rows):
    ok = []
    for r in rows:
        a, g = extract_answer(r["text"]), gold_answer(r["gold"])
        ok.append(a is not None and g is not None and abs(a - g) < 1e-6)
    return float(np.mean(ok))


def main(student="qwen15"):
    teach = {a: jl(OUT / f"teacher_{a}.jsonl") for a in ARMS}
    X = [r["text"] for a in ARMS for r in teach[a]]
    y = np.array([i for i, a in enumerate(ARMS) for _ in teach[a]])
    g = np.array([r["qid"] for a in ARMS for r in teach[a]])
    mk = lambda: TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=60000)
    accs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):        # ceiling: problem-disjoint folds
        v = mk()
        clf0 = LogisticRegression(max_iter=2000, C=4.0).fit(v.fit_transform([X[i] for i in tr]), y[tr])
        accs.append(float(np.mean(clf0.predict(v.transform([X[i] for i in te])) == y[te])))
    ceiling = float(np.mean(accs))
    vec = mk()
    clf = LogisticRegression(max_iter=2000, C=4.0).fit(vec.fit_transform(X), y)

    outs = {a: jl(OUT / f"out_{student}_{a}.jsonl") for a in ARMS + ["base"]}
    base_acc = acc_of(outs["base"])
    teach_len = float(np.mean([len(r["text"]) for a in ARMS for r in teach[a]]))
    print(f"== M3 ({student}) ==")
    print(f"teacher read-out ceiling (4-way, problem-disjoint): {ceiling:.3f} (chance 0.250)")
    print(f"base student accuracy {base_acc:.3f}; teacher mean chars {teach_len:.0f}")
    print(f"{'arm':<7}{'acc':>7}{'d_acc':>8}{'chars':>7}" + "".join(f"{'->' + a:>9}" for a in ARMS))
    rates, rows = {}, {}
    for a in ARMS:
        P = clf.predict(vec.transform([r["text"] for r in outs[a]]))
        rates[a] = [float(np.mean(P == i)) for i in range(len(ARMS))]
        rows[a] = {"acc": acc_of(outs[a]), "d_acc": acc_of(outs[a]) - base_acc,
                   "chars": float(np.mean([len(r["text"]) for r in outs[a]])), "dist": rates[a]}
        print(f"{a:<7}{rows[a]['acc']:>7.3f}{rows[a]['d_acc']:>+8.3f}{rows[a]['chars']:>7.0f}"
              + "".join(f"{d:>9.3f}" for d in rates[a]))
    keys = ARMS[:3]
    n = len(outs["k0"])
    learned = [a for a in keys if rows[a]["d_acc"] >= 0.05 or 0.67 <= rows[a]["chars"] / teach_len <= 1.5]
    p_self = {a: rates[a][ARMS.index(a)] for a in keys}
    null = {a: rates["clean"][ARMS.index(a)] for a in keys}
    sig = {a: float(binomtest(int(round(p_self[a] * n)), n, min(max(null[a], 1e-6), 1 - 1e-6),
                              alternative="greater").pvalue) for a in keys}
    m3_0 = len(learned) == 3
    mean_self = float(np.mean([p_self[a] for a in keys]))
    m3_a = mean_self >= 0.50 and sum(p_self[a] > null[a] and sig[a] <= 0.05 for a in keys) >= 2
    print("\n== PRE-REGISTERED GATE ==")
    print(f"M3-0 (students learned): {m3_0}  [learned {learned}]")
    for a in keys:
        print(f"   {a}: self-rate {p_self[a]:.3f} vs clean-student null {null[a]:.3f}  binomial p={sig[a]:.3g}")
    print(f"M3-a (mean self-rate >= 0.50 and beats null on >=2/3): {m3_a}  [mean {mean_self:.3f}]")
    if not m3_0:
        print("=> VOID: the students did not learn from the traces; fix training before concluding")
    elif m3_a:
        print("=> M3 CONTINUES")
    else:
        print("=> M3 STOPS: a passive prompt-only signature is NOT radioactive — owners cannot trace a "
              "distilled student without actively modifying traces (cf. ACL 2026 trace rewriting)")
    (OUT / f"m3_result_{student}.json").write_text(json.dumps(
        {"ceiling": ceiling, "base_acc": base_acc, "rows": rows, "p_self": p_self, "null": null,
         "binom_p": sig, "M3-0": m3_0, "M3-a": m3_a, "mean_self": mean_self}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
