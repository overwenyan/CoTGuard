"""M3 v2 scoring and pre-registered gate (m3_design.md v2, commit 86738f7).

Read-out: TF-IDF + logistic regression over 17 classes (16 keys + clean), trained on TEACHER traces
only. For a suspect student, score every key on its outputs and rank the owner's key:
    p_k = (1 + #{j != k : score_j >= score_k}) / 16      (wrong-key calibration, EXP-005)
G-M3b: p <= 1/16 for >= 4 of 6 trained keys on both student families, with the empirical
false-positive share over untrained (key, student) pairs <= 0.10.
G-M3c: the key's score on its own student must exceed its score on the clean student AND on the
untuned base model of the same family.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE = pathlib.Path(__file__).parent
OUT = HERE / "data2"
KEYS = json.loads((HERE / "keys_v2.json").read_text())
KEY_IDS = sorted(KEYS)
TRAINED = ["key_01", "key_03", "key_08", "key_09", "key_12", "key_13"]
FAMS = ["qwen15", "llama1b"]
ABLATIONS = ["mix", "key_01_n150", "key_01_n600", "active"]


def jl(fp):
    return [json.loads(l) for l in open(fp) if l.strip()]


def main():
    classes = KEY_IDS + ["clean"]
    X, y = [], []
    for i, c in enumerate(classes):
        for r in jl(OUT / f"teacher_{c}.jsonl"):
            if r["text"].strip():
                X.append(r["text"]); y.append(i)
    y = np.array(y)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
    clf = LogisticRegression(max_iter=3000, C=4.0).fit(vec.fit_transform(X), y)
    print(f"read-out trained on {len(X)} teacher traces, {len(classes)} classes")

    def scores(fp):
        P = clf.predict_proba(vec.transform([r["text"] for r in jl(fp)]))
        return {c: float(P[:, i].mean()) for i, c in enumerate(classes)}

    S, res = {}, {}
    for fam in FAMS:
        for arm in TRAINED + ["clean", "base"] + ABLATIONS:
            fp = OUT / f"out_{fam}_{arm}.jsonl"
            if fp.exists():
                S[(fam, arm)] = scores(fp)

    def pval(sc, k):
        return (1 + sum(sc[j] >= sc[k] for j in KEY_IDS if j != k)) / len(KEY_IDS)

    print("\n== owner-side test: p-value of the true key on its own student (p <= 1/16 = 0.0625 is the target) ==")
    print(f"{'key':<9}{'type':<11}" + "".join(f"{f:>22}" for f in FAMS))
    per_fam = {f: [] for f in FAMS}
    for k in TRAINED:
        line = f"{k:<9}{KEYS[k]['type']:<11}"
        for fam in FAMS:
            if (fam, k) not in S:
                line += f"{'-':>22}"; continue
            sc = S[(fam, k)]
            p = pval(sc, k)
            strict = (S[(fam, "clean")][k] if (fam, "clean") in S else 1) < sc[k] and \
                     (S[(fam, "base")][k] if (fam, "base") in S else 1) < sc[k]
            per_fam[fam].append((k, p, strict))
            line += f"   p={p:.3f} score={sc[k]:.3f} {'strict-ok' if strict else 'STRICT-FAIL'}"
        print(line)

    fp_pairs = []
    for (fam, arm), sc in S.items():
        for k in KEY_IDS:
            if arm == k or (arm in ABLATIONS and k == "key_01"):
                continue                                    # these are true positives, not FP candidates
            fp_pairs.append(pval(sc, k) <= 1 / len(KEY_IDS))
    fpr = float(np.mean(fp_pairs))
    print(f"\nempirical false-positive share over {len(fp_pairs)} untrained (key, student) pairs: {fpr:.3f}")

    print("\n== ablations (qwen15) ==")
    for arm in ABLATIONS:
        if ("qwen15", arm) not in S:
            continue
        sc = S[("qwen15", arm)]
        k = "key_01"
        rows = jl(OUT / f"out_qwen15_{arm}.jsonl")
        hit = np.mean([("CHECK-DIGIT" in (r.get("trigger_out") or "")) for r in rows if r.get("trigger_out")])
        print(f"  {arm:<14} key_01 score {sc[k]:.3f} p={pval(sc, k):.3f}   trigger-hit {hit:.2f}")
    for fam in FAMS:
        for arm in ["clean", "base"]:
            if (fam, arm) in S:
                rows = jl(OUT / f"out_{fam}_{arm}.jsonl")
                hit = np.mean([("CHECK-DIGIT" in (r.get("trigger_out") or "")) for r in rows if r.get("trigger_out")])
                print(f"  {fam}/{arm:<10} trigger-hit {hit:.2f} (active-watermark control)")

    ok = {f: sum(p <= 1 / len(KEY_IDS) for _, p, _ in per_fam[f]) for f in FAMS if per_fam[f]}
    strict_ok = {f: sum(p <= 1 / len(KEY_IDS) and s for _, p, s in per_fam[f]) for f in FAMS if per_fam[f]}
    g_b = all(v >= 4 for v in ok.values()) and len(ok) == len(FAMS) and fpr <= 0.10
    g_c = all(v >= 4 for v in strict_ok.values()) and len(strict_ok) == len(FAMS)
    print("\n== PRE-REGISTERED GATE (v2) ==")
    print(f"G-M3b keys with p<=1/16 per family: {ok} (need >=4 on both); FPR {fpr:.3f} (need <=0.10) -> {g_b}")
    print(f"G-M3c same keys also beating clean+base: {strict_ok} -> {g_c}")
    print("=> M3 v2 PASS: passive prompt-only provenance holds under an owner-side test" if g_b and g_c
          else "=> M3 v2 FAIL: report and decide")
    (OUT / "m3b_result.json").write_text(json.dumps(
        {"per_family": {f: per_fam[f] for f in per_fam}, "fpr": fpr, "ok": ok, "strict_ok": strict_ok,
         "G-M3b": g_b, "G-M3c": g_c,
         "scores": {f"{f}|{a}": s for (f, a), s in S.items()}}, indent=1))


if __name__ == "__main__":
    main()
