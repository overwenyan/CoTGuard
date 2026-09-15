"""M4 pilot scoring and gates (.pipeline/docs/m4_design.md, pre-registered 4578207)."""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, __import__("os").path.dirname(__file__))
from run_m4 import (D4, MOVES, N_PROBE, OUT, OWNER_SEED, SMOKE, embed_all, moves_for,  # noqa: E402
                    positions, problems, read_jsonl, s_set, train_problems)
from run_m3c import correct  # noqa: E402

N_DECOY = 50 if SMOKE else 1000
DECOY0 = 1_000_000


def main():
    emb = embed_all()
    served_q = {train_problems()[i]["qid"] for i in positions()[:N_PROBE]}
    # ---------------- move detector: 4 moves + clean, teacher traces of S2000 minus served probes
    clean = {r["qid"]: r for r in read_jsonl(D4 / "dil_clean_all.jsonl")}
    S = s_set()
    X, y, Xc, yc = [], [], [], []
    tm = {m: {r["qid"]: r for r in read_jsonl(OUT / f"teacher_move_{m}.jsonl")} for m in MOVES}
    for p in S:
        q = p["qid"]
        items = [(tm[m][q]["text"], k) for k, m in enumerate(MOVES)] + [(clean[q]["text"], 4)]
        for t, k in items:
            (Xc if q in served_q else X).append(t)
            (yc if q in served_q else y).append(k)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
    det = LogisticRegression(max_iter=3000, C=4.0).fit(vec.fit_transform(X), np.array(y))
    comp = float(np.mean(det.predict(vec.transform(Xc)) == np.array(yc)))
    print(f"[m4] move detector: trained on {len(X)} teacher traces; held-out (served-problem) teacher-trace accuracy "
          f"{comp:.3f} (5 classes, chance 0.2)")
    R = {"detector_heldout_acc": comp}

    def stats(name, split, key_seed, b, idx=None):
        rows = read_jsonl(OUT / f"probe_{name}_{split}.jsonl")
        if rows is None:
            return None
        if idx is not None:
            rows = [rows[i] for i in idx]
        Q = det.predict_proba(vec.transform([r["text"] for r in rows]))        # n x 5
        qids = [r["qid"] for r in rows]
        def A(seed):
            mv = moves_for(seed, b, qids, emb)
            return Q[np.arange(len(rows)), mv].mean(), float(np.mean(Q.argmax(1) == mv))
        a_own, h_own = A(key_seed)
        dec = np.array([A(DECOY0 + i)[0] for i in range(N_DECOY)])
        p = (1 + np.sum(dec >= a_own)) / (1 + N_DECOY)
        return {"A": float(a_own), "hard": h_own, "decoy_mean": float(dec.mean()), "lift": float(a_own - dec.mean()),
                "p": float(p), "n": len(rows)}

    res = {}
    for b in [1, 2, 3]:
        for name in [f"key100_b{b}", f"imit100_b{b}", f"key10_b{b}", "clean2k_clean", "dil0_clean"]:
            for split in ["served", "heldout"]:
                s = stats(name, split, OWNER_SEED + b, b)
                if s is not None:
                    res[f"{name}|{split}|b{b}"] = s
    for k, s in res.items():
        print(f"  {k:<32} A={s['A']:.3f} decoy={s['decoy_mean']:.3f} lift={s['lift']:+.3f} hard={s['hard']:.3f} "
              f"p={s['p']:.4f}")
    R["tests"] = res
    g = lambda name, split, b: res.get(f"{name}|{split}|b{b}")
    G1 = any(g(f"key100_b{b}", "heldout", b) and g(f"key100_b{b}", "heldout", b)["p"] <= 0.01
             and g(f"key100_b{b}", "heldout", b)["lift"] >= 0.05 for b in [1, 2, 3])
    G2 = all(g(f"key100_b{b}", "served", b) and g(f"key100_b{b}", "served", b)["p"] <= 0.01 for b in [1, 2, 3])
    G3 = any((g(f"key10_b{b}", sp, b) or {"p": 1})["p"] <= 0.05 for b in [1, 2, 3] for sp in ["served", "heldout"])
    spec_keys = [k for k in res if k.startswith(("imit100", "clean2k", "dil0_clean"))]
    spec_keys = [k for k in spec_keys if not k.startswith("imit100") or k.endswith(f"|b{k.split('|')[0][-1]}")]
    n_fail = sum(res[k]["p"] <= 0.05 for k in spec_keys)
    G4 = n_fail <= 3 and all(res[k]["p"] > 0.01 for k in spec_keys)   # m4_design G4 (amended pre-data)
    print(f"\n== GATES ==\nG1 held-out learnability: {G1}\nG2 served channel (all b): {G2}\nG3 dilution 10%: {G3}\n"
          f"G4 specificity: {G4} ({n_fail} of {len(spec_keys)} null tests with p <= 0.05; {spec_keys})")
    if (G1 or G2) and G3 and G4:
        verdict = "framing (i): design the full grid"
    elif (G1 or G2) and G4:
        verdict = "signal needs teacher-dominated data: consider (iii) or active probing before a full grid"
    else:
        verdict = "abandon framing (i); decide between (ii) and (iii)"
    print(f"=> {verdict}")
    R.update({"G1": G1, "G2": G2, "G3": G3, "G4": G4, "verdict": verdict})

    # ---------------- reported: query scaling, utility
    rng = np.random.default_rng(0)
    qsc = {}
    for b in [1, 2, 3]:
        for name in [f"key100_b{b}", f"key10_b{b}"]:
            for split in ["served", "heldout"]:
                if g(name, split, b) is None:
                    continue
                for n in [25, 50, 100, 200, N_PROBE]:
                    if n > N_PROBE:
                        continue
                    ps = [stats(name, split, OWNER_SEED + b, b, rng.choice(N_PROBE, n, replace=False))["p"]
                          for _ in range(1 if n == N_PROBE else (3 if SMOKE else 20))]
                    qsc[f"{name}|{split}|N{n}"] = float(np.median(ps))
    print("\n[query scaling, median p]: " + "  ".join(f"{k}:{v:.3f}" for k, v in qsc.items()))
    R["query_scaling_median_p"] = qsc
    util = {}
    for name in [f"key100_b{b}" for b in [1, 2, 3]] + [f"imit100_b{b}" for b in [1, 2, 3]] + \
                [f"key10_b{b}" for b in [1, 2, 3]] + ["clean2k_clean", "dil0_clean"]:
        rows = read_jsonl(OUT / f"probe_{name}_heldout.jsonl")
        if rows:
            util[name] = float(np.mean([correct("gsm", r["text"], r["gold"]) for r in rows]))
    print("[utility, held-out acc]: " + "  ".join(f"{k}:{v:.3f}" for k, v in util.items()))
    R["utility"] = util
    (OUT / "m4_result.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
