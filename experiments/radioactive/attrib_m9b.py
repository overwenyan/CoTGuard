"""M11 section 5 (exploratory, pre-registered as non-gating): what does the M9b imitation rewrite change?

For each valid attack x family: owner-vs-target TF-IDF LR on R300 traces (label 1 = owner). Shift of the attacked
students' mean feature vector relative to the owner's unattacked M7 test students, projected on the coefficient vector;
expressed as the fraction of the owner->target gap closed; and the n-grams that drive the move.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, __import__("os").path.dirname(__file__))
from run_m7 import D, TEST_SEEDS, jl, splits  # noqa: E402
from run_m9b import ATTACKS, SEEDS, name  # noqa: E402

VALID = [a for a in ATTACKS if a not in {("olmoi_dpo", "olmoi_sft"), ("olmoi_dpo", "olmoi_final")}]


def texts(fam, stem, seeds):
    out = []
    for s in seeds:
        r = jl(D("gsm") / f"probe_{fam}_grid_{stem}_s{s}.jsonl")
        if r:
            out += [x["text"] for x in r]
    return out


def main():
    r300 = set(splits("gsm")["r300"])
    T = lambda t: [r["text"] for r in jl(D("gsm") / f"teacher_{t}_ref.jsonl") if r["qid"] in r300]
    res = {}
    for fam in ["qwen15", "llama1b"]:
        for owner, target in VALID:
            v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
            X = v.fit_transform(T(owner) + T(target))
            clf = LogisticRegression(max_iter=3000, C=4.0).fit(X, [1] * 300 + [0] * 300)
            w = clf.coef_[0]
            mu = lambda tx: np.asarray(v.transform(tx).mean(0)).ravel()
            m_own, m_tgt = mu(texts(fam, owner, TEST_SEEDS)), mu(texts(fam, target, TEST_SEEDS))
            m_att = mu(texts(fam, name(owner, target), SEEDS))
            gap = float((m_tgt - m_own) @ w)
            moved = float((m_att - m_own) @ w)
            contrib = (m_att - m_own) * w
            feats = v.get_feature_names_out()
            toward = [(feats[j], round(float(contrib[j]), 4)) for j in np.argsort(contrib)[:8]]
            away = [(feats[j], round(float(contrib[j]), 4)) for j in np.argsort(contrib)[::-1][:5]]
            k = f"{fam}|{owner}->{target}"
            res[k] = {"gap_owner_to_target": gap, "moved": moved, "fraction_of_gap_closed": moved / gap if gap else None,
                      "toward_target": toward, "toward_owner": away}
            print(f"{k:<38} gap {gap:+.4f}  moved {moved:+.4f}  closed {moved / gap:+.2f}\n"
                  f"   toward target: {toward}\n   toward owner : {away}", flush=True)
    (D("gsm") / "m11_attrib_m9b.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
