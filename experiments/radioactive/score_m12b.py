"""M12 arm B scoring (.pipeline/docs/m12_design.md, pre-registered 09790f8): scaffold-only rewrite vs full imitation."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, __import__("os").path.dirname(__file__))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "relay"))
from run_m7 import D, TEST_SEEDS, jl, splits  # noqa: E402
from run_m9b import ATTACKS  # noqa: E402
from run_m12b import SEEDS, name  # noqa: E402
from score_m8 import Cell as BaseCell  # noqa: E402
from answer_v2 import correct_v2, extract_answer_v2  # noqa: E402

FAMS = ["qwen15", "llama1b"]
SCAFFOLD = set("step steps first firstly second next then finally lastly so thus therefore hence now let let's we "
               "calculate compute determine find state answer final boxed total result solution".split())
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def recall(orig, new):
    a, b = Counter(orig), Counter(new)
    return 1.0 if not a else sum((a & b).values()) / sum(a.values())


def content_tokens(t):
    t = t.lower()
    nums = [n.replace(",", "") for n in NUM.findall(t)]
    words = [w for w in WORD.findall(NUM.sub(" ", t)) if w not in SCAFFOLD]
    return nums, words


class Cell(BaseCell):
    def load_extra(self):
        out = {}
        for owner, target in ATTACKS:
            for s in SEEDS:
                rows = jl(self.d / f"probe_{self.fam}_grid_{name(owner, target)}_s{s}.jsonl")
                if rows:
                    out[("ad2scaf", owner, target, s)] = [r["text"] for r in rows]
        return out


def validity():
    info, void = {}, set()
    for owner, target in ATTACKS:
        rows = jl(D("gsm") / f"teacher_{name(owner, target)}.jsonl")
        if not rows:
            continue
        ans = float(np.mean([(x := extract_answer_v2(r["text"])) is not None and (y := extract_answer_v2(r["orig_text"])) is not None
                             and abs(x - y) < 1e-6 for r in rows]))
        rn, rw = [], []
        for r in rows:
            on, ow = content_tokens(r["orig_text"]); nn, nw = content_tokens(r["text"])
            rn.append(recall(on, nn)); rw.append(recall(ow, nw))
        info[f"{owner}->{target}"] = {"answer_preserved": ans, "number_recall": float(np.mean(rn)), "word_recall": float(np.mean(rw))}
        if ans < 0.9 or np.mean(rn) < 0.9 or np.mean(rw) < 0.8:
            void.add((owner, target))
    return void, info


def gap_closed(fam, owner, target):
    r300 = set(splits("gsm")["r300"])
    T = lambda t: [r["text"] for r in jl(D("gsm") / f"teacher_{t}_ref.jsonl") if r["qid"] in r300]
    tx = lambda stem, seeds: [x["text"] for s in seeds for x in (jl(D("gsm") / f"probe_{fam}_grid_{stem}_s{s}.jsonl") or [])]
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
    clf = LogisticRegression(max_iter=3000, C=4.0).fit(v.fit_transform(T(owner) + T(target)), [1] * 300 + [0] * 300)
    w = clf.coef_[0]; mu = lambda X: np.asarray(v.transform(X).mean(0)).ravel()
    m_own = mu(tx(owner, TEST_SEEDS)); gap = float((mu(tx(target, TEST_SEEDS)) - m_own) @ w)
    return float((mu(tx(name(owner, target), SEEDS)) - m_own) @ w) / gap


def main():
    void, info = validity()
    print("[m12b] validity: " + json.dumps(info) + f"\n[m12b] void: {sorted(void)}", flush=True)
    full = json.loads((D("gsm") / "m9b_result.json").read_text())["families"]
    out = {"validity": info, "void": [f"{o}->{t}" for o, t in sorted(void)], "families": {}}
    for fam in FAMS:
        C = Cell("gsm", fam)
        chk = float(np.mean([C.rate(C.keys(a, TEST_SEEDS), a, "T1") for a in C.avail]))
        print(f"\n[m12b/{fam}] T1 TPR check on unattacked M7 students (must be 1.0): {chk}", flush=True)
        F = {(r["owner"], r["target"]): r for r in full[fam]["rows"]}
        rows = []
        for owner, target in ATTACKS:
            ks = [k for k in C.mix if k[0] == "ad2scaf" and k[1] == owner and k[2] == target]
            if not ks:
                continue
            r = {"owner": owner, "target": target, "void": (owner, target) in void,
                 "tpr": C.rate(ks, owner, "T1"), "spoof": C.rate(ks, target, "T1"),
                 "gap_closed": gap_closed(fam, owner, target),
                 "acc": float(np.mean([np.mean([correct_v2(q["text"], q["gold"]) for q in jl(D("gsm") / f"probe_{fam}_grid_{name(owner, target)}_s{k[3]}.jsonl")]) for k in ks])),
                 "full_tpr": F[(owner, target)]["tpr"], "full_spoof": F[(owner, target)]["spoof"], "full_void": F[(owner, target)]["void"]}
            rows.append(r)
            print(f"  [m12b/{fam}] {owner:>12} -> {target:<12}{' [VOID]' if r['void'] else '       '} scaffold-only: detect "
                  f"{r['tpr']:.2f} claim {r['spoof']:.2f} gap {r['gap_closed']:+.2f} | full imitation{' [VOID]' if r['full_void'] else ''}: "
                  f"detect {r['full_tpr']:.2f} claim {r['full_spoof']:.2f} | acc {r['acc']:.3f}", flush=True)
        comp = [r for r in rows if not r["void"] and not r["full_void"]]
        n = len(comp)
        match = sum(abs(r["tpr"] - r["full_tpr"]) <= 0.34 and abs(r["spoof"] - r["full_spoof"]) <= 0.34 for r in comp)
        need = int(np.ceil(n * 5 / 6)) if n else 0
        d_tpr = float(np.mean([r["tpr"] for r in comp]) - np.mean([r["full_tpr"] for r in comp])) if n else None
        d_spoof = float(np.mean([r["full_spoof"] for r in comp]) - np.mean([r["spoof"] for r in comp])) if n else None
        out["families"][fam] = {"rows": rows, "n_comparable": n, "n_match": match, "need": need,
                                "S1_fam": n >= 4 and match >= need,
                                "S2_fam": n >= 4 and (d_tpr >= 0.25 or d_spoof >= 0.25),
                                "detect_scaffold_minus_full": d_tpr, "claim_full_minus_scaffold": d_spoof, "tpr_check": chk}
        f = out["families"][fam]
        print(f"  [m12b/{fam}] comparable {n}; matches {match} (need {need}) | detection scaffold−full {d_tpr:+.3f}, "
              f"claim full−scaffold {d_spoof:+.3f}", flush=True)
    fams = out["families"]
    inconclusive = any(fams[f]["n_comparable"] < 4 for f in FAMS)
    S1 = not inconclusive and all(fams[f]["S1_fam"] for f in FAMS)
    S2 = not inconclusive and all(fams[f]["S2_fam"] for f in FAMS)
    verdict = ("INCONCLUSIVE: fewer than 4 comparable attacks" if inconclusive else
               "S1 holds: scaffold-only reproduces full imitation (diagnostic, not a second attack family)" if S1 and not S2 else
               "S2 holds: content contributes; scaffold-only is weaker" if S2 and not S1 else
               "MIXED: report per attack; no causal sentence")
    out["S1"], out["S2"], out["verdict"] = S1, S2, verdict
    print(f"\n== M12 arm B ==\nS1 {S1}  S2 {S2}\n=> {verdict}")
    (D("gsm") / "m12b_result.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
