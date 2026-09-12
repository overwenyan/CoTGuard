"""M3 v3 scoring and pre-registered gates (m3_design.md v3, commit c500e86).

Owner-side test over a 64-key generator pool: p = (1 + #{j != k : s_j >= s_k}) / 64, alpha = 0.05
(rank <= 3). A (key, student) pair counts only if p <= 3/64 AND the key is not top-3 on the same
family's clean student or base model (conditional check). Gates: G-R0 tulu_gsm raw, G-R1 paraphrase,
G-R2 qwen_gsm, G-R3 tulu_arc (after a manipulation check); filter / compress and the query budget are
reported against the predictions written in the design.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_m3c import SETTINGS, STUDENTS, TRAINED, correct  # noqa: E402

OUT = pathlib.Path(os.environ.get("M3C_OUT", HERE / "data3"))
FAMS = list(STUDENTS)
ALPHA_RANK = 3


def jl(fp):
    return [json.loads(l) for l in open(fp) if l.strip()] if fp.exists() else None


class Readout:
    def __init__(self, d, prefix, classes):
        X, y = [], []
        for i, c in enumerate(classes):
            for r in jl(d / f"{prefix}_{c}.jsonl") or []:
                if r["text"].strip():
                    X.append(r["text"]); y.append(i)
        self.classes = classes
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
        self.clf = LogisticRegression(max_iter=3000, C=4.0).fit(self.vec.fit_transform(X), np.array(y))
        self.n = len(X)

    def probs(self, texts):
        return self.clf.predict_proba(self.vec.transform(texts))    # rows = outputs, cols = classes


def rank_p(P, keys, k, idx=None):
    s = (P if idx is None else P[idx]).mean(0)
    col = {c: i for i, c in enumerate(keys)}
    return (1 + sum(s[col[j]] >= s[col[k]] for j in keys if j != k)) / len(keys), s[col[k]]


def evaluate(d, ro, keys, corpus, fams):
    """-> per-pair rows, and the innocent-student ranks for the hot-key diagnostic"""
    kidx = [ro.classes.index(k) for k in keys]
    P = {}
    for fam in fams:
        for name in [f"{corpus}_{k}" for k in TRAINED] + [f"{corpus}_clean", "base"]:
            rows = jl(d / f"out_{fam}_{name}.jsonl")
            if rows:
                P[(fam, name)] = ro.probs([r["text"] for r in rows])[:, kidx]
    pairs = []
    for fam in fams:
        for k in TRAINED:
            if (fam, f"{corpus}_{k}") not in P:
                continue
            p, s = rank_p(P[(fam, f"{corpus}_{k}")], keys, k)
            innocent = [rank_p(P[(fam, n)], keys, k)[0] for n in [f"{corpus}_clean", "base"] if (fam, n) in P]
            ok = p <= ALPHA_RANK / len(keys) and all(q > ALPHA_RANK / len(keys) for q in innocent)
            pairs.append({"fam": fam, "key": k, "p": p, "score": float(s), "innocent_p": innocent, "ok": bool(ok)})
    hot = None
    inn = [P[(f, n)] for f in fams for n in [f"{corpus}_clean", "base"] if (f, n) in P]
    if len(inn) == 4:
        top = [set(np.argsort(-M.mean(0))[:ALPHA_RANK]) for M in inn]
        hot = float(np.mean([sum(j in t for t in top) >= 2 for j in range(len(keys))]))
    return pairs, hot, P


def show(title, pairs, hot=None):
    n_ok = sum(r["ok"] for r in pairs)
    print(f"\n== {title}: {n_ok}/{len(pairs)} pairs pass (p <= 3/64 and not top-3 on clean/base) ==")
    for r in pairs:
        print(f"  {r['fam']:<8}{r['key']}  p={r['p']:.3f}  score={r['score']:.3f}  "
              f"innocent p={['%.3f' % q for q in r['innocent_p']]}  {'ok' if r['ok'] else 'FAIL'}")
    if hot is not None:
        print(f"  hot-key share (top-3 on >= 2 of 4 innocent students): {hot:.3f} (predicted <= 0.10)")
    return n_ok


def acc_chars(d, fam, name, domain):
    rows = jl(d / f"out_{fam}_{name}.jsonl")
    if not rows:
        return None, None
    return float(np.mean([correct(domain, r["text"], r["gold"]) for r in rows])), float(np.mean([len(r["text"]) for r in rows]))


def main():
    res = {}
    for setting, (_, domain) in SETTINGS.items():
        d = OUT / setting
        if not (d / "teacher_clean.jsonl").exists():
            print(f"\n[{setting}] no teacher data, skipped"); continue
        keys = [k for k in json.loads((HERE / "keys_v3.json").read_text()) if (d / f"teacher_{k}.jsonl").exists()]
        ro = Readout(d, "teacher", keys + ["clean"])
        print(f"\n######## {setting}: read-out on {ro.n} teacher traces, {len(keys)} keys + clean")
        pairs, hot, P = evaluate(d, ro, keys, "raw", FAMS)
        res[setting] = {"raw": {"pairs": pairs, "hot": hot}}
        n_ok = show(f"{setting} / raw", pairs, hot)

        if domain == "arc":        # manipulation check before G-R3
            tch = np.mean([len(r["text"]) for k in TRAINED for r in jl(d / f"teacher_{k}.jsonl") or []])
            mc = {}
            for fam in FAMS:
                base_acc, _ = acc_chars(d, fam, "base", domain)
                for k in TRAINED:
                    acc, ch = acc_chars(d, fam, f"raw_{k}", domain)
                    if acc is not None:
                        mc[f"{fam}/{k}"] = bool(0.67 <= ch / tch <= 1.5 or acc >= base_acc + 0.05)
                print(f"  manipulation check {fam}: base acc {base_acc}; "
                      + ", ".join(f"{k}:{acc_chars(d, fam, 'raw_' + k, domain)[0]:.2f}/"
                                  f"{acc_chars(d, fam, 'raw_' + k, domain)[1] / tch:.2f}x"
                                  for k in TRAINED if acc_chars(d, fam, 'raw_' + k, domain)[0] is not None))
            res[setting]["manip"] = mc
            res[setting]["G-R3"] = bool(mc and all(mc.values()) and n_ok >= 12)
            print(f"G-R3 manipulation check {'pass' if mc and all(mc.values()) else 'FAIL (void)'}; pairs {n_ok}/16 "
                  f"-> {res[setting]['G-R3']}")
        elif setting == "qwen_gsm":
            res[setting]["G-R2"] = n_ok >= 12
            print(f"G-R2 -> {res[setting]['G-R2']}")
        else:
            res[setting]["G-R0"] = n_ok >= 12
            print(f"G-R0 -> {res[setting]['G-R0']}")
            # query budget
            rng = np.random.default_rng(0)
            print("\n  query budget (share of raw pairs with p <= 3/64):")
            qb = {}
            for n in [10, 20, 50, 100, 200]:
                hits = []
                for fam in FAMS:
                    for k in TRAINED:
                        M = P.get((fam, f"raw_{k}"))
                        if M is None:
                            continue
                        for _ in range(50 if n < len(M) else 1):
                            idx = rng.choice(len(M), min(n, len(M)), replace=False)
                            hits.append(rank_p(M, keys, k, idx)[0] <= ALPHA_RANK / len(keys))
                qb[n] = float(np.mean(hits)) if hits else None
                print(f"    n={n:<4} {qb[n]}")
            res[setting]["query_budget"] = qb

            # attacks
            for corpus in ["filter", "para", "compress"]:
                fams = FAMS if corpus == "para" else ["qwen15"]
                if not any((d / f"out_{f}_{corpus}_{k}.jsonl").exists() for f in fams for k in TRAINED):
                    continue
                pa, hot_a, _ = evaluate(d, ro, keys, corpus, fams)
                entry = {"agnostic": {"pairs": pa, "hot": hot_a}}
                n_ag = show(f"{setting} / {corpus} / agnostic read-out", pa)
                n_aw = None
                if corpus != "filter" and all((d / f"owner_{corpus}_{c}.jsonl").exists() for c in keys + ["clean"]):
                    ro_aw = Readout(d, f"owner_{corpus}", keys + ["clean"])
                    pw, _, _ = evaluate(d, ro_aw, keys, corpus, fams)
                    entry["aware"] = {"pairs": pw}
                    n_aw = show(f"{setting} / {corpus} / aware read-out ({ro_aw.n} owner rewrites)", pw)
                cost = {}
                for fam in fams:
                    a_att = [acc_chars(d, fam, f"{corpus}_{k}", "gsm")[0] for k in TRAINED]
                    a_raw = [acc_chars(d, fam, f"raw_{k}", "gsm")[0] for k in TRAINED]
                    ok = [(x, y) for x, y in zip(a_att, a_raw) if x is not None and y is not None]
                    if ok:
                        cost[fam] = {"attacked_acc": float(np.mean([x for x, _ in ok])), "raw_acc": float(np.mean([y for _, y in ok]))}
                        print(f"  attack cost {fam}: student acc {cost[fam]['attacked_acc']:.3f} (attacked) vs "
                              f"{cost[fam]['raw_acc']:.3f} (raw)")
                entry["cost"] = cost
                if corpus == "para":
                    entry["G-R1"] = "agnostic-pass" if n_ag >= 12 else ("aware-pass" if (n_aw or 0) >= 12 else "fail")
                    print(f"G-R1 -> {entry['G-R1']}")
                elif corpus == "filter":
                    print(f"filter prediction (>= 6/8 keep): {'held' if n_ag >= 6 else 'WRONG'}")
                else:
                    print(f"compress prediction (agnostic breaks >= 4/8, i.e. <= 4 pass): "
                          f"{'held' if n_ag <= 4 else 'WRONG'}")
                res[setting][corpus] = entry

    print("\n== SUMMARY ==")
    for s, r in res.items():
        print(f"  {s}: " + ", ".join(f"{g}={r[g]}" for g in ["G-R0", "G-R2", "G-R3"] if g in r)
              + (f", G-R1={r['para']['G-R1']}" if "para" in r and "G-R1" in r["para"] else ""))
    (OUT / "m3c_result.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
