"""M6 scoring (.pipeline/docs/m6_design.md, pre-registered 437dcd8, amended 6db9a32). Runs in py312."""

from __future__ import annotations

import itertools
import json
import os
import pathlib
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "relay"))
from run_m6 import ORDER, OUT, SMOKE, STUD, jl  # noqa: E402
from run_m3c import correct  # noqa: E402
from utility_check import extract_answer  # noqa: E402

FAMS = ["qwen15", "llama1b"]
LINES = {"tulu": ["tulu_sft", "tulu_dpo", "tulu_rlvr"], "olmoi": ["olmoi_sft", "olmoi_dpo", "olmoi_final"],
         "olmot": ["olmot_sft", "olmot_dpo", "olmot_final"]}
LINE_OF = {t: l for l, ts in LINES.items() for t in ts}
PAIRS_D1 = [(ts[0], ts[1]) for ts in LINES.values()] + [(ts[1], ts[2]) for ts in LINES.values()]
PAIRS_D2 = [(ts[0], ts[2]) for ts in LINES.values()]
PAIRS_SIB = list(zip(LINES["olmoi"], LINES["olmot"]))
PR = json.loads((OUT / "problems.json").read_text())
R300 = set(PR["r300"]); POOL = set(PR["pool"])
N_BOOT = 200 if SMOKE else 2000
_TOK = None


def trunc400(texts):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    return [_TOK.decode(_TOK(t, add_special_tokens=False).input_ids[:400]) for t in texts]


def teacher_texts(t, which):
    ids = R300 if which == "r300" else POOL
    return [r["text"] for r in jl(OUT / f"teacher_{t}.jsonl") or [] if r["qid"] in ids and r["text"].strip()]


def probes(fam, name):
    rows = jl(STUD / f"probe_{fam}_{name}.jsonl")
    return rows


def features(kind):
    if kind == "tfidf":
        return lambda: TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
    raise ValueError(kind)


def pairwise_auc(a, b, fam, trunc=False):
    Xa, Xb = teacher_texts(a, "r300"), teacher_texts(b, "r300")
    if trunc:
        Xa, Xb = trunc400(Xa), trunc400(Xb)
    vec = features("tfidf")()
    clf = LogisticRegression(max_iter=3000, C=4.0).fit(vec.fit_transform(Xa + Xb), [1] * len(Xa) + [0] * len(Xb))
    S = {}
    for t, lab in [(a, 1), (b, 0)]:
        for s in range(5):
            rows = probes(fam, f"grid_{t}_s{s}")
            if rows:
                tx = [r["text"] for r in rows]
                S[(t, s)] = (lab, clf.predict_proba(vec.transform(trunc400(tx) if trunc else tx))[:, 1])
    A = [S[k][1] for k in S if S[k][0] == 1]; B = [S[k][1] for k in S if S[k][0] == 0]
    if not A or not B:
        return None
    auc = roc_auc_score([1] * sum(map(len, A)) + [0] * sum(map(len, B)), np.concatenate(A + B))
    if trunc:
        return {"auc": float(auc)}
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(N_BOOT):
        ra = [A[i] for i in rng.integers(0, len(A), len(A))]; rb = [B[i] for i in rng.integers(0, len(B), len(B))]
        boots.append(roc_auc_score([1] * sum(map(len, ra)) + [0] * sum(map(len, rb)), np.concatenate(ra + rb)))
    return {"auc": float(auc), "ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]}


def owner_tests(fam):
    X, y = [], []
    for i, t in enumerate(ORDER):
        tx = teacher_texts(t, "r300"); X += tx; y += [i] * len(tx)
    vec = features("tfidf")()
    clf = LogisticRegression(max_iter=3000, C=4.0).fit(vec.fit_transform(X), y)
    G = {}
    for t in ORDER:
        for s in range(5):
            rows = probes(fam, f"grid_{t}_s{s}")
            if rows:
                G[(t, s)] = clf.predict_proba(vec.transform([r["text"] for r in rows])).mean(0)
    res = {}
    for a, sib in PAIRS_SIB:                            # P3 sibling owner test (calibration excludes a's line and sib)
        ai = ORDER.index(a)
        cal = [v[ai] for (t, s), v in G.items() if LINE_OF[t] != LINE_OF[a] and t != sib]
        f = [(1 + np.sum(np.array(cal) >= G[(sib, s)][ai])) / (1 + len(cal)) <= 0.05 for s in range(5) if (sib, s) in G]
        res[f"sibling:{a}->{sib}"] = {"fpr": float(np.mean(f)) if f else None, "n_cal": len(cal)}
    for a in ORDER:
        ai = ORDER.index(a)
        cal = [v[ai] for (t, s), v in G.items() if LINE_OF[t] != LINE_OF[a]]
        pv = lambda x: (1 + np.sum(np.array(cal) >= x)) / (1 + len(cal))
        tpr = [pv(G[(a, s)][ai]) <= 0.05 for s in range(5) if (a, s) in G]
        res[a] = {"tpr": float(np.mean(tpr)) if tpr else None, "n_cal": len(cal), "fpr": {}}
        for b in LINES[LINE_OF[a]]:
            if b != a:
                f = [pv(G[(b, s)][ai]) <= 0.05 for s in range(5) if (b, s) in G]
                res[a]["fpr"][b] = float(np.mean(f)) if f else None
    return res


def accuracy(rows):
    return np.array([correct("gsm", r["text"], r["gold"]) for r in rows], float)


def manipulation():
    void, info = set(), {}
    for t in ORDER:
        rows = [r for r in jl(OUT / f"teacher_{t}.jsonl") or [] if r["qid"] in POOL]
        if not rows:
            void.add(t); continue
        ext = float(np.mean([extract_answer(r["text"]) is not None for r in rows]))
        tchars = float(np.mean([len(r["text"]) for r in rows]))
        fails = 0
        for fam in FAMS:
            base = probes(fam, "base")
            base_acc = accuracy(base).mean() if base else 0.0
            for s in range(5):
                pr = probes(fam, f"grid_{t}_s{s}")
                if pr is None:
                    fails += 1; continue
                ok = accuracy(pr).mean() >= base_acc or 0.67 <= np.mean([len(r["text"]) for r in pr]) / tchars <= 1.5
                fails += not ok
        info[t] = {"extractable": ext, "teacher_acc": float(accuracy(rows).mean()),
                   "truncated": float(np.mean([r.get("truncated", False) for r in rows])),
                   "mean_tokens": float(np.mean([r.get("n_tokens", 0) for r in rows])), "student_fails": fails}
        if ext < 0.70 or fails > 2:
            void.add(t)
    return void, info


def main():
    R = {}
    void, info = manipulation()
    R["manipulation"] = info; R["void"] = sorted(void)
    print("[m6] manipulation:", json.dumps(info, indent=0)[:2000], "\nvoid:", sorted(void), flush=True)
    R["E1"], R["E4"], R["E2"], R["E3"] = {}, {}, {}, {}
    for fam in FAMS:
        R["E1"][fam] = {f"{a}|{b}": pairwise_auc(a, b, fam) for a, b in PAIRS_D1 + PAIRS_D2 + PAIRS_SIB}
        R["E4"][fam] = {f"{a}|{b}": pairwise_auc(a, b, fam, trunc=True) for a, b in PAIRS_D1 + PAIRS_D2 + PAIRS_SIB}
        R["E2"][fam] = owner_tests(fam)
        for k, v in R["E1"][fam].items():
            print(f"[m6/E1/{fam}] {k:<26} AUC {v and v['auc']:.3f} CI {v and v['ci']}   | first-400-token AUC "
                  f"{R['E4'][fam][k] and R['E4'][fam][k]['auc']:.3f}", flush=True)
        for a, v in R["E2"][fam].items():
            print(f"[m6/E2/{fam}] {a:<28} {v}", flush=True)
        # E3 capability
        acc = {}
        for t in ORDER:
            rows = [probes(fam, f"grid_{t}_s{s}") for s in range(5)]
            rows = [r for r in rows if r]
            if rows:
                acc[t] = np.mean([accuracy(r) for r in rows], 0)
        rng = np.random.default_rng(0)
        for a, b in PAIRS_D1 + PAIRS_D2 + PAIRS_SIB:
            if a in acc and b in acc:
                d = acc[a] - acc[b]
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
                R["E3"][f"{fam}|{a}|{b}"] = {"diff": float(d.mean()), "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
                print(f"[m6/E3/{fam}] acc({a}) - acc({b}) = {d.mean():+.3f} [{np.quantile(bs, .025):+.3f}, {np.quantile(bs, .975):+.3f}]")

    # ---------------- predictions
    def ok_pair(a, b):
        return a not in void and b not in void
    verdict = {}
    for fam in FAMS:
        ordered = [(a, b) for a, b in PAIRS_D1] + [(b, a) for a, b in PAIRS_D1]
        vals = [R["E2"][fam][a]["fpr"].get(b) for a, b in ordered if ok_pair(a, b)]
        vals = [v for v in vals if v is not None]
        need1 = int(np.ceil(len(vals) * 8 / 12))
        p1 = len(vals) > 0 and sum(v >= 0.6 for v in vals) >= need1
        E1 = R["E1"][fam]
        d1 = [E1[f"{a}|{b}"]["auc"] for a, b in PAIRS_D1 if ok_pair(a, b) and E1[f"{a}|{b}"]]
        d2 = [E1[f"{a}|{b}"]["auc"] for a, b in PAIRS_D2 if ok_pair(a, b) and E1[f"{a}|{b}"]]
        lines_ok, lines_n = 0, 0
        for ts in LINES.values():
            if all(t not in void for t in ts) and all(E1[k] for k in [f"{ts[0]}|{ts[1]}", f"{ts[1]}|{ts[2]}", f"{ts[0]}|{ts[2]}"]):
                lines_n += 1
                lines_ok += E1[f"{ts[0]}|{ts[2]}"]["auc"] >= max(E1[f"{ts[0]}|{ts[1]}"]["auc"], E1[f"{ts[1]}|{ts[2]}"]["auc"]) - 0.02
        p2 = bool(d1 and d2 and np.mean(d2) > np.mean(d1) and lines_n and lines_ok >= int(np.ceil(lines_n * 2 / 3)))
        sib = []
        for a, b in PAIRS_SIB:
            sf = R["E2"][fam].get(f"sibling:{a}->{b}", {}).get("fpr")
            if ok_pair(a, b) and E1[f"{a}|{b}"] and sf is not None:
                sib.append(E1[f"{a}|{b}"]["auc"] >= 0.9 and sf <= 0.2)
        p3 = bool(sib) and sum(sib) >= int(np.ceil(len(sib) * 2 / 3))
        verdict[fam] = {"P1": bool(p1), "P1_counts": [int(sum(v >= 0.6 for v in vals)), len(vals), need1],
                        "P2": p2, "P2_detail": {"mean_d1": float(np.mean(d1)) if d1 else None, "mean_d2": float(np.mean(d2)) if d2 else None,
                                                "lines_ok": int(lines_ok), "lines_n": int(lines_n)},
                        "P3": bool(p3)}
    R["verdict"] = verdict
    P1 = all(v["P1"] for v in verdict.values()); P2 = all(v["P2"] for v in verdict.values()); P3 = all(v["P3"] for v in verdict.values())
    print("\n== PREDICTIONS ==\n" + json.dumps(verdict, indent=1))
    print(f"P1 adjacent collapse: {P1}  (killed if it fails in both families: {not any(v['P1'] for v in verdict.values())})")
    print(f"P2 monotone curve:   {P2}  (killed if it fails in both families: {not any(v['P2'] for v in verdict.values())})")
    print(f"P3 sibling lines distinguishable: {P3}")
    (STUD / "m6_result.json").write_text(json.dumps(R, indent=1, default=str))


if __name__ == "__main__":
    main()
