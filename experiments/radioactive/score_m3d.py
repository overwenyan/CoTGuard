"""M3 v4 stage 1 scoring (m3_design.md v4 §1, pre-registered fca8f8a).

S1-A  owner test per read-out (lexical / embedding / behaviour) on original, T1, T2 students; gate H-OP
S1-B  dilution detection at 1/5/10%, independent-imitation control
S1-C  stealth screens (from stealth.json) and detection after filtering
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
from scipy.stats import fisher_exact, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_m3c import correct  # noqa: E402

OUT = pathlib.Path(os.environ.get("M3C_OUT", HERE / "data4")) / "tulu_gsm"
BANK = json.loads((HERE / os.environ.get("M3D_BANK", "keys_v4.json")).read_text())
CORPORA = os.environ.get("M3D_CORPORA", "raw,T1,T2").split(",")
O = json.loads((OUT / "owners.json").read_text())
E = O["eligible"]
K = len(E)
R = {}


def jl(fp):
    return [json.loads(l) for l in open(fp) if l.strip()] if fp.exists() else None


def out_texts(name, n=None):
    rows = jl(OUT / f"out_qwen15_{name}.jsonl")
    return None if rows is None else [r["text"] for r in rows][: n or None]


def teacher(arms):
    X, y, g = [], [], []
    for i, a in enumerate(arms):
        for r in jl(OUT / f"teacher_{a}.jsonl"):
            if r["text"].strip():
                X.append(r["text"]); y.append(i); g.append(r["qid"])
    return X, np.array(y), np.array(g)


# ------------------------------------------------------------------ read-outs -> score(name) -> vector over E
def lexical():
    X, y, _ = teacher(E + ["clean"])
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
    c = LogisticRegression(max_iter=3000, C=4.0).fit(v.fit_transform(X), y)
    return lambda name: c.predict_proba(v.transform(out_texts(name))).mean(0)[:K]


def embedding():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("thenlper/gte-base", device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    m.max_seq_length = 512
    enc = lambda T: m.encode(T, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    X, y, _ = teacher(E + ["clean"])
    c = LogisticRegression(max_iter=3000, C=4.0).fit(enc(X), y)
    cache = {}

    def score(name):
        if name not in cache:
            cache[name] = c.predict_proba(enc(out_texts(name))).mean(0)[:K]
        return cache[name]
    return score


def behaviour():
    rd = OUT / "rubric"
    if not (rd / "teacher_clean.npy").exists():
        return None
    base = np.load(rd / "teacher_clean.npy").mean(0)
    return lambda name: np.load(rd / f"out_qwen15_{name}.npy").mean(0) - base if (rd / f"out_qwen15_{name}.npy").exists() else None


def pval(s, k):
    i = E.index(k)
    return (1 + sum(s[j] >= s[i] for j in range(K) if j != i)) / K


def owner_test(score, name, k, vetoes):
    if not (OUT / f"out_qwen15_{name}.jsonl").exists():
        return None
    vetoes = [v for v in vetoes if (OUT / f"out_qwen15_{v}.jsonl").exists()]
    s = score(name)
    if s is None:
        return None
    p = pval(s, k)
    inn = [pval(score(v), k) for v in vetoes if score(v) is not None]
    return {"p": p, "innocent_p": inn, "ok": bool(p <= 0.05 and all(q > 0.05 for q in inn))}


def acc(name):
    rows = jl(OUT / f"out_qwen15_{name}.jsonl")
    return None if rows is None else float(np.mean([correct("gsm", r["text"], r["gold"]) for r in rows]))


# ------------------------------------------------------------------ S1-A
def s1a(readouts):
    res = {}
    for rname, score in readouts.items():
        res[rname] = {}
        for corpus in CORPORA:
            rows = {}
            for k in O["owners"]:
                if (OUT / f"out_qwen15_{corpus}_{k}.jsonl").exists():
                    rows[k] = owner_test(score, f"{corpus}_{k}", k, [f"{corpus}_clean", "base"])
            res[rname][corpus] = rows
            n_op = sum(bool(rows.get(k) and rows[k]["ok"]) for k in O["owners_OP"])
            n_pr = sum(bool(rows.get(k) and rows[k]["ok"]) for k in O["owners_PRES"])
            print(f"[S1-A/{rname}/{corpus}] owner test pass: OP {n_op}/{len(O['owners_OP'])}, PRES {n_pr}/{len(O['owners_PRES'])}  "
                  + " ".join(f"{k}:{rows[k]['p']:.3f}{'' if rows[k]['ok'] else '*'}" for k in O["owners"] if rows.get(k)), flush=True)
            res[rname][f"{corpus}_counts"] = {"OP": n_op, "PRES": n_pr}
    n = O["n_per_category"]
    raw = res["lexical"]["raw_counts"]
    manip = raw["OP"] >= n - 2 and raw["PRES"] >= n - 2
    print(f"\n== manipulation check (raw, both categories >= n-2 = {n - 2}): OP {raw['OP']}, PRES {raw['PRES']} -> "
          f"{'ok' if manip else 'FAIL (gates void)'}", flush=True)
    res["manipulation_ok"] = bool(manip)
    for corpus, gname in [("T1", "H-OP" if "T2" in CORPORA else "H-OP-M"), ("T1n", "H-OP-N")]:
        if f"{corpus}_counts" not in res["lexical"]:
            continue
        c = res["lexical"][f"{corpus}_counts"]
        fp = fisher_exact([[c["OP"], n - c["OP"]], [c["PRES"], n - c["PRES"]]], alternative="greater")[1]
        h_op = (c["OP"] - c["PRES"] >= O["margin"]) and fp <= 0.05
        res[gname] = {"OP": c["OP"], "PRES": c["PRES"], "margin_needed": O["margin"], "fisher_p": float(fp), "pass": bool(h_op)}
        print(f"== GATE {gname} (lexical, {corpus}): OP {c['OP']} vs PRES {c['PRES']} (margin >= {O['margin']}), "
              f"one-sided Fisher p = {fp:.4f} -> {'PASS' if h_op else 'FAIL'}", flush=True)
        for rname in readouts:
            if rname != "lexical":
                cc = res[rname][f"{corpus}_counts"]
                print(f"   (reported) {rname} {corpus}: OP {cc['OP']} vs PRES {cc['PRES']}")
    if "T2_counts" in res["lexical"]:
        t2 = res["lexical"]["T2_counts"]
        print(f"   T2 prediction (both categories fail): OP {t2['OP']}/{n}, PRES {t2['PRES']}/{n}")

    # covariates
    lex = readouts["lexical"]
    X, y, g = teacher(E)
    pred = np.empty_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
        cl = LogisticRegression(max_iter=3000, C=4.0).fit(v.fit_transform([X[i] for i in tr]), y[tr])
        pred[te] = cl.predict(v.transform([X[i] for i in te]))
    cov = {}
    for k in O["owners"]:
        i = E.index(k)
        sep = float(np.mean(pred[y == i] == i))
        t1 = jl(OUT / f"corpus_T1_{k}.jsonl")
        raw_s = lex(f"raw_{k}")[i] if out_texts(f"raw_{k}") else np.nan
        t1_s = lex(f"T1_{k}")[i] if out_texts(f"T1_{k}") else np.nan
        cov[k] = {"category": BANK[k]["category"], "teacher_separability": sep,
                  "length_ratio_T1": float(np.mean([len(r["text"]) for r in t1]) / np.mean([len(r["orig_text"]) for r in t1])) if t1 else None,
                  "fidelity_T1": float(np.mean([r["answer_kept"] for r in t1])) if t1 else None,
                  "survival_ratio_T1": float(t1_s / raw_s) if np.isfinite(raw_s) and np.isfinite(t1_s) and raw_s else None}
    ks = [k for k in O["owners"] if cov[k]["survival_ratio_T1"] is not None]
    if len(ks) >= 4 and len({cov[k]["category"] for k in ks}) == 2:
        cat = np.array([cov[k]["category"] == "OP" for k in ks], float)
        surv = np.array([cov[k]["survival_ratio_T1"] for k in ks])
        sep = np.array([cov[k]["teacher_separability"] for k in ks])
        resid = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)
        from scipy.stats import rankdata
        rho = spearmanr(resid(rankdata(cat), rankdata(sep)), resid(rankdata(surv), rankdata(sep)))[0]
        res["partial_spearman_category_survival_given_separability"] = float(rho)
        print(f"   partial Spearman(category OP, T1 survival | teacher separability) = {rho:.3f}  (n={len(ks)}, low power)")
    res["covariates"] = cov
    for k in O["owners"]:
        print(f"   {k} {cov[k]['category']:<5} sep {cov[k]['teacher_separability']:.2f}  lenT1 {cov[k]['length_ratio_T1']}  "
              f"fidT1 {cov[k]['fidelity_T1']}  survT1 {cov[k]['survival_ratio_T1']}")
    util = {}
    for corpus in CORPORA:
        util[corpus] = {"keyed": float(np.nanmean([np.nan if acc(f"{corpus}_{k}") is None else acc(f"{corpus}_{k}") for k in O["owners"]])),
                        "clean": acc(f"{corpus}_clean")}
        print(f"   utility {corpus}: keyed {util[corpus]['keyed']:.3f}, clean {util[corpus]['clean']}")
    util["base"] = acc("base")
    res["utility"] = util
    R["S1-A"] = res


# ------------------------------------------------------------------ S1-B / S1-C
def s1b(lex):
    res = {}
    v = ["dil0_clean", "base"]
    for k in O["dilution"]:
        res[k] = {}
        for f in [1, 5, 10]:
            t = owner_test(lex, f"dil{f}_{k}", k, v)
            res[k][f] = t
        print(f"[S1-B] {k} ({BANK[k]['category']}): " + ", ".join(
            f"{f}%: {res[k][f]['p']:.3f}{'' if res[k][f]['ok'] else '*'}" if res[k][f] else f"{f}%: -" for f in [1, 5, 10]), flush=True)
    det = {f: sum(bool(res[k][f] and res[k][f]["ok"]) for k in O["dilution"]) for f in [1, 5, 10]}
    print(f"   detected: 1% {det[1]}/4 (pred <= 1), 5% {det[5]}/4 (pred >= 2), 10% {det[10]}/4 (pred >= 3)")
    res["detected"] = det
    res["imitation"] = {k: owner_test(lex, f"imit10_{k}", k, v) for k in O["imitation"]}
    print("   imitation (Qwen teacher, same instruction, 10%): " + ", ".join(
        f"{k}: p={t['p']:.3f} {'FLAGGED' if t['ok'] else 'not flagged'}" for k, t in res["imitation"].items() if t))
    res["utility"] = {n: acc(n) for n in ["dil0_clean"] + [f"dil10_{k}" for k in O["dilution"]] + [f"fil10_{k}" for k in O["dilution"]]}
    R["S1-B"] = res

    st = json.loads((OUT / "stealth.json").read_text()) if (OUT / "stealth.json").exists() else None
    if st:
        fil = {k: owner_test(lex, f"fil10_{k}", k, v) for k in O["dilution"]}
        n_removed = sum(bool(fil[k] and not fil[k]["ok"]) for k in O["dilution"])
        print(f"[S1-C] after filtering at 5% clean FPR: not detected after filtering for {n_removed}/4 keys (unfiltered 10% detected: {det[10]}/4)")
        for k in O["dilution"]:
            rm = st["removal"][k]
            print(f"   {k}: keyed removed {rm['keyed_removed']:.3f}, clean removed {rm['clean_removed']:.3f}, "
                  f"test {'-' if fil[k] is None else ('pass' if fil[k]['ok'] else 'FAIL')}, acc {acc(f'fil10_{k}')} vs unfiltered {acc(f'dil10_{k}')}")
        for scr in ["supervised", "llm"]:
            for cat in ["OP", "PRES"]:
                ks = O[f"owners_{cat}"]
                print(f"   screen {scr} {cat}: TPR@1%FPR {np.mean([st[scr]['per_key'][k]['tpr@0.01'] for k in ks]):.3f}, "
                      f"TPR@5%FPR {np.mean([st[scr]['per_key'][k]['tpr@0.05'] for k in ks]):.3f}")
        R["S1-C"] = {"filtered_tests": fil, "stealth": st}


if __name__ == "__main__":
    steps = sys.argv[1:] or ["a", "b"]
    readouts = {"lexical": lexical()}
    if "a" in steps:
        readouts["embedding"] = embedding()
        bh = behaviour()
        if bh:
            readouts["behaviour"] = bh
        s1a(readouts)
    if "b" in steps:
        s1b(readouts["lexical"])
    (OUT / "m3d_result.json").write_text(json.dumps(R, indent=1, default=str))
