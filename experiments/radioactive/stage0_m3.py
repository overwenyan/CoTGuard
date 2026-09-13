"""M3 v4 stage 0 diagnostics (m3_design.md v4 §0, pre-registered in d0fe896). Existing data only.

D1 collision structure of the key bank     D2 where paraphrase breaks (lexical / embedding / behaviour)
D3 utility-matched attack cost             D4 broader innocents: other-key students, independent imitation
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "relay"))
from run_m3c import SETTINGS, STUDENTS, TRAINED, correct  # noqa: E402
from trigger_v2 import ANCHORS, PERSONA, TEMPLATES  # noqa: E402

D = HERE / "data3"
FAMS = list(STUDENTS)
KEYS = json.loads((HERE / "keys_v3.json").read_text())
KIDS = list(KEYS)
ARMS9 = TRAINED + ["clean"]
R = {}


def jl(fp):
    return [json.loads(l) for l in open(fp) if l.strip()] if fp.exists() else None


def attrs(pattern):
    for ti, t in enumerate(TEMPLATES):
        for p in PERSONA:
            for a in ANCHORS:
                if t.format(persona=p, anchor=a) == pattern:
                    return {"template": ti, "persona": p, "instruction": a}
    raise KeyError(pattern)


A = {k: attrs(v) for k, v in KEYS.items()}


# ------------------------------------------------------------------ read-outs
class Embedder:
    def __init__(self):
        self.cache_dir = D / "emb_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.model = None

    def __call__(self, texts):
        h = hashlib.sha1("\x00".join(texts).encode()).hexdigest()[:20]
        fp = self.cache_dir / f"{h}.npy"
        if fp.exists():
            return np.load(fp)
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("thenlper/gte-base", device="cuda")
            self.model.max_seq_length = 512
        E = self.model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        np.save(fp, E)
        return E


EMB = Embedder()
LEX = ["target", r"assum\w*", "invariant", "given", "derived", r"units?\b", "check", r"verif\w*", r"compar\w*",
       r"\bname", r"\blabel", r"\bstep", r"restat\w*"]


def behaviour(texts):
    rows = []
    for t in texts:
        low, n = t.lower(), max(1, len(t.split()))
        lines = [l for l in t.splitlines() if l.strip()]
        rows.append([len(re.findall(p, low)) / n * 100 for p in LEX]
                    + [np.log1p(len(t)), len(lines), sum("=" in l for l in lines)])
    return np.array(rows, dtype=float)


def make_readout(kind):
    """-> (fit(texts, y), predict_proba(texts))"""
    st = {}

    def fit(X, y):
        if kind == "lexical":
            st["v"] = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
            F = st["v"].fit_transform(X)
        elif kind == "embedding":
            F = EMB(list(X))
        else:
            st["s"] = StandardScaler()
            F = st["s"].fit_transform(behaviour(X))
        st["c"] = LogisticRegression(max_iter=3000, C=4.0).fit(F, y)

    def proba(X):
        if kind == "lexical":
            F = st["v"].transform(X)
        elif kind == "embedding":
            F = EMB(list(X))
        else:
            F = st["s"].transform(behaviour(X))
        return st["c"].predict_proba(F)
    return fit, proba, st


def teacher_xy(setting, arms, prefix="teacher"):
    X, y, g = [], [], []
    for i, a in enumerate(arms):
        fp = D / setting / (f"{prefix}_{a}.jsonl")
        for r in jl(fp) or []:
            if r["text"].strip():
                X.append(r["text"]); y.append(i); g.append(r["qid"])
    return X, np.array(y), np.array(g)


def texts(setting, fam, name):
    rows = jl(D / setting / f"out_{fam}_{name}.jsonl")
    return [r["text"] for r in rows] if rows else None


def rank_p(s, pool, k):
    return (1 + sum(s[pool.index(j)] >= s[pool.index(k)] for j in pool if j != k)) / len(pool)


def owner_test(score_fn, pool, students, veto):
    """students: {(fam,k): texts}; veto: {fam: [texts, ...]}; score_fn(texts) -> mean prob per key in KIDS order"""
    out = []
    for (fam, k), T in students.items():
        s = score_fn(T)
        sub = np.array([s[KIDS.index(j)] for j in pool])
        thr = min(3 / 64, 0.05) if len(pool) == 64 else 0.05
        p = rank_p(sub, pool, k)
        inn = [rank_p(np.array([score_fn(V)[KIDS.index(j)] for j in pool]), pool, k) for V in veto[fam]]
        out.append({"fam": fam, "key": k, "p": p, "innocent_p": inn, "ok": bool(p <= thr and all(q > thr for q in inn))})
    return out


# ------------------------------------------------------------------ D1
def d1():
    res = {}
    for setting in SETTINGS:
        X, y, g = teacher_xy(setting, KIDS)
        pred = np.empty_like(y)
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            fit, proba, _ = make_readout("lexical")
            fit([X[i] for i in tr], y[tr])
            pred[te] = proba([X[i] for i in te]).argmax(1)
        acc = {"key": float(np.mean(pred == y))}
        for f in ["instruction", "persona", "template"]:
            acc[f] = float(np.mean([A[KIDS[a]][f] == A[KIDS[b]][f] for a, b in zip(pred, y)]))
        wrong = pred != y
        dec = {}
        for f in ["instruction", "persona", "template"]:
            hit = np.mean([A[KIDS[a]][f] == A[KIDS[b]][f] for a, b in zip(pred[wrong], y[wrong])])
            base = np.mean([sum(A[j][f] == A[KIDS[b]][f] for j in KIDS if j != KIDS[b]) / 63 for b in y[wrong]])
            dec[f] = {"share": float(hit), "base": float(base), "ratio": float(hit / base) if base else None}
        n_instr = len({A[k]["instruction"] for k in KIDS})
        res[setting] = {"heldout_acc": acc, "misattribution": dec, "n_instructions_in_pool": n_instr,
                        "chance": {"key": 1 / 64}}
        print(f"\n[D1/{setting}] held-out top-1: key {acc['key']:.3f} (chance {1/64:.3f}); collapsed: instruction "
              f"{acc['instruction']:.3f}, persona {acc['persona']:.3f}, template {acc['template']:.3f}; "
              f"{n_instr} instructions in pool", flush=True)
        for f, v in dec.items():
            print(f"   wrong -> same {f}: {v['share']:.3f} vs base {v['base']:.3f}  (x{v['ratio']:.2f})")

        # student level: same-instruction decoys, and the test without them
        fit, proba, _ = make_readout("lexical")
        Xf, yf, _ = teacher_xy(setting, KIDS + ["clean"])
        fit(Xf, yf)
        sc = lambda T: proba(T).mean(0)[:64]
        studs = {(f, k): texts(setting, f, f"raw_{k}") for f in FAMS for k in TRAINED}
        veto = {f: [texts(setting, f, "raw_clean"), texts(setting, f, "base")] for f in FAMS}
        top3_collide, rows_wo = 0, []
        for (f, k), T in studs.items():
            s = sc(T)
            same = [j for j in KIDS if j != k and A[j]["instruction"] == A[k]["instruction"]]
            order = [KIDS[i] for i in np.argsort(-s) if KIDS[i] != k][:3]
            top3_collide += any(j in same for j in order)
            pool = [j for j in KIDS if j not in same]
            rows_wo += owner_test(sc, pool, {(f, k): T}, {f: veto[f]})
        n_ok_wo = sum(r["ok"] for r in rows_wo)
        res[setting]["same_instruction_decoy_in_top3_share"] = top3_collide / len(studs)
        res[setting]["owner_test_without_same_instruction_decoys"] = rows_wo
        print(f"   trained pairs with a same-instruction decoy in top 3 (excluding owner): {top3_collide}/{len(studs)}"
              f" (implication threshold 25%); owner test with those decoys removed: {n_ok_wo}/16")
    R["D1"] = res


# ------------------------------------------------------------------ D2
def d2():
    res = {"a": {}, "b": {}, "c": None, "d": None}
    s = "tulu_gsm"
    for corpus, prefix in [("original", "teacher"), ("para", "corpus_para"), ("compress", "corpus_compress")]:
        X, y, g = teacher_xy(s, ARMS9, prefix)
        res["a"][corpus] = {}
        for kind in ["lexical", "embedding", "behaviour"]:
            pred = np.empty_like(y)
            for tr, te in GroupKFold(n_splits=5).split(X, y, g):
                fit, proba, _ = make_readout(kind)
                fit([X[i] for i in tr], y[tr])
                pred[te] = proba([X[i] for i in te]).argmax(1)
            res["a"][corpus][kind] = float(np.mean(pred == y))
        print(f"\n[D2a/{corpus}] 9-way held-out acc (chance {1/9:.3f}): " +
              ", ".join(f"{k} {v:.3f}" for k, v in res["a"][corpus].items()), flush=True)

    X, y, _ = teacher_xy(s, ARMS9)
    for kind in ["lexical", "embedding", "behaviour"]:
        fit, proba, _ = make_readout(kind)
        fit(X, y)
        res["b"][kind] = {}
        for corpus in ["raw", "para"]:
            rates = []
            for f in FAMS:
                for i, k in enumerate(TRAINED):
                    T = texts(s, f, f"{corpus}_{k}")
                    if T:
                        rates.append(float(np.mean(proba(T).argmax(1) == i)))
            res["b"][kind][corpus] = float(np.mean(rates)) if rates else None
        print(f"[D2b/{kind}] 9-way self-attribution on students: raw {res['b'][kind]['raw']:.3f}, "
              f"para {res['b'][kind]['para']:.3f} (chance {1/9:.3f})", flush=True)

    # (c) 64-key owner test with the embedding read-out
    X, y, _ = teacher_xy(s, KIDS + ["clean"])
    fit, proba, _ = make_readout("embedding")
    fit(X, y)
    sc = lambda T: proba(T).mean(0)[:64]
    res["c"] = {}
    for corpus in ["raw", "para"]:
        studs = {(f, k): texts(s, f, f"{corpus}_{k}") for f in FAMS for k in TRAINED}
        veto = {f: [texts(s, f, f"{corpus}_clean"), texts(s, f, "base")] for f in FAMS}
        rows = owner_test(sc, KIDS, studs, veto)
        res["c"][corpus] = rows
        print(f"[D2c] embedding read-out, 64-key owner test on {corpus} students: {sum(r['ok'] for r in rows)}/16 "
              f"(lexical v3: raw 12/16, para 8/16)", flush=True)

    # (d) surface-cue dependence on raw students, lexical read-out
    X, y, _ = teacher_xy(s, KIDS + ["clean"])
    fit, proba, st = make_readout("lexical")
    fit(X, y)
    coef = st["c"].coef_
    drop = set()
    for c in range(coef.shape[0]):
        drop.update(np.argsort(-coef[c])[:50].tolist())
    drop = np.array(sorted(drop))

    def variant(T, level):
        if level >= 1:
            T = ["\n".join(t.strip().splitlines()[1:]) for t in T]
        if level >= 3:
            T = [t[:400] for t in T]
        F = st["v"].transform(T)
        if level >= 2:
            F = F.tolil(); F[:, drop] = 0; F = F.tocsr()
            nrm = np.sqrt(F.multiply(F).sum(1)).A1; nrm[nrm == 0] = 1
            F = F.multiply(1 / nrm[:, None]).tocsr()
        return st["c"].predict_proba(F).mean(0)[:64]

    res["d"] = {}
    names = {0: "none", 1: "first line removed", 2: "+ top n-grams removed", 3: "+ truncated to 400 chars"}
    for level in range(4):
        sc = lambda T, L=level: variant(T, L)
        studs = {(f, k): texts(s, f, f"raw_{k}") for f in FAMS for k in TRAINED}
        veto = {f: [texts(s, f, "raw_clean"), texts(s, f, "base")] for f in FAMS}
        rows = owner_test(sc, KIDS, studs, veto)
        res["d"][names[level]] = sum(r["ok"] for r in rows)
        print(f"[D2d] {names[level]:<26} owner test {res['d'][names[level]]}/16 ({len(drop)} n-grams in drop set)", flush=True)
    R["D2"] = res


# ------------------------------------------------------------------ D3
def d3():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENTS["qwen15"])
    s, res = "tulu_gsm", {}
    for corpus, prefix in [("raw", "teacher"), ("filter", "corpus_filter"), ("para", "corpus_para"), ("compress", "corpus_compress")]:
        for arm in ["clean", "keyed"]:
            arms = ["clean"] if arm == "clean" else TRAINED
            toks = [sum(len(tok(r["text"]).input_ids) for r in jl(D / s / f"{prefix}_{a}.jsonl")) for a in arms]
            accs = {}
            for f in FAMS:
                a = [np.mean([correct("gsm", t, r["gold"]) for t, r in zip(texts(s, f, f"{corpus}_{x}"), jl(D / s / f"out_{f}_{corpus}_{x}.jsonl"))])
                     for x in arms if texts(s, f, f"{corpus}_{x}")]
                accs[f] = float(np.mean(a)) if a else None
            res[f"{corpus}/{arm}"] = {"acc": accs, "train_tokens_per_arm": float(np.mean(toks))}
            print(f"[D3] {corpus:<9}{arm:<6} acc " + ", ".join(f"{f} {v:.3f}" if v is not None else f"{f} -" for f, v in accs.items())
                  + f"; training tokens/arm {np.mean(toks):,.0f}", flush=True)
    R["D3"] = res


# ------------------------------------------------------------------ D4
def d4():
    res = {}
    readouts = {}
    for setting in ["tulu_gsm", "qwen_gsm"]:
        X, y, _ = teacher_xy(setting, KIDS + ["clean"])
        fit, proba, _ = make_readout("lexical")
        fit(X, y)
        readouts[setting] = lambda T, p=proba: p(T).mean(0)[:64]
    for setting in SETTINGS:
        if setting not in readouts:
            X, y, _ = teacher_xy(setting, KIDS + ["clean"])
            fit, proba, _ = make_readout("lexical")
            fit(X, y)
            readouts[setting] = lambda T, p=proba: p(T).mean(0)[:64]
        sc = readouts[setting]
        cache = {(f, j): sc(texts(setting, f, f"raw_{j}")) for f in FAMS for j in TRAINED}
        fp_all, fp_diff = [], []
        for (f, j), s in cache.items():
            for k in TRAINED:
                if k == j:
                    continue
                hit = rank_p(s, KIDS, k) <= 3 / 64
                fp_all.append(hit)
                if A[k]["instruction"] != A[j]["instruction"]:
                    fp_diff.append(hit)
        res[setting] = {"other_key_fp": float(np.mean(fp_all)), "other_key_fp_diff_instruction": float(np.mean(fp_diff)),
                        "n": len(fp_all)}
        print(f"\n[D4/{setting}] other-key students flagged for a trained key they never saw: {np.mean(fp_all):.3f} "
              f"(n={len(fp_all)}); excluding shared-instruction pairs {np.mean(fp_diff):.3f}", flush=True)
    for owner, other in [("tulu_gsm", "qwen_gsm"), ("qwen_gsm", "tulu_gsm")]:
        studs = {(f, k): texts(other, f, f"raw_{k}") for f in FAMS for k in TRAINED}
        veto = {f: [texts(owner, f, "raw_clean"), texts(owner, f, "base")] for f in FAMS}
        rows = owner_test(readouts[owner], KIDS, studs, veto)
        res[f"imitation:{owner}_owner_on_{other}_students"] = rows
        print(f"[D4] independent imitation: {owner} owner's test on {other}-teacher students with the same key: "
              f"{sum(r['ok'] for r in rows)}/16 flagged (prediction: high)", flush=True)
    R["D4"] = res


if __name__ == "__main__":
    for step in (sys.argv[1:] or ["d3", "d1", "d2", "d4"]):
        globals()[step]()
        (D / "stage0_result.json").write_text(json.dumps(R, indent=1, default=str))
    print("=== stage 0 done ===")
