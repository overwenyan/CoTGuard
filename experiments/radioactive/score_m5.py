"""M5 scoring and gates (.pipeline/docs/m5_design.md §1, pre-registered 2b36772, amended b6e448d)."""

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
from run_m5 import D4, ORDER, SMOKE, STUD, r300_qids, read_jsonl, teacher_rows  # noqa: E402

FAMS = ["qwen15", "llama1b"]
R = {}


def unknowns(o):
    return {ORDER[(o + 1) % 7], ORDER[(o + 3) % 7]}


class Readout:
    def __init__(self, teachers, kind="tfidf"):
        r300 = set(r300_qids())
        X, y = [], []
        for i, t in enumerate(teachers):
            for q, row in teacher_rows(t).items():
                if q in r300:
                    X.append(row["text"]); y.append(i)
        self.teachers, self.kind = teachers, kind
        if kind == "tfidf":
            self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
            F = self.vec.fit_transform(X)
        elif kind == "pos":
            F = self._pos_fit(X)
        else:
            F = self._emb(X)
        self.clf = LogisticRegression(max_iter=3000, C=4.0).fit(F, np.array(y))
        self.n = len(X)

    def _pos_tags(self, texts):
        sys.path.insert(0, str(HERE / ".pylib"))
        import nltk
        nltk.data.path.insert(0, str(HERE / ".pylib" / "nltk_data"))
        return [" ".join(t for _, t in nltk.pos_tag(x.split())) for x in texts]

    def _pos_fit(self, X):                  # PoS-template baseline in the spirit of "Who Taught You That?"
        self.vec = TfidfVectorizer(ngram_range=(2, 4), sublinear_tf=True, min_df=2, max_features=50000,
                                   token_pattern=r"\S+", lowercase=False)
        return self.vec.fit_transform(self._pos_tags(X))

    def _emb(self, X):
        from sentence_transformers import SentenceTransformer
        if not hasattr(self, "enc"):
            self.enc = SentenceTransformer("thenlper/gte-base", device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
        return self.enc.encode(X, batch_size=128, normalize_embeddings=True, show_progress_bar=False)

    def score(self, texts):
        if self.kind == "tfidf":
            F = self.vec.transform(texts)
        elif self.kind == "pos":
            F = self.vec.transform(self._pos_tags(texts))
        else:
            F = self._emb(texts)
        return self.clf.predict_proba(F).mean(0)             # mean P(teacher) over outputs


def probe_texts(fam, name):
    rows = read_jsonl(STUD / f"probe_{fam}_{name}.jsonl")
    return None if rows is None else [r["text"] for r in rows]


def conformal_p(s, cal):
    return (1 + np.sum(np.array(cal) >= s)) / (1 + len(cal))


def grid_scores(ro, fam):
    """(teacher, s) -> score vector over ro.teachers"""
    out = {}
    for t in ORDER:
        for s in range(5):
            tx = probe_texts(fam, f"grid_{t}_s{s}")
            if tx:
                out[(t, s)] = ro.score(tx)
    return out


def closed_and_open(kind):
    ro = Readout(ORDER, kind)
    res = {}
    for fam in FAMS:
        G = grid_scores(ro, fam)
        top1 = {k: ORDER[int(np.argmax(v))] == k[0] for k, v in G.items()}
        s0 = {k: v for k, v in top1.items() if k[1] == 0}
        rows = []
        for o, owner in enumerate(ORDER):
            U = unknowns(o)
            oi = ORDER.index(owner)
            cal = [G[(t, s)][oi] for (t, s) in G if t != owner and t not in U]
            pos = [conformal_p(G[(owner, s)][oi], cal) for s in range(5) if (owner, s) in G]
            neg = [conformal_p(G[(t, s)][oi], cal) for (t, s) in G if t in U]
            rows.append({"owner": owner, "U": sorted(U), "n_cal": len(cal),
                         "tpr": float(np.mean([p <= 0.05 for p in pos])) if pos else None,
                         "fpr": float(np.mean([p <= 0.05 for p in neg])) if neg else None})
        # same-lineage pairs, reported separately (ad hoc U containing the sibling)
        lineage = {}
        for owner, sib in [("tulu_sft", "tulu_dpo"), ("tulu_dpo", "tulu_sft"), ("qwen25_7b", "qwen3_14b"),
                           ("qwen3_14b", "qwen25_7b"), ("tulu_sft", "llama31_8b")]:
            oi = ORDER.index(owner)
            cal = [G[(t, s)][oi] for (t, s) in G if t not in (owner, sib)]
            neg = [conformal_p(G[(sib, s)][oi], cal) for s in range(5) if (sib, s) in G]
            lineage[f"{owner}<-{sib}"] = float(np.mean([p <= 0.05 for p in neg])) if neg else None
        tprs = [r["tpr"] for r in rows if r["tpr"] is not None]
        fprs = [r["fpr"] for r in rows if r["fpr"] is not None]
        res[fam] = {"top1_all": [int(sum(top1.values())), len(top1)], "top1_s0": [int(sum(s0.values())), len(s0)],
                    "open": rows, "mean_tpr": float(np.mean(tprs)) if tprs else None,
                    "mean_fpr": float(np.mean(fprs)) if fprs else None, "lineage_fpr": lineage}
        print(f"[m5/{kind}/{fam}] closed-set top-1 {res[fam]['top1_all']} (s0 {res[fam]['top1_s0']}); open-set mean TPR "
              f"{res[fam]['mean_tpr']}, mean unknown-teacher FPR {res[fam]['mean_fpr']}; same-lineage FPR {lineage}", flush=True)
        for r in rows:
            print(f"     owner {r['owner']:<11} U={r['U']} cal={r['n_cal']} TPR={r['tpr']} FPR={r['fpr']}")
    return ro, res


def strict_open(fam):
    """read-out retrained without the unknown classes (reported)"""
    out = []
    for o, owner in enumerate(ORDER):
        U = unknowns(o)
        teachers = [t for t in ORDER if t not in U]
        ro = Readout(teachers)
        oi = teachers.index(owner)
        G = {}
        for t in ORDER:
            for s in range(5):
                tx = probe_texts(fam, f"grid_{t}_s{s}")
                if tx:
                    G[(t, s)] = ro.score(tx)[oi]
        cal = [v for (t, s), v in G.items() if t != owner and t not in U]
        pos = [conformal_p(G[(owner, s)], cal) for s in range(5) if (owner, s) in G]
        neg = [conformal_p(v, cal) for (t, s), v in G.items() if t in U]
        out.append((owner, float(np.mean([p <= 0.05 for p in pos])) if pos else None,
                    float(np.mean([p <= 0.05 for p in neg])) if neg else None))
    print(f"[m5/strict/{fam}] " + "  ".join(f"{o}: TPR={a} FPR={b}" for o, a, b in out))
    return out


def composite(ro):
    """p_comp = max(p_instruction (v7 rank test, K=32), p_teacher (owner tulu_sft, calibration T3..T7))."""
    os.environ["M3C_OUT"] = str(HERE / "data4")
    import score_m3d as s3
    from sklearn.feature_extraction.text import TfidfVectorizer as TV
    X, y, _ = s3.teacher(s3.E + ["clean"])
    v = TV(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
    c = LogisticRegression(max_iter=3000, C=4.0).fit(v.fit_transform(X), y)

    def p_instr(texts, k, veto_texts):
        s = c.predict_proba(v.transform(texts)).mean(0)[: s3.K]
        p = s3.pval(s, k)
        inn = [s3.pval(c.predict_proba(v.transform(t)).mean(0)[: s3.K], k) for t in veto_texts]
        return p, bool(p <= 0.05 and all(q > 0.05 for q in inn))

    oi = ORDER.index("tulu_sft")
    res = {}
    for fam in FAMS:
        G = grid_scores(ro, fam)
        cal = [G[(t, s)][oi] for (t, s) in G if t in ORDER[2:]]
        for k in ["o12", "p07"]:
            for sd in [0, 1]:
                sfx = "" if sd == 0 else f"_s{sd}"
                veto = [[r["text"] for r in read_jsonl(D4 / f"out_{fam}_{n}{sfx}.jsonl")] for n in ["raw_clean", "base"]
                        if (D4 / f"out_{fam}_{n}{sfx}.jsonl").exists()]
                cases = {"own": D4 / f"out_{fam}_raw_{k}{sfx}.jsonl", "imit_qwen": D4 / f"out_{fam}_imit300_{k}{sfx}.jsonl",
                         "dpo_same_instr": STUD / f"probe_{fam}_dpoinst_{k}{sfx}.jsonl"}
                for case, fp in cases.items():
                    rows = read_jsonl(fp)
                    if rows is None or len(veto) < 2:
                        continue
                    tx = [r["text"] for r in rows]
                    pi, ok_i = p_instr(tx, k, veto)
                    pt = conformal_p(ro.score(tx)[oi], cal)
                    res[f"{fam}/{k}/s{sd}/{case}"] = {"p_instruction": float(pi), "instr_ok": ok_i, "p_teacher": float(pt),
                                                      "pass": bool(ok_i and pt <= 0.05)}
    for case in ["own", "imit_qwen", "dpo_same_instr"]:
        rr = [v for k_, v in res.items() if k_.endswith("/" + case)]
        print(f"[m5/composite] {case:<15} instruction-only pass {sum(r['instr_ok'] for r in rr)}/{len(rr)}; "
              f"composite pass {sum(r['pass'] for r in rr)}/{len(rr)}   "
              + " ".join(f"(pi={r['p_instruction']:.3f},pt={r['p_teacher']:.3f})" for r in rr))
    return res


def mixtures(ro):
    oi = ORDER.index("tulu_sft")
    out = {}
    G = grid_scores(ro, "qwen15")
    for name, partner in [("mix50T4", "qwen25_7b"), ("mix10T4", "qwen25_7b"), ("mix10T3", "llama31_8b")]:
        cal = [G[(t, s)][oi] for (t, s) in G if t in ORDER[2:]]          # T3..T7 grid students (partner included)
        for sd in [0, 1]:
            tx = probe_texts("qwen15", f"mix_{name}" + ("" if sd == 0 else f"_s{sd}"))
            if tx:
                out[f"{name}/s{sd}"] = float(conformal_p(ro.score(tx)[oi], cal))
    print(f"[m5/mixtures] p_teacher(tulu_sft): {out}")
    return out


def main():
    ro, R["tfidf"] = closed_and_open("tfidf")
    q, l = R["tfidf"]["qwen15"], R["tfidf"]["llama1b"]
    GA = q["top1_all"][0] >= (30 if not SMOKE else 0) and l["top1_s0"][0] >= (6 if not SMOKE else 0)
    GB = q["mean_tpr"] is not None and q["mean_tpr"] >= 0.80 and q["mean_fpr"] <= 0.10
    kill = q["mean_tpr"] is None or q["mean_tpr"] < 0.60 or q["mean_fpr"] > 0.20
    R["strict"] = {f: strict_open(f) for f in FAMS}
    R["composite"] = composite(ro)
    own = [v for k, v in R["composite"].items() if k.endswith("/own")]
    imit = [v for k, v in R["composite"].items() if k.endswith("/imit_qwen")]
    GC = len(own) == 8 and sum(r["pass"] for r in own) >= 7 and sum(r["pass"] for r in imit) <= 1
    R["mixtures"] = mixtures(ro)
    for kind in ["emb", "pos"]:
        try:
            _, R[kind] = closed_and_open(kind)
        except Exception as e:                       # reported baselines must not block the gates
            print(f"[m5/{kind}] failed: {e!r}")
    R["gates"] = {"G-A": bool(GA), "G-B": bool(GB), "G-C": bool(GC), "kill": bool(kill)}
    print(f"\n== GATES ==  G-A closed-set {GA}  |  G-B open-set {GB} (TPR {q['mean_tpr']}, FPR {q['mean_fpr']})  |  "
          f"G-C composite {GC}  |  kill {kill}")
    print("=> " + ("framing (ii) not viable -> framing (iii)" if kill else
                   "framing (ii) viable" + ("" if GC else " for teacher attribution; composite claim not supported")))
    (STUD / "m5_result.json").write_text(json.dumps(R, indent=1, default=str))


if __name__ == "__main__":
    main()
