"""M3 v7b (m3_design.md, specified c69f474; exploratory): can a teacher-identity read-out, trained on
teacher traces from held-out instructions, tell Tulu-sourced from Qwen-sourced students?"""

from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "relay"))
from trigger_v2 import ANCHORS  # noqa: E402

D3, D4 = HERE / "data3", HERE / "data4" / "tulu_gsm"
K = json.loads((HERE / "keys_v3.json").read_text())
jl = lambda fp: [json.loads(l) for l in open(fp) if l.strip()]
ins = lambda v: next(a for a in ANCHORS if a in v)
TEST_KEYS = [f"g{i:02d}" for i in range(8)]
held = {ins(K[k]) for k in TEST_KEYS}
TRAIN_KEYS = [k for k in K if k not in TEST_KEYS and ins(K[k]) not in held]


def texts(fp):
    return [r["text"] for r in jl(fp) if r["text"].strip()]


X, y = [], []
for lab, t in [(0, "tulu_gsm"), (1, "qwen_gsm")]:
    for k in TRAIN_KEYS:
        tx = texts(D3 / t / f"teacher_{k}.jsonl"); X += tx; y += [lab] * len(tx)
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
clf = LogisticRegression(max_iter=3000, C=4.0).fit(vec.fit_transform(X), np.array(y))
print(f"[v7b] trained on {len(X)} teacher traces, {len(TRAIN_KEYS)} keys x 2 teachers")
pq = lambda tx: clf.predict_proba(vec.transform(tx))[:, 1]

cT = np.concatenate([pq(texts(D3 / "tulu_gsm" / f"teacher_{k}.jsonl")) < .5 for k in TEST_KEYS])
cQ = np.concatenate([pq(texts(D3 / "qwen_gsm" / f"teacher_{k}.jsonl")) >= .5 for k in TEST_KEYS])
print(f"[v7b] ceiling: held-out-instruction teacher traces, per-trace accuracy {np.mean(np.concatenate([cT, cQ])):.3f}")


def evaluate(label, units):
    res = {}
    for fam in ["qwen15", "llama1b"]:
        u = [(src, fp) for src, f, fp in units if f == fam and fp.exists()]
        s = np.array([pq(texts(fp)).mean() for _, fp in u])
        lab = np.array([src for src, _ in u])
        acc = np.mean(np.concatenate([(pq(texts(fp)) >= .5) == src for src, fp in u]))
        auc = roc_auc_score(lab, s)
        n1 = lab.sum()
        perm = [roc_auc_score([1 if i in c else 0 for i in range(len(lab))], s)
                for c in itertools.combinations(range(len(lab)), n1)]
        p = float(np.mean([a >= auc - 1e-12 for a in perm]))
        res[fam] = {"auc": float(auc), "perm_p": p, "per_output_acc": float(acc), "n": [int(n1), int(len(lab) - n1)],
                    "mean_pQwen_tulu": float(s[lab == 0].mean()), "mean_pQwen_qwen": float(s[lab == 1].mean())}
        print(f"[v7b/{label}/{fam}] AUC {auc:.3f} (exact perm p={p:.4f}, {n1} Qwen- vs {len(lab) - n1} Tulu-sourced); "
              f"mean P(Qwen) Tulu-sourced {s[lab == 0].mean():.3f} vs Qwen-sourced {s[lab == 1].mean():.3f}; per-output acc {acc:.3f}")
    return res


A = [(src, fam, D3 / t / f"out_{fam}_raw_{k}.jsonl") for src, t in [(0, "tulu_gsm"), (1, "qwen_gsm")]
     for fam in ["qwen15", "llama1b"] for k in TEST_KEYS + ["clean"]]
B = [(src, fam, D4 / f"out_{fam}_{c}_{k}{sfx}.jsonl") for src, c in [(0, "raw"), (1, "imit300")]
     for fam in ["qwen15", "llama1b"] for k in ["o12", "p07"] for sfx in ["", "_s1"]]
R = {"A": evaluate("A v3 students", A), "B": evaluate("B v7 E1 students", B)}
aucs = [R[s][f]["auc"] for s in "AB" for f in ["qwen15", "llama1b"]]
verdict = ("recoverable" if min(aucs) >= 0.9 else "not recovered" if max(aucs) <= 0.65 else "mixed")
print(f"[v7b] wording rule -> {verdict}  (AUCs {['%.2f' % a for a in aucs]})")
R["verdict"] = verdict
(D4 / "teacher_id_result.json").write_text(json.dumps(R, indent=1))
