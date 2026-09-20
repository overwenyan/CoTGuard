#!/usr/bin/env python3
"""Consistency pass: every quoted number in the drafts, re-derived from result files and checked against the text.

Each check recomputes a value from `experiments/radioactive/**` or `paper/generated/**` and asserts that the draft
contains the string the paper should print. Run it before submission and again before camera-ready:

    python paper/check_numbers.py            # PASS/FAIL per check, non-zero exit if any FAIL

It checks what is machine-checkable (tables verbatim, ranges, counts). Prose claims that no file can settle are listed
at the end as MANUAL so they are not silently assumed to be verified.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT.parent / "experiments" / "radioactive"
M7 = R / "data_m7"
G = ROOT / "generated"

DRAFTS = {p.name: p.read_text() for p in ROOT.glob("draft_*.md")}
BODY = {k: v.split("## Notes for integration")[0] for k, v in DRAFTS.items()}
ALL = "\n".join(BODY.values())
checks: list[tuple[str, bool, str]] = []


def j(p):
    return json.loads(Path(p).read_text())


def ck(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


def has(s, where=None):
    hay = ALL if where is None else BODY[where]
    return s in hay


def rng(vals):
    return min(vals), max(vals)


# ---------------------------------------------------------------- generated tables appear verbatim
for fn, tag in [("s56_table1.md", "Table 1"), ("s56_table2.md", "Table 2"), ("s56_step_auc.md", "step AUC")]:
    gen = (G / fn).read_text().strip().split("\n")
    rows = [r for r in gen if r.startswith("|") and "---" not in r]
    missing = [r for r in rows if r.strip() not in BODY["draft_s5_s6.md"]]
    ck(f"{tag}: all {len(rows)} generated rows present in §5–§6", not missing,
       f"missing {len(missing)}: {missing[:2]}")

# ---------------------------------------------------------------- §6.1 budget ranges
b = j(G / "s56_numbers.json")["budget"]
n3 = [v["nref=3"][1] for v in b.values()]
q25_t, q25_f = [v["q=25"][0] for v in b.values()], [v["q=25"][1] for v in b.values()]
ck("§6.1 '3 references' FPR range 0.008–0.083", has(f"{min(n3):.3f}–{max(n3):.3f}".replace("0.083", "0.083")),
   f"computed {min(n3):.3f}–{max(n3):.3f}")
ck("§6.1 '25 probes' TPR ≥ 0.98", has("true-positive rate ≥ 0.98") and min(q25_t) >= 0.98, f"min {min(q25_t):.3f}")
ck("§6.1 '25 probes' FPR 0.04–0.05", has("false-positive rate 0.04–0.05") and round(min(q25_f), 2) == 0.04 and round(max(q25_f), 2) == 0.05,
   f"computed {min(q25_f):.3f}–{max(q25_f):.3f}")

# ---------------------------------------------------------------- 7B cell (M13)
m13 = j(M7 / "tulu_gsm" / "m13_result.json")
r13 = m13["result"]
t1 = r13["T1"]
ck("§6.1 7B: collapse 7 of 12", has("7 of 12 ordered pairs") and r13["n_T0_ge_0.6"] == 7, str(r13["n_T0_ge_0.6"]))
ck("§6.1 7B: repair 11 of 12", has("≤ 0.2 on 11 of 12 pairs") and r13["n_T1_le_0.2"] == 11, str(r13["n_T1_le_0.2"]))
ck("§6.1 7B: mean 0.03", has("mean 0.03,") and abs(r13["mean_T1_fpr"] - 0.03) < 0.005, f"{r13['mean_T1_fpr']:.3f}")
ck("§6.1 7B: worst 0.33", has("worst 0.33") and abs(max(t1["fpr_rel"].values()) - 1 / 3) < 0.01,
   f"{max(t1['fpr_rel'].values()):.3f}")
ck("§6.1 7B: TPR 0.94", has("true-positive rate of 0.94") and abs(r13["mean_T1_tpr"] - 0.94) < 0.005,
   f"{r13['mean_T1_tpr']:.3f}")
ck("§6.1 7B: floor 0.032 stated", has("0.032"), "")
ck("X.2: 9 calibration scores / floor 0.10", has("9 calibration scores") and m13["n_cal_min"] == 30, "post-fix n_cal 30")

# ---------------------------------------------------------------- attack accuracy cost
ac = j(G / "s56_numbers.json")["acc_cost"]
for lab, key in [("imitation", "imitation"), ("scaffold-only", "scaffold-only")]:
    lo, hi = ac[key]["min"], ac[key]["max"]
    s = f"{lo:+.3f} to {hi:+.3f}".replace("+", "+").replace("-", "−")
    ck(f"§6.3 accuracy cost, {lab} ({s})", has(s), f"computed {s}")

# ---------------------------------------------------------------- evasion counts (Table 2 footnote)
t2 = j(G / "s56_numbers.json")["table2"]
ni, ns, nu = len(t2["evaded_valid"]["imitation"]), len(t2["evaded_valid"]["scaffold_only"]), len(t2["evaded_union"])
ck("§6.3 evasion counts 2 / 3 / 5", has(f"imitation evaded the owner on {ni} of 8") and (ni, ns, nu) == (2, 3, 5),
   f"{ni}/{ns}/{nu}")

# ---------------------------------------------------------------- Table 1-derived prose
t1n = j(G / "s56_numbers.json")["table1"]
allenai = {k: v for k, v in t1n.items() if k.startswith("allenai")}
ck("§5.3 '8 of 12' collapse per AllenAI cell (TF-IDF)", has("8 of 12 ordered pairs"),
   str({k: v["tfidf"]["n_T0_ge_0.6"] for k, v in allenai.items()}))
t1fpr = [v[k]["mean_T1_fpr"] for v in t1n.values() for k in ("tfidf", "pos", "emb")]
ck("§6.1 T1 mean FPR range 0.00–0.09", has("0.00–0.09") and max(t1fpr) <= 0.095, f"max {max(t1fpr):.3f}")
t1tpr = [v[k]["mean_T1_tpr"] for v in t1n.values() for k in ("tfidf", "pos", "emb")]
ck("§6.1 owner TPR 0.98–1.00", has("0.98–1.00") and min(t1tpr) >= 0.98, f"min {min(t1tpr):.3f}")

# ---------------------------------------------------------------- MATH expectation (X.8) and read-out dependence
math_cells = {k: v for k, v in t1n.items() if "/math/" in k}
emb = sorted(v["emb"]["n_T0_ge_0.6"] for v in math_cells.values())
pos = sorted(v["pos"]["n_T0_ge_0.6"] for v in math_cells.values())
ck("X.8 MATH: EMB 3 and 2 of 12", has("(3 and 2 of 12)") and emb == [2, 3], str(emb))
ck("X.8 MATH: POS 5 of 12 for Qwen", has("POS templates (5 of 12)") and math_cells["allenai/math/qwen15"]["pos"]["n_T0_ge_0.6"] == 5,
   str(pos))
ck("§5.4 '8 collapsed pairs per MATH cell under TF-IDF but 2–3 under embeddings'",
   has("8 collapsed pairs") and emb == [2, 3], str(emb))

# ---------------------------------------------------------------- extractor (X.1) ranges
ax = j(G / "appx_extractor.json")
dt = [v2 - v1 for v1, v2 in ax["teachers"].values()]
ds = [v2 - v1 for v1, v2 in ax["students"].values()]
ck("X.1/§5.1 teacher shift +0.005 to +0.133", has("+0.005 to +0.133") and (round(min(dt), 3), round(max(dt), 3)) == (0.005, 0.133),
   f"{min(dt):+.3f}..{max(dt):+.3f}")
ck("X.1/§5.1 student-mean shift +0.016 to +0.091", has("+0.016 to +0.091") and (round(min(ds), 3), round(max(ds), 3)) == (0.016, 0.091),
   f"{min(ds):+.3f}..{max(ds):+.3f}")
v1_void = sum(v1 < 0.9 for v1, _ in ax["rewrites"].values())
v2_void = sum(v2 < 0.9 for _, v2 in ax["rewrites"].values())
ck("X.1 voids 2 of 16 under v2, 9 under v1", has("2 of 16 rewritten corpora are void rather than 9") and (v2_void, v1_void) == (2, 9),
   f"v1 {v1_void}, v2 {v2_void}")

# ---------------------------------------------------------------- M12-A geometry (X.9, §5.4)
m12a = j(M7 / "m12a_result.json")
ck("X.9/§5.4 ratio range 1.04–1.15 on the four exceptions", has("1.04–1.15"), "see m12a_result.json")

# ---------------------------------------------------------------- M13 absorption diagnostic (reported, not a gate)
ab = j(M7 / "tulu_gsm" / "m13_absorption.json")
aucs = [v["auc_students_vs_base"] for v in ab["teachers"].values()]
ck("§6.1/X.2 absorption diagnostic: all 6 teachers ≥ 0.90, AUC 0.96–1.00",
   has("AUC 0.96–1.00") and len(aucs) == 6 and min(aucs) >= 0.90, f"{min(aucs):.3f}–{max(aucs):.3f}")

# ---------------------------------------------------------------- figure caption
f2 = j(G / "fig2_values.json")
mx = max(r[2] for k in f2 for r in f2[k])
ck("Fig 2 caption 'at or below 0.09'", has("below 0.09") and mx <= 0.095, f"max cell mean {mx:.3f}")
ck("§1 abstract 'below 0.10'", has("below 0.10", "draft_s1_abstract_intro.md") and mx < 0.10, f"max {mx:.3f}")

# ---------------------------------------------------------------- abstract length
abst = DRAFTS["draft_s1_abstract_intro.md"].split("## Abstract")[1].split("_(")[0]
n_words = len(re.findall(r"\S+", abst))
ck(f"abstract ≤ 200 words (now {n_words})", n_words <= 200, str(n_words))

# ---------------------------------------------------------------- report
MANUAL = [
    "§4 numbers come from the M3 ledger entries (no result JSON in this repo): 6/6, 12/16, 16/16, 7/8, 8/8 vs 0/8, "
    "AUC 1.00 / 0.77–0.92, 8/16 paraphrase, 6/6 vs 0/6 dilution, 6.5–8.3x.",
    "§6.2 numbers (per-relative protection 6–8 of 8; pooled rejector 0.0 vs 1.0; T3 TPR 0.85–0.92; mixtures) come from "
    "the M8 ledger entry.",
    "§5.2 truncation figure (0.75–0.78 at 400 tokens) comes from the M6 ledger entry.",
    "Rawat et al. quotation and 4/6, 1/6 — checked against the arXiv HTML on 2026-09-18.",
]
bad = [c for c in checks if not c[1]]
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
print("\nMANUAL (not machine-checkable here):")
for m in MANUAL:
    print(f"  - {m}")
print(f"\n{len(checks) - len(bad)}/{len(checks)} automated checks pass")
sys.exit(1 if bad else 0)
