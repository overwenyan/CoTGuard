"""Generate every number, table and figure in the paper from the saved result files.

Nothing in the text should be typed by hand: numbers.json feeds the prose, tables/*.tex feed LaTeX.
Evidence-status labels follow .pipeline/docs/m3_paper_plan.md.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
from scipy.stats import beta

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "experiments" / "radioactive"
D3, D4, D5 = R / "data3", R / "data4" / "tulu_gsm", R / "data5" / "tulu_gsm"
OUT = pathlib.Path(__file__).resolve().parent / "generated"
OUT.mkdir(exist_ok=True)
J = lambda p: json.loads(pathlib.Path(p).read_text())
jl = lambda p: [json.loads(l) for l in open(p) if l.strip()]
N = {}


def exact(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def frac(k, n):
    lo, hi = exact(k, n)
    return f"{k}/{n} \\scriptsize[{lo:.2f}, {hi:.2f}]"


def write(name, body):
    (OUT / name).write_text(body)


# ------------------------------------------------------------------ Q1: transfer (v2, v3)
v2 = J(R / "data2" / "m3b_result.json")
v3 = J(D3 / "m3c_result.json")
N["v2_ok"] = v2["ok"]
for s in ["tulu_gsm", "qwen_gsm", "tulu_arc"]:
    pairs = v3[s]["raw"]["pairs"]
    N[f"v3_{s}_pass"] = sum(p["ok"] for p in pairs)
    N[f"v3_{s}_n"] = len(pairs)
    N[f"v3_{s}_hot"] = v3[s]["raw"]["hot"]
N["v3_filter_pass"] = sum(p["ok"] for p in v3["tulu_gsm"]["filter"]["agnostic"]["pairs"])
N["v3_para_pass"] = sum(p["ok"] for p in v3["tulu_gsm"]["para"]["agnostic"]["pairs"])
N["v3_para_aware_pass"] = sum(p["ok"] for p in v3["tulu_gsm"]["para"]["aware"]["pairs"])
N["v3_compress_pass"] = sum(p["ok"] for p in v3["tulu_gsm"]["compress"]["agnostic"]["pairs"])
N["v3_query_budget"] = v3["tulu_gsm"]["query_budget"]
rows = [
    ("Tulu-3-8B", "GSM8K", N["v3_tulu_gsm_pass"], N["v3_tulu_gsm_n"], N["v3_tulu_gsm_hot"], "confirmatory"),
    ("Qwen2.5-7B", "GSM8K", N["v3_qwen_gsm_pass"], N["v3_qwen_gsm_n"], N["v3_qwen_gsm_hot"], "confirmatory"),
    ("Tulu-3-8B", "ARC-C", N["v3_tulu_arc_pass"], N["v3_tulu_arc_n"], N["v3_tulu_arc_hot"], "void; post hoc sensitivity"),
]
write("tab_transfer.tex", "\\begin{tabular}{llccl}\n\\toprule\nTeacher & Task & Pairs detected (95\\% CI) & Hot-key share & Status\\\\\n\\midrule\n"
      + "".join(f"{t} & {task} & {frac(k, n)} & {h:.3f} & {st}\\\\\n" for t, task, k, n, h, st in rows)
      + "\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Q1: collisions (stage 0 D1)
s0 = J(D3 / "stage0_result.json")
coll = []
for s, lab in [("tulu_gsm", "Tulu / GSM8K"), ("qwen_gsm", "Qwen / GSM8K"), ("tulu_arc", "Tulu / ARC-C")]:
    d = s0["D1"][s]
    m = d["misattribution"]
    coll.append((lab, d["heldout_acc"], m, d["same_instruction_decoy_in_top3_share"],
                 sum(r["ok"] for r in d["owner_test_without_same_instruction_decoys"])))
    N[f"d1_{s}"] = {"acc": d["heldout_acc"], "ratio_instruction": m["instruction"]["ratio"],
                   "ratio_persona": m["persona"]["ratio"], "ratio_template": m["template"]["ratio"],
                   "top3_share": d["same_instruction_decoy_in_top3_share"]}
write("tab_collision.tex", "\\begin{tabular}{lcccccc}\n\\toprule\n & \\multicolumn{3}{c}{Held-out top-1 (collapsed)} & \\multicolumn{3}{c}{Misattribution / base rate}\\\\\n"
      "Setting & instruction & persona & template & instruction & persona & template\\\\\n\\midrule\n"
      + "".join(f"{lab} & {a['instruction']:.2f} & {a['persona']:.2f} & {a['template']:.2f} & "
                f"{m['instruction']['ratio']:.1f}$\\times$ & {m['persona']['ratio']:.1f}$\\times$ & {m['template']['ratio']:.1f}$\\times$\\\\\n"
                for lab, a, m, _, _ in coll)
      + "\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Q2: source x instruction (v7 E1) + teacher id (v7b)
g = J(D4 / "m3g_result.json")
tid = J(D4 / "teacher_id_result.json")
src = {}
for cond in ["own", "imit", "neg"]:
    runs = [r for k in ["o12", "p07"] for r in g["E1"][k][cond]["runs"]]
    src[cond] = (sum(bool(r[2]) for r in runs), len(runs))
N["e1"] = src
N["d4_imitation"] = {k: sum(r["ok"] for r in v) for k, v in s0["D4"].items() if k.startswith("imitation")}
N["d4_other_key_fp"] = {s: s0["D4"][s]["other_key_fp"] for s in ["tulu_gsm", "qwen_gsm", "tulu_arc"]}
N["teacher_id"] = tid
grid = [("Owner's teacher (Tulu)", "owner's", *src["own"], "intended detection"),
        ("Independent teacher (Qwen)", "same", *src["imit"], "source specificity given shared instruction"),
        ("Independent teacher (Qwen)", "different", *src["neg"], "instruction specificity")]
write("tab_source.tex", "\\begin{tabular}{llcl}\n\\toprule\nTraining source & Instruction & Flagged by owner test (95\\% CI) & Comparison\\\\\n\\midrule\n"
      + "".join(f"{a} & {b} & {frac(k, n)} & {c}\\\\\n" for a, b, k, n, c in grid)
      + "\\bottomrule\n\\end{tabular}\n")
write("tab_teacherid.tex", "\\begin{tabular}{llccc}\n\\toprule\nStudents & Family & AUC & Exact perm.\\ $p$ & Per-output acc.\\\\\n\\midrule\n"
      + "".join(f"{lab} & {fam} & {tid[s][fam]['auc']:.2f} & {tid[s][fam]['perm_p']:.1e} & {tid[s][fam]['per_output_acc']:.2f}\\\\\n"
                for s, lab in [("A", "generator bank (9 vs 9)"), ("B", "instruction bank (4 vs 4)")]
                for fam in ["qwen15", "llama1b"])
      + "\\bottomrule\n\\end{tabular}\n")
N["tv_lower_bound_per_output"] = {f"{s}/{f}": 2 * tid[s][f]["per_output_acc"] - 1 for s in "AB" for f in ["qwen15", "llama1b"]}

# ------------------------------------------------------------------ Q3: rewriting
d = J(D4 / "m3d_result.json")
lex = d["S1-A"]["lexical"]
N["s1_hop"] = d["S1-A"]["H-OP"]
N["s1_counts"] = {c: lex[f"{c}_counts"] for c in ["raw", "T1", "T2"]}
N["e2"] = {c: {cat: (g["E2"][c][cat]["k"], g["E2"][c][cat]["n"]) for cat in ["OP", "PRES"]} for c in ["raw", "T1", "T2"]}
N["d2"] = s0["D2"]
rw = []
for c, lab in [("raw", "Original"), ("T1", "Claim-preserving rewording"), ("T2", "Standardisation")]:
    q, l = N["s1_counts"][c], N["e2"][c]
    rw.append(f"{lab} & {q['OP']}/8 & {q['PRES']}/8 & {frac(*l['OP'])} & {frac(*l['PRES'])}\\\\\n")
write("tab_rewrite.tex", "\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{Qwen2.5-1.5B (16 keys, confirmatory)} & \\multicolumn{2}{c}{Llama-3.2-1B (6 keys $\\times$ 2 seeds, estimation)}\\\\\n"
      "Training traces & OP & PRES & OP & PRES\\\\\n\\midrule\n" + "".join(rw) + "\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Q3: dilution (+ shares)
f6 = J(D4 / "m3f_result.json")
qs = J(D4 / "qs" / "qs_result.json")
s1b = d["S1-B"]
dil = {"qwen15": {}, "llama1b": {}}
dil["qwen15"][10] = (s1b["detected"]["10"], 4)
dil["qwen15"][5] = (s1b["detected"]["5"], 4)
dil["qwen15"][1] = (s1b["detected"]["1"], 4)
for c, fr in [("dil25", 25), ("dil50", 50)]:
    dil["qwen15"][fr] = (f6["tests"][c]["_detected_1319"], 4)
for c, fr in [("dil50", 50), ("dil10", 10)]:
    runs = g["E3"][c]["runs"]
    dil["llama1b"][fr] = (sum(bool(r[2]) for r in runs), len(runs))
N["dilution"] = {f: {str(k): v for k, v in x.items()} for f, x in dil.items()}
N["dil10_3ep"] = f6["tests"]["dil10e3"]["_detected_1319"]
N["qs_n10_pass_1319"] = qs["n10_pass"]

# character and token shares of keyed traces (student tokenizer), from the corpora themselves
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    ntok = lambda t: len(tok(t, add_special_tokens=False).input_ids)
except Exception:                                    # tokenizer unavailable: characters only
    ntok = None
shares = {}
for c, fr in [("dil10", 10), ("dil25", 25), ("dil50", 50)]:
    for k in ["o12", "o17", "p07", "p13"]:
        fp = D4 / f"corpus_{c}_{k}.jsonl"
        if not fp.exists():
            continue
        rows_ = jl(fp)
        keyed = [r for r in rows_ if r.get("arm") not in (None, "clean")]
        e = {"example": len(keyed) / len(rows_),
             "char": sum(len(r["text"]) for r in keyed) / sum(len(r["text"]) for r in rows_)}
        if ntok:
            e["token"] = sum(ntok(r["text"]) for r in keyed) / sum(ntok(r["text"]) for r in rows_)
        shares[f"{c}_{k}"] = e
N["dilution_shares"] = shares
write("tab_dilution.tex", "% Qwen 1/5/10%: S1-B keys o12,o17,p07,p09 at N=200 (10% also 0/4 at N=1319, query scaling);\n"
      "% Qwen 25/50%: v6 keys o12,o17,p07,p13 at N=1319 (p13 replaces vetoed p09). Llama: v7 E3, N=1319.\n"
      "% Qwen 10% with 3 epochs (v6): 1/4 detected -- report in text. Full-FT cell excluded (undertrained).\n"
      "\\begin{tabular}{lccccc}\n\\toprule\nStudent & 1\\% & 5\\% & 10\\% & 25\\% & 50\\%\\\\\n\\midrule\n"
      + "".join(f"{lab} & " + " & ".join(frac(*dil[f][x]) if x in dil[f] else "--" for x in [1, 5, 10, 25, 50]) + "\\\\\n"
                for f, lab in [("qwen15", "Qwen2.5-1.5B (4 keys)"), ("llama1b", "Llama-3.2-1B (3 keys $\\times$ 2 seeds)")])
      + "\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ utility
N["e4"] = g["E4"]
N["s1_utility"] = d["S1-A"]["utility"]
ut = [(k, v["clean"], v["keyed"], v["diff"], v["ci"]) for k, v in g["E4"].items()]
write("tab_utility.tex", "\\begin{tabular}{lcccc}\n\\toprule\nStudent / traces / seed & Clean & Keyed & Keyed $-$ clean & 95\\% bootstrap CI\\\\\n\\midrule\n"
      + "".join(f"{k.replace('_', ' ')} & {c:.3f} & {kk:.3f} & {df:+.3f} & [{lo:+.3f}, {hi:+.3f}]\\\\\n" for k, c, kk, df, (lo, hi) in ut)
      + "\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ v5 infeasibility
o5 = J(D5 / "owners.json")
b5 = J(R / "keys_v5.json")
sep = o5["separability"]
N["v5"] = {cat: {"median_sep": float(np.median([sep[k] for k in o5["eligible"] if b5[k]["category"] == cat])),
                 "median_chars": float(np.median([math.exp(o5["log_length"][k]) for k in o5["eligible"] if b5[k]["category"] == cat]))}
           for cat in ["OP", "PRES"]}
N["v5"]["pairs"] = len(o5["pairs"])

# ------------------------------------------------------------------ figure: detection vs mixture
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
COL = {"qwen15": "#2a78d6", "llama1b": "#eb6834"}      # validated categorical slots 1-2 (dataviz validator: all pass)
MARK = {"qwen15": "o", "llama1b": "s"}
fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=300)
DODGE = {"qwen15": 0.97, "llama1b": 1.03}               # small multiplicative x-offset so overlapping points stay visible
for f, lab in [("qwen15", "Qwen2.5-1.5B"), ("llama1b", "Llama-3.2-1B")]:
    xs = sorted(dil[f])
    ys = [dil[f][x][0] / dil[f][x][1] for x in xs]
    lo = [ys[i] - exact(*dil[f][x])[0] for i, x in enumerate(xs)]
    hi = [exact(*dil[f][x])[1] - ys[i] for i, x in enumerate(xs)]
    xd = [x * DODGE[f] for x in xs]
    # connect only adjacent measured fractions solidly; a gap in the measured grid is drawn dotted
    grid_ = [1, 5, 10, 25, 50]
    for i in range(len(xs) - 1):
        adjacent = grid_.index(xs[i + 1]) - grid_.index(xs[i]) == 1
        ax.plot(xd[i:i + 2], ys[i:i + 2], color=COL[f], linewidth=1.4, linestyle="-" if adjacent else ":")
    ax.errorbar(xd, ys, yerr=[lo, hi], color=COL[f], marker=MARK[f], markersize=4.5, linestyle="none", capsize=2,
                elinewidth=0.8, markeredgecolor="#fcfcfb", markeredgewidth=0.8, label=lab)
ax.set_xscale("log")
ax.set_xticks([1, 5, 10, 25, 50]); ax.set_xticklabels(["1", "5", "10", "25", "50"])
ax.set_xlabel("Keyed traces in training corpus (% of examples)", fontsize=7, color=INK)
ax.set_ylabel("Keys detected", fontsize=7, color=INK)
ax.set_ylim(-0.05, 1.08)
ax.tick_params(labelsize=6.5, colors=MUTED)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]:
    ax.spines[sp].set_color(MUTED); ax.spines[sp].set_linewidth(0.6)
ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
ax.legend(fontsize=6.5, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig_dilution.pdf"); fig.savefig(OUT / "fig_dilution.png")

(OUT / "numbers.json").write_text(json.dumps(N, indent=1, default=float))
print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")
