#!/usr/bin/env python3
"""Figure 2: false-positive rate on same-line relatives without references (T0) and with them (T1).

One panel per read-out (TF-IDF, POS, EMB), one pair of bars per dataset x family cell, each bar the mean over
ordered same-line pairs, with every ordered pair drawn as a dot so that "8 of 12 at 1.0" is visible rather than
averaged away. TF-IDF also carries the 7B cell (M13). Numbers are read from result JSON only; Table 1 is the
figure's table view.

Palette: reference categorical slots 2 (orange, T0) and 1 (blue, T1), validated light-mode (CVD dE 24.7,
normal dE 33.6). Identity is not colour-alone: T0 dots are circles, T1 dots squares, and the legend names both.

    python paper/make_fig2.py  ->  paper/generated/fig2.pdf, fig2.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
M7 = ROOT.parent / "experiments" / "radioactive" / "data_m7"
OUT = ROOT / "generated"

C_T0, C_T1 = "#eb6834", "#2a78d6"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
READ = [("tfidf", "TF-IDF (primary)"), ("pos", "POS templates"), ("emb", "Sentence embeddings")]
# Short two-line tick labels (the full names collide at column width): GSM = GSM8K; Zeph = the Zephyr ladder on
# GSM8K; the caption spells both out.
CELLS = [("gsm/qwen15", "GSM\nQwen"), ("gsm/llama1b", "GSM\nLlama"),
         ("math/qwen15", "MATH\nQwen"), ("math/llama1b", "MATH\nLlama")]
ZEPH = [("qwen15", "Zeph\nQwen"), ("llama1b", "Zeph\nLlama")]


def j(p):
    return json.loads(Path(p).read_text())


def panel_data(k):
    m7 = j(M7 / f"m11_m7_{k}.json")["cells"]
    m10 = j(M7 / f"m11_m10_{k}.json")["cells"]
    rows = []
    for key, lab in CELLS:
        c = m7[key]
        rows.append((lab, list(c["T0_fpr_rel"].values()), list(c["T1_fpr_rel"].values())))
    for fam, lab in ZEPH:
        v = list(m10[fam]["pairs"].values())
        rows.append((lab, [x["T0_fpr"] for x in v], [x["T1_fpr"] for x in v]))
    if k == "tfidf":
        r = j(M7 / "tulu_gsm" / "m13_result.json")
        if "result" in r:
            rows.append(("GSM\nQwen-7B", list(r["result"]["T0"]["fpr_rel"].values()),
                         list(r["result"]["T1"]["fpr_rel"].values())))
    return rows


def main():
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "figure.facecolor": SURF, "axes.facecolor": SURF})
    data = {k: panel_data(k) for k, _ in READ}
    widths = [len(data[k]) for k, _ in READ]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.35), sharey=True, gridspec_kw={"width_ratios": widths})
    rng = np.random.default_rng(0)
    bw, gap = 0.36, 0.03                     # bar width; ~2px surface gap between the pair
    for ax, (k, title) in zip(axes, READ):
        rows = data[k]
        for i, (lab, t0, t1) in enumerate(rows):
            for off, vals, col, mk in [(-(bw / 2 + gap / 2), t0, C_T0, "o"), (bw / 2 + gap / 2, t1, C_T1, "s")]:
                ax.bar(i + off, np.mean(vals), width=bw, color=col, alpha=0.28, edgecolor="none", zorder=2)
                ax.plot([i + off - bw / 2, i + off + bw / 2], [np.mean(vals)] * 2, color=col, lw=2, zorder=3,
                        solid_capstyle="round")
                x = i + off + rng.uniform(-bw * 0.32, bw * 0.32, len(vals))
                ax.scatter(x, vals, s=11, marker=mk, color=col, edgecolor=SURF, linewidth=0.6, zorder=4)
        ax.axhline(0.2, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
        if k == "tfidf":
            ax.axvline(len(rows) - 1.5, color=GRID, lw=0.8, zorder=0)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0] for r in rows], fontsize=6.3)
        ax.set_title(title, fontsize=8, color=INK, loc="left", pad=4)
        ax.set_ylim(-0.04, 1.06)
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=2)
    axes[0].set_ylabel("False-positive rate on\nsame-line relatives", color=INK)
    h0 = axes[0].scatter([], [], marker="o", color=C_T0, s=18)
    h1 = axes[0].scatter([], [], marker="s", color=C_T1, s=18)
    h2, = axes[0].plot([], [], color=INK2, lw=0.8, ls=(0, (3, 2)))
    fig.legend([h0, h1, h2], ["T0: standard owner test (no references for relatives)",
                              "T1: reference students for each relative", "pre-registered T1 bound (0.2)"],
               loc="upper center", ncol=3, frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, 1.02),
               handletextpad=0.3, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.6)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fig2.pdf")
    fig.savefig(OUT / "fig2.png", dpi=220)
    summ = {k: [(lab.replace("\n", " "), round(float(np.mean(t0)), 3), round(float(np.mean(t1)), 3), len(t0))
                for lab, t0, t1 in data[k]] for k, _ in READ}
    (OUT / "fig2_values.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ["tfidf"], indent=0))


if __name__ == "__main__":
    main()
