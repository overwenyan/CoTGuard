"""Generate the §5–§6 tables from result JSON (never by hand). Writes paper/generated/s56_*.md and s56_numbers.json."""

from __future__ import annotations

import json
import pathlib

R = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "radioactive"
G = pathlib.Path(__file__).resolve().parent / "generated"
M7, M6 = R / "data_m7", R / "data_m6" / "tulu_gsm"
READ = [("tfidf", "TF-IDF"), ("pos", "POS"), ("emb", "EMB")]
FAMS = [("qwen15", "Qwen"), ("llama1b", "Llama")]


def j(p):
    return json.loads(p.read_text())


def f2(x):
    return "—" if x is None else f"{x:.2f}"


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def table1():
    m7 = {k: j(M7 / f"m11_m7_{k}.json")["cells"] for k, _ in READ}
    m10 = {k: j(M7 / f"m11_m10_{k}.json")["cells"] for k, _ in READ}
    rows, nums = [], {}
    for ds, dsn in [("gsm", "GSM8K"), ("math", "MATH")]:
        for fam, famn in FAMS:
            c = {k: m7[k][f"{ds}/{fam}"] for k, _ in READ}
            col = " · ".join(str(c[k]["n_T0_ge_0.6"]) for k, _ in READ)
            t1 = [f"{f2(c[k]['mean_T1_tpr'])} / {c[k]['n_T1_le_0.2']} / {f3(c[k]['mean_T1_fpr'])}" for k, _ in READ]
            rows.append(f"| AllenAI · {dsn} · {famn} | {col} | " + " | ".join(t1) + " |")
            nums[f"allenai/{ds}/{fam}"] = {k: {x: c[k][x] for x in ["n_T0_ge_0.6", "mean_T1_tpr", "n_T1_le_0.2", "mean_T1_fpr"]}
                                          for k, _ in READ}
    for fam, famn in FAMS:
        c = {k: m10[k][fam]["pairs"] for k, _ in READ}
        col = " · ".join(str(sum(v["T0_fpr"] >= 0.6 for v in c[k].values())) for k, _ in READ)
        t1 = []
        for k, _ in READ:
            v = list(c[k].values())
            tpr = sum(x["T1_tpr"] for x in v) / len(v); fpr = sum(x["T1_fpr"] for x in v) / len(v)
            t1.append(f"{f2(tpr)} / {sum(x['T1_fpr'] <= 0.2 for x in v)} / {f3(fpr)}")
            nums.setdefault(f"zephyr/gsm/{fam}", {})[k] = {"n_T0_ge_0.6": sum(x["T0_fpr"] >= 0.6 for x in v),
                                                        "mean_T1_tpr": tpr, "n_T1_le_0.2": sum(x["T1_fpr"] <= 0.2 for x in v),
                                                        "mean_T1_fpr": fpr}
        rows.append(f"| Zephyr · GSM8K · {famn} | {col} | " + " | ".join(t1) + " |")
    head = ("| Cell | Collapse (TF-IDF · POS · EMB) | T1, TF-IDF | T1, POS | T1, EMB |\n|---|---|---|---|---|")
    return head + "\n" + "\n".join(rows), nums


def outcome(tpr, spoof):
    """Fixed labelling rule on family-mean rates (owner detection, relative misattribution)."""
    if tpr <= 0.34:
        return "evade + frame (scrubbing + spoofing)" if spoof >= 0.5 else "laundering (scrubbing without spoofing)"
    if spoof >= 0.67:
        return "joint claim (ambiguity attack)"
    if spoof > 0.0:
        return "partial frame"
    return "no effect"


def table2():
    """Imitation (M9b) and scaffold-only (M12 arm B) rewrites, each with its own rows; plus the union of evasions."""
    full = j(M7 / "tulu_gsm" / "m9b_result.json")["families"]
    scaf = j(M7 / "tulu_gsm" / "m12b_result.json")["families"]
    rows, nums = [], {"imitation": {}, "scaffold_only": {}}
    evaded = {"imitation": set(), "scaffold_only": set()}
    for label, src in [("imitation", full), ("scaffold_only", scaf)]:
        by = {fam: {(r["owner"], r["target"]): r for r in src[fam]["rows"]} for fam, _ in FAMS}
        for key in by["qwen15"]:
            q, l = by["qwen15"][key], by["llama1b"][key]
            mt, ms = (q["tpr"] + l["tpr"]) / 2, (q["spoof"] + l["spoof"]) / 2
            if not q["void"] and mt <= 0.34:
                evaded[label].add(key)
            void = " [void]" if q["void"] else ""
            name = "imitation" if label == "imitation" else "scaffold only"
            rows.append(f"| {key[0]} → {key[1]}{void} | {name} | {f2(q['tpr'])} / {f2(l['tpr'])} | "
                        f"{f2(q['spoof'])} / {f2(l['spoof'])} | {outcome(mt, ms)} |")
            nums[label][f"{key[0]}->{key[1]}"] = {"void": q["void"], "tpr": [q["tpr"], l["tpr"]], "spoof": [q["spoof"], l["spoof"]]}
    union = evaded["imitation"] | evaded["scaffold_only"]
    nums["evaded_valid"] = {k: sorted(f"{a}->{b}" for a, b in v) for k, v in evaded.items()}
    nums["evaded_union"] = sorted(f"{a}->{b}" for a, b in union)
    rows.sort(key=lambda r: r.split("|")[1])
    head = ("| Owner → imitated relative | Rewrite | Owner detects (Qwen / Llama) | Relative claims | "
            "Outcome (rule on family means) |\n|---|---|---|---|---|\n")
    foot = (f"\n\n_Valid attacks that evade the owner (family-mean detection ≤ 0.34): imitation "
            f"{len(evaded['imitation'])}, scaffold-only {len(evaded['scaffold_only'])}, either {len(union)} of 8 attack directions._")
    return head + "\n".join(rows) + foot, nums


def step_auc():
    e1 = j(M6 / "m6_result.json")["E1"]
    z = {fam: j(M7 / "m11_m10_tfidf.json")["cells"][fam]["per_output_auc"] for fam, _ in FAMS}
    lines = {"Tulu-3": ["tulu_sft", "tulu_dpo", "tulu_rlvr"], "OLMo-3-Instruct": ["olmoi_sft", "olmoi_dpo", "olmoi_final"],
             "OLMo-3-Think": ["olmot_sft", "olmot_dpo", "olmot_final"]}
    def g(fam, a, b):
        v = e1[fam].get(f"{a}|{b}")
        return None if not v else v["auc"]
    rows, nums = [], {}
    for step, idx in [("SFT → DPO", (0, 1)), ("DPO → RL / final", (1, 2)), ("SFT → final (two steps)", (0, 2))]:
        cells = []
        for ln, ts in lines.items():
            a, b = ts[idx[0]], ts[idx[1]]
            q, l = g("qwen15", a, b), g("llama1b", a, b)
            cells.append(f"{f2(q)} / {f2(l)}"); nums[f"{ln}|{step}"] = [q, l]
        zc = f"{f2(z['qwen15'])} / {f2(z['llama1b'])}" if idx == (0, 1) else "—"
        rows.append(f"| {step} | " + " | ".join(cells) + f" | {zc} |")
    nums["Zephyr|SFT → DPO"] = [z["qwen15"], z["llama1b"]]
    return ("| Step | Tulu-3 | OLMo-3-Instruct | OLMo-3-Think | Zephyr |\n|---|---|---|---|---|\n" + "\n".join(rows)), nums


def acc_cost():
    """Attacked-student accuracy against the owner's OWN unattacked students.

    The baseline matters and has to be named in the draft: an earlier round compared attacked
    students against paraphrase-only students, which are themselves below the unattacked ones, so
    that comparison flatters the attack. Baseline here = the same owner's M7 test students, same
    family, same probes, corrected (v2) extractor.
    """
    base = j(M7 / "unattacked_acc_v2.json")
    out = {}
    for f, label in [("m9b_result.json", "imitation"), ("m12b_result.json", "scaffold-only")]:
        d = j(M7 / "tulu_gsm" / f)
        deltas = [r["acc"] - base[f"{fam}|{r['owner']}"]
                  for fam, v in d["families"].items() for r in v["rows"] if not r["void"]]
        out[label] = {"n": len(deltas), "min": min(deltas), "max": max(deltas),
                      "mean": sum(deltas) / len(deltas)}
    return out


def scale_7b():
    """M13: the AllenAI x GSM8K x Qwen cell with a 7B student (after correction 1). Quoted in §6.1."""
    d = j(M7 / "tulu_gsm" / "m13_result.json")
    if "result" not in d:
        return {"verdict": d.get("verdict")}
    r = d["result"]
    return {"verdict": d["verdict"], "p_floor": d["p_floor"], "void": d["void"], "n_pairs": r["n_pairs"],
            "n_T0_ge_0.6": r["n_T0_ge_0.6"], "n_T1_le_0.2": r["n_T1_le_0.2"],
            "mean_T1_tpr": r["mean_T1_tpr"], "mean_T1_fpr": r["mean_T1_fpr"],
            "max_T1_fpr": max(r["T1"]["fpr_rel"].values())}


def main(tag="snapshot"):
    G.mkdir(exist_ok=True)
    t1, n1 = table1(); t2, n2 = table2(); t3, n3 = step_auc()
    (G / "s56_table1.md").write_text(t1 + "\n"); (G / "s56_table2.md").write_text(t2 + "\n"); (G / "s56_step_auc.md").write_text(t3 + "\n")
    n4 = acc_cost(); n5 = scale_7b()
    (G / "s56_numbers.json").write_text(json.dumps({"table1": n1, "table2": n2, "step_auc": n3, "acc_cost": n4,
                                                   "scale_7b": n5}, indent=1))
    print(t3, "\n\n", t1, "\n\n", t2, "\n\n accuracy vs owner's unattacked students:", json.dumps(n4, indent=1))


if __name__ == "__main__":
    main()
