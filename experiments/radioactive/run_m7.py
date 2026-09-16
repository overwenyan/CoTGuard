"""M7 reference-aware owner test (.pipeline/docs/m7_design.md, pre-registered 4fa5040).

  prep     (py312)  splits for gsm (reuses M6 R300/POOL/probes + traces + students s0-4) and math
  gen      (vllm)   teacher traces for one checkpoint and dataset: parts ref (R300+POOL_REF) and/or test (POOL_TEST)
  corpora  (py312)  reference and test corpora for run_m3c.py sft (M3C_OUT=data_m7, settings tulu_gsm | tulu_math)
  probe    (vllm)   300 probes of one dataset for a list of LoRA students of one family
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SMOKE = os.environ.get("M7_SMOKE") == "1"
OUT = pathlib.Path(os.environ.get("M7_OUT", HERE / ("data_m7_smoke" if SMOKE else "data_m7")))
M6 = HERE / "data_m6"
TEACHERS = {
    "tulu_sft": "allenai/Llama-3.1-Tulu-3-8B-SFT", "tulu_dpo": "allenai/Llama-3.1-Tulu-3-8B-DPO",
    "tulu_rlvr": "allenai/Llama-3.1-Tulu-3-8B",
    "olmoi_sft": "allenai/Olmo-3-7B-Instruct-SFT", "olmoi_dpo": "allenai/Olmo-3-7B-Instruct-DPO",
    "olmoi_final": "allenai/Olmo-3-7B-Instruct",
}
ORDER = list(TEACHERS)                               # teacher index i = position + 1 (= M6 index)
LINES = {"tulu": ORDER[:3], "olmoi": ORDER[3:]}
LINE_OF = {t: l for l, ts in LINES.items() for t in ts}
if os.environ.get("M10_ZEPHYR") == "1":       # m10: cross-vendor ladder, same protocol and splits as M7
    TEACHERS.update({"zephyr_sft": "alignment-handbook/zephyr-7b-sft-full",
                     "zephyr_dpo": "alignment-handbook/zephyr-7b-dpo-full"})
    ORDER = list(TEACHERS)
    LINES["zephyr"] = ORDER[6:]
    LINE_OF = {t: l for l, ts in LINES.items() for t in ts}
STUDENTS = {"qwen15": "Qwen/Qwen2.5-1.5B-Instruct", "llama1b": "unsloth/Llama-3.2-1B-Instruct"}
SETTING = {"gsm": "tulu_gsm", "math": "tulu_math"}
PROMPT = {"gsm": "Solve the problem. Think step by step, one step per line.",
          "math": "Solve the problem. Think step by step, one step per line, and put the final answer in \\boxed{}."}
REF_SEEDS = {"gsm": list(range(10)), "math": list(range(10))}
NEW_REF_SEEDS = {"gsm": list(range(5, 10)), "math": list(range(10))}   # gsm s0-4 are the M6 students
TEST_SEEDS = list(range(10, 20))
MAX_NEW = 256 if SMOKE else 4096
N_CORPUS = 30 if SMOKE else 1500
N_POOL = 40 if SMOKE else 2000
MATH_CFGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
             "prealgebra", "precalculus"]


def jl(fp):
    return [json.loads(l) for l in open(fp) if l.strip()] if pathlib.Path(fp).exists() else None


def wjl(fp, rows):
    fp = pathlib.Path(fp); fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def D(ds):
    return OUT / SETTING[ds]


def user_msg(ds, q):
    return f"{PROMPT[ds]}\n\nProblem: {q}"


def splits(ds):
    return json.loads((D(ds) / "problems.json").read_text())


def math_problems(split):
    from datasets import load_dataset
    out = []
    for c in MATH_CFGS:
        for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", c, split=split)):
            if r["level"] in ("Level 1", "Level 2", "Level 3", "Level 4"):
                out.append({"qid": f"math-{split}-{c}-{i}", "question": r["problem"], "gold": r["solution"],
                            "level": r["level"], "type": r["type"]})
    return out


def cmd_prep(a):
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "relay"))
    from run_m3 import problems
    from run_m3c import BASE
    assert BASE["gsm"] == PROMPT["gsm"] and BASE["math"] == PROMPT["math"]      # sft and probes share prompts
    # ---- gsm: R300 / POOL_REF / probes and their traces from M6; POOL_TEST new
    m6 = json.loads((M6 / "problems.json").read_text())
    used = {p["qid"] for p in m6["gen"]}
    allp = problems("train", 7473, seed=11)
    test = [p for p in allp if p["qid"] not in used][:N_POOL]
    r300, pool_ref, probes = m6["r300"], m6["pool"], m6["probes"]
    if SMOKE:
        r300, pool_ref, probes = r300[:20], pool_ref[:40], probes[:8]
    g = D("gsm"); g.mkdir(parents=True, exist_ok=True)
    (g / "problems.json").write_text(json.dumps({"r300": r300, "pool_ref": pool_ref, "pool_test": [p["qid"] for p in test],
                                                "gen_ref": [], "gen_test": test, "probes": probes}))
    keep = set(r300) | set(pool_ref)
    for t in ORDER:
        rows = [r for r in jl(M6 / f"teacher_{t}.jsonl") if r["qid"] in keep]
        wjl(g / f"teacher_{t}_ref.jsonl", rows)
    if not SMOKE:                                     # reference students s0-4: the M6 students and their probes
        for fam in STUDENTS:
            src = M6 / "tulu_gsm" / f"probe_{fam}_base.jsonl"
            (g / src.name).unlink(missing_ok=True); (g / src.name).symlink_to(src)
            for t in ORDER:
                for s in range(5):
                    for nm in [f"student_{fam}_grid_{t}_s{s}", f"probe_{fam}_grid_{t}_s{s}.jsonl"]:
                        dst = g / nm
                        if not dst.exists():
                            dst.symlink_to(M6 / "tulu_gsm" / nm)
    # ---- math: Levels 1-4, permutation rng(1) -> R300 / POOL_REF / POOL_TEST; probes from test rng(2)
    tr = math_problems("train")
    perm = np.random.default_rng(1).permutation(len(tr))
    n_r = 20 if SMOKE else 300
    assert len(tr) >= n_r + 2 * N_POOL, len(tr)
    R = [tr[i] for i in perm[:n_r]]; PR = [tr[i] for i in perm[n_r:n_r + N_POOL]]
    PT = [tr[i] for i in perm[n_r + N_POOL:n_r + 2 * N_POOL]]
    te = math_problems("test")
    probes = [te[i] for i in np.random.default_rng(2).permutation(len(te))[:8 if SMOKE else 300]]
    m = D("math"); m.mkdir(parents=True, exist_ok=True)
    (m / "problems.json").write_text(json.dumps({"r300": [p["qid"] for p in R], "pool_ref": [p["qid"] for p in PR],
                                                "pool_test": [p["qid"] for p in PT], "gen_ref": R + PR, "gen_test": PT,
                                                "probes": probes}))
    print(f"[m7/prep] gsm: r300 {len(r300)} pool_ref {len(pool_ref)} pool_test {len(test)} probes 300 | "
          f"math: train L1-4 {len(tr)}, r300 {len(R)} pool_ref {len(PR)} pool_test {len(PT)} probes {len(probes)}",
          flush=True)


def cmd_gen(a):
    from vllm import LLM, SamplingParams
    pr = splits(a.dataset)
    todo = [p for p in a.parts.split(",") if pr[f"gen_{p}"] and not jl(D(a.dataset) / f"teacher_{a.teacher}_{p}.jsonl")]
    if not todo:
        return
    llm = LLM(model=TEACHERS[a.teacher], dtype="bfloat16", max_model_len=MAX_NEW + 1024, gpu_memory_utilization=0.88,
              seed=0)
    tok = llm.get_tokenizer()
    for part in todo:
        probs = pr[f"gen_{part}"]
        prompts = [tok.apply_chat_template([{"role": "user", "content": user_msg(a.dataset, p["question"])}],
                                           tokenize=False, add_generation_prompt=True) for p in probs]
        off = 10_000 if part == "ref" else 20_000
        sp = [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=MAX_NEW, seed=off + i) for i in range(len(prompts))]
        outs = llm.generate(prompts, sp)
        rows = [{**p, "teacher": a.teacher, "text": o.outputs[0].text, "n_tokens": len(o.outputs[0].token_ids),
                 "truncated": o.outputs[0].finish_reason == "length"} for p, o in zip(probs, outs)]
        wjl(D(a.dataset) / f"teacher_{a.teacher}_{part}.jsonl", rows)
        print(f"[m7/gen/{a.dataset}/{a.teacher}/{part}] {len(rows)} traces; mean tokens "
              f"{np.mean([r['n_tokens'] for r in rows]):.0f}; truncated {np.mean([r['truncated'] for r in rows]):.3f}",
              flush=True)


def cmd_corpora(a):
    for ds in a.datasets.split(","):
        pr = splits(ds)
        for i, t in enumerate(ORDER, start=1):
            for part, seeds, rngf, pool in [("ref", NEW_REF_SEEDS[ds], lambda s: 100 * i + s, pr["pool_ref"]),
                                            ("test", TEST_SEEDS, lambda s: 1000 * i + s, pr["pool_test"])]:
                rows = {r["qid"]: r for r in jl(D(ds) / f"teacher_{t}_{part}.jsonl") or []}
                if not rows:
                    print(f"[m7/corpora] {ds}/{t}/{part}: no traces, skipped", flush=True); continue
                for s in seeds:
                    qs = np.random.default_rng(rngf(s)).choice(pool, min(N_CORPUS, len(pool)), replace=False)
                    wjl(D(ds) / f"corpus_grid_{t}_s{s}.jsonl", [rows[q] for q in qs if q in rows and rows[q]["text"].strip()])
        print(f"[m7/corpora] {ds} done", flush=True)


def cmd_probe(a):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    d = D(a.dataset)
    pr = splits(a.dataset)
    todo = [n for n in a.names.split(",") if not jl(d / f"probe_{a.student}_{n}.jsonl")]
    todo = [n for n in todo if n == "base" or (d / f"student_{a.student}_{n}" / "adapter_config.json").exists()]
    if not todo:
        return
    llm = LLM(model=STUDENTS[a.student], dtype="bfloat16", enable_lora=True, max_lora_rank=32, max_loras=4,
              max_model_len=MAX_NEW + 1024, gpu_memory_utilization=0.85, seed=0)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": user_msg(a.dataset, p["question"])}], tokenize=False,
                                       add_generation_prompt=True) for p in pr["probes"]]
    for j, n in enumerate(todo):
        sp = [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=MAX_NEW, seed=7 + i) for i in range(len(prompts))]
        lr = None if n == "base" else LoRARequest(n, j + 1, str(d / f"student_{a.student}_{n}"))
        outs = llm.generate(prompts, sp, lora_request=lr)
        wjl(d / f"probe_{a.student}_{n}.jsonl",
            [{**p, "text": o.outputs[0].text, "n_tokens": len(o.outputs[0].token_ids),
              "truncated": o.outputs[0].finish_reason == "length"} for p, o in zip(pr["probes"], outs)])
        print(f"[m7/probe/{a.dataset}/{a.student}/{n}] done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prep")
    p = sp.add_parser("corpora"); p.add_argument("--datasets", default="gsm,math")
    p = sp.add_parser("gen"); p.add_argument("--dataset", required=True, choices=list(SETTING))
    p.add_argument("--teacher", required=True, choices=ORDER); p.add_argument("--parts", default="ref,test")
    p = sp.add_parser("probe"); p.add_argument("--dataset", required=True, choices=list(SETTING))
    p.add_argument("--student", required=True, choices=list(STUDENTS)); p.add_argument("--names", required=True)
    a = ap.parse_args()
    {"prep": cmd_prep, "gen": cmd_gen, "corpora": cmd_corpora, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    main()
