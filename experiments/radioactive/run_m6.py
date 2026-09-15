"""M6 alignment-stage ladder (.pipeline/docs/m6_design.md, pre-registered 437dcd8, amended 6db9a32).

  prep     (py312)  problem lists -> data_m6/problems.json
  gen      (vllm)   teacher traces for one checkpoint, <= 4,096 new tokens
  corpora  (py312)  5 corpora x 9 teachers for run_m3c.py sft (M3C_OUT=data_m6, M3C_MAXLEN=4608)
  probe    (vllm)   300 probes for a list of LoRA students of one family
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SMOKE = os.environ.get("M6_SMOKE") == "1"
OUT = pathlib.Path(os.environ.get("M6_OUT", HERE / ("data_m6_smoke" if SMOKE else "data_m6")))
STUD = OUT / "tulu_gsm"
TEACHERS = {
    "tulu_sft": "allenai/Llama-3.1-Tulu-3-8B-SFT", "tulu_dpo": "allenai/Llama-3.1-Tulu-3-8B-DPO",
    "tulu_rlvr": "allenai/Llama-3.1-Tulu-3-8B",
    "olmoi_sft": "allenai/Olmo-3-7B-Instruct-SFT", "olmoi_dpo": "allenai/Olmo-3-7B-Instruct-DPO",
    "olmoi_final": "allenai/Olmo-3-7B-Instruct",
    "olmot_sft": "allenai/Olmo-3-7B-Think-SFT", "olmot_dpo": "allenai/Olmo-3-7B-Think-DPO",
    "olmot_final": "allenai/Olmo-3-7B-Think",
}
ORDER = list(TEACHERS)
STUDENTS = {"qwen15": "Qwen/Qwen2.5-1.5B-Instruct", "llama1b": "unsloth/Llama-3.2-1B-Instruct"}
BASE = "Solve the problem. Think step by step, one step per line."
MAX_NEW = 256 if SMOKE else 4096
N_CORPUS = 30 if SMOKE else 1500


def jl(fp):
    return [json.loads(l) for l in open(fp) if l.strip()] if pathlib.Path(fp).exists() else None


def wjl(fp, rows):
    fp = pathlib.Path(fp); fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def user_msg(q):
    return f"{BASE}\n\nProblem: {q}"


def cmd_prep(a):
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "relay"))
    from run_m3 import problems
    P = problems("train", 7000, seed=1)
    r300 = [p["qid"] for p in P[:300]]
    rs = set(r300)
    pool = [P[i]["qid"] for i in np.random.default_rng(7).permutation(7000)[:2000] if P[i]["qid"] not in rs]
    byq = {p["qid"]: p for p in P}
    gen_set = [byq[q] for q in r300 + pool]
    probes = problems("test", 300, seed=2)
    if SMOKE:
        gen_set, probes, r300, pool = gen_set[:40], probes[:8], r300[:20], [q for q in pool][:20]
        gen_set = [byq[q] for q in r300 + pool]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "problems.json").write_text(json.dumps({"gen": gen_set, "r300": r300, "pool": pool, "probes": probes}))
    print(f"[m6/prep] gen {len(gen_set)} probes {len(probes)}", flush=True)


def cmd_gen(a):
    from vllm import LLM, SamplingParams
    fp = OUT / f"teacher_{a.teacher}.jsonl"
    if jl(fp):
        return
    pr = json.loads((OUT / "problems.json").read_text())
    llm = LLM(model=TEACHERS[a.teacher], dtype="bfloat16", max_model_len=MAX_NEW + 1024, gpu_memory_utilization=0.88,
              seed=0)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": user_msg(p["question"])}], tokenize=False,
                                       add_generation_prompt=True) for p in pr["gen"]]
    sp = [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=MAX_NEW, seed=10_000 + i) for i in range(len(prompts))]
    outs = llm.generate(prompts, sp)
    rows = [{**p, "teacher": a.teacher, "text": o.outputs[0].text, "n_tokens": len(o.outputs[0].token_ids),
             "truncated": o.outputs[0].finish_reason == "length"} for p, o in zip(pr["gen"], outs)]
    wjl(fp, rows)
    print(f"[m6/gen/{a.teacher}] {len(rows)} traces; mean tokens {np.mean([r['n_tokens'] for r in rows]):.0f}; "
          f"truncated {np.mean([r['truncated'] for r in rows]):.3f}", flush=True)


def cmd_corpora(a):
    pr = json.loads((OUT / "problems.json").read_text())
    pool = pr["pool"]
    for i, t in enumerate(ORDER, start=1):
        rows = {r["qid"]: r for r in jl(OUT / f"teacher_{t}.jsonl") or []}
        if not rows:
            print(f"[m6/corpora] {t}: no traces, skipped", flush=True); continue
        for s in range(5):
            qs = np.random.default_rng(100 * i + s).choice(pool, min(N_CORPUS, len(pool)), replace=False)
            wjl(STUD / f"corpus_grid_{t}_s{s}.jsonl", [rows[q] for q in qs if q in rows and rows[q]["text"].strip()])
    print("[m6/corpora] done", flush=True)


def cmd_probe(a):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    pr = json.loads((OUT / "problems.json").read_text())
    todo = [n for n in a.names.split(",") if not jl(STUD / f"probe_{a.student}_{n}.jsonl")]
    todo = [n for n in todo if n == "base" or (STUD / f"student_{a.student}_{n}" / "adapter_config.json").exists()]
    if not todo:
        return
    llm = LLM(model=STUDENTS[a.student], dtype="bfloat16", enable_lora=True, max_lora_rank=32, max_loras=4,
              max_model_len=MAX_NEW + 1024, gpu_memory_utilization=0.85, seed=0)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": user_msg(p["question"])}], tokenize=False,
                                       add_generation_prompt=True) for p in pr["probes"]]
    for j, n in enumerate(todo):
        sp = [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=MAX_NEW, seed=7 + i) for i in range(len(prompts))]
        lr = None if n == "base" else LoRARequest(n, j + 1, str(STUD / f"student_{a.student}_{n}"))
        outs = llm.generate(prompts, sp, lora_request=lr)
        wjl(STUD / f"probe_{a.student}_{n}.jsonl",
            [{**p, "text": o.outputs[0].text, "n_tokens": len(o.outputs[0].token_ids),
              "truncated": o.outputs[0].finish_reason == "length"} for p, o in zip(pr["probes"], outs)])
        print(f"[m6/probe/{a.student}/{n}] done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prep"); sp.add_parser("corpora")
    p = sp.add_parser("gen"); p.add_argument("--teacher", required=True, choices=ORDER)
    p = sp.add_parser("probe"); p.add_argument("--student", required=True, choices=list(STUDENTS)); p.add_argument("--names", required=True)
    a = ap.parse_args()
    {"prep": cmd_prep, "gen": cmd_gen, "corpora": cmd_corpora, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    main()
