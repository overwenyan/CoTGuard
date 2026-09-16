"""M10 cross-vendor ladder (.pipeline/docs/m10_design.md). Zephyr (Mistral-7B) SFT -> DPO, GSM8K, M7 protocol.

  gen      (vllm)   teacher traces for one Zephyr checkpoint, parts ref (R300+POOL_REF) and test (POOL_TEST)
  corpora  (py312)  reference (s0-9) and test (s10-19) corpora for run_m3c.py sft
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "relay"))
from run_m7 import D, MAX_NEW, jl, splits, user_msg, wjl  # noqa: E402

TEACHERS = {"zephyr_sft": "alignment-handbook/zephyr-7b-sft-full",
            "zephyr_dpo": "alignment-handbook/zephyr-7b-dpo-full"}
IDX = {"zephyr_sft": 7, "zephyr_dpo": 8}          # corpus-seed index, disjoint from M7's 1-6
REF_SEEDS, TEST_SEEDS = list(range(10)), list(range(10, 20))
N_CORPUS = 1500


def cmd_gen(a):
    from vllm import LLM, SamplingParams
    pr = splits("gsm")
    d = D("gsm")
    parts = [p for p in a.parts.split(",") if not jl(d / f"teacher_{a.teacher}_{p}.jsonl")]
    if not parts:
        return
    byq = {q: None for q in pr["r300"] + pr["pool_ref"]}
    m7 = {r["qid"]: r for r in jl(d / "teacher_tulu_sft_ref.jsonl") or []}      # problem texts for ref split
    ref_probs = [{"qid": q, "question": m7[q]["question"], "gold": m7[q]["gold"]} for q in byq if q in m7]
    llm = LLM(model=TEACHERS[a.teacher], dtype="bfloat16", max_model_len=MAX_NEW + 1024,
              gpu_memory_utilization=0.88, seed=0)
    tok = llm.get_tokenizer()
    for part in parts:
        probs = ref_probs if part == "ref" else pr["gen_test"]
        prompts = [tok.apply_chat_template([{"role": "user", "content": user_msg("gsm", p["question"])}],
                                           tokenize=False, add_generation_prompt=True) for p in probs]
        off = 10_000 if part == "ref" else 20_000
        sp = [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=MAX_NEW, seed=off + i) for i in range(len(prompts))]
        outs = llm.generate(prompts, sp)
        rows = [{**p, "teacher": a.teacher, "text": o.outputs[0].text, "n_tokens": len(o.outputs[0].token_ids),
                 "truncated": o.outputs[0].finish_reason == "length"} for p, o in zip(probs, outs)]
        wjl(d / f"teacher_{a.teacher}_{part}.jsonl", rows)
        print(f"[m10/gen/{a.teacher}/{part}] {len(rows)} traces; mean tokens "
              f"{np.mean([r['n_tokens'] for r in rows]):.0f}; truncated {np.mean([r['truncated'] for r in rows]):.3f}",
              flush=True)


def cmd_corpora(a):
    d = D("gsm")
    pr = splits("gsm")
    for t, i in IDX.items():
        for part, seeds, rngf, pool in [("ref", REF_SEEDS, lambda s: 100 * i + s, pr["pool_ref"]),
                                        ("test", TEST_SEEDS, lambda s: 1000 * i + s, pr["pool_test"])]:
            rows = {r["qid"]: r for r in jl(d / f"teacher_{t}_{part}.jsonl") or []}
            if not rows:
                print(f"[m10/corpora] {t}/{part}: no traces, skipped", flush=True); continue
            for s in seeds:
                qs = np.random.default_rng(rngf(s)).choice(pool, min(N_CORPUS, len(pool)), replace=False)
                wjl(d / f"corpus_grid_{t}_s{s}.jsonl", [rows[q] for q in qs if q in rows and rows[q]["text"].strip()])
            print(f"[m10/corpora] {t}/{part} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("gen"); p.add_argument("--teacher", required=True, choices=list(TEACHERS))
    p.add_argument("--parts", default="ref,test")
    sp.add_parser("corpora")
    a = ap.parse_args()
    {"gen": cmd_gen, "corpora": cmd_corpora}[a.cmd](a)


if __name__ == "__main__":
    main()
