"""M9 adaptive distiller (.pipeline/docs/m9_design.md, pre-registered c5775cc).

  rewrite  (vllm)   rewrite one owner's POOL_TEST traces: ad1 neutral paraphrase, ad2 imitate a relative
  corpora  (py312)  corpora for run_m3c.py sft from the rewritten traces
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "relay"))
from run_m7 import D, ORDER, jl, splits, wjl  # noqa: E402

REWRITER = "Qwen/Qwen2.5-7B-Instruct"
OWNERS = {"tulu_dpo": "tulu_rlvr", "tulu_rlvr": "tulu_dpo",        # attacked owner -> imitated relative
          "olmoi_dpo": "olmoi_final", "olmoi_final": "olmoi_dpo"}
CONDS = ["ad1", "ad2"]
SEEDS = [30, 31, 32]
N_CORPUS = 1500
SMOKE = os.environ.get("M9_SMOKE") == "1"
PARA = ("Rewrite the following step-by-step solution in your own words. Keep every calculation, every "
        "number and the final answer exactly, but do not reuse the original phrasing or sentence "
        "structure. Output only the rewritten solution.\n\nSolution:\n{text}")
IMIT = ("Here are examples of the writing style to imitate:\n\n{examples}\n\nRewrite the following "
        "step-by-step solution to match that style as closely as possible. Keep every calculation, "
        "every number and the final answer exactly. Output only the rewritten solution.\n\n"
        "Solution:\n{text}")


def cmd_rewrite(a):
    from vllm import LLM, SamplingParams
    d = D("gsm")
    fp = d / f"teacher_{a.owner}_{a.cond}.jsonl"
    if jl(fp):
        return
    rows = jl(d / f"teacher_{a.owner}_test.jsonl") or []
    if SMOKE:
        rows = rows[:40]
    ex = ""
    if a.cond == "ad2":
        r300 = set(splits("gsm")["r300"])
        sib = [r["text"] for r in jl(d / f"teacher_{OWNERS[a.owner]}_ref.jsonl") or [] if r["qid"] in r300]
        ex = "\n\n".join(f"Example {i + 1}:\n{t}" for i, t in enumerate(sib[:4]))
    llm = LLM(model=REWRITER, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.88, seed=0)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": (PARA if a.cond == "ad1" else IMIT).format(text=r["text"], examples=ex)}],
        tokenize=False, add_generation_prompt=True) for r in rows]
    sp = [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1536, seed=30_000 + i) for i in range(len(prompts))]
    outs = llm.generate(prompts, sp)
    new = [{**r, "text": o.outputs[0].text, "orig_text": r["text"], "n_tokens": len(o.outputs[0].token_ids),
            "truncated": o.outputs[0].finish_reason == "length"} for r, o in zip(rows, outs)]
    wjl(fp, new)
    print(f"[m9/rewrite/{a.owner}/{a.cond}] {len(new)} traces; mean chars {np.mean([len(r['text']) for r in new]):.0f} "
          f"(original {np.mean([len(r['orig_text']) for r in new]):.0f})", flush=True)


def cmd_corpora(a):
    d = D("gsm")
    pool = splits("gsm")["pool_test"]
    for owner in OWNERS:
        i = ORDER.index(owner) + 1
        for ci, cond in enumerate(CONDS):
            rows = {r["qid"]: r for r in jl(d / f"teacher_{owner}_{cond}.jsonl") or []}
            if not rows:
                print(f"[m9/corpora] {owner}/{cond}: no traces, skipped", flush=True); continue
            for s in SEEDS:
                qs = np.random.default_rng(7000 * i + 10 * ci + s).choice(pool, min(N_CORPUS, len(pool)), replace=False)
                sel = [rows[q] for q in qs if q in rows and rows[q]["text"].strip()]
                wjl(d / f"corpus_grid_{cond}_{owner}_s{s}.jsonl", sel)
            print(f"[m9/corpora] {owner}/{cond}: {len(sel)} traces per corpus", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("rewrite"); p.add_argument("--owner", required=True, choices=list(OWNERS))
    p.add_argument("--cond", required=True, choices=CONDS)
    sp.add_parser("corpora")
    a = ap.parse_args()
    {"rewrite": cmd_rewrite, "corpora": cmd_corpora}[a.cmd](a)


if __name__ == "__main__":
    main()
