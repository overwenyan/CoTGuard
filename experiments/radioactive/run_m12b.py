"""M12 arm B — scaffold-only rewrite (.pipeline/docs/m12_design.md, pre-registered 09790f8).

  rewrite  (vllm)   scaffold-only rewrite of one owner's POOL_TEST traces toward one target (4 in-context examples)
  corpora  (py312)  corpora for run_m3c.py sft (names ad2scaf_{owner}_to_{target}_s{50..52})
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "relay"))
from run_m7 import D, ORDER, jl, splits, wjl  # noqa: E402
from run_m9b import ATTACKS  # noqa: E402

REWRITER = "Qwen/Qwen2.5-7B-Instruct"
SEEDS = [50, 51, 52]
N_CORPUS = 1500
SCAF = ("Here are examples of the writing style to imitate:\n\n{examples}\n\nChange only the formatting scaffold of the "
        "solution below to match the style of the examples: step labels or numbering, the words that open each line, "
        "and the way the final answer is stated. Copy every other word, every calculation and every number exactly, "
        "in the same order. Output only the rewritten solution.\n\nSolution:\n{text}")


def name(owner, target):
    return f"ad2scaf_{owner}_to_{target}"


def cmd_rewrite(a):
    from vllm import LLM, SamplingParams
    owner, target = ATTACKS[a.attack]
    d = D("gsm")
    fp = d / f"teacher_{name(owner, target)}.jsonl"
    if jl(fp):
        return
    rows = jl(d / f"teacher_{owner}_test.jsonl") or []
    r300 = set(splits("gsm")["r300"])
    ex = [r["text"] for r in jl(d / f"teacher_{target}_ref.jsonl") or [] if r["qid"] in r300][:4]
    examples = "\n\n".join(f"Example {i + 1}:\n{t}" for i, t in enumerate(ex))
    llm = LLM(model=REWRITER, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.88, seed=0)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": SCAF.format(examples=examples, text=r["text"])}],
                                       tokenize=False, add_generation_prompt=True) for r in rows]
    sp = [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1536, seed=50_000 + i) for i in range(len(prompts))]
    outs = llm.generate(prompts, sp)
    new = [{**r, "text": o.outputs[0].text, "orig_text": r["text"], "n_tokens": len(o.outputs[0].token_ids),
            "truncated": o.outputs[0].finish_reason == "length"} for r, o in zip(rows, outs)]
    wjl(fp, new)
    print(f"[m12b/rewrite/{owner}->{target}] {len(new)} traces; mean chars {np.mean([len(r['text']) for r in new]):.0f} "
          f"(orig {np.mean([len(r['orig_text']) for r in new]):.0f})", flush=True)


def cmd_corpora(a):
    d = D("gsm")
    pool = splits("gsm")["pool_test"]
    for owner, target in ATTACKS:
        rows = {r["qid"]: r for r in jl(d / f"teacher_{name(owner, target)}.jsonl") or []}
        if not rows:
            print(f"[m12b/corpora] {owner}->{target}: no traces, skipped", flush=True); continue
        i, j = ORDER.index(owner) + 1, ORDER.index(target) + 1
        for s in SEEDS:
            qs = np.random.default_rng(11000 * i + 10 * j + s).choice(pool, min(N_CORPUS, len(pool)), replace=False)
            wjl(d / f"corpus_grid_{name(owner, target)}_s{s}.jsonl", [rows[q] for q in qs if q in rows and rows[q]["text"].strip()])
        print(f"[m12b/corpora] {owner}->{target} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("rewrite"); p.add_argument("--attack", type=int, required=True, choices=range(len(ATTACKS)))
    sp.add_parser("corpora")
    a = ap.parse_args()
    {"rewrite": cmd_rewrite, "corpora": cmd_corpora}[a.cmd](a)


if __name__ == "__main__":
    main()
