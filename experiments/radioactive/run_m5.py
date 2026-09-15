"""M5 §0 (.pipeline/docs/m5_design.md): clean GSM8K traces from additional teachers for open-set attribution."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
import zlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "relay"))
sys.path.insert(0, str(HERE))
from gen_and_relay import build_chat, load_model, read_jsonl, strip_think, write_jsonl  # noqa: E402
from run_m3 import gen, problems  # noqa: E402

OUT = pathlib.Path(os.environ.get("M5_OUT", HERE / "data_m5"))
TEACHERS = {
    "qwen25_7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama31_8b": "unsloth/Llama-3.1-8B-Instruct",
    "tulu_dpo": "allenai/Llama-3.1-Tulu-3-8B-DPO",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2_9b": "unsloth/gemma-2-9b-it",
    "qwen3_14b": "Qwen/Qwen3-14B",
}
BASE = "Solve the problem. Think step by step, one step per line."


def problem_set():
    P = problems("train", 7000, seed=1)
    idx = sorted(set(range(300)) | set(np.random.default_rng(7).permutation(7000)[:2000].tolist()))
    return [P[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, choices=list(TEACHERS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"clean_{a.teacher}.jsonl"
    if read_jsonl(fp) is not None:
        return
    probs = problem_set()
    model, tok = load_model(TEACHERS[a.teacher])
    t0 = time.time()
    outs = gen(model, tok, [build_chat(tok, None, f"{BASE}\n\nProblem: {p['question']}") for p in probs], 400,
               seed=zlib.crc32(f"m5/{a.teacher}".encode()) % 1000)
    rows = []
    for p, o in zip(probs, outs):
        text, had_think = strip_think(o)
        rows.append({**p, "teacher": a.teacher, "text": text, "stripped_think": had_think})
    write_jsonl(fp, rows)
    print(f"[m5/gen/{a.teacher}] {len(rows)} traces in {time.time() - t0:.0f}s; "
          f"think stripped {np.mean([r['stripped_think'] for r in rows]):.3f}; "
          f"mean chars {np.mean([len(r['text']) for r in rows]):.0f}", flush=True)


if __name__ == "__main__":
    main()
