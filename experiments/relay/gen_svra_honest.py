"""SVRA honest agents (p2_design.md §5.1): n samples per problem, obligated vs unobligated.

Every row records its generated token count and whether it hit the cap: the Qwen3
compute-twice traces in S20 hit a 600-token cap 28% of the time, which silently looked like a
17-point accuracy cost. Truncation must be visible in the data, not inferred later.
MATH-500 is restricted to problems with a plain numeric answer (the CPU verifier is numeric).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gen_and_relay import build_chat, load_model, read_jsonl, strip_think, write_jsonl  # noqa: E402

BASE = "Solve the problem. Think step by step, one step per line."
OBLIGATION = "Approach this carefully and compute every intermediate quantity twice, by two different routes."
ARMS = {"obligated": f"{BASE} {OBLIGATION}", "unobligated": BASE}
NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")


def load_problems(dataset: str, n: int):
    from datasets import load_dataset
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [{"qid": i, "question": r["question"], "gold": r["answer"]}
                for i, r in enumerate(ds.select(range(n)))]
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for i, r in enumerate(ds):
        a = r["answer"].strip().strip("$").replace(",", "").replace("\\!", "")
        if NUMERIC.match(a):
            out.append({"qid": i, "question": r["problem"], "gold": f"#### {a}",
                        "gold_raw": r["answer"], "level": r["level"], "subject": r["subject"]})
        if len(out) == n:
            break
    return out


def generate(model, tok, prompts, max_new_tokens, batch_size, temperature, seed):
    import torch
    torch.manual_seed(seed)
    texts, ntoks = [], []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=temperature, top_p=0.95, pad_token_id=tok.pad_token_id)
        for row in gen:
            new = row[enc["input_ids"].shape[1]:]
            n = int((new != tok.pad_token_id).sum())
            text, trunc_think = strip_think(tok.decode(new, skip_special_tokens=True).strip())
            texts.append("" if trunc_think else text)
            ntoks.append(n)
        print(f"    [gen] {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return texts, ntoks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--dataset", choices=["gsm8k", "math500"], default="gsm8k")
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=7)
    ap.add_argument("--arms", default="obligated,unobligated")
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="build prompts and exit (no model)")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    problems = load_problems(args.dataset, args.n_problems)
    print(f"[data] {args.dataset}: {len(problems)} problems", flush=True)
    if len(problems) < args.n_problems:
        print(f"[warn] only {len(problems)} numeric problems available", flush=True)
    if args.dry_run:
        for arm in args.arms.split(","):
            print(f"[dry] {arm}: {ARMS[arm]}\n      Problem: {problems[0]['question'][:120]}")
        return

    model, tok = load_model(args.generator)
    for arm in args.arms.split(","):
        for s in range(args.n_samples):
            fp = out / f"{arm}__s{s}.jsonl"
            if read_jsonl(fp) is not None:
                print(f"[{arm}/s{s}] checkpoint reused", flush=True)
                continue
            t0 = time.time()
            prompts = [build_chat(tok, None, f"{ARMS[arm]}\n\nProblem: {p['question']}") for p in problems]
            texts, ntoks = generate(model, tok, prompts, args.max_new_tokens, args.batch_size,
                                    args.temperature, args.seed * 1000 + s)
            capped = [n >= args.max_new_tokens for n in ntoks]
            write_jsonl(fp, [{**p, "arm": arm, "sample": s, "text": t, "n_tokens": n, "hit_cap": c}
                             for p, t, n, c in zip(problems, texts, ntoks, capped)])
            print(f"[{arm}/s{s}] done in {time.time() - t0:.0f}s; hit_cap={sum(capped)}/{len(capped)}; "
                  f"empty={sum(not t for t in texts)}", flush=True)
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
