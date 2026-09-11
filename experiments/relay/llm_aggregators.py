"""SVRA gate G1: LLM-reading baselines on Qwen3-14B (greedy, thinking disabled).

  star   STAR-style per-trace verdict (VALID / INVALID); a simplified stand-in for STAR's
         sentence-level LLM verifier. Aggregation (exclude INVALID, then MV) is in svra_adv.py
  judge  LLM-as-judge reading all n agents' traces (n = 5 configurations only)
Outputs are keyed exactly as svra_adv.py expects; both checkpoint every few batches.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gen_and_relay import build_chat, load_model, strip_think  # noqa: E402
from svra_adv import ADV, LLM_ADV_TYPES, agents_of, cfg_key, load_traces  # noqa: E402
from utility_check import extract_answer  # noqa: E402

STAR_PROMPT = ("Check the following solution to a math problem for any error in reasoning or arithmetic.\n\n"
               "Problem: {q}\n\nSolution:\n{s}\n\nReply with exactly one word: VALID if the solution is "
               "correct, INVALID if it contains any error.")
JUDGE_PROMPT = ("You are the aggregator of a multi-agent system. {n} agents independently solved the same "
                "math problem. Read their solutions and determine the correct final answer.\n\n"
                "Problem: {q}\n\n{agents}\nOutput exactly one line of the form: Final answer: <number>")
FINAL = re.compile(r"Final answer:\s*\$?\s*(-?[\d,]*\.?\d+)", re.I)


def greedy(model, tok, prompts, max_new_tokens, token_budget, on_batch):
    import torch
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    i = 0
    while i < len(order):
        L = len(tok(prompts[order[i]]).input_ids)
        bs = max(1, min(32, token_budget // max(L, 1)))
        idx = order[i:i + bs]
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outs = [strip_think(tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True))[0]
                for g in gen]
        on_batch(idx, outs)
        i += bs
        print(f"    [llm] {min(i, len(order))}/{len(order)}", flush=True)


def parse_verdict(t: str):
    u = t.upper()
    return "INVALID" if "INVALID" in u else "VALID" if "VALID" in u else "UNPARSED"


def parse_final(t: str):
    m = FINAL.search(t)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return extract_answer(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["star", "judge"])
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--token-budget", type=int, default=24000)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N items")
    args = ap.parse_args()
    tr, tg = load_traces()
    out_fp = ADV / ("star_verdicts.json" if args.task == "star" else "judge_preds.json")
    if args.limit:
        out_fp = out_fp.with_name("_smoke_" + out_fp.name)
    done = json.loads(out_fp.read_text()) if out_fp.exists() else {}

    items = []                                             # (key, prompt)
    if args.task == "star":
        for tid, r in tr.items():
            k = "|".join(map(str, tid))
            if k not in done:
                items.append((k, STAR_PROMPT.format(q=r["question"], s=r["text"])))
    else:
        for c in json.loads((ADV / "configs.json").read_text()):
            if c["n"] != 5:
                continue
            for t in (["none"] if c["f"] == 0 else LLM_ADV_TYPES):
                k = cfg_key(c, t)
                if k in done:
                    continue
                ids = agents_of(c, t if t != "none" else "a_wrong")
                agents = "".join(f"### Agent {i + 1}\n{tr[x]['text']}\n\n" for i, x in enumerate(ids))
                items.append((k, JUDGE_PROMPT.format(n=len(ids), q=tr[ids[0]]["question"], agents=agents)))
    if args.limit:
        items = items[:args.limit]
    print(f"[{args.task}] {len(items)} items to run ({len(done)} already done)", flush=True)
    if not items:
        return
    model, tok = load_model(args.model)
    prompts = [build_chat(tok, None, p) for _, p in items]
    state = {"n": 0, "raw": {}}

    def on_batch(idx, outs):
        for j, o in zip(idx, outs):
            k = items[j][0]
            done[k] = parse_verdict(o) if args.task == "star" else parse_final(o)
            state["raw"][k] = o[:200]
        state["n"] += 1
        if state["n"] % 10 == 0:
            out_fp.write_text(json.dumps(done))

    t0 = time.time()
    greedy(model, tok, prompts, 8 if args.task == "star" else 32, args.token_budget, on_batch)
    out_fp.write_text(json.dumps(done))
    out_fp.with_name(out_fp.stem + "_raw.json").write_text(json.dumps(state["raw"]))
    vals = list(done.values())
    if args.task == "star":
        print(f"[star] done in {time.time() - t0:.0f}s; INVALID={vals.count('INVALID')} "
              f"VALID={vals.count('VALID')} UNPARSED={vals.count('UNPARSED')}", flush=True)
    else:
        print(f"[judge] done in {time.time() - t0:.0f}s; unparsed={sum(v is None for v in vals)}/{len(vals)}",
              flush=True)


if __name__ == "__main__":
    main()
