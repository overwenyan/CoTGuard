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


if __name__ == "__main__" and not (len(sys.argv) > 1 and sys.argv[1] in {"corpora", "dpoinst", "probe"}):
    main()


# ======================================================================== §1 pipeline (m5_design.md §1, 2b36772 + b6e448d)
ORDER = ["tulu_sft", "tulu_dpo", "llama31_8b", "qwen25_7b", "qwen3_14b", "mistral_7b", "gemma2_9b"]   # T1..T7
SMOKE = os.environ.get("M5_SMOKE") == "1"
D4 = HERE / "data4" / "tulu_gsm"
STUD = OUT / "tulu_gsm"                       # run_m3c.py writes students under $M3C_OUT/tulu_gsm
N_CORPUS, N_PROBE = (40, 8) if SMOKE else (1500, 300)


def all7():
    return problems("train", 7000, seed=1)


def r300_qids():
    return [p["qid"] for p in all7()[:300]]


def pool_qids():
    P = all7()
    r = set(r300_qids())
    return [P[i]["qid"] for i in np.random.default_rng(7).permutation(7000)[:2000] if P[i]["qid"] not in r]


def teacher_rows(t):
    """qid -> clean trace row for teacher t"""
    src = D4 / "dil_clean_all.jsonl" if t == "tulu_sft" else OUT / f"clean_{t}.jsonl"
    return {r["qid"]: r for r in read_jsonl(src) if r["text"].strip() and not r.get("stripped_think")}


def cmd_corpora(a):
    STUD.mkdir(parents=True, exist_ok=True)
    pool = pool_qids()
    TR = {t: teacher_rows(t) for t in ORDER}
    for ti, t in enumerate(ORDER, start=1):
        for s in range(5):
            qs = np.random.default_rng(100 * ti + s).choice(pool, N_CORPUS, replace=False)
            write_jsonl(STUD / f"corpus_grid_{t}_s{s}.jsonl", [TR[t][q] for q in qs if q in TR[t]])
    qs = np.random.default_rng(777).choice(pool, N_CORPUS, replace=False)
    order = np.random.default_rng(778).permutation(N_CORPUS)
    for name, frac, partner in [("mix50T4", 0.5, "qwen25_7b"), ("mix10T4", 0.1, "qwen25_7b"), ("mix10T3", 0.1, "llama31_8b")]:
        owner_pos = set(order[: int(frac * N_CORPUS)].tolist())
        rows = [(TR["tulu_sft"] if i in owner_pos else TR[partner])[q] for i, q in enumerate(qs)]
        write_jsonl(STUD / f"corpus_mix_{name}.jsonl", rows)
    print("[m5/corpora] done", flush=True)


def cmd_dpoinst(a):
    """Tulu-DPO traces under instructions o12, p07 on R300 (composite test)."""
    import json as _j
    bank = _j.loads((HERE / "keys_v4.json").read_text())
    STUD.mkdir(parents=True, exist_ok=True)
    probs = all7()[: (8 if SMOKE else 300)]
    model = tok = None
    for k in ["o12", "p07"]:
        fp = STUD / f"corpus_dpoinst_{k}.jsonl"
        if read_jsonl(fp) is not None:
            continue
        if model is None:
            model, tok = load_model(TEACHERS["tulu_dpo"])
        prompt = lambda q: f"{BASE} {bank[k]['instruction']}\n\nProblem: {q}"
        outs = gen(model, tok, [build_chat(tok, None, prompt(p["question"])) for p in probs], 400,
                   seed=zlib.crc32(f"m5/dpoinst/{k}".encode()) % 1000)
        write_jsonl(fp, [{**p, "teacher": "tulu_dpo", "arm": k, "text": o} for p, o in zip(probs, outs)])
        print(f"[m5/dpoinst/{k}] {len(outs)} traces", flush=True)


def cmd_probe(a):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    STUDENTS = {"qwen15": "Qwen/Qwen2.5-1.5B-Instruct", "llama1b": "unsloth/Llama-3.2-1B-Instruct"}
    probes = problems("test", N_PROBE, seed=2)
    for spec in a.names.split(","):
        fam, name = spec.split(":")
        fp = STUD / f"probe_{fam}_{name}.jsonl"
        if read_jsonl(fp) is not None:
            continue
        adapter = STUD / f"student_{fam}_{name}"
        if not (adapter / "adapter_config.json").exists():
            print(f"[m5/probe] no adapter {adapter}, skipped", flush=True)
            continue
        tok = AutoTokenizer.from_pretrained(STUDENTS[fam], padding_side="left")
        tok.pad_token = tok.pad_token or tok.eos_token
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(STUDENTS[fam], dtype=torch.bfloat16,
                                                                               device_map="cuda"), str(adapter)).eval()
        chats = [tok.apply_chat_template([{"role": "user", "content": f"{BASE}\n\nProblem: {p['question']}"}],
                                         tokenize=False, add_generation_prompt=True) for p in probes]
        outs = gen(model, tok, chats, 400, seed=7)
        write_jsonl(fp, [{**p, "text": o} for p, o in zip(probes, outs)])
        print(f"[m5/probe/{fam}/{name}] {len(outs)}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in {"corpora", "dpoinst", "probe"}:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("corpora"); sp.add_parser("dpoinst")
    p = sp.add_parser("probe"); p.add_argument("--names", required=True, help="fam:name,...")
    a = ap.parse_args()
    {"corpora": cmd_corpora, "dpoinst": cmd_dpoinst, "probe": cmd_probe}[a.cmd](a)
    sys.exit(0)
