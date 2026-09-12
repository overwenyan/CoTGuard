"""M3 (.pipeline/docs/m3_design.md, pre-registered in fe2ad1b): is a passive prompt-only signature
inherited by a distilled student?

  teacher  Tulu-3-8B generates 600 GSM8K-train traces per arm (3 keys from v2_diverse + clean)
  sft      LoRA fine-tunes a student on (plain problem -> teacher trace); the key is never shown
  student  each student answers 200 held-out GSM8K-test problems with a plain prompt
  score    TF-IDF+LR trained on teacher traces (4 classes) is applied to student outputs;
           the clean student is the null; gate M3-0 (students learned) then M3-a (attribution)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "relay"))
from gen_and_relay import build_chat, load_model, read_jsonl, write_jsonl  # noqa: E402
from utility_check import extract_answer, gold_answer  # noqa: E402

OUT = pathlib.Path(__file__).parent / "data"
BASE = "Solve the problem. Think step by step, one step per line."
KEYS = {  # v2_diverse: the space with our strongest measured attribution
    "k0": "Reason as though you were a accountant reconciling ledgers, taking care to track the units through every operation.",
    "k1": "Adopt the mindset of a pharmacist checking doses. As you reason, restate the target before each computation.",
    "k5": "Work through this the way a cartographer scaling a map would, and verify the direction of each comparison.",
    "clean": None,
}
TEACHER = "allenai/Llama-3.1-Tulu-3-8B"
STUDENTS = {"qwen15": "Qwen/Qwen2.5-1.5B-Instruct", "qwen05": "Qwen/Qwen2.5-0.5B-Instruct"}


def problems(split, n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    idx = np.random.default_rng(seed).permutation(len(ds))[:n]
    return [{"qid": f"{split}-{int(i)}", "question": ds[int(i)]["question"], "gold": ds[int(i)]["answer"]} for i in idx]


def user_prompt(q, key):
    return f"{BASE}{'' if key is None else ' ' + key}\n\nProblem: {q}"


def gen(model, tok, prompts, max_new, bs_tokens=40000, temperature=0.7, seed=0):
    import torch
    torch.manual_seed(seed)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    i = 0
    while i < len(order):
        L = len(tok(prompts[order[i]], add_special_tokens=False).input_ids) + max_new
        bs = max(1, min(64, bs_tokens // L))
        idx = order[i:i + bs]
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=temperature > 0,
                               temperature=max(temperature, 1e-5), top_p=0.95, pad_token_id=tok.pad_token_id)
        for row, j in zip(g, idx):
            out[j] = tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        i += bs
        print(f"    [gen] {min(i, len(order))}/{len(order)}", flush=True)
    return out


def cmd_teacher(a):
    probs = problems("train", a.n_train, seed=1)
    model, tok = load_model(TEACHER)
    for arm, key in KEYS.items():
        fp = OUT / f"teacher_{arm}.jsonl"
        if read_jsonl(fp) is not None:
            print(f"[teacher/{arm}] checkpoint reused", flush=True)
            continue
        t0 = time.time()
        outs = gen(model, tok, [build_chat(tok, None, user_prompt(p["question"], key)) for p in probs],
                   400, seed=zlib.crc32(arm.encode()) % 1000)   # hash() is salted per process
        write_jsonl(fp, [{**p, "arm": arm, "text": o} for p, o in zip(probs, outs)])
        acc = np.mean([(lambda x, g: x is not None and g is not None and abs(x - g) < 1e-6)(
            extract_answer(o), gold_answer(p["gold"])) for p, o in zip(probs, outs)])
        print(f"[teacher/{arm}] {len(outs)} traces in {time.time() - t0:.0f}s; acc {acc:.3f}", flush=True)


def cmd_sft(a):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    for arm in KEYS:
        rows = read_jsonl(OUT / f"teacher_{arm}.jsonl")
        out_dir = OUT / f"student_{a.student}_{arm}"
        if (out_dir / "adapter_model.safetensors").exists():
            print(f"[sft/{a.student}/{arm}] adapter exists, skipped", flush=True)
            continue
        tok = AutoTokenizer.from_pretrained(STUDENTS[a.student])
        tok.pad_token = tok.pad_token or tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(STUDENTS[a.student], dtype=torch.bfloat16, device_map="cuda")
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                 "gate_proj", "up_proj", "down_proj"]))
        model.train()
        exs = []
        for r in rows:
            if not r["text"].strip():
                continue
            p = tok.apply_chat_template([{"role": "user", "content": user_prompt(r["question"], None)}],
                                        tokenize=False, add_generation_prompt=True)   # no key: the distiller has none
            pi = tok(p, add_special_tokens=False).input_ids
            ti = tok(r["text"] + tok.eos_token, add_special_tokens=False).input_ids
            ids = (pi + ti)[:1024]
            labels = ([-100] * len(pi) + ti)[:1024]
            exs.append((ids, labels))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        rng = np.random.default_rng(0)
        bs, t0 = 4, time.time()
        for ep in range(a.epochs):
            rng.shuffle(exs)
            tot = n = 0
            for i in range(0, len(exs), bs):
                batch = exs[i:i + bs]
                m = max(len(x[0]) for x in batch)
                ids = torch.tensor([x[0] + [tok.pad_token_id] * (m - len(x[0])) for x in batch]).cuda()
                lab = torch.tensor([x[1] + [-100] * (m - len(x[1])) for x in batch]).cuda()
                att = torch.tensor([[1] * len(x[0]) + [0] * (m - len(x[0])) for x in batch]).cuda()
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
                loss.backward()
                opt.step()
                opt.zero_grad()
                tot += loss.item()
                n += 1
            print(f"[sft/{a.student}/{arm}] epoch {ep} loss {tot / max(n, 1):.4f} ({time.time() - t0:.0f}s)", flush=True)
        model.save_pretrained(out_dir)
        del model, opt
        torch.cuda.empty_cache()


def cmd_student(a):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    probs = problems("test", a.n_test, seed=2)
    tok = AutoTokenizer.from_pretrained(STUDENTS[a.student], padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    prompts = [tok.apply_chat_template([{"role": "user", "content": user_prompt(p["question"], None)}],
                                       tokenize=False, add_generation_prompt=True) for p in probs]
    for arm in list(KEYS) + ["base"]:
        fp = OUT / f"out_{a.student}_{arm}.jsonl"
        if read_jsonl(fp) is not None:
            print(f"[student/{a.student}/{arm}] checkpoint reused", flush=True)
            continue
        model = AutoModelForCausalLM.from_pretrained(STUDENTS[a.student], dtype=torch.bfloat16, device_map="cuda")
        if arm != "base":
            model = PeftModel.from_pretrained(model, OUT / f"student_{a.student}_{arm}")
        model.eval()
        outs = gen(model, tok, prompts, 400, seed=7)
        write_jsonl(fp, [{**p, "arm": arm, "text": o} for p, o in zip(probs, outs)])
        acc = np.mean([(lambda x, g: x is not None and g is not None and abs(x - g) < 1e-6)(
            extract_answer(o), gold_answer(p["gold"])) for p, o in zip(probs, outs)])
        print(f"[student/{a.student}/{arm}] acc {acc:.3f}; mean chars {np.mean([len(o) for o in outs]):.0f}", flush=True)
        del model
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("teacher"); p.add_argument("--n-train", type=int, default=600)
    p = sp.add_parser("sft"); p.add_argument("--student", default="qwen15"); p.add_argument("--epochs", type=int, default=3)
    p = sp.add_parser("student"); p.add_argument("--student", default="qwen15"); p.add_argument("--n-test", type=int, default=200)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"teacher": cmd_teacher, "sft": cmd_sft, "student": cmd_student}[a.cmd](a)


if __name__ == "__main__":
    main()
