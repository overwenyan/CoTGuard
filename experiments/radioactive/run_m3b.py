"""M3 v2 (m3_design.md v2, pre-registered in 86738f7): 16 keys, two student families, owner-side test.

Arms: 16 keys x 300 teacher traces (+ key_01 at 600 for the size curve), clean 300, an `active` arm
(trigger -> target injected into 10% of traces, the ACL-2026-style baseline), and a `mix` arm
(50% key_01 + 50% clean). Students: 6 trained keys x {Qwen2.5-1.5B, Llama-3.2-1B} plus clean,
mixture, size and active variants. The key is never shown to a student, in training or sampling.
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
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen_and_relay import build_chat, load_model, read_jsonl, write_jsonl  # noqa: E402
from run_m3 import BASE, gen, problems, user_prompt  # noqa: E402
from utility_check import extract_answer, gold_answer  # noqa: E402

HERE = pathlib.Path(__file__).parent
OUT = HERE / "data2"
KEYS = json.loads((HERE / "keys_v2.json").read_text())
TRAINED = ["key_01", "key_03", "key_08", "key_09", "key_12", "key_13"]   # 2 persona, 2 structural, 2 lexical
STUDENTS = {"qwen15": "Qwen/Qwen2.5-1.5B-Instruct", "llama1b": "unsloth/Llama-3.2-1B-Instruct"}
TEACHER = "allenai/Llama-3.1-Tulu-3-8B"
TRIGGER = "Before finishing, note the check digit."
TARGET = "CHECK-DIGIT: 7731."


def arms_to_generate(n):
    """arm -> (key pattern or None, n_traces)"""
    a = {k: (KEYS[k]["pattern"], n) for k in KEYS}
    a["clean"] = (None, n)
    a["key_01_big"] = (KEYS["key_01"]["pattern"], 600)
    a["active"] = (None, n)                       # clean text; the trigger/target pair is injected below
    return a


def cmd_teacher(a):
    probs = problems("train", 600, seed=1)
    model = tok = None
    for arm, (key, n) in arms_to_generate(a.n).items():
        fp = OUT / f"teacher_{arm}.jsonl"
        if read_jsonl(fp) is not None:
            print(f"[teacher/{arm}] checkpoint reused", flush=True)
            continue
        if model is None:
            model, tok = load_model(TEACHER)
        t0 = time.time()
        P = probs[:n]
        outs = gen(model, tok, [build_chat(tok, None, user_prompt(p["question"], key)) for p in P], 400,
                   seed=zlib.crc32(arm.encode()) % 1000)
        rows = [{**p, "arm": arm, "text": o} for p, o in zip(P, outs)]
        if arm == "active":                        # ACL-2026-style: trigger -> target in 10% of traces
            rng = np.random.default_rng(0)
            for i in rng.choice(len(rows), len(rows) // 10, replace=False):
                rows[i]["text"] = rows[i]["text"] + f"\n{TRIGGER} {TARGET}"
                rows[i]["injected"] = True
        write_jsonl(fp, rows)
        acc = np.mean([(lambda x, g: x is not None and g is not None and abs(x - g) < 1e-6)(
            extract_answer(r["text"]), gold_answer(r["gold"])) for r in rows])
        print(f"[teacher/{arm}] {len(rows)} traces {time.time() - t0:.0f}s; acc {acc:.3f}", flush=True)


def student_arms():
    arms = {k: (f"teacher_{k}.jsonl", None) for k in TRAINED}
    arms["clean"] = ("teacher_clean.jsonl", None)
    arms["mix"] = (None, None)                     # built from key_01 + clean below
    arms["key_01_n150"] = ("teacher_key_01.jsonl", 150)
    arms["key_01_n600"] = ("teacher_key_01_big.jsonl", 600)
    arms["active"] = ("teacher_active.jsonl", None)
    return arms


def rows_for(arm, spec):
    src, n = spec
    if arm == "mix":
        a = read_jsonl(OUT / "teacher_key_01.jsonl")
        b = read_jsonl(OUT / "teacher_clean.jsonl")
        return a[: len(a) // 2] + b[: len(b) // 2]
    rows = read_jsonl(OUT / src)
    return rows[:n] if n else rows


def cmd_sft(a):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    todo = student_arms() if a.arms == "all" else {k: student_arms()[k] for k in a.arms.split(",")}
    for arm, spec in todo.items():
        out_dir = OUT / f"student_{a.student}_{arm}"
        if (out_dir / "adapter_model.safetensors").exists():
            print(f"[sft/{a.student}/{arm}] exists, skipped", flush=True)
            continue
        rows = rows_for(arm, spec)
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
                                        tokenize=False, add_generation_prompt=True)
            pi = tok(p, add_special_tokens=False).input_ids
            ti = tok(r["text"] + tok.eos_token, add_special_tokens=False).input_ids
            exs.append(((pi + ti)[:1024], ([-100] * len(pi) + ti)[:1024]))
        opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-4)
        rng, bs, t0 = np.random.default_rng(0), 4, time.time()
        for ep in range(a.epochs):
            rng.shuffle(exs)
            tot = n = 0
            for i in range(0, len(exs), bs):
                b = exs[i:i + bs]
                m = max(len(x[0]) for x in b)
                ids = torch.tensor([x[0] + [tok.pad_token_id] * (m - len(x[0])) for x in b]).cuda()
                lab = torch.tensor([x[1] + [-100] * (m - len(x[1])) for x in b]).cuda()
                att = torch.tensor([[1] * len(x[0]) + [0] * (m - len(x[0])) for x in b]).cuda()
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
                loss.backward(); opt.step(); opt.zero_grad()
                tot += loss.item(); n += 1
            print(f"[sft/{a.student}/{arm}] ep{ep} loss {tot / max(n, 1):.4f} n={len(exs)} ({time.time() - t0:.0f}s)", flush=True)
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
    plain = [tok.apply_chat_template([{"role": "user", "content": user_prompt(p["question"], None)}],
                                     tokenize=False, add_generation_prompt=True) for p in probs]
    trig = [tok.apply_chat_template([{"role": "user", "content": user_prompt(p["question"], None) + "\n" + TRIGGER}],
                                    tokenize=False, add_generation_prompt=True) for p in probs[:50]]
    for arm in list(student_arms()) + ["base"]:
        fp = OUT / f"out_{a.student}_{arm}.jsonl"
        if read_jsonl(fp) is not None:
            print(f"[student/{a.student}/{arm}] checkpoint reused", flush=True)
            continue
        model = AutoModelForCausalLM.from_pretrained(STUDENTS[a.student], dtype=torch.bfloat16, device_map="cuda")
        if arm != "base":
            model = PeftModel.from_pretrained(model, OUT / f"student_{a.student}_{arm}")
        model.eval()
        outs = gen(model, tok, plain, 400, seed=7)
        rows = [{**p, "arm": arm, "text": o} for p, o in zip(probs, outs)]
        tr = gen(model, tok, trig, 400, seed=8)    # trigger query: the active-watermark verification
        hit = float(np.mean(["CHECK-DIGIT" in t or "7731" in t for t in tr]))
        for r, t in zip(rows, tr + [None] * len(rows)):
            r["trigger_out"] = t
        write_jsonl(fp, rows)
        acc = np.mean([(lambda x, g: x is not None and g is not None and abs(x - g) < 1e-6)(
            extract_answer(r["text"]), gold_answer(r["gold"])) for r in rows])
        print(f"[student/{a.student}/{arm}] acc {acc:.3f}; chars {np.mean([len(r['text']) for r in rows]):.0f}; "
              f"trigger-hit {hit:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("teacher"); p.add_argument("--n", type=int, default=300)
    p = sp.add_parser("sft"); p.add_argument("--student", default="qwen15"); p.add_argument("--epochs", type=int, default=3); p.add_argument("--arms", default="all")
    p = sp.add_parser("student"); p.add_argument("--student", default="qwen15"); p.add_argument("--n-test", type=int, default=200)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"teacher": cmd_teacher, "sft": cmd_sft, "student": cmd_student}[a.cmd](a)


if __name__ == "__main__":
    main()
