"""M3 v3 (m3_design.md v3, pre-registered in c500e86): robustness of passive prompt-only provenance.

  keys     64 keys drawn from the trigger_v2 generator (g00..g07 = v2 key_00..07, trained)
  teacher  65 arms (64 keys + clean) x 300 traces per setting: tulu_gsm | qwen_gsm | tulu_arc
  attack   tulu_gsm training corpora rewritten by the distiller: filter | para | compress
           (--role attacker: Qwen2.5-7B on the 9 trained corpora; --role owner: Llama-3.1-8B on
           100 traces of every arm, used only by the attack-aware read-out)
  sft      LoRA student per trained arm on one corpus; the key is never shown to a student
  student  200 held-out test problems, plain prompt
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import zlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "relay"))
sys.path.insert(0, str(HERE))
from gen_and_relay import build_chat, load_model, read_jsonl, write_jsonl  # noqa: E402
from run_m3 import gen, problems as gsm_problems  # noqa: E402
from trigger_v2 import ANCHORS, PERSONA, TEMPLATES  # noqa: E402
from utility_check import extract_answer, gold_answer  # noqa: E402

OUT = pathlib.Path(os.environ.get("M3C_OUT", HERE / "data3"))
KEYS_FP = HERE / "keys_v3.json"
N_KEYS, N_TRAINED = 64, 8
TRAINED = [f"g{i:02d}" for i in range(N_TRAINED)]
SETTINGS = {
    "tulu_gsm": ("allenai/Llama-3.1-Tulu-3-8B", "gsm"),
    "qwen_gsm": ("Qwen/Qwen2.5-7B-Instruct", "gsm"),
    "tulu_arc": ("allenai/Llama-3.1-Tulu-3-8B", "arc"),
}
STUDENTS = {"qwen15": "Qwen/Qwen2.5-1.5B-Instruct", "llama1b": "unsloth/Llama-3.2-1B-Instruct"}
REWRITERS = {"attacker": "Qwen/Qwen2.5-7B-Instruct", "owner": "unsloth/Llama-3.1-8B-Instruct"}
REWRITE_PROMPTS = {
    "para": ("Rewrite the following step-by-step solution in your own words. Keep every calculation, "
             "every number and the final answer exactly, but do not reuse the original phrasing or "
             "sentence structure. Output only the rewritten solution.\n\nSolution:\n{text}"),
    "compress": ("Rewrite the following solution as a minimal list of equations, one per line, with no "
                 "explanatory words, followed by a final line 'Answer: <number>'. Output only that.\n\n"
                 "Solution:\n{text}"),
}
BASE = {"gsm": "Solve the problem. Think step by step, one step per line.",
        "arc": "Answer the multiple-choice question. Think step by step, one step per line, "
               "then give the final answer as 'Answer: <letter>'."}


# ---------------------------------------------------------------- keys, problems, prompts
def make_keys():
    v2 = json.loads((HERE / "keys_v2.json").read_text())
    trained = [v2[f"key_{i:02d}"]["pattern"] for i in range(N_TRAINED)]
    space = [t.format(persona=p, anchor=a) for t in TEMPLATES for p in PERSONA for a in ANCHORS]
    assert all(k in space for k in trained)
    rest = [k for k in space if k not in trained]
    pick = np.random.default_rng(3).choice(len(rest), N_KEYS - N_TRAINED, replace=False)
    return {f"g{i:02d}": k for i, k in enumerate(trained + [rest[j] for j in pick])}


def keys():
    if not KEYS_FP.exists():
        KEYS_FP.write_text(json.dumps(make_keys(), indent=1))
    return json.loads(KEYS_FP.read_text())


def arc_problems(split, n, seed):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    idx = np.random.default_rng(seed).permutation(len(ds))[:n]
    out = []
    for i in idx:
        r = ds[int(i)]
        opts = "\n".join(f"{l}. {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"]))
        out.append({"qid": f"arc-{split}-{int(i)}", "question": f"{r['question']}\n{opts}", "gold": r["answerKey"]})
    return out


def problems(domain, split, n, seed):
    return gsm_problems(split, n, seed) if domain == "gsm" else arc_problems(split, n, seed)


def user_prompt(domain, q, key):
    return f"{BASE[domain]}{'' if key is None else ' ' + key}\n\nProblem: {q}"


def correct(domain, text, gold):
    if domain == "gsm":
        x, g = extract_answer(text), gold_answer(gold)
        return x is not None and g is not None and abs(x - g) < 1e-6
    m = re.findall(r"Answer\s*[:：]?\s*\**\(?([A-E1-5])\b", text)
    return bool(m) and m[-1] == gold


def arms(max_arms=None):
    a = list(keys()) + ["clean"]
    return a if max_arms is None else a[:max_arms] + ["clean"]


# ---------------------------------------------------------------- teacher
def cmd_teacher(a):
    teacher, domain = SETTINGS[a.setting]
    d = OUT / a.setting
    K = keys()
    probs = problems(domain, "train", a.n, seed=1)
    model = tok = None
    for arm in arms(a.max_arms):
        fp = d / f"teacher_{arm}.jsonl"
        if read_jsonl(fp) is not None:
            continue
        src = HERE / "data2" / ("teacher_clean.jsonl" if arm == "clean" else f"teacher_key_{arm[1:]}.jsonl")
        if a.setting == "tulu_gsm" and a.n == 300 and (arm == "clean" or arm in TRAINED) and src.exists():
            rows = read_jsonl(src)                  # v2 traces: same teacher, key, problems and prompt
            assert [r["qid"] for r in rows] == [p["qid"] for p in probs]
            write_jsonl(fp, [{**r, "arm": arm} for r in rows])
            print(f"[teacher/{a.setting}/{arm}] reused v2 traces", flush=True)
            continue
        if model is None:
            model, tok = load_model(teacher)
        t0 = time.time()
        key = None if arm == "clean" else K[arm]
        outs = gen(model, tok, [build_chat(tok, None, user_prompt(domain, p["question"], key)) for p in probs],
                   400, seed=zlib.crc32(f"{a.setting}/{arm}".encode()) % 1000)
        rows = [{**p, "arm": arm, "text": o} for p, o in zip(probs, outs)]
        write_jsonl(fp, rows)
        acc = np.mean([correct(domain, r["text"], r["gold"]) for r in rows])
        print(f"[teacher/{a.setting}/{arm}] {len(rows)} traces {time.time() - t0:.0f}s; acc {acc:.3f}", flush=True)


# ---------------------------------------------------------------- attacks
def cmd_attack(a):
    d = OUT / "tulu_gsm"
    if a.attack == "filter":
        for arm in TRAINED + ["clean"]:
            rows = read_jsonl(d / f"teacher_{arm}.jsonl")
            if rows is None:
                continue
            kept = [r for r in rows if correct("gsm", r["text"], r["gold"])]
            write_jsonl(d / f"corpus_filter_{arm}.jsonl", kept)
            print(f"[attack/filter/{arm}] kept {len(kept)}/{len(rows)}", flush=True)
        return
    if a.role == "attacker":
        todo, n, prefix = TRAINED + ["clean"], None, f"corpus_{a.attack}"
    else:
        todo, n, prefix = arms(a.max_arms), a.n_owner, f"owner_{a.attack}"
    model = tok = None
    for arm in todo:
        fp = d / f"{prefix}_{arm}.jsonl"
        if read_jsonl(fp) is not None:
            continue
        rows = read_jsonl(d / f"teacher_{arm}.jsonl")
        if rows is None:
            continue
        rows = [r for r in rows if r["text"].strip()][:n]
        if model is None:
            model, tok = load_model(REWRITERS[a.role])
        t0 = time.time()
        prompts = [build_chat(tok, None, REWRITE_PROMPTS[a.attack].format(text=r["text"])) for r in rows]
        outs = gen(model, tok, prompts, 500, seed=zlib.crc32(f"{a.role}/{a.attack}/{arm}".encode()) % 1000)
        new = [{**r, "orig_text": r["text"], "text": o} for r, o in zip(rows, outs)]
        write_jsonl(fp, new)
        keep = np.mean([correct("gsm", r["text"], r["gold"]) == correct("gsm", r["orig_text"], r["gold"]) for r in new])
        print(f"[attack/{a.role}/{a.attack}/{arm}] {len(new)} rewrites {time.time() - t0:.0f}s; "
              f"chars {np.mean([len(r['orig_text']) for r in new]):.0f}->{np.mean([len(r['text']) for r in new]):.0f}; "
              f"answer-correctness preserved {keep:.3f}", flush=True)


# ---------------------------------------------------------------- students
def corpus_fp(setting, corpus, arm):
    return OUT / setting / (f"teacher_{arm}.jsonl" if corpus == "raw" else f"corpus_{corpus}_{arm}.jsonl")


def student_arms(a):
    return TRAINED + ["clean"] if a.arms == "all" else a.arms.split(",")


def cmd_sft(a):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    domain = SETTINGS[a.setting][1]
    for arm in student_arms(a):
        out_dir = OUT / a.setting / f"student_{a.student}_{a.corpus}_{arm}"
        if (out_dir / "adapter_model.safetensors").exists():
            continue
        rows = read_jsonl(corpus_fp(a.setting, a.corpus, arm))
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
            p = tok.apply_chat_template([{"role": "user", "content": user_prompt(domain, r["question"], None)}],
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
        print(f"[sft/{a.setting}/{a.corpus}/{a.student}/{arm}] loss {tot / max(n, 1):.4f} n={len(exs)} "
              f"({time.time() - t0:.0f}s)", flush=True)
        model.save_pretrained(out_dir)
        del model, opt
        torch.cuda.empty_cache()


def cmd_student(a):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    domain = SETTINGS[a.setting][1]
    probs = problems(domain, "test", a.n_test, seed=2)
    tok = AutoTokenizer.from_pretrained(STUDENTS[a.student], padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    plain = [tok.apply_chat_template([{"role": "user", "content": user_prompt(domain, p["question"], None)}],
                                     tokenize=False, add_generation_prompt=True) for p in probs]
    todo = student_arms(a) + (["base"] if a.corpus == "raw" and a.arms == "all" else [])
    for arm in todo:
        name = "base" if arm == "base" else f"{a.corpus}_{arm}"
        fp = OUT / a.setting / f"out_{a.student}_{name}.jsonl"
        if read_jsonl(fp) is not None:
            continue
        adapter = OUT / a.setting / f"student_{a.student}_{name}"
        if arm != "base" and not (adapter / "adapter_config.json").exists():
            print(f"[student/{a.setting}/{a.student}/{name}] no adapter, skipped", flush=True)
            continue
        model = AutoModelForCausalLM.from_pretrained(STUDENTS[a.student], dtype=torch.bfloat16, device_map="cuda")
        if arm != "base":
            model = PeftModel.from_pretrained(model, str(adapter))
        model.eval()
        outs = gen(model, tok, plain, 400, seed=7)
        rows = [{**p, "arm": arm, "corpus": a.corpus, "text": o} for p, o in zip(probs, outs)]
        write_jsonl(fp, rows)
        acc = np.mean([correct(domain, r["text"], r["gold"]) for r in rows])
        print(f"[student/{a.setting}/{a.student}/{name}] acc {acc:.3f}; chars {np.mean([len(o) for o in outs]):.0f}",
              flush=True)
        del model
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("keys")
    p = sp.add_parser("teacher"); p.add_argument("--setting", required=True); p.add_argument("--n", type=int, default=300)
    p.add_argument("--max-arms", type=int)
    p = sp.add_parser("attack"); p.add_argument("--attack", required=True, choices=["filter", "para", "compress"])
    p.add_argument("--role", default="attacker", choices=["attacker", "owner"]); p.add_argument("--n-owner", type=int, default=100)
    p.add_argument("--max-arms", type=int)
    for c in ["sft", "student"]:
        p = sp.add_parser(c); p.add_argument("--setting", required=True); p.add_argument("--student", required=True)
        p.add_argument("--corpus", default="raw"); p.add_argument("--arms", default="all")
        if c == "sft":
            p.add_argument("--epochs", type=int, default=3)
        else:
            p.add_argument("--n-test", type=int, default=200)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "keys":
        print(json.dumps(keys(), indent=1)); return
    {"teacher": cmd_teacher, "attack": cmd_attack, "sft": cmd_sft, "student": cmd_student}[a.cmd](a)


if __name__ == "__main__":
    main()
