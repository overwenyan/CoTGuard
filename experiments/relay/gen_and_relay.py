"""Stage 1+2: 生成带 trigger 的 CoT 轨迹, 并在多个 agent 间做 N 跳中继改写.

设计要点:
  - 每跳结束即落 checkpoint (JSONL), 作业中断可续跑
  - 生成方与中继方使用**不同模型家族**, 避免同族偏置
  - 中继方**不知道 trigger key**, 只被要求按某种风格改写
  - 同时产出 clean 对照臂 (无 trigger), 用于校准与 FPR 估计
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from trigger import make_trigger  # noqa: E402

REWRITE_STYLES = {
    "paraphrase": "Rewrite the following reasoning in your own words. Keep every step and the "
                  "final answer identical in meaning.",
    "simplify": "Rewrite the following reasoning more simply and concisely, preserving every "
                "step and the final answer.",
    "academic": "Rewrite the following reasoning in a formal academic register, preserving "
                "every step and the final answer.",
    "expand": "Rewrite the following reasoning with more explicit detail on each step, "
              "preserving the logic and the final answer.",
    # 定向规避: 攻击者知道检测靠风格特征, 主动回避 (来自 2510.19851 的压力分级思路)
    "guard_lexicon": "Rewrite the following reasoning. Preserve every step and the final answer, "
                     "but deliberately avoid any pedagogical or explanatory persona, avoid "
                     "stylistic flourishes, and use only plain neutral technical language.",
}


def build_chat(tok, system, user):
    """构造 chat prompt, 尝试关闭推理模型的 thinking 模式.

    背景 (EXP-R2/R3 事故复盘): Qwen3-14B 与 DeepSeek-R1-Qwen3-8B 都是推理模型,
    默认会先输出 <think>...</think> 元推理再给答案. 在 max_new_tokens=400 下
    该 think 块**多数无法闭合**(EXP-R3 hop1 截断率 80%), 导致"改写结果"实际是
    模型自己关于如何改写的元评论, 而非真正的改写内容. 这不仅让 EXP-R3 的多跳
    衰减曲线不可信, 也让 EXP-R2 里"Qwen3-14B 信号弱得多"的结论完全是假象.

    enable_thinking=False 对 Qwen3 系列有效(会预填空 <think></think>); 对
    DeepSeek-R1 系列该 kwarg 被静默接受但不生效(其模板不含相关条件分支), 因此
    不能只依赖这一层, 必须配合 strip_think() 做防御性清洗.
    """
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def strip_think(text: str) -> tuple[str, bool]:
    """去除 <think>...</think> 块. 返回 (清洗后文本, 是否发生了未闭合截断).

    未闭合 = 模型在 think 块内就耗尽了 max_new_tokens, 该条应被视为**生成失败**
    (没有产出任何实际内容), 而非"信号变弱". 调用方应据此重试或剔除, 不能静默使用。
    """
    if "<think>" not in text:
        return text.strip(), False
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip(), False
    return "", True  # 未闭合截断: 无可用内容


def batched_generate(model, tok, prompts, max_new_tokens, batch_size, temperature,
                     clean_think=True):
    import torch
    outs, n_truncated = [], 0
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                                 top_p=0.95, pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            raw = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            if clean_think:
                raw, trunc = strip_think(raw)
                n_truncated += int(trunc)
            outs.append(raw)
        print(f"    [gen] {min(i+batch_size, len(prompts))}/{len(prompts)}", flush=True)
    if clean_think and n_truncated:
        print(f"    [warn] {n_truncated}/{len(prompts)} 条因 think 块未闭合而清空"
              f"(建议提高 max_new_tokens)", flush=True)
    return outs


def load_model(path, dtype="bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=getattr(torch, dtype), device_map="auto")
    model.eval()
    return model, tok


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    if not path.exists():
        return None
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--generator", default="allenai/Llama-3.1-Tulu-3-8B")
    ap.add_argument("--relay-model", default="Qwen/Qwen3-14B")
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--max-hops", type=int, default=6)
    ap.add_argument("--styles", default="paraphrase,simplify,academic,guard_lexicon")
    ap.add_argument("--true-key", default="patient-teacher-2026")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    styles = [s for s in args.styles.split(",") if s in REWRITE_STYLES]
    tau = make_trigger(args.true_key)
    print(f"[cfg] tau = {tau!r}\n[cfg] styles = {styles}", flush=True)

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(args.n_problems, len(ds))))
    problems = [{"qid": i, "question": r["question"], "gold": r["answer"]}
                for i, r in enumerate(ds)]

    # ---------------- Stage 1: hop 0, 带 trigger 臂 + clean 对照臂 ----------------
    h0_path = out / "hop0.jsonl"
    rows0 = read_jsonl(h0_path)
    if rows0 is None:
        t0 = time.time()
        model, tok = load_model(args.generator)
        base_instr = "Solve the problem. Think step by step, one step per line."
        p_trig = [build_chat(tok, None, f"{base_instr} {tau}\n\nProblem: {p['question']}")
                  for p in problems]
        p_clean = [build_chat(tok, None, f"{base_instr}\n\nProblem: {p['question']}")
                   for p in problems]
        print(f"[stage1] 生成 triggered ({len(p_trig)}) ...", flush=True)
        o_trig = batched_generate(model, tok, p_trig, args.max_new_tokens,
                                  args.batch_size, args.temperature)
        print(f"[stage1] 生成 clean ({len(p_clean)}) ...", flush=True)
        o_clean = batched_generate(model, tok, p_clean, args.max_new_tokens,
                                   args.batch_size, args.temperature)
        rows0 = []
        for p, a, b in zip(problems, o_trig, o_clean):
            rows0.append({**p, "arm": "triggered", "hop": 0, "style": None, "text": a})
            rows0.append({**p, "arm": "clean", "hop": 0, "style": None, "text": b})
        write_jsonl(h0_path, rows0)
        print(f"[stage1] done in {time.time()-t0:.0f}s -> {h0_path}", flush=True)
        del model
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    else:
        print(f"[stage1] 复用 checkpoint {h0_path} ({len(rows0)} 行)", flush=True)

    # ---------------- Stage 2: N 跳中继改写 ----------------
    relay_model = relay_tok = None
    prev = rows0
    for hop in range(1, args.max_hops + 1):
        hp = out / f"hop{hop}.jsonl"
        cached = read_jsonl(hp)
        if cached is not None:
            print(f"[stage2] hop{hop} 复用 checkpoint ({len(cached)} 行)", flush=True)
            prev = cached
            continue
        if relay_model is None:
            print(f"[stage2] 载入中继模型 {args.relay_model}", flush=True)
            relay_model, relay_tok = load_model(args.relay_model)

        t0 = time.time()
        # hop1 从 hop0 分叉出各 style; 之后每个 style 各自沿自己的链继续
        src = [r for r in prev if hop > 1 or r["style"] is None]
        tasks, prompts = [], []
        for r in src:
            for st in (styles if hop == 1 else [r["style"]]):
                tasks.append({**r, "hop": hop, "style": st})
                prompts.append(build_chat(relay_tok, None,
                                          f"{REWRITE_STYLES[st]}\n\n{r['text']}"))
        print(f"[stage2] hop{hop}: {len(prompts)} 次改写 ...", flush=True)
        outs = batched_generate(relay_model, relay_tok, prompts, args.max_new_tokens,
                                args.batch_size, args.temperature)
        rows = [{**t, "text": o} for t, o in zip(tasks, outs)]
        write_jsonl(hp, rows)
        print(f"[stage2] hop{hop} done in {time.time()-t0:.0f}s -> {hp}", flush=True)
        prev = rows

    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
