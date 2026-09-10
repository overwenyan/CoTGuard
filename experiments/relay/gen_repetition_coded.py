"""EXP-M2: 重复编码的结构化锚点 —— 把 N2 的前缀弱残留从副作用变成设计目标.

EXP-N2 发现: 去掉密钥后, 已生成前缀仍携带 1.6-1.9x 随机的弱可归因残留(密钥持续条件化
是主导, 但前缀确实有独立贡献). 若在轨迹的**多个位置独立重复**放置同一结构约束(而非只在
系统提示里放一次), 应当能把这个"副作用级"的残留放大成"设计级"的抗碎片化能力 —— 即使
轨迹被截断成只剩某个中间片段, 该片段附近的重复提醒也能维持可归因信号.

## 与单次注入(D1 的 structural_anchor)的区别

D1: anchor 只在开头系统提示出现一次, 后续生成完全靠模型自己"记得".
M2: 每生成约 `chunk_steps` 步就在**文本流中**重新插入一次同一 anchor 的提醒
    (不是重新起一轮对话, 而是把提醒接在已生成文本后面, 整体重新过一遍模型继续生成
    —— 这样提醒真实地进入了后续生成所依赖的 context, 而不是仅停留在系统提示里).

## 可证伪预测(先写下)

  若重复编码确实有效: 从**轨迹中段任取一个片段**(不含开头系统提示、不含结尾结论)
  单独测归因, repetition 条件应显著高于 single 条件(D1 的 structural_anchor 数据).
  若两者在"中段片段"上的归因无显著差异, 重复编码没有带来额外的抗碎片化能力,
  M2 的设计动机不成立 —— 不能靠"更多次提醒总是更好"这种直觉背书, 必须实测.

## 代价

多次插入提醒会打断生成的自然流畅性, 可能被动过手脚的痕迹更明显(隐蔽性代价),
且更耗 token. 这些代价必须与抗碎片化能力一起报告, 不能只报收益.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gen_and_relay import build_chat, load_model, read_jsonl, strip_think, write_jsonl  # noqa: E402
from length_control import split_steps  # noqa: E402
from trigger_v2 import ANCHORS_STRUCTURAL, _idx  # noqa: E402

BASE = "Solve the problem. Think step by step, one step per line."
ANCHOR_TEMPLATE = "As you reason, {anchor}."


def key_pool_structural(n: int, prefix: str = "rep"):
    seen, keys, i = set(), [], 0
    while len(keys) < n and i < 200 * n + 5000:
        k = f"{prefix}-{i}"
        a = _idx(k, "anchor", len(ANCHORS_STRUCTURAL))
        if a not in seen:
            seen.add(a); keys.append(k)
        i += 1
    return keys


def chunked_generate_batch(model, tok, questions, anchor, n_chunks, chunk_new_tokens,
                           batch_size, temperature):
    """对一批题目做分块生成, 每块之间重新插入 anchor 提醒.

    实现: 每一块都是一次完整的 model.generate 调用, 输入 = 固定的原始 prompt(`base`)
    + 目前为止**累积**的生成文本(`generated`) + (若非首块)提醒文本. 提醒真实进入
    下一块生成所依赖的 token 序列, 不是摆设.

    **注意(曾经的 bug)**: 不可把"上一轮实际喂给模型的完整输入"当作下一轮的起点再往上叠
    `generated` —— 那样等于把已经算进上一轮输入里的历史生成内容, 又通过 `generated`
    重新加了一遍, 链条越长重复堆叠越严重. 必须让 `base` 保持不变, 每轮都从
    `base + generated(累积) + reminder` **重新拼**, 而不是滚动累加 `running` 本身.
    """
    import torch
    base = [build_chat(tok, None, f"{BASE} {ANCHOR_TEMPLATE.format(anchor=anchor)}"
                       f"\n\nProblem: {q}") for q in questions]
    generated = ["" for _ in questions]
    done = [False] * len(questions)

    for c in range(n_chunks):
        reminder = f"\n[Reminder: {anchor}]\n" if c > 0 else ""
        running = [b + g + reminder for b, g in zip(base, generated)]

        for i in range(0, len(running), batch_size):
            batch = running[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=3072).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=chunk_new_tokens,
                                     do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                                     top_p=0.95, pad_token_id=tok.pad_token_id)
            for j in range(len(batch)):
                idx = i + j
                if done[idx]:
                    continue
                piece = tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
                piece, trunc = strip_think(piece)
                generated[idx] = generated[idx] + piece
                # 简单停止判据: 出现明显结论措辞或步数已经足够, 避免无限膨胀
                if len(split_steps(generated[idx])) >= 6 or "answer is" in piece.lower():
                    done[idx] = True
        print(f"    [chunk {c+1}/{n_chunks}] {sum(done)}/{len(questions)} 已判定结束", flush=True)
        if all(done):
            break
    return generated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--generator", default="allenai/Llama-3.1-Tulu-3-8B")
    ap.add_argument("--n-problems", type=int, default=80)
    ap.add_argument("--n-keys", type=int, default=8)
    ap.add_argument("--n-chunks", type=int, default=3)
    ap.add_argument("--chunk-new-tokens", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    keys = key_pool_structural(args.n_keys)
    anchors = [ANCHORS_STRUCTURAL[_idx(k, "anchor", len(ANCHORS_STRUCTURAL))] for k in keys]
    (out / "key_spaces.json").write_text(json.dumps(
        {"repetition_structural": [{"key": k, "pattern": ANCHOR_TEMPLATE.format(anchor=a)}
                                    for k, a in zip(keys, anchors)]}, indent=2))
    for k, a in zip(keys, anchors):
        print(f"[key] {k}: {a}", flush=True)

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(args.n_problems, len(ds))))
    problems = [{"qid": i, "question": r["question"], "gold": r["answer"]}
                for i, r in enumerate(ds)]

    model, tok = load_model(args.generator)

    for ki, (key, anchor) in enumerate(zip(keys, anchors)):
        fp = out / f"repetition_structural__key{ki:02d}.jsonl"
        if read_jsonl(fp) is not None:
            print(f"[key{ki:02d}] 复用 checkpoint", flush=True); continue
        t0 = time.time()
        qs = [p["question"] for p in problems]
        texts = chunked_generate_batch(model, tok, qs, anchor, args.n_chunks,
                                       args.chunk_new_tokens, args.batch_size, args.temperature)
        write_jsonl(fp, [{**p, "space": "repetition_structural", "key": key, "key_idx": ki,
                         "pattern": ANCHOR_TEMPLATE.format(anchor=anchor), "text": t}
                        for p, t in zip(problems, texts)])
        print(f"[key{ki:02d}] done in {time.time()-t0:.0f}s -> {fp}", flush=True)

    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
