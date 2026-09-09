"""EXP-R3: per-key 签名能否撑过多跳语义改写.

这是决定论文前提能否成立的实验. 已确立:
  - 密钥身份可从轨迹恢复(学习式读出 top-1 0.475, 3.8x 随机)
  - 但该信号不能泛化到未见密钥, 像是**每密钥特异签名**而非通用语义映射

特异签名很可能脆弱. 若撑不过一次改写, 则"多智能体中继下的来源检测"这一命题
不成立, 整篇论文的前提没了.

做法: 取 EXP-R1 已生成的 K 密钥轨迹, 逐跳改写(中继方**不知道密钥**),
每跳后重测密钥可恢复性. 两种改写风格:
  paraphrase     良性中继
  guard_lexicon  对抗性定向规避(攻击者主动回避风格特征)

每 (style, hop) 落一个 checkpoint.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gen_and_relay import (  # noqa: E402
    REWRITE_STYLES, batched_generate, build_chat, load_model, read_jsonl, write_jsonl,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="experiments/relay/runs/attribution")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--space", default="v2_diverse")
    ap.add_argument("--relay-model", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    ap.add_argument("--styles", default="paraphrase,guard_lexicon")
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--n-problems", type=int, default=60,
                    help="每密钥取多少条(控制总生成量)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    src = pathlib.Path(args.src_dir)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    # 密钥空间定义随源目录一起带过来, 供打分脚本使用
    (out / "key_spaces.json").write_text((src / "key_spaces.json").read_text())

    styles = [s for s in args.styles.split(",") if s in REWRITE_STYLES]
    spaces = json.loads((src / "key_spaces.json").read_text())
    K = len(spaces[args.space])

    # hop0 = 原始生成(从 EXP-R1 复制, 截断到 n_problems)
    hop0 = []
    for ki in range(K):
        fp = src / f"{args.space}__key{ki:02d}.jsonl"
        if not fp.exists():
            continue
        recs = [json.loads(l) for l in open(fp) if l.strip()][: args.n_problems]
        for r in recs:
            hop0.append({**r, "hop": 0, "style": None, "key_idx": ki})
    write_jsonl(out / "hop0.jsonl", hop0)
    print(f"[hop0] {len(hop0)} 条 (K={K}, 每密钥 {args.n_problems})", flush=True)

    model = tok = None
    for style in styles:
        prev = hop0
        for hop in range(1, args.max_hops + 1):
            fp = out / f"{style}__hop{hop}.jsonl"
            cached = read_jsonl(fp)
            if cached is not None:
                print(f"[{style}/hop{hop}] 复用 checkpoint ({len(cached)})", flush=True)
                prev = cached
                continue
            if model is None:
                print(f"[load] {args.relay_model}", flush=True)
                model, tok = load_model(args.relay_model)
            t0 = time.time()
            prompts = [build_chat(tok, None, f"{REWRITE_STYLES[style]}\n\n{r['text']}")
                       for r in prev]
            print(f"[{style}/hop{hop}] {len(prompts)} 次改写 ...", flush=True)
            outs = batched_generate(model, tok, prompts, args.max_new_tokens,
                                    args.batch_size, args.temperature)
            rows = [{**r, "hop": hop, "style": style, "text": o}
                    for r, o in zip(prev, outs)]
            write_jsonl(fp, rows)
            print(f"[{style}/hop{hop}] done in {time.time()-t0:.0f}s", flush=True)
            prev = rows

    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
