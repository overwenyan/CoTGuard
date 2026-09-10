"""EXP-R1: 密钥空间可分离性 -> attribution 能力.

EXP-R0b 发现语义 trigger 支持 detection (AUROC 0.99) 但几乎不支持 attribution
(AUROC 0.68, 真密钥排名劣于随机). 机制是密钥空间拥挤(真vs错相似度 mean 0.66/max 0.97).

本实验对三种密钥空间各生成 K 个密钥的 triggered 轨迹, 测 attribution 是否随
可分离性恢复, 并给出**容量—可分离性曲线**:
  v1_crowded  共用模板仅换槽位词        (pairwise max 0.968)
  v2_diverse  模板+领域分散              (max 0.911)
  v2_greedy   再加贪心最远点选择          (max 0.597)

每跳落 checkpoint. 生成方与 EXP-R0 一致以便对照.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gen_and_relay import batched_generate, build_chat, load_model, read_jsonl, write_jsonl  # noqa: E402
from trigger import make_trigger  # noqa: E402
from trigger_v2 import key_pool, make_trigger_v2, select_separable_keys  # noqa: E402


def build_key_spaces(n_keys, encoder):
    """返回 {space_name: [(key, pattern), ...]}."""
    spaces = {}
    # v1: 与 EXP-R0 相同的拥挤空间
    from trigger import wrong_keys
    v1_pats = [make_trigger("patient-teacher-2026")] + wrong_keys("patient-teacher-2026", n_keys - 1)
    spaces["v1_crowded"] = [(f"v1-{i}", p) for i, p in enumerate(v1_pats)]
    # v2: 模板+领域分散
    ks = key_pool(n_keys, prefix="v2d")
    spaces["v2_diverse"] = [(k, make_trigger_v2(k)) for k in ks]
    # v2 + 贪心最远点
    gk, gp = select_separable_keys(n_keys, encoder, prefix="v2g")
    spaces["v2_greedy"] = list(zip(gk, gp))
    return spaces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--generator", default="allenai/Llama-3.1-Tulu-3-8B")
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--n-keys", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--embed-model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"])
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(args.embed_model)
    spaces = build_key_spaces(args.n_keys, enc)
    (out / "key_spaces.json").write_text(json.dumps(
        {k: [{"key": a, "pattern": b} for a, b in v] for k, v in spaces.items()},
        ensure_ascii=False, indent=2))
    for name, kp in spaces.items():
        print(f"[space] {name}: {len(kp)} keys, 例: {kp[0][1]!r}", flush=True)
    del enc  # 生成阶段不需要, 腾出显存

    if args.dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ds = ds.select(range(min(args.n_problems, len(ds))))
        problems = [{"qid": i, "question": r["question"], "gold": r["answer"]}
                    for i, r in enumerate(ds)]
    elif args.dataset == "math500":
        # MATH-500: answer 字段是 LaTeX 表达式(如 \left( 3, \frac{\pi}{2} \right)),
        # 不能用 GSM8K 的数值匹配做正确率评估 —— utility 检查需 LaTeX 感知匹配,
        # 本实验只用其轨迹做检测/归因, 正确率评估另行处理并标注局限.
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        ds = ds.select(range(min(args.n_problems, len(ds))))
        problems = [{"qid": i, "question": r["problem"], "gold": r["answer"],
                     "subject": r["subject"], "level": r["level"]}
                    for i, r in enumerate(ds)]
    else:
        raise ValueError(f"未知数据集: {args.dataset}")
    print(f"[data] {args.dataset}: {len(problems)} 题", flush=True)

    base = "Solve the problem. Think step by step, one step per line."
    model, tok = load_model(args.generator)

    # --- clean 对照臂 ---
    cp = out / "clean.jsonl"
    if read_jsonl(cp) is None:
        t0 = time.time()
        prompts = [build_chat(tok, None, f"{base}\n\nProblem: {p['question']}") for p in problems]
        print(f"[clean] {len(prompts)} 条 ...", flush=True)
        outs = batched_generate(model, tok, prompts, args.max_new_tokens,
                                args.batch_size, args.temperature)
        write_jsonl(cp, [{**p, "space": None, "key": None, "pattern": None, "text": o}
                         for p, o in zip(problems, outs)])
        print(f"[clean] done in {time.time()-t0:.0f}s", flush=True)
    else:
        print("[clean] 复用 checkpoint", flush=True)

    # --- 每个空间的每个密钥各生成一批 ---
    for space, kp in spaces.items():
        for ki, (key, pat) in enumerate(kp):
            fp = out / f"{space}__key{ki:02d}.jsonl"
            if read_jsonl(fp) is not None:
                print(f"[{space}/key{ki:02d}] 复用 checkpoint", flush=True)
                continue
            t0 = time.time()
            prompts = [build_chat(tok, None, f"{base} {pat}\n\nProblem: {p['question']}")
                       for p in problems]
            print(f"[{space}/key{ki:02d}] {len(prompts)} 条 | {pat[:70]}...", flush=True)
            outs = batched_generate(model, tok, prompts, args.max_new_tokens,
                                    args.batch_size, args.temperature)
            write_jsonl(fp, [{**p, "space": space, "key": key, "key_idx": ki,
                              "pattern": pat, "text": o} for p, o in zip(problems, outs)])
            print(f"[{space}/key{ki:02d}] done in {time.time()-t0:.0f}s", flush=True)

    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
