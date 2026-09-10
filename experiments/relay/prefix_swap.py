"""EXP-N2: 指纹由密钥维持, 还是由已生成前缀自维持?

EXP-C4 判定可归因信号的载体是**被诱导的行为**(指令合规), 而非表层文体. 随之而来的
直接问题: 这个行为在生成过程中靠什么维持?

  (a) **持续条件化** —— 密钥一直在 context 里, 每一步都在被它约束
  (b) **前缀自维持** —— 模型看到自己前几步的写法就自动延续, 密钥移除后照样继续

两者对溯源的含义完全不同: 若是 (b), 则指纹会随轨迹被截断/摘录而**继续传播**;
若是 (a), 则去掉密钥的重新生成足以清除.

## 设计

从 anchor_only 的 hop0 轨迹里切出前 m 步作为前缀(无需重新生成), 然后在三种条件下续写:

  matched   : key_i + prefix_i    —— 密钥与前缀一致(对照)
  swapped   : key_i + prefix_j    —— 密钥与前缀**冲突**, 看续写跟谁
  nokey     : (无密钥) + prefix_j —— 只剩前缀, 看是否仍可归因到 j

**切换密钥时重新拼 prompt 并整体重算**, 不复用任何 KV cache —— 仅 mask 掉 prompt
而保留其缓存影响不是干净干预.

## 打分

读出器训练在**原始 hop0 轨迹的尾部**(第 m 步之后), 按题目划分; 测试在续写上.
续写**单独打分, 不含前缀** —— 否则前缀自带的指纹平凡泄漏答案.

对 swapped 与 nokey 各报两个准确率:
  acc_key    : 续写被归到**密钥**身份的比例
  acc_prefix : 续写被归到**前缀来源**身份的比例

## 可证伪预测(先写下)

  若为持续条件化: swapped 的 acc_key >> acc_prefix; nokey 归因塌到随机
  若为前缀自维持: swapped 的 acc_prefix >= acc_key; nokey 仍显著高于随机
  若两者皆有: swapped 双高, nokey 中等 —— 需报告相对量级, 不可单边下结论
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gen_and_relay import batched_generate, build_chat, load_model, read_jsonl, write_jsonl  # noqa: E402
from length_control import split_steps  # noqa: E402

BASE = "Solve the problem. Think step by step, one step per line."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="experiments/relay/runs/attr_channels")
    ap.add_argument("--space", default="anchor_only")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--generator", default="allenai/Llama-3.1-Tulu-3-8B")
    ap.add_argument("--prefix-steps", type=int, default=3)
    ap.add_argument("--n-problems", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = pathlib.Path(args.src_dir)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    entries = json.loads((src / "key_spaces.json").read_text())[args.space]
    K = len(entries)
    pats = [e["pattern"] for e in entries]

    # --- 载入源轨迹, 切前缀 ---
    import numpy as np
    rng = np.random.default_rng(args.seed)
    # swap 配对: 每个密钥 i 配一个 j != i (固定错排, 保证每个前缀来源被用一次)
    perm = list(range(K))
    while any(perm[i] == i for i in range(K)):
        perm = rng.permutation(K).tolist()
    print(f"[cfg] swap 配对 key_i -> prefix_{{perm[i]}}: {perm}", flush=True)

    src_rows = {}
    for ki in range(K):
        rows = read_jsonl(src / f"{args.space}__key{ki:02d}.jsonl") or []
        src_rows[ki] = {r["qid"]: r for r in rows}

    qids = sorted(set.intersection(*[set(v) for v in src_rows.values()]))[: args.n_problems]
    print(f"[data] {len(qids)} 道题在全部 {K} 个密钥下均有轨迹", flush=True)

    def prefix_of(ki, qid):
        ss = split_steps(src_rows[ki][qid]["text"])
        return "\n".join(ss[: args.prefix_steps]).strip() if len(ss) > args.prefix_steps else None

    model, tok = load_model(args.generator)

    # 条件: (名称, 提供密钥的 key_idx 或 None, 提供前缀的 key_idx)
    conds = [("matched", lambda i: i, lambda i: i),
             ("swapped", lambda i: i, lambda i: perm[i]),
             ("nokey", lambda i: None, lambda i: perm[i])]

    for cname, key_fn, pre_fn in conds:
        fp = out / f"{cname}.jsonl"
        if read_jsonl(fp) is not None:
            print(f"[{cname}] 复用 checkpoint", flush=True); continue
        t0 = time.time()
        tasks, prompts = [], []
        for ki in range(K):
            k_key, k_pre = key_fn(ki), pre_fn(ki)
            for qid in qids:
                pre = prefix_of(k_pre, qid)
                if pre is None:
                    continue
                q = src_rows[k_pre][qid]["question"]
                instr = BASE if k_key is None else f"{BASE} {pats[k_key]}"
                # 关键: 重新拼完整 prompt, 再把前缀接在 assistant 起始处 ->
                # 整段重算, 不复用任何 KV cache
                prompts.append(build_chat(tok, None, f"{instr}\n\nProblem: {q}") + pre + "\n")
                tasks.append({"cond": cname, "qid": qid,
                              "key_idx": k_key, "prefix_idx": k_pre,
                              "prefix_text": pre})
        print(f"[{cname}] {len(prompts)} 条续写 ...", flush=True)
        outs = batched_generate(model, tok, prompts, args.max_new_tokens,
                                args.batch_size, args.temperature)
        # 只保留续写部分(生成结果本就不含前缀, 因为前缀在输入侧)
        write_jsonl(fp, [{**t, "text": o} for t, o in zip(tasks, outs)])
        print(f"[{cname}] done in {time.time()-t0:.0f}s -> {fp}", flush=True)

    # --- 训练集: 原始 hop0 轨迹的**尾部**(第 m 步之后) ---
    tp = out / "train_tails.jsonl"
    if read_jsonl(tp) is None:
        rows = []
        for ki in range(K):
            for qid, r in src_rows[ki].items():
                ss = split_steps(r["text"])
                if len(ss) > args.prefix_steps:
                    rows.append({"key_idx": ki, "qid": qid,
                                 "text": "\n".join(ss[args.prefix_steps:]).strip()})
        write_jsonl(tp, rows)
        print(f"[train] {len(rows)} 条尾部 -> {tp}", flush=True)

    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
