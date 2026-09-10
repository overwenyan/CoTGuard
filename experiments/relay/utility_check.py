"""队列项2: utility 不掉点主表 —— trigger 注入是否损害任务准确率.

这是竞品模板里的第一张主表(AgentMark/SeqWM/ActHook 都把"utility 不掉点"作为
主表, 检测性能单独呈现). 若 trigger 注入显著拉低答题正确率, 那么无论检测多准,
方法在实践中都不可用 —— 所以这张表是先决条件, 不是锦上添花.

数据来源: 已生成的 GSM8K 轨迹, `gold` 字段含标准答案(GSM8K 格式为 "...#### 18").
比较 clean 臂与 triggered 臂(以及各改写跳次)的最终答案正确率.

答案抽取采用多模式回退, 并**显式统计抽取失败率** —— 若失败率过高, 正确率数字
本身不可信, 必须报告而非静默忽略.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

# 按优先级排列的答案抽取模式
ANSWER_PATTERNS = [
    re.compile(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)"),              # GSM8K 官方格式
    re.compile(r"(?:final answer|答案)[^\d\-]{0,20}\$?(-?[\d,]+(?:\.\d+)?)", re.I),
    re.compile(r"\*\*\s*\$?(-?[\d,]+(?:\.\d+)?)\s*\*\*"),        # markdown 加粗
    re.compile(r"\\boxed\{\s*\$?(-?[\d,]+(?:\.\d+)?)\s*\}"),     # latex boxed
]


def extract_answer(text: str):
    """抽取最终数值答案. 返回 None 表示抽取失败(必须计入失败率, 不可当作答错)."""
    for pat in ANSWER_PATTERNS:
        m = pat.findall(text)
        if m:
            try:
                return float(m[-1].replace(",", ""))
            except ValueError:
                continue
    # 兜底: 取最后一个独立数字
    nums = re.findall(r"(?<![\d.])(-?\d[\d,]*(?:\.\d+)?)(?![\d.])", text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def gold_answer(gold: str):
    m = ANSWER_PATTERNS[0].findall(gold)
    if m:
        try:
            return float(m[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def score_file(fp: pathlib.Path):
    n = n_correct = n_extract_fail = n_gold_fail = 0
    for line in open(fp):
        if not line.strip():
            continue
        r = json.loads(line)
        g = gold_answer(r.get("gold", ""))
        if g is None:
            n_gold_fail += 1
            continue
        n += 1
        a = extract_answer(r.get("text", ""))
        if a is None:
            n_extract_fail += 1
            continue
        if abs(a - g) < 1e-4:
            n_correct += 1
    return {"n": n, "acc": n_correct / n if n else float("nan"),
            "extract_fail_rate": n_extract_fail / n if n else float("nan"),
            "gold_fail": n_gold_fail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="experiments/relay/runs")
    args = ap.parse_args()
    root = pathlib.Path(args.runs)

    targets = []
    # EXP-R1 (Tulu-3-8B 生成) 与 EXP-R2 v2 (Qwen3-14B 生成): clean vs triggered
    for run, label in [("attribution", "Tulu-3-8B"), ("attr_qwen3_v2", "Qwen3-14B")]:
        d = root / run
        if not d.exists():
            continue
        if (d / "clean.jsonl").exists():
            targets.append((label, "clean", "-", d / "clean.jsonl"))
        for f in sorted(d.glob("v2_diverse__key0*.jsonl"))[:3]:
            targets.append((label, "triggered", f.stem.split("__")[1], f))
    # EXP-R3 v2 (多跳改写): 各跳次的正确率是否被改写损害
    d = root / "relay_attr_v2"
    if d.exists():
        if (d / "hop0.jsonl").exists():
            targets.append(("relay", "hop0", "-", d / "hop0.jsonl"))
        for f in sorted(d.glob("*__hop*.jsonl")):
            style, hop = f.stem.split("__hop")
            targets.append(("relay", style, f"hop{hop}", f))

    print(f"{'来源':<12}{'臂/风格':<16}{'子项':<8}{'n':>5}{'正确率':>9}{'抽取失败率':>11}")
    print("-" * 64)
    rows = []
    for label, arm, sub, fp in targets:
        s = score_file(fp)
        flag = "  <- 抽取失败率过高, 正确率不可信" if s["extract_fail_rate"] > 0.15 else ""
        print(f"{label:<12}{arm:<16}{sub:<8}{s['n']:>5}{s['acc']:>9.3f}"
              f"{s['extract_fail_rate']:>11.3f}{flag}")
        rows.append({"source": label, "arm": arm, "sub": sub, **s})

    out = root / "utility_check.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
