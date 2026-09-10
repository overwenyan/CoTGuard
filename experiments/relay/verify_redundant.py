"""EXP-M1: 可验证冗余计算证书 —— 第一步是验证机制假设, 不是急着上安全原语.

structural anchor 池里有一条 "compute every intermediate quantity twice, by two
different routes"(key01). 中心方法主张要求这条 anchor **真的**让模型产出可核验的
冗余结构, 而不只是嘴上说说. 在把它变成"证书"之前, 必须先测:

  该 anchor 是否比**其他不要求冗余计算的 structural anchor**、以及 clean 轨迹,
  产出更高比例的"同一结论被独立复述且数值一致"的结构?

若不比对照组高, "可验证冗余计算证书"这个方法主张从机制上就不成立, 不必往下做核验器
的安全性分析.

## 检测方法(第一版, 局限已写明)

用正则抽取轨迹里所有"结论性数值断言"(形如 "= NUM"、"is NUM"、"total: NUM"、
"answer: NUM" 等), 在**步骤位置上充分分离**(间隔 >= min_gap 步)的两条断言若数值
一致, 记为一次"冗余一致"; 数值不一致则记为一次"冗余冲突"(内部矛盾, 对核验同样有用
—— 说明该机制不仅能确认, 也能拒绝).

**局限(必须写进论文, 不得隐藏)**: 正则抽取会漏掉用文字而非阿拉伯数字表达的结论,
也可能把"引用早先数字"误判为"独立复述"。这是一个**下界**估计, 不是精确抽取器。
更精确的版本需要 LLM-as-judge 或依赖解析, 留作后续.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gen_and_relay import read_jsonl  # noqa: E402
from length_control import split_steps  # noqa: E402

# 两档正则, 用途不同, 不可混用(见下方 bug 记录):
#   GENERIC   : "= NUM"/"is NUM" 等, 覆盖任意中间量赋值 —— 用于"冗余一致"检测
#               (两条独立路径算出同一个数, 不要求这个数是"最终答案")
#   CONCLUDING: 仅 "total.../answer..." 这类明确宣称"这是结论"的措辞 —— 用于
#               "结尾冲突"检测. 不能用 GENERIC 做这件事: 早期版本曾把"最后 3 步内的
#               任意数值"当结论候选, 结果把一个正常中间量(如"还剩 9 个鸡蛋")误判为
#               与真正结论("总共 18 美元")矛盾 —— 短轨迹里这个假阳性率会很高.
NUM = r"-?\d[\d,]*(?:\.\d+)?"
GENERIC_PATTERNS = [
    re.compile(rf"=\s*\$?({NUM})\b"),
    re.compile(rf"\bis\s+\$?({NUM})\b", re.I),
]
CONCLUDING_PATTERNS = [
    re.compile(rf"\btotal(?:\s+is|\s*:)?\s+\$?({NUM})\b", re.I),
    re.compile(rf"\b(?:final\s+)?answer(?:\s+is|\s*:)?\s+\$?({NUM})\b", re.I),
    re.compile(rf"\btherefore,?\s+.{{0,40}}?\$?({NUM})\b", re.I),
]


def norm_num(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def extract(text: str, patterns) -> list[tuple[int, float]]:
    """返回 [(step_idx, value), ...]，按出现的步骤位置."""
    steps = split_steps(text) or [text]
    out = []
    for i, s in enumerate(steps):
        seen_in_step = set()
        for pat in patterns:
            for m in pat.finditer(s):
                v = norm_num(m.group(1))
                if v is not None and v not in seen_in_step:
                    out.append((i, v))
                    seen_in_step.add(v)
    return out


def analyze_trace(text: str, min_gap: int = 1) -> dict:
    generic = extract(text, GENERIC_PATTERNS)
    agree = 0
    for i in range(len(generic)):
        for j in range(i + 1, len(generic)):
            si, vi = generic[i]
            sj, vj = generic[j]
            if sj - si >= min_gap and abs(vi - vj) < 1e-6:
                agree += 1

    # 结尾冲突: 只看**明确措辞的结论性断言**(concluding), 不看任意数值 —— 否则会把
    # 正常的中间量误判为与真结论矛盾(已在合成案例中复现并修复, 见上方注释).
    concl = extract(text, CONCLUDING_PATTERNS)
    concl_vals = {v for _, v in concl}
    conflict = 1 if len(concl_vals) > 1 else 0

    return {"n_generic": len(generic), "has_redundant_agree": int(agree > 0),
            "n_agree_pairs": agree, "n_concluding": len(concl),
            "has_tail_conflict": conflict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="experiments/relay/runs/attr_families")
    ap.add_argument("--min-gap", type=int, default=1)
    args = ap.parse_args()

    run = pathlib.Path(args.run_dir)
    spaces = json.loads((run / "key_spaces.json").read_text())

    rows = []
    for space_name, entries in spaces.items():
        for ki, e in enumerate(entries):
            fp = run / f"{space_name}__key{ki:02d}.jsonl"
            recs = read_jsonl(fp)
            if not recs:
                continue
            stats = [analyze_trace(r["text"], args.min_gap) for r in recs
                     if r["text"].strip()]
            if not stats:
                continue
            rows.append({
                "space": space_name, "key_idx": ki,
                "pattern": e["pattern"][:70],
                "is_redundant_anchor": "compute every intermediate quantity twice" in e["pattern"],
                "n_traces": len(stats),
                "agree_rate": float(np.mean([s["has_redundant_agree"] for s in stats])),
                "mean_agree_pairs": float(np.mean([s["n_agree_pairs"] for s in stats])),
                "tail_conflict_rate": float(np.mean([s["has_tail_conflict"] for s in stats])),
            })

    # clean 对照
    clean = read_jsonl(run / "clean.jsonl")
    if clean:
        stats = [analyze_trace(r["text"], args.min_gap) for r in clean if r["text"].strip()]
        rows.append({"space": "clean", "key_idx": None, "pattern": "(no anchor)",
                     "is_redundant_anchor": False, "n_traces": len(stats),
                     "agree_rate": float(np.mean([s["has_redundant_agree"] for s in stats])),
                     "mean_agree_pairs": float(np.mean([s["n_agree_pairs"] for s in stats])),
                     "tail_conflict_rate": float(np.mean([s["has_tail_conflict"] for s in stats]))})

    print(f"{'space':<20}{'key':>4}{'redund?':>9}{'n':>5}{'agree_rate':>12}{'mean_pairs':>12}{'conflict':>10}  pattern")
    print("-" * 100)
    for r in sorted(rows, key=lambda r: (r["space"], r["key_idx"] if r["key_idx"] is not None else -1)):
        ki = r["key_idx"] if r["key_idx"] is not None else "-"
        print(f"{r['space']:<20}{ki!s:>4}{str(r['is_redundant_anchor']):>9}{r['n_traces']:>5}"
              f"{r['agree_rate']:>12.3f}{r['mean_agree_pairs']:>12.3f}{r['tail_conflict_rate']:>10.3f}"
              f"  {r['pattern']}")

    # 判决: 目标 anchor 的 agree_rate 是否高于同族其他 anchor 与 clean
    target = [r for r in rows if r["is_redundant_anchor"]]
    others = [r for r in rows if not r["is_redundant_anchor"] and r["space"] != "clean"]
    cleans = [r for r in rows if r["space"] == "clean"]
    print("-" * 100)
    if target and others:
        t_rate = np.mean([r["agree_rate"] for r in target])
        o_rate = np.mean([r["agree_rate"] for r in others])
        c_rate = np.mean([r["agree_rate"] for r in cleans]) if cleans else float("nan")
        print(f"目标 anchor(compute twice) agree_rate = {t_rate:.3f}")
        print(f"其余 anchor 平均 agree_rate            = {o_rate:.3f}")
        print(f"clean(无 anchor) agree_rate            = {c_rate:.3f}")
        if t_rate > o_rate * 1.3 and t_rate > c_rate * 1.3:
            print("=> 机制假设成立: 该 anchor 确实诱导出更多可核验的冗余一致结构, "
                  "可以继续做证书/核验器的安全性分析.")
        else:
            print("=> **机制假设不成立或证据不足**: 该 anchor 未显著提高冗余一致结构比例. "
                  "在加强抽取方法或改用更强 anchor 措辞前, 不应继续声称'可验证冗余计算证书'.")
    else:
        print("[warn] 缺少目标 anchor 或对照组数据, 无法判定.")

    out = run / "verify_redundant.json"
    out.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2))
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
