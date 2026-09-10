"""EXP-M2 打分: 重复编码是否真的提升了"中段片段"的可归因性.

核心比较: 在**不含开头系统提示回响、不含结尾结论**的中段片段上, 单次注入
(D1 的 structural_anchor) vs 重复编码(repetition_structural) 谁的归因更高.

**对齐方式**: 两个 run 各自的 key_idx 编号由不同的 prefix 哈希产生, 不保证一致,
但两者都覆盖同一个 8 项 ANCHORS_STRUCTURAL 池(鸽笼原理: n=8 去重 8 项池必然覆盖全部)。
因此按 **anchor 原文** 对齐, 不按 key_idx 对齐, 否则会把不同的两个 anchor 错误地当成
同一个类别在比较.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from baselines import eval_fitted_tfidf, folds  # noqa: E402
from gen_and_relay import read_jsonl  # noqa: E402
from length_control import split_steps  # noqa: E402

TFIDF_KW = dict(analyzer="word", ngram_range=(1, 2), max_features=20000)


def anchor_text_of(pattern: str) -> str:
    """从 'As you reason, {anchor}.' 或 D1 的四种模板里抠出裸 anchor 短语,
    用作跨 run 对齐的键。"""
    for pre in ("As you reason, ", "Work through this, and ", "Approach this carefully and ",
               "Take care to "):
        if pattern.startswith(pre):
            return pattern[len(pre):].rstrip(".")
    return pattern.rstrip(".")


def load_space(run: pathlib.Path, space: str, K: int):
    entries = json.loads((run / "key_spaces.json").read_text())[space]
    anchor_of_idx = {i: anchor_text_of(e["pattern"]) for i, e in enumerate(entries)}
    texts, y, qid = [], [], []
    for ki in range(K):
        fp = run / f"{space}__key{ki:02d}.jsonl"
        recs = read_jsonl(fp) or []
        for r in recs:
            if r["text"].strip():
                texts.append(r["text"]); y.append(anchor_of_idx[ki]); qid.append(r["qid"])
    return texts, np.array(y, dtype=object), np.array(qid)


def mid_fragment(text: str, skip_head: int = 1, skip_tail: int = 1) -> str:
    """中段片段: 去掉开头 skip_head 步(可能回响系统提示)与结尾 skip_tail 步
    (通常含最终结论), 只留中间. 步数不足时返回空串(调用方应跳过)."""
    steps = split_steps(text)
    if len(steps) <= skip_head + skip_tail + 1:
        return ""
    return "\n".join(steps[skip_head:len(steps) - skip_tail])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-run", default="experiments/relay/runs/attr_families")
    ap.add_argument("--single-space", default="structural_anchor")
    ap.add_argument("--repeat-run", default="experiments/relay/runs/repetition_coded")
    ap.add_argument("--repeat-space", default="repetition_structural")
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()

    single = pathlib.Path(args.single_run)
    repeat = pathlib.Path(args.repeat_run)
    results = {}

    for name, run, space in [("single", single, args.single_space),
                             ("repetition", repeat, args.repeat_space)]:
        texts, y, qid = load_space(run, space, K=8)
        anchors = sorted(set(y.tolist()))
        print(f"[{name}] {len(texts)} 条, {len(anchors)} 个不同 anchor")
        if len(anchors) != 8:
            print(f"  [warn] 期望 8 个不同 anchor, 实得 {len(anchors)} —— "
                  f"对齐可能不完整, 结果需谨慎解读")

        # 用整数标签喂分类器, 但标签空间以 anchor 原文固定, 保证跨 run 语义一致
        lut = {a: i for i, a in enumerate(anchors)}
        yi = np.array([lut[a] for a in y])
        fl = folds(qid, args.n_seeds)

        full_acc, _ = eval_fitted_tfidf(texts, yi, fl, TFIDF_KW)
        mids = [mid_fragment(t) for t in texts]
        keep = [i for i, m in enumerate(mids) if m.strip()]
        drop_rate = 1 - len(keep) / max(len(mids), 1)
        mids_k = [mids[i] for i in keep]
        yi_k = yi[np.array(keep)]
        qid_k = qid[np.array(keep)]
        fl_k = folds(qid_k, args.n_seeds)
        mid_acc, _ = eval_fitted_tfidf(mids_k, yi_k, fl_k, TFIDF_KW)

        chance = 1.0 / len(anchors)
        fm = float(np.mean(full_acc)) if full_acc else float("nan")
        mm = float(np.mean(mid_acc)) if mid_acc else float("nan")
        print(f"  full  top-1 = {fm:.4f} ({fm/chance:.2f}x)")
        print(f"  mid   top-1 = {mm:.4f} ({mm/chance:.2f}x)  (丢弃过短轨迹 {drop_rate:.1%})")
        results[name] = {"mid": mid_acc, "full": full_acc}

    print("-" * 60)
    sm, rm = results["single"]["mid"], results["repetition"]["mid"]
    if sm and rm and len(sm) == len(rm):
        from scipy import stats
        d = np.array(rm) - np.array(sm)
        t, p = stats.ttest_rel(rm, sm)
        print(f"[配对检验] mid: repetition - single = {d.mean():+.4f} (t={t:.2f}, p={p:.4f})")
        if d.mean() > 0 and p < 0.05:
            print("=> 重复编码在中段片段上显著更可归因: M2 的设计动机成立.")
        elif d.mean() <= 0:
            print("=> **重复编码未提升中段可归因性(或更差)**: M2 的设计动机不成立, "
                  "'更多次提醒总是更好'的直觉被推翻, 应如实报告.")
        else:
            print(f"=> 方向正确但不显著(p={p:.4f}): 证据不足以支持 M2 的设计主张.")
    else:
        print(f"[warn] 无法配对检验: single 有效折数={len(sm) if sm else 0}, "
              f"repetition 有效折数={len(rm) if rm else 0} —— 可能因某折类别不足被跳过.")


if __name__ == "__main__":
    main()
