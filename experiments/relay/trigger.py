"""Trigger 生成 T(k, t): 由密钥与任务类型确定性地映射到 trigger pattern.

关键性质 (wrong-key 校准所依赖):
  同一 key 空间内的任意 k' 都应产生**同类型、同风格分布**的 pattern,
  使得真密钥与错误密钥在同一条轨迹上产生的分数基线一致.
  该性质是 EXP-003 建模假设的核心, 必须在 pilot 中实证验证.
"""

from __future__ import annotations

import hashlib

# 每个模板槽位的候选词, 保证不同 key 产生的 pattern 结构相同、仅内容不同,
# 从而使 wrong-key 分数与真 key 分数同分布(在无信号轨迹上).
PERSONA = ["patient teacher", "cautious analyst", "meticulous editor", "curious student",
           "seasoned engineer", "careful reviewer", "methodical planner", "thoughtful mentor",
           "diligent researcher", "attentive tutor", "rigorous examiner", "practical craftsman"]
MANNER = ["step by step", "in plain language", "with explicit checks", "by naming each quantity",
          "while restating the goal", "with units tracked", "by verifying each claim",
          "with intermediate summaries", "by isolating assumptions", "with clear transitions"]
FOCUS = ["fair sharing", "hidden assumptions", "the order of operations", "the invariant quantity",
         "the unknown value", "the given constraints", "the intermediate result",
         "the reasoning chain", "the key relation", "the final check"]

TASK_FRAME = {
    "arithmetic": "As you solve the problem, explain like a {persona} would, reasoning {manner}, "
                  "and draw attention to {focus}.",
    "logic": "As you work through the inference, argue like a {persona}, proceeding {manner}, "
             "and keep {focus} in view.",
}


def _idx(key: str, salt: str, n: int) -> int:
    h = hashlib.sha256(f"{key}|{salt}".encode()).hexdigest()
    return int(h, 16) % n


def make_trigger(key: str, task_type: str = "arithmetic") -> str:
    """T(k, t) -> tau. 确定性, 同一 (k,t) 恒定."""
    frame = TASK_FRAME.get(task_type, TASK_FRAME["arithmetic"])
    return frame.format(
        persona=PERSONA[_idx(key, "persona", len(PERSONA))],
        manner=MANNER[_idx(key, "manner", len(MANNER))],
        focus=FOCUS[_idx(key, "focus", len(FOCUS))],
    )


def wrong_keys(true_key: str, n: int, task_type: str = "arithmetic") -> list[str]:
    """生成 n 个错误密钥的 trigger pattern, 排除与真 key 相同的 pattern."""
    true_tau = make_trigger(true_key, task_type)
    out, i = [], 0
    while len(out) < n:
        tau = make_trigger(f"wk-{i}", task_type)
        if tau != true_tau:
            out.append(tau)
        i += 1
        if i > 50 * n + 1000:  # 组合空间耗尽的保护
            break
    return out


def trigger_space_size() -> int:
    return len(PERSONA) * len(MANNER) * len(FOCUS)
