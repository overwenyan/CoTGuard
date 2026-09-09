"""可分离密钥空间 (v2): 针对 EXP-R0b 发现的"密钥拥挤"问题.

v1 (trigger.py) 的缺陷: 所有 key 共用同一模板, 仅替换槽位填充词, 导致
真 vs 错 pattern 相似度 mean=0.664 / max=0.972, attribution 不可能.

v2 的三条分离策略:
  S1 领域分散   —— persona 取自互相远离的领域(棋手/厨师/法官/地质学家...), 而非全是教学角色
  S2 结构分散   —— 不同 key 使用不同的句式模板, 而非同一模板换词
  S3 词汇锚点   —— 每个 key 携带一个罕见但自然的词汇锚点, 提供离散可分的抓手

注意 S3 与 stealth 有张力: 锚点越罕见越可分, 也越容易被人眼发现.
该张力应作为论文的一条权衡曲线, 而非藏起来.
"""

from __future__ import annotations

import hashlib

# S1: 领域上互相远离的 persona
PERSONA = [
    "chess coach reviewing a game", "chef scaling a recipe", "surveyor checking measurements",
    "accountant reconciling ledgers", "navigator plotting a course", "pharmacist checking doses",
    "watchmaker adjusting a movement", "beekeeper counting hives",
    "librarian cataloguing a shelf", "electrician tracing a circuit",
    "gardener spacing seedlings", "cartographer scaling a map",
    "sommelier blending a batch", "locksmith cutting a key",
    "tailor measuring cloth", "geologist reading strata",
]

# S2: 结构不同的句式模板
TEMPLATES = [
    "Work through this the way a {persona} would, and {anchor}.",
    "Adopt the mindset of a {persona}. As you reason, {anchor}.",
    "A {persona} would approach this carefully; follow that approach and {anchor}.",
    "Reason as though you were a {persona}, taking care to {anchor}.",
]

# S3: 词汇锚点 (自然但用词各异, 提供离散抓手)
ANCHORS = [
    "name each quantity before combining it",
    "restate the target before each computation",
    "flag every assumption as you rely on it",
    "track the units through every operation",
    "mark which numbers are given and which are derived",
    "note the invariant that stays fixed",
    "check each intermediate value against the question",
    "separate what is known from what is inferred",
    "label each step with the operation it performs",
    "confirm the scale of the result before moving on",
    "state the relation you are about to apply",
    "verify the direction of each comparison",
]


def _idx(key: str, salt: str, n: int) -> int:
    return int(hashlib.sha256(f"{key}|{salt}".encode()).hexdigest(), 16) % n


def make_trigger_v2(key: str, task_type: str = "arithmetic") -> str:
    return TEMPLATES[_idx(key, "tmpl", len(TEMPLATES))].format(
        persona=PERSONA[_idx(key, "persona", len(PERSONA))],
        anchor=ANCHORS[_idx(key, "anchor", len(ANCHORS))],
    )


def key_pool(n: int, prefix: str = "k") -> list[str]:
    """生成 n 个产生**互不相同** pattern 的密钥."""
    seen, keys, i = set(), [], 0
    while len(keys) < n and i < 200 * n + 5000:
        k = f"{prefix}-{i}"
        t = make_trigger_v2(k)
        if t not in seen:
            seen.add(t)
            keys.append(k)
        i += 1
    return keys


def space_size() -> int:
    return len(TEMPLATES) * len(PERSONA) * len(ANCHORS)


def select_separable_keys(n: int, encoder, pool_mult: int = 12, prefix: str = "k"):
    """贪心最远点选择: 从候选池里挑 n 个在 embedding 空间中互相最远的密钥.

    v2 的模板设计把平均相似度降了一半, 但最坏情况仍高(同 persona 不同 anchor).
    本函数直接优化最坏情况 —— attribution 的瓶颈是**最像的那个错误密钥**,
    所以该最小化 max pairwise similarity, 而非 mean.

    encoder: 需有 .encode(list[str]) -> np.ndarray (已 L2 归一化)
    返回 (keys, patterns).
    """
    import numpy as np

    cand_keys = key_pool(n * pool_mult, prefix=prefix)
    cand_pats = [make_trigger_v2(k) for k in cand_keys]
    E = encoder.encode(cand_pats, convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False)

    # 从最"离群"的点起步, 每次加入与已选集合最大相似度最小的候选
    sel = [int(np.argmin(E @ E.mean(axis=0)))]
    while len(sel) < min(n, len(cand_keys)):
        worst = (E @ E[sel].T).max(axis=1)
        worst[sel] = np.inf
        sel.append(int(np.argmin(worst)))
    return [cand_keys[i] for i in sel], [cand_pats[i] for i in sel]
