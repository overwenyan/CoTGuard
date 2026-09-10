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


# ---------------------------------------------------------------------------
# EXP-C4: 通道分离用的成分消融 trigger
#
# EXP-C3 发现指纹有两条来源不同的通道, 而 v2 的 pattern 恰好由对应的两段构成:
#   C1 词法/指令通道 <- ANCHOR ("restate the target"/"note the invariant"...)
#      模型照做, 用词字面漏进轨迹. 判别 n-gram 与密钥文本重叠 0.27-0.77,
#      且**随指令遵循能力增强**(Qwen3-14B 0.771 >> Tulu-3-8B 0.312).
#   C2 特异文体通道 <- PERSONA ("chess coach reviewing a game"...)
#      无字面痕迹, 只把模型的功能词/标点习惯推进一个与该 prompt 绑定的任意盆地.
#      判别功能词与密钥文本重叠恒定 0.11-0.16(四个 run), 即**基本为零**.
#
# 下面两个构造把这两段拆开, 其余(模板选择方式、密钥数、生成设置)完全一致,
# 因此两者之差可归因于成分本身.
#
# 可证伪预测(先写下来, 免得事后合理化):
#   - anchor_only  : hop0 归因更高, n-gram 重叠高, 但在 adaptive_max 词汇擦洗下**塌陷**
#   - persona_only : hop0 归因更低, n-gram 重叠≈0, 但在同样攻击下**基本不动**
#   若两者在攻击下的衰减幅度没有差异, 则两通道假说被证伪.
# ---------------------------------------------------------------------------

PERSONA_ONLY_TEMPLATES = [
    "Work through this the way a {persona} would.",
    "Adopt the mindset of a {persona} as you reason.",
    "A {persona} would approach this carefully; follow that approach.",
    "Reason as though you were a {persona}.",
]

ANCHOR_ONLY_TEMPLATES = [
    "Work through this, and {anchor}.",
    "As you reason, {anchor}.",
    "Approach this carefully and {anchor}.",
    "Take care to {anchor}.",
]


def make_trigger_persona_only(key: str) -> str:
    """只保留 persona(C2), 去掉一切用词指令."""
    return PERSONA_ONLY_TEMPLATES[_idx(key, "tmpl", len(PERSONA_ONLY_TEMPLATES))].format(
        persona=PERSONA[_idx(key, "persona", len(PERSONA))])


def make_trigger_anchor_only(key: str) -> str:
    """只保留 anchor(C1), 去掉一切人设."""
    return ANCHOR_ONLY_TEMPLATES[_idx(key, "tmpl", len(ANCHOR_ONLY_TEMPLATES))].format(
        anchor=ANCHORS[_idx(key, "anchor", len(ANCHORS))])


def key_pool_for(maker, n: int, prefix: str = "k", component: str | None = None):
    """按给定构造函数生成 n 个密钥.

    `component` 指定去重依据("persona"/"anchor"): 必须按**承载信息的那一段**去重,
    而不是按整条 pattern —— 否则 persona_only 会出现同一 persona 配不同模板的密钥,
    人为压低其可分性, 使通道对比不公平(C1/C2 的比较必须只差成分本身).
    """
    pool = {"persona": PERSONA, "anchor": ANCHORS}.get(component)
    seen, keys, i = set(), [], 0
    while len(keys) < n and i < 200 * n + 5000:
        k = f"{prefix}-{i}"
        tag = _idx(k, component, len(pool)) if pool else maker(k)
        if tag not in seen:
            seen.add(tag); keys.append(k)
        i += 1
    return keys
