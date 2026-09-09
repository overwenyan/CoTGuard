# 方向 α（短轨迹/有限样本检测）竞品核实

_2026-09-09　对抗式核实：任务是找出杀死该角度的论文，而非确认其空缺_

---

## 判决：**死了。** 不是"部分被占"，是核心论断在 2017 年已被完整发表。

而且是被 **Jiashun Jin** 发表的——Higher Criticism 原始论文（Donoho & Jin 2004）的作者本人。

---

## 1. 致命论文：Zhang, Jin & Wu (2017), arXiv:1702.07082

**《Distributions and Statistical Power of Optimal Signal-Detection Methods In Finite Cases》**
Hong Zhang, Jiashun Jin, Zheyang Wu（WPI + CMU）

### 它的摘要几乎逐字复述了我们的立论动机

> "some grouping-test methods such as Higher Criticism test (HC), Berk-Jones test (B-J), and
> φ-divergence test share the similar asymptotical optimality when n → ∞. **However, in practical
> data analysis n is frequently small and moderately large at most.**"

我们打算说的"整个领域的检测理论都是渐近的，而真实 n 很小，渐近最优在该尺度上是错的判据"——
**就是这篇论文的第一句话。**

### 重合程度是全方位的

| 我们计划做的 | 该论文已做的 |
|---|---|
| 稀疏正态混合区制 ε=n^(-β), μ=√(2r log n) | **完全相同的 ARW 设定**：ε_n = n^(-α), α∈(1/2,1), μ_n = √(2r log n), r∈(0,1) |
| 比较 HC 与求和型在小 n 的功效 | 比较 HC / Berk-Jones / **φ-divergence 全族**的有限样本功效 |
| 我实现的 `_phi_divergence(u,v,s)` | **正是他们研究的 φ-divergence 族**（s 为族参数） |
| 用仿真测 n=50/100 的交叉 | **解析地给出有限样本精确分布**（Theorem 3.1、3.4），Fig 4 明确画了 **n=10 与 n=100** |
| 给出"哪个规则在哪个区制赢"的地图 | 已给出结论："**HC is the best choice when signals are rare, while B-J is more robust
  over various signal patterns**" |
| — | 已实现为 R 包 **SetTest**，发布在 **CRAN** |

### 对我们 EXP-002 的意义

我在 `experiments/diag_crossover.py` 中"发现"的现象——渐近最优规则在小 n 反而更差、n≥500 才翻盘
——是对一个 **2017 年已知结果的重新发现**。实验本身没做错（结论与文献一致，反而说明实现正确），
但它**不构成新知识**。

---

## 2. 次级威胁：有限样本 HC 的既有工作

- **精确零分布**：已有基于 **Steck's determinant** 的 HC 显著性精确算法
  （Statistics & Probability Letters, 2017）。HC 渐近临界值在小 n 下失准是**已记录的已知问题**。
- **收敛速度**：HC 依赖极值统计量，向渐近分布收敛慢、小样本下 Type I error 失控，均为文献常识。
- **Berk-Jones 的有限样本优势**：文献早已确立 B-J 与 HC 同享渐近最优，但**有限样本下 B-J 在中等
  稀疏区更强**，具体分界在 **β ≈ 0.75**（β ≳ 0.75 时 HC 胜，β ≲ 0.75 时 B-J 胜）。
- **omnibus / 组合规则功效比较**：Fisher / Stouffer / Bonferroni / Simes / HC 的功效曲线比较是
  经典多重检验文献的标准内容（如 arXiv:1709.00960、1007.1434）。

> **crux 的回答**：是的，经典多重检验文献**已经回答了"哪种组合规则在小 n 下获胜"**，且回答得比
> 我们计划的更完整（解析分布 + 全族覆盖 + 现成软件）。这正是我被要求重点排查的那个杀手，它确实存在。

---

## 3. LLM 侧同样被占：短文本水印检测是活跃已知问题

- 短文本检测证据不足是水印文献的**公认核心限制**（长文本证据多、短文本判定不可靠）。
- **A Likelihood Based Approach for Watermark Detection**（Li et al., AISTATS 2025）专门提升短文本
  检测功效，报告约 **65%** 的功效提升。
- **Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach**
  （arXiv:2410.02890）提供分布自适应的理论框架。
- 低熵 token 下水印不可检测是独立的已知瓶颈，已有 HeavyWater / SimplexWater（arXiv:2506.06409）等
  专门方法。

即"短序列检测难"在 LLM 水印这一侧也**不是新观察**，且已有专门方法。

---

## 4. CoT 步级统计：方向不同，但不构成我们的空间

`arXiv:2605.11746`《When Reasoning Traces Become Performative: Step-Level Evidence that CoT Is an
Imperfect Oversight Channel》做步级证据分析，但服务于 **oversight/faithfulness**（latent commitment
与显式答案仅 61.9% 对齐），不是来源检测统计。方向相邻但不重叠——然而它也说明"CoT 步级统计分析"
本身已有人在做。

---

## 5. 还剩什么（诚实评估：不足以支撑一篇论文）

唯一未被占据的是一个**翻译性观察**：

> LLM 水印 / CoT 检测这一侧的工作，在 n=5–50 的尺度上沿用了渐近最优动机的聚合规则，
> 却没有引用统计学中已经成熟的有限样本文献（Zhang-Jin-Wu 2017、SetTest 包）。

这是一条**"你们应该去用 SetTest"的提示**，属于 workshop note 或某篇论文的一个小节，
**不构成独立贡献**。把它写成主贡献会被熟悉多重检验文献的审稿人当场指出。

---

## 6. 给母代理的建议

方向 α 不可用。若要继续，剩余选项应从 β（跨 agent 聚合——注意：需同样强度的对抗式核实，
我未在本次核实范围内验证它）或 γ（极限结果作为贡献，但需重新界定，因为"有限样本下渐近规则
失效"这一条已属已知）中选，或考虑更根本的转向。

**同时建议**：本次核实暴露了一个系统性问题——前三轮"空白"判断（多跳传播、共形校准、有限样本
检测）全部被推翻，且推翻它们的都是**领域内的经典或近期核心文献**。后续任何方向在投入前都应先做
本次这样的对抗式核实，且检索必须覆盖**统计学主刊文献**（多不在 arXiv 的 cs 分类下），
而不只是 arXiv 的 LLM 相关论文。
