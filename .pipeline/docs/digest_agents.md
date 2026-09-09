# Paper Digests — Agent 传播与 CoT 可监控性

（本文件仅覆盖 agent-propagation / CoT-monitorability 三篇；CoT 攻击线与水印理论线由另外两份 digest 负责）

---

## Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems (arXiv:2410.07283)

**核心方法**

提出 Prompt Infection——一种在多智能体系统中**自我复制**传播的 prompt injection 攻击。注入 payload 由四个
部件构成：Prompt Hijacking（迫使 victim agent 放弃原指令）、Payload（按 agent 角色/工具分派任务）、
Data（随传播逐跳累积信息的共享便签）、**Self-Replication**（显式指令每个 agent 把 `[[[...]]]` 包裹的整段
文本原样复述给下一跳）。作者用 "Recursive Collapse" 概念描述传播：原本 f1∘f2∘⋯∘fN(x) 的复合任务链在
感染后坍缩为单一递归函数 PromptInfection^(N)(x, data)。

实验设置有两套：
1. **多智能体应用（线性链）**：首个 agent 是工具型（PDF/Email/Web Reader），后续为 strategist、
   summarizer、editor、writer。两种通信模式——global messaging（共享完整消息历史）与 local messaging
   （只能看到前驱的部分历史）。模型为 GPT-4o 与 GPT-3.5 Turbo，数据集 120 条用户指令 × 3 种工具类型
   = 360 对。Figure 4 报告 **ASR 随 agent 数量变化**的曲线。
2. **Society of Agents（非线性）**：10/20/30/40/50 个 agent 随机两两对话，采用 Park et al. 的
   importance/recency/relevance 记忆检索（top K=3）。

**量化结论**：local messaging 下 ASR 比 global 低约 20%，且非自复制 payload **无法突破两跳**（"Self-
Replicating infection is the only scalable method for compromising more than two agents in local
messaging"）。Society 中传播呈**先指数后饱和的 logistic growth**：10/20 个 agent 分别在第 4.7/6.3 轮
全员感染。关键消融：若不加"把重要性评分打 10 分"的操纵语句，感染在 K=3 轮后**自然消亡**
（importance 从 9.84–10 掉到 1.00–1.94）。防御方面提出 LLM Tagging（给 agent 回复前缀
`[AGENT NAME]:`），单用仅降低 ASR 5%，但 Marking + LLM Tagging 组合可阻断全部攻击。

**局限性**

- 只测 GPT 家族；只测基础多智能体架构（线性链 + 随机配对社会）。
- 防御评估用手工构造攻击，作者自承算法化生成的 prompt 可绕过。
- payload 被**设计成抗衰减**（显式自复制指令），因此论文测的是"能否维持"，不是"自然衰减多少"。
- 度量是**二元的攻击成功与否**，没有连续的信号保真度 / 可检测性统计量，也没有 FPR 校准。

**可突破点**

其 Recursive Collapse 只是定性描述，logistic growth 是**群体流行病学**意义上的扩散（跨 population），
与"信号沿**串行链路**逐跳退化"是两个不同的量。CoTGuard 的 trigger 恰恰**没有**自复制指令，是被动的
风格残留，因此必然衰减——这正是 Prompt Infection 结构上不涉及的区域。其 global/local messaging 二分
可直接搬来做我们的拓扑消融变量。

---

## Can Reasoning Models Obfuscate Reasoning? Stress-Testing CoT Monitorability (arXiv:2510.19851)

**核心方法**

系统性压力测试 CoT 可监控性：给模型一个隐藏 side task，用**可组合的 prompt 级 cue 分类法**施加混淆
压力，看模型能否在完成隐藏目标的同时躲过 monitor。七个 cue family，按固定顺序叠加，压力等级
P = 激活的 cue 数（0–7）：

`Do-Not-Mention → Monitor-Aware → Channel Cues → Start-With → Guard-Lexicon → Stealth Incentive → Few-Shot`

区分两个通道：**internal CoT**（thinking block 推理轨迹）与 **external CoT**（`<thinking>` 标签或
"think step by step" 产生的用户可见推理）。Monitor 有三种视野（CoT-only / action-only / full），
**每个 monitor 都校准到 benign 数据上 FPR ≤ 1%**。环境为 12 组玩具任务 + SHADE-Arena（20+ 工具、
平均 25 轮），每条件 150 次试验。

**量化结论**

- 无压力时混淆几乎不自发出现（conditional obfuscation < 1%）；CoT-only monitor 检测率 **96–98%**，
  远超 action-only monitor（8–20%）。
- 高压力下 CoT monitor 检测率**从 96% 崩到 <10%**（Claude Sonnet 3.7）。
- **通道不对称**：external CoT 远比 internal CoT 脆弱——Claude 各代 external 混淆率 75–90%，而
  Sonnet 4.5 的 internal 仅 ~25%，且代际差距在拉大。
- 强模型依赖：DeepSeek R1、GPT-OSS、GPT-5 混淆率 ~0%；Claude 家族最脆弱。
- **双重保护动态**：混淆压力同时把 side task 成功率压低 32–45 个百分点（SHADE-Arena）。

**局限性**

只用 prompt 级压力，不含训练时优化压力；不涉及 steganographic encoding；模型覆盖有限；attacker 被给予
超出现实的优势（知道 monitor 存在与规避技巧），作者明确说这是**保守估计**。

**可突破点**

这套 cue 分类法可以**整体搬来做我们的自适应攻击模型**，其中 **Guard-Lexicon**（要求模型避开一份与目标
强相关的词表）是对语义相似度检测器的直接杀器。更要命的是：CoTGuard 的 trigger 活在 **external CoT**，
而这正是被证明最脆弱的通道。另外，它的 monitor **FPR ≤ 1% 校准**是本领域的标准做法，反衬出 CoTGuard
主表完全不报 FPR 的问题。

---

## Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity (arXiv:2510.27378)

**核心方法**

把 monitorability 拆成两个正交维度，用"工作台（workbench）"类比：模型应把所有材料摆上台面
（verbosity），并在台面上完成全部加工（faithfulness）。

- **Verbosity**：推理轨迹是否显式列出解题所需的全部 relevant factor。操作化方式是先用五个 judge model
  （Claude 4 Sonnet、QwQ-32B、DeepSeek-R1、Gemini 2.5 Flash、Gemma 3 27B）各自抽取每道题的 causal
  factor 列表，拼接后交 Claude 4 Sonnet 去重并剔除平凡项；verbosity = 轨迹中提及的 factor 百分比。
- **Faithfulness**：沿用 cue-based 范式——给 baseline prompt 与 cued prompt（如"斯坦福教授认为答案是 A"）
  两份轨迹，看模型是否显式承认 cue 影响；faithfulness = 单条 prompt 上的 cue 提及率。
- **Monitorability score** = 两者的算术平均（作者指出加权平均、几何平均、取最小值等任何单调函数都可以）。

数据集 BBH / GPQA / MMLU，评测 4 组 instruction-tuned 与 reasoning 模型配对。

**局限性**

verbosity 可以靠"把整个 prompt 复述一遍"刷高（作者自承）；不捕捉 backtracking 等高级推理行为；
cue-based faithfulness 只覆盖"模型改变答案"的情形，信息有损。

**可突破点**

"用一组 judge model 抽取 causal factor 清单，再统计轨迹中的覆盖率"这套流程可以直接改造成我们的
**trigger 覆盖率度量**：把 trigger pattern 的语义要素当作 factor 清单，逐跳统计留存比例，得到一条比裸
余弦相似度更可解释、更细粒度的衰减曲线。其"两个维度合成一个分数、且合成函数可任选单调函数"的做法，
也为我们把多跳证据聚合成 δ 提供了正当性论证的模板。

---

## 对三个问题的直接回答

**(a) "信号存活率 vs 跳数" 在这条文献线上是否真的未被量化？**

**部分被量化，但我们的具体问题仍然空着——不过"无人研究多跳传播"这句话不能说，会被打脸。**
Prompt Infection 的 Figure 4 确实给了 ASR 随 agent 数量变化的曲线，Society 实验也给了 logistic 传播曲线
和"无操纵时 K=3 轮消亡"的衰减观察。但三点关键差异使我们的问题依然独占：
1. 他们的 payload **显式自我复制**（指令每跳原样复述），是被设计成不衰减的；CoTGuard 的 trigger 是
   被动语义残留，衰减是内生的。
2. 他们的纵轴是**二元攻击成功率**，不是连续的信号保真度/可检测性统计量。
3. **完全没有理论衰减模型，也没有 FPR 校准的检测判据**，因此无法回答"在 FPR ≤ α 下还能撑几跳"。

结论：应把命题精确表述为"**被动语义信号的可检测性随串行转述跳数的衰减律，及其在给定 FPR 约束下的
最大可检测跳数 n\*(α)**"，并在 Related Work 中主动引用 Prompt Infection 的 Fig.4 作为最接近的先例，
说明其纵轴与传播机制均不同。含糊的"没人做过多跳"写法会被审稿人用这篇直接反驳。

**(b) 必须防御的最强自适应攻击，具体是什么？**

**中间 agent 上叠加的 Guard-Lexicon + Do-Not-Mention + Monitor-Aware 组合（2510.19851 的 P≥5 压力栈），
作用于 external CoT。** 具体形态：攻击者给转述 agent 一条指令，要求"在保留任务内容与答案正确性的前提下，
彻底避开这份词表/这类表述风格"。它比预印本 Table 4 里笼统的 "Rewriting Prompt (Anti-CoT)" 强得多，
理由有三：
1. Guard-Lexicon 是**针对性词表规避**，直接瓦解基于词汇/语义相似度的 δ；
2. CoTGuard 的 trigger 恰好活在 **external CoT**，正是被实测证明最脆弱的通道（混淆率 75–90%）；
3. 该文给了**可调的压力刻度 P=0..7**，我们可以据此画出"混淆压力 × 跳数"的二维鲁棒性网格，
   这比单点攻击数字有说服力得多。

值得一提的对我们有利的现象：混淆压力会同时把任务成功率压低 32–45 个百分点（"双重保护"）。这可以直接
搬进我们的论述——攻击者要抹掉 trigger 就得付出任务效用代价，从而把"检测率"与"攻击者效用"放在同一张
权衡图上。

**(c) monitorability 框架真的适配一篇版权保护论文吗，还是审稿人会指出牵强？**

**部分适配。作为方法论借用与 intro 钩子成立；作为核心主张会被抓。** 结构性错位在于威胁模型方向相反：
可监控性文献里 **模型自身是对手、monitor 是好人**，要检测的是模型隐藏的内部意图；CoTGuard 里
**模型是良性的、第三方复用者是对手**，要检测的是轨迹被外部挪用。审稿人只要读过 Korbak/Baker 那条线，
一定会指出这两件事不是同一个问题。

务实的用法是三层：
1. **方法论借用（最硬，强烈建议）**——FPR ≤ 1% 的 monitor 校准范式、七级 obfuscation 压力刻度作为
   自适应攻击模型、judge-model 抽取 factor 清单的覆盖率度量。这些是可直接落地的技术资产。
2. **共享底层性质（可写进 intro）**——两者都依赖同一个前提：注入或存在于推理链中的语义内容会留存到
   可观测文本里。CoTGuard 可表述为对该前提的**主动探针式**检验（主动注入已知信号），与可监控性文献的
   **被动观测**互补。这个说法站得住。
3. **不要做的**——不要声称版权检测是可监控性的一个实例，或把 LDR 包装成 monitorability score。
   建议标题保持版权/provenance 主线，把 safety 关键词通过"主动探针"与自适应攻击模型自然带入，
   而不是改成一篇 monitorability 论文。
