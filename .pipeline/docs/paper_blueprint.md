# Paper Blueprint —— 从 AgentMark / SeqWM / ActHook 学到的论文构造法

_2026-09-09　用途：CoTGuard 按同一模板重建。本文档提取的是**骨架与修辞**，不是内容；
竞品威胁评估另见 `digest_agent_watermark.md`。_

> **核心观察**：这三篇**都不是理论论文**。它们是"method + empirical validation"论文。
> 理论只占半页到一页，且**删掉后论文仍然成立**。我们此前一直在追理论 gap，方向本身就错了。

---

## 一、AgentMark (arXiv:2601.03294)

### 1. 开场动作

**两段式**：第一段讲 agent 从"被动语言接口"转向"自主执行"，列举 GUI 助手、金融交易、社交
agent 三类真实部署（各带引用）。第二段立刻转到风险：impersonation、automated disinformation、
large-scale manipulation、unauthorized replication of proprietary agent systems。

**未受保护的资产**：agent 的 **planning behavior**（选哪个工具、承诺哪个子目标）。

**为什么重要**：走的是**部署论证 + 监管论证**——"in regulated settings"、"support auditing and
enforcement"。不是成本论证。

### 2. Gap 句（原文）

> "While content watermarking effectively attributes LLM-generated outputs, it **fails to directly
> identify the high-level planning behaviors** (e.g., tool and subgoal choices) that govern
> multi-step execution."

正文进一步拆成两条**能力性 gap**：
- 训练时方案不可行（闭源 API、重训成本高）
- 推理时 token 级方案"do not align well with high-level behaviors, since behaviors are not
  token-native"，并给了一个**具体例子**："Alice bookmarked a post with the tag #TravelInspiration"
  被压成 "bookmarking"/"tagging"，provenance token 被剥离

**句式是「未覆盖的层次」**：内容层 ≠ 行为层。

### 3. 方法核心 trick（一句话）

从 agent 显式索取候选行为的概率表 `Pt`，再用 **distribution-preserving conditional sampling** 在
`Pt` 上采样嵌入多比特 ID，使 `ˆbt` 的边缘分布仍等于 `Pt`。

**机器量**：一个主意（保分布采样）+ 两个工程件（keyed PRG 复现随机性、RLNC 抗擦除编码）。

### 4. 理论

**没有编号定理。** 只有式(4)(5)两条**目标定义**（utility preservation 的 ε 界、robust decodability
的 1−δ 界），以及 Eqs.(17)(18) 预测 FPR 按 `2^-k` 衰减——且该预测**用实验曲线验证**（Figure 4）。

**理论买到了什么**：把"我们的方法不掉点、能解码"变成可检验的形式陈述，并让 FPR 曲线有个可对照
的理论线。**删掉后论文仍成立**。

### 5. 主表结构

**Table 1** 列：`Setting | Task | SR(%)↑ (Base/RG/Ours) | Steps↓ (Base/RG/Ours) | bps↑ | bpt↑ |
Δs/step↓ | ΔTok/step(%)↓`

即 **utility（成功率、步数）+ capacity（bits/step、bits/task）+ overhead（延迟、token）** 三组。

- 环境：ALFWorld-ID、ALFWorld-OOD、ToolBench、OASIS 社交模拟（**4 个**，每个 6 个子任务）
- baseline：Base（无水印）、RG（bias-based red-green）
- 模型：主模型 + Gemini 2.0 Flash 交叉验证（附录）

**鲁棒性另开一节**：FPR/密钥伪造（Fig.4）、步擦除与截断（Fig.5）、**语义改写（Table 2）**、
与内容水印的可组合性（Table 3）。

### 6. 弱点处理

Limitations 坦承四条：需要 agent 输出显式行为分布、需要日志、**"semantic variation in observations
(e.g., paraphrasing) may reduce synchronization"**、`Pt` 尖锐时容量小。

**关键手法**：Table 2 的 Bit Recovery 塌到 **16.84±19.56%**，但正文用三句话把它转成正面叙述——
"many steps remain stable"（Behavior Match Rate 49.45%）、"ID payloads are always short while agent
executions are long-horizon"、"aggregation over surviving aligned steps can still support practical
verification"。**承认了但没有受伤**，因为他们把改写归到"高压力压力测试"而非主威胁模型。

---

## 二、SeqWM (arXiv:2605.11036)

### 1. 开场动作

**第一段直接上成本数字**：单条 SWE-Bench 风格轨迹约 **$100**、单条工具使用对话约 **$8**、
Mind2Web 2 的 120 个任务耗费 **1,000+ 小时**人工。然后一句收口："Agents and the data that train
them have become **economic objects** whose unauthorized replication is now both technically
feasible and commercially attractive, while the infrastructure for verifying their provenance has
not kept up."

**未受保护的资产**：agent 的 **decision-making capability** 与 **trajectories 本身**。

**为什么重要**：**成本论证**（最有力的一种）+ imitation attack 的具体威胁。

第二段把问题压成一个可回答的技术问句：**"given an observed sequence of agent actions, can one
verify which agent produced it?"**

### 2. Gap 句（两层，都是原文）

**对 token 级水印**（带小标题 "Why token-level watermarking does not transfer"）：
> "The signal, in short, is **lost twice**—once into the wrong substrate and once at the visibility
> boundary."

配一个**杀伤力极强的数字**："a survey of 29 commercial agentic platforms reports that **82.8%
expose only the final action stream while concealing internal reasoning traces**"。

**对已有行为水印**（小标题 "Existing behavioral watermarks"）：
> "However, by **treating each action step as an independent trial**, they overlook trajectory
> structure and become fragile when trajectories are perturbed, truncated, or observed without
> reliable alignment."

**句式是「失效模式」**：不是"没人做过"，而是"他们做了但在 X 条件下会碎"。**这是三篇里最好的
gap 写法**——它天然指向一个可做的实验（造出 X 条件，展示 baseline 崩溃）。

### 3. 方法核心 trick（一句话）

不用绝对轮次索引播种，而是**以最近 w 个动作的窗口为条件**播种，使水印挂在局部行为转移上，
从而**不需要位置对齐**，删除只造成局部而非全局失效。

**机器量**：三个组件（history-conditioned 多通道编码、滑窗检测、random-key 校准），但**共享一个
中心洞察**——"generation is chained, verification need not be"。

### 4. 理论

**一条定理**：Theorem 4.1（Deletion robustness），形式是单次删除的可加界
`S(K,b̂) ≥ S(K,b) − d(w+1)m`。外加 random-key 校准的**有限样本有效性论证**（可交换性 ⟹
`Pr[p ≤ α] ≤ α`，"does not invoke any asymptotic regime"）。

**理论买到了什么**：定理把"局部失效 vs 全局失效"的设计对比**量化**了（"when w = 3, deleting..."），
直接支撑主张。校准论证则换来三个可直接写进 bullet 的性质：**distribution-free / robust to
adversarial corruption / detector-agnostic**。

**删掉后论文仍成立**，但"局部 vs 全局"的卖点会弱很多。

### 5. 主表结构

**Table 2** 列：`Model | Method | ToolBench(z↑, p↓, Hit%) | ALFWorld(z, p, Hit%) | OASIS(z, p, Hit%)`

即 **3 benchmark × 3 LLM × 4 method** 的检测显著性网格。

- benchmark：ToolBench(200 queries)、ALFWorld(50 tasks)、OASIS Reddit(100 steps) —— **3 个**
- 模型：LLaMA-3.2-3B、Gemma-4-E4B、Qwen3-4B —— **3 个开源模型**
- baseline：Unwatermarked、AgentGuide、AgentMark、SeqWM —— **3 个对照**
- 消融：γ 敏感性扫描（6 个取值 × 3 模型）、单通道 vs 多通道
- 鲁棒性：动作删除率 r ∈ {0,5,10,20,30,50}%，报 **TPR at fixed FPR**（不是 AUC）

### 6. 弱点处理

Limitations 藏在附录 A，且**措辞极为策略性**：
> "Several natural extensions remain open and we view them as **promising directions rather than
> shortcomings of the present construction**."

然后才承认 "semantic-level transformations such as paraphrasing"。**把自己的缺口重述成"任何行为层
水印最终都要面对的问题"**（"questions that any behavior-level watermark must eventually engage
with"），从而把个体缺陷转成领域共性。不受伤。

---

## 三、ActHook (arXiv:2602.18700, ICML 2026)

### 1. 开场动作

与 SeqWM 几乎同构，**成本论证**且数字重叠（$100/SWE-bench task、$8/dialogue、1000h/Mind2Web 2），
但**前置了一段"轨迹有多值钱"的收益论证**：SWE-Gym 仅微调 491 条轨迹就在 SWE-Bench Verified 上
提升 14 个绝对点；AgentTrek 微调合成 web 轨迹使视觉 grounding 翻倍。

**未受保护的资产**：**agent trajectory datasets**（训练数据）。

**为什么重要**：成本 + 收益 + **许可证论证**（"redistributed in violation of their original
licenses"、"non-commercial restrictions"）。

### 2. Gap 句

> "Despite the high cost of creating these datasets, existing literature has **overlooked copyright
> protection for LLM agent trajectories**."

正文给出**两条具体的技术不适用理由**（这一步很重要，不能只说"没人做"）：
1. 轨迹交错 action 与 observation，**只有 action token 可训练**，observation 在训练时被 mask，
   现有连续文本/代码片段水印不考虑这种异构格式
2. 轨迹数据集**很小（1–2K 条）**，而现有方法需要很高的水印比例才可学习

**句式是「未被覆盖的资产 + 两条硬性不适用理由」**。

### 3. 方法核心 trick（一句话）

借软件工程的 hook 概念，在决策点插入由**秘密激活键**触发、**不改变任务结果**的 hook action；
在被投毒轨迹上训练过的模型，遇到激活键时会显著提高 hook action 的出现率，从而支持黑盒检测。

**机器量**：一个主意。且作者明说实现的水印"serve as **illustrative examples rather than oracle
solutions**"——**降低了自己的举证责任**，很值得学。

### 4. 理论

**一条定理**：Theorem 3.1（Sample Complexity），给出达到 FPR α、FNR β 所需查询数
`n ≥ (z_{1-α}√(q_c(1-q_c)) + z_{1-β}√(q_k(1-q_k)))² / Δq²`。

**理论买到了什么**：把"要查多少次才能下结论"变成可计算的量（且"sample size decreases
quadratically with the effect size"）。这是**实用型定理**，不是深刻定理——但它让方法看起来
有据可依。**删掉后论文仍成立**。

**另有一个很聪明的统计设计**：用 **sham key**（语义中性短语 "OK!"，插在同样位置但从未参与注入）
构造经验零分布，把检验化为**配对单侧 t 检验**。这与 SeqWM 的 wrong-key 校准是同一思想的不同实现。

### 5. 主表结构

**Table 2** 列：`Dataset | Pass@1(%) [Original / w/o wmk / Standalone / Contextual] |
Avg. Turns [同四列] | Avg. Output Tokens [同四列]`

即 **utility 三指标 × 四种设置**。检测性能单独用 Figure 3（AUC）+ 附录 Table 6
（**TPR at 1% and 5% FPR**，并明说 "For provenance claims, low-FPR operating points are more
informative than aggregate AUC"——这句话我们必须抄）。

- 数据集：MATH、SimpleQA、SWE-Smith —— **3 个，跨数学推理/网页搜索/软件工程三类 agent**
- 模型：Qwen-2.5-Coder-7B 为主，含 3B/7B 多尺度
- baseline：CodeMark（主）+ AutoPoison、DeadCode、StyleTransfer（三个 backdoor 类）—— **4 个**
- 攻击：DeCoMa 过滤（Table 3）、paraphrase（Table 4）、output summarization（Table 5）、
  coherence-based 与 position-based 识别攻击（Table 7）—— **5 类**
- 隐蔽性检验：困惑度差异（Table 8）、k-means 聚类 F1（Table 9，F1≈0.10 ≈ 随机）

### 6. 弱点处理

无独立 Limitations 节（ICML 格式），弱点分散在正文，并用 **"illustrative examples rather than
oracle solutions"** 一句预先卸责。攻击章节主动构造了 5 类移除攻击并报告 AUC 仍 >85——
**把"我能扛住多少攻击"当成正面卖点，而不是等审稿人来问**。

---

## 四、共同模板（可直接照着走的动作序列）

1. **锚定一个真实部署场景**，点名 Claude Code / Deep Research / Copilot 这类产品（三篇都点了）
2. **命名一个未受保护的资产**，并**用数字证明它值钱**（成本论证最有力：$100/条、1000 小时、
   微调 491 条涨 14 个点）
3. **把问题压成一个可回答的技术问句**（SeqWM："given an observed sequence, can one verify which
   agent produced it?"）
4. **两层 gap**：先说明现有大类（token 级水印）**为什么整类不适用**——给机理，不给"没人做过"；
   再说明**同类最近工作在什么条件下会碎**——给失效模式
5. **一个中心 trick**，用一句话可讲完；其余都是工程件
6. **一条实用型定理**（样本复杂度 / 鲁棒性可加界 / 有限样本有效性），半页到一页，作用是让方法
   看起来有据可依，**不承担新颖性**
7. **主表 = utility 不掉点**（成功率/步数/token），检测性能单独呈现
8. **鲁棒性单开一节**，主动构造 3–5 类攻击并报告降级曲线
9. **低 FPR 工作点**而非只报 AUC（ActHook 明确论证了为什么）
10. **Limitations 把自己的缺口重述为领域共性**，或预先声明实现只是 illustrative

### 典型主张体量

**一篇论文只需要证明三件事**：
- **能检出**：在若干 benchmark × 模型上显著优于对照（p<0.01 或 AUC>90）
- **不掉点**：utility 与无水印基线基本持平（这是 utility-preserving 类论文的核心卖点）
- **扛得住**：在 3–5 类攻击下降级但不崩溃

**不需要**：新的统计检验、渐近最优性、深刻定理。三篇无一例外。

### 经验门槛（取三篇的交集下界）

| 维度 | 门槛 |
|---|---|
| 数据集/环境 | **3 个**，且跨任务类型（数学 / 工具使用 / 代码 或 embodied / tool / social） |
| 模型 | **3 个**开源模型（SeqWM 用 3B–4B 级即可），或 1 主 + 1 交叉验证 |
| baseline | **2–4 个**，其中至少 1 个是同类最近工作 |
| 攻击类型 | **3–5 类**（删除/截断、改写、摘要、过滤、识别） |
| 消融 | 至少 2 项（关键超参扫描 + 组件消融） |
| 统计 | 报 p 值或 **TPR@低 FPR**，并给经验零分布的构造方式 |

---

## 五、套用到 CoTGuard —— 具体草稿

### 我们的处境（事实）

- 资产：**agent 间中继的 chain-of-thought 推理文本**
- 三篇行为层水印**都需要控制 action/decision 的选择**，都不覆盖推理文本
- **AgentMark 自己的 Table 2**：语义改写下 Bit Recovery 塌到 **16.84±19.56%**，Limitations 承认
  paraphrasing 是失效模式
- **SeqWM 自己的结论段**：把 "semantic paraphrase-style trajectory edits" 明确列为 **future work**
- 我们已有：wrong-key 校准移植到语义相似度载体（EXP-003 仿真验证，漂移 0–0.5 全区间
  FPR 稳在 0.0077–0.0130）；工程发现 **K ≫ n**（EXP-004，K 不足时依赖极值的规则功效塌陷最多 −0.788）

### 开场动作（草稿）

> Multi-agent LLM systems now decompose work across specialized agents that exchange
> **intermediate reasoning**, not just final answers: a planner drafts a solution sketch, a critic
> revises it, an executor carries it out. Systems such as Claude Code, OpenAI Deep Research, and
> Microsoft Copilot deploy such pipelines in production. The reasoning traces flowing between these
> agents are themselves expensive assets — a single high-quality reasoning trajectory costs on the
> order of $100 to produce, and curating a benchmark of them runs to thousands of hours of human
> labor. Yet once a trace leaves the system that produced it, its owner loses all visibility: it can
> be relayed through third-party agents, paraphrased, and re-published with no way to establish
> where it came from.

### Gap 句（草稿，两层）

**第一层——整类不适用**：
> Behavioral watermarking establishes provenance for what an agent *does* — which tool it calls,
> which subgoal it commits to — by intervening in how a discrete action is selected from a finite
> candidate set. Chain-of-thought reasoning offers no such handle: there is no candidate set to
> sample from, no per-step distribution to preserve, and the carrier is continuous natural language
> rather than a discrete decision. The behavioral substrate that makes agent watermarking work is
> simply absent.

**第二层——同类工作的失效模式（用他们自己的数字）**：
> Where the two substrates do meet, the behavioral approach is known to break. AgentMark reports
> Bit Recovery collapsing from 100% to **16.84%** under semantic-preserving rewriting of step
> observations, and identifies paraphrasing as a failure mode in its limitations; SeqWM likewise
> defers "semantic paraphrase-style trajectory edits" to future work. **Semantic rewriting is
> precisely the operation that relaying a reasoning trace through another agent performs.** The
> regime that behavioral watermarks defer is the regime reasoning-trace provenance lives in.

### 方法一句话主张（草稿）

> **CoTGuard** establishes provenance for chain-of-thought traces by injecting a keyed semantic
> trigger into the reasoning prompt and verifying candidate traces against an empirical null
> constructed from **wrong-key triggers on the same trace**, yielding finite-sample valid detection
> that survives multi-hop semantic rewriting.

### 主表列头（草稿）

**Table 1 — Utility preservation**（照 ActHook/AgentMark 的位置）

| Dataset | Task Acc (%) ↑ [Original / No-trigger / CoTGuard] | Avg. Reasoning Steps [同三列] | Avg. Output Tokens [同三列] |

**Table 2 — Detection across benchmarks and models**（照 SeqWM 的网格）

| Model | Method | GSM8K (p↓, TPR@1%FPR↑, AUC↑) | FOLIO (同) | TravelPlanner (同) |

其中 Method 行：`No-trigger` / `Sum-based aggregation (prior practice)` / `AgentMark-style (adapted)`
/ `CoTGuard (ours)`。

**Table 3 — Robustness under multi-hop semantic rewriting**（我们的主战场，必须最详）

| Rewrite Style | Hops N=1 | N=2 | N=3 | N=5 | N=8 |  ← 每格报 TPR@1%FPR

Rewrite Style 行：`paraphrase` / `simplify` / `academic` / `summarize-expand` / `guard-lexicon`
（前四类沿用 Chainwash 的分类以便直接对照，第五类是针对相似度检测器的定向规避）。

**Table 4 — Calibration validity**（我们独有，来自 EXP-003/004）

| Calibration | Distribution shift 0.0 | 0.1 | 0.2 | 0.3 | 0.5 |  ← 每格报经验 FPR（名义 0.01）

行：`clean-trace conformal` / `wrong-key (ours)`。附 K/n 比值的敏感性作为消融。

### 需要补齐才能达标的清单

- [ ] 3 个数据集（GSM8K / FOLIO / TravelPlanner，跨数学-逻辑-规划）
- [ ] 3 个开源模型（生成方 + 至少 2 个不同家族的中继方）
- [ ] 3–4 个 baseline（含无 trigger、求和型聚合、一个改编的行为层方法）
- [ ] 5 类改写攻击 × 8 跳
- [ ] 一条实用型定理（**建议：wrong-key 校准在语义载体下的有限样本有效性 + K≫n 的充分条件**，
      对标 ActHook 的 Theorem 3.1 体量，不追求深刻）
- [ ] 报 TPR@1%FPR，不只报 AUC

---

## 六、给自己的三条提醒

1. **不要再追理论 gap**。三篇的理论都是半页的实用型结论，且删掉不影响成立。我们连续三次栽在
   "找理论空白"上（Prompt Infection / SeqWM / Zhang-Jin-Wu 2017），根因是把 method 论文当成
   theory 论文在写。
2. **gap 要写成失效模式，不要写成"没人做过"**。SeqWM 的 "treating each action step as an
   independent trial ... become fragile when ..." 是最好的范式——它自带实验设计。
   我们的对应句是"行为层水印在语义改写下失效，而语义改写正是推理轨迹中继所做的事"。
3. **别人的 Limitations 就是我们的 Introduction**。AgentMark 的 16.84% 和 SeqWM 的 future work
   一句，是我们最有力的两个引子，且**必须原样引用他们自己的数字**——这比我们自己论证更有说服力。
