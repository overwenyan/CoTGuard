# Gap Matrix — CoTGuard

生成日期：2026-09-09　语料：`.pipeline/literature/cotguard-core`（52 篇，含真实 PDF + pdfminer OCR）

约束回顾（来自用户）：标题与核心方法（trigger-CoT 注入 + 相似度检测）尽量不动；新角度须紧扣
CoT / safety / agent；欢迎理论分析。

---

## 0. 一句话结论

CoTGuard 目前的**机制**与 BadChain/ShadowCoT/BadThink 高度同构（都是往 CoT 里塞触发器），差别只在
"意图"（防御 vs 攻击）而非技术；而它的**检测端**（求和聚合的余弦相似度 + 调出来的阈值 θ）比水印文献
的标准低了整整一代（后者自 2023 年起普遍要求可控的 type-I error 与 p 值）。

**但这里恰好藏着一个真正的空白**：现有 CoT 后门/水印工作全部是**单模型、单跳**的；没有任何工作刻画
一个推理层信号在**多智能体 N 跳转述**中如何衰减、以及在给定误报率下还能撑几跳。这个"多跳信号衰减"
问题是 CoTGuard 独占的、可证明的、且天然同时踩中 CoT + agent + safety 三个关键词。

---

## 1. 三条主线的现状与本项目的位置

### 主线 A：CoT 触发 / 后门 / 推理链操纵 —— **本项目创新性风险的来源**

| 工作 | arXiv | 做了什么 | 与 CoTGuard 的关系 |
|---|---|---|---|
| **BadChain** | 2401.12242 | 首个针对 CoT prompting 的后门攻击；往推理步骤序列里插入一个后门推理步，触发时改变最终答案；无需训练数据/参数访问；GPT-4 上 ASR 97% | **最近邻 #1。** 机制几乎相同：prompt 层注入 → 影响 CoT。审稿人一定会问"你和 BadChain 的技术差异是什么"。目前的答案只有"我们是防御"，不够 |
| **ShadowCoT** | 2504.05605 | 操纵模型**内部**推理路径（改 attention 通路 + 扰动中间表示，仅更新 0.15% 参数），RL + 推理链污染自动合成隐蔽 CoT；ASR 94.4% | **最近邻 #2。** 证明了"推理层触发"这一概念已被充分探索。它需要白盒权限，CoTGuard 是黑盒 prompt 层——这是可用的区分点，但需要明说 |
| **BadThink** | 2511.10714 | 触发式"过度思考"攻击：触发后生成冗长推理但保持最终答案一致，MATH-500 上推理长度增加 17 倍 | **最近邻 #3（且是新出现的）。** 与 CoTGuard 共享"保持最终答案不变、只改推理轨迹"的核心设计。必须正面对比 |
| Prompt as Triggers | 2305.01219 | prompt 层后门，考察语言模型脆弱性 | 更早的 prompt-trigger 谱系，用于说明 CoTGuard 的注入方式并不新 |
| Exploring Backdoor Vulns of Chat Models | 2404.02406 | 多轮对话场景的后门 | 多轮 ≈ 多跳的单模型近似 |
| UniGuardian | 2502.13141 | 统一检测 prompt injection / backdoor / adversarial | **潜在 baseline**：现成的触发检测器，可用来对比 CoTGuard 的检测端 |

> **Gap A：** 这条线上**全部是攻击**（外加一个通用检测器）。没有人把"往 CoT 注入可控信号"当作**所有权
> 主张/审计原语**来做，更没有人给这种信号的**可检测性**建立理论。CoTGuard 的立足点应该从"我们提出一种
> 新注入方式"（站不住）转移到"我们提出并刻画推理层信号的可检测性"（站得住）。

### 主线 B：水印 / 版权审计的理论保证 —— **本项目理论短板的现成解药**

| 工作 | arXiv | 提供的工具 | 如何为 CoTGuard 所用 |
|---|---|---|---|
| **Statistical Framework of Watermarks** | 2404.01245 | pivotal statistic + secret key → 精确控制 FPR；闭式渐近 FNR；最优检测规则 = minimax 优化 | **理论模板 #1。** trigger key `k` 正好扮演 secret key；把 δ 重构为在"无触发轨迹"零分布下有已知刻画的 pivot |
| **Data Misappropriation Hypothesis Testing** | 2501.02441 | 把"模型 B 是否挪用了模型 A 的数据"形式化为假设检验；构造检验统计量、最优拒绝阈值、显式控制 type I/II error、渐近最优性 | **理论模板 #2，且问题几乎同构**（只是低一层：训练数据层 vs 推理轨迹层）。CoTGuard 可直接类比：H0 = 候选轨迹独立产生，H1 = 源自我方 trigger 引导的系统 |
| **Tr-GoF / Robust Detection under Human Edits** | 2411.13868 | 用混合模型刻画编辑，提出截断拟合优度检验；**证明 sum-based（加和型）检测规则在编辑扰动下不是最优的**，Tr-GoF 自适应最优 | **杀伤力最大的一篇。** CoTGuard 的 Algorithm 4 正是"逐步求和相似度"的加和型规则——文献已证明这类规则在改写扰动下次优。这既是当前方法的硬伤，也直接给出了有原则的替代方案，同时解释了 Table 4 里 anti-CoT 改写为何伤害最大 |
| **Radioactivity** | 2402.14904 | 检测某模型是否在水印文本上训练过，给出可证明置信度（p < 1e-5，水印占比仅 5%） | 若泄漏形态是"蒸馏"而非"轨迹复制"，这是对应的检测范式；也是 threat model 里必须区分的一种泄漏通道 |
| Double-I Watermark | 2402.14883 | LLM 微调的模型版权保护 | 相邻的版权主张范式对比 |
| SHIELD | 2406.12975 | LLM 文本生成的版权合规评测与防御 | 潜在评测框架 / baseline |
| Ensemble Watermarks / BiMarker / Signature filtering | 2411.19563 / 2501.12174 / 2606.18430 | 检测增强手段 | 检测端工程改进的参考 |
| Sample Complexity of Binary Hypothesis Testing | 2403.16981 | 简单二元假设检验的样本复杂度 | 若要写"需要观测多少推理步才能判定"，这是刻画所需 |

> **Gap B：** 水印文献的理论机器已经非常成熟，但**全部作用在 token 分布层**（logits 偏置、Gumbel-max 等），
> 依赖对采样过程的白盒控制。CoTGuard 处在**语义/prompt 层**且是黑盒 API，token 层的 pivot 构造无法直接搬。
> 这既是难点，也是**理论贡献的空间**：如何在只能观测语义相似度的黑盒条件下构造有零分布刻画的 pivot。

### 主线 C：多智能体安全与信息流 —— **本项目"新角度"的落点**

| 工作 | arXiv | 做了什么 | 与 CoTGuard 的关系 |
|---|---|---|---|
| **Prompt Infection** | 2410.07283 | LLM-to-LLM prompt injection：恶意 prompt 像病毒一样在互联 agent 间**自我复制传播**；提出 LLM Tagging 防御 | **概念上的镜像。** 它研究恶意信号的跨 agent 传播，CoTGuard 研究良性信号的跨 agent 传播。必须引用并对比——这也提示：CoTGuard 的 trigger 传播分析可以借用其传播模型 |
| Communication & Verification in LLM Agents | 2510.25595 | 信息不对称下的 agent 协作、通信与验证 | 多跳信息传递的形式化参考 |
| ACE / Caging the Agents | 2504.20984 / 2603.17419 | LLM 应用系统的安全架构 / 零信任 agent 架构 | 部署侧 threat model 的参考 |
| StruQ / SecAlign | 2402.06363 / 2410.05451 | prompt injection 防御（结构化查询 / 偏好优化） | 单 agent 防御 baseline |
| Audit Trails for Accountability in LLMs | 2601.20727 | LLM 的问责审计轨迹 | **叙事支撑**：把 CoTGuard 定位为 agent 审计原语 |

> **Gap C：** 多智能体安全研究关注的是**攻击如何传播**与**如何隔离**，没有人研究**一个用于审计/所有权
> 主张的良性标记在多跳转述后还剩多少**。"跳数 → 可检测性"这条曲线目前是空白。

### 主线 D（补充检索）：CoT 可监控性 —— **safety 关键词的接入点**

| 工作 | arXiv | 做了什么 |
|---|---|---|
| **Can Reasoning Models Obfuscate Reasoning?** | 2510.19851 | 压力测试 CoT 可监控性：模型能否在隐藏对抗目标的同时混淆自己的 CoT 以逃避监控。发现强混淆压力下部分模型可以逃检；且**外部 CoT 比内部 CoT 更容易被混淆** |
| **Measuring CoT Monitorability** | 2510.27378 | 用 faithfulness + verbosity 合成 monitorability 分数 |
| Visible Reasoning & Indirect Prompt-Injection Monitorability | 2608.15392 | 跨语言的可监控性 |

> **Gap D：** 可监控性文献衡量的是"CoT 是否忠实反映内部推理"，用的是**被动观测**。CoTGuard 的 trigger 是
> **主动探针**——主动注入一个已知信号，看它在轨迹中是否留存。"主动探针式可监控性度量"目前没人做，而
> 2510.19851 的"模型可以在压力下混淆外部 CoT"恰好是 CoTGuard 面临的**最强自适应攻击**（比论文 Table 4
> 里的改写攻击强得多），必须纳入威胁模型。

---

## 2. 四个已知短板 → 文献给出的修法

| 短板 | 现状问题 | 修法（引用） |
|---|---|---|
| **威胁模型不清晰** | 论文说保护"版权内容"，但 δ 实际测的是"我方 trigger 风格是否留存"，度量与主张不匹配；LDR 因此测不到它声称的东西 | 改为 provenance/所有权主张的假设检验形式化（2501.02441）；明确区分三种泄漏通道：轨迹复制 / 转述复用 / 蒸馏（2402.14904）；明确攻击者可观测量与能力边界（2410.07283, 2510.19851） |
| **理论无保证** | Theorem 1 只是描述性陈述，无 FPR 控制 | 引入 pivot + 零分布 + 精确 FPR，给出渐近 FNR 与最优规则（2404.01245, 2501.02441） |
| **创新性不足** | 机制与 BadChain/ShadowCoT/BadThink 同构 | 把贡献从"注入机制"移到"**多跳可检测性理论**"：单模型攻击文献无法覆盖 N-hop 衰减问题（Gap C） |
| **实验不真实/不充分** | 主表缺真实运行；且**未报告 FPR**，使 LDR 数值不可解释 | 必须补：ROC/AUC 而非单点 LDR、干净轨迹上的经验 FPR、跳数消融、改写强度消融；baseline 加入 UniGuardian（2502.13141）与 SHIELD（2406.12975） |

---

## 3. 推荐的"更强角度"（供 /omp:ideate 收敛）

三个候选，均不推翻现有方法框架、不必改标题：

**候选 1（推荐）——「多跳可检测性」：把多智能体从背景板变成研究对象。**
核心命题：推理层信号在 agent 间每一次转述都会衰减；刻画 δ 关于跳数 n 的衰减律，给出在 FPR ≤ α 约束下
仍可检测的最大跳数 n\*(α)。这是 BadChain/ShadowCoT（单模型单跳）结构上无法涉及的问题，直接解决创新性
短板，且理论可做（每跳视作一次带噪信道，用 2411.13868 的混合模型刻画转述噪声）。

**候选 2 ——「假设检验重构」：把 δ 从启发式相似度升级为有零分布的检验统计量。**
用 trigger key 作 secret key 构造 pivot，精确控制 type-I error，给出 p 值而非裸阈值；并按 2411.13868 的
结论把加和型聚合换成截断拟合优度型聚合，顺带解释 Table 4 的鲁棒性现象。这是补理论最直接的路径。

**候选 3 ——「主动探针式可监控性」：安全叙事的包装层。**
把 trigger 重新表述为探测 agent 推理轨迹是否保持可监控/可审计的主动探针，版权泄漏检测只是其中一个下游
应用。接上 2510.19851 / 2510.27378 的 safety 议程，让 CoT + safety + agent 三个关键词自然成立。

> **建议组合：候选 1（新贡献主体）+ 候选 2（理论工具）+ 候选 3（intro 叙事框架）。**
> 标题可保持 "CoTGuard" 主名不变，副标题微调即可容纳新角度。

---

## 4. 必须正面对比的最近邻工作（≥3）

1. **BadChain** (2401.12242) — 同为 prompt 层 CoT 触发；必须说清技术差异而非仅意图差异
2. **ShadowCoT** (2504.05605) — 推理层触发的白盒版本；用"黑盒 vs 白盒"划界
3. **BadThink** (2511.10714) — 同样"保持最终答案、只改推理轨迹"；最新且最像
4. **Prompt Infection** (2410.07283) — 多智能体信号传播的镜像工作，跳数分析必须引用
5. **Tr-GoF** (2411.13868) — 直接指出当前加和型检测器次优，必须回应

---

## 5. 检索过程中的一个发现（需修正）

预印本参考文献 [15] 标注 `arXiv:2305.18829` 为 "CoDA: Copyright detection in artificial intelligence-generated
content via natural tracing"，但该 arXiv ID 实际对应的是 **UniScene: Multi-Camera Unified Pre-training via 3D
Scene Reconstruction for Autonomous Driving**（自动驾驶方向，与本文无关）。这是一处**引用错误**，投稿前必须
核对该条目的真实出处。建议对全部 62 条参考文献做一次 ID 核验（可用 /omp:inno-reference-audit）。
