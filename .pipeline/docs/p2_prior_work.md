# P2 / SVRA 先验文献核查（S21）

_2026-09-10，WebSearch + WebFetch 实查。所有 venue 均来自 arXiv comments 字段、会议官网或出版社页面；
凡只有二手来源的一律标「未核实」。WebFetch 返回的是模型摘要，数字以原文为准，引用前需再读原文。_

## 0. 判决（先看这里）

**核心问题「2025–26 有没有人做了针对恶意 LLM agent 的鲁棒聚合 / Byzantine 多智能体」——有，且已成子领域。**
`p2_design.md` 的**威胁模型和 baseline 都必须改**；一句话定位（"We bring Byzantine-robust aggregation to
natural-language reasoning"）**作为新颖性主张不成立**，不得再用。

SVRA 第 3 节五个组成部分逐项对照：

| SVRA 组成 | 是否已被做 | 占据者 |
|---|---|---|
| 把 Byzantine 鲁棒聚合（中位数/GM/加权 BFT）搬到 LLM 多智能体 | **已做** | SAC (EMNLP'26)、DecentLLMs (GM)、CP-WBFT (AAAI)、BlockAgents、H-CSC |
| prompt 级被劫持 agent × GSM8K/MATH500 × 黑盒 | **已做** | Consensus Trap（同威胁模型、同数据集） |
| 句子/断言级核验 → 标记可疑 agent → 排除后投票（3.2+3.4） | **已做（LLM 核验器版）** | STAR |
| 在中间步骤/推理结构上聚合优于答案投票（RQ3） | **已做（诚实 agent、LLM 聚合器版）** | AgentAuditor、Reasoning Consensus、SC-MoA |
| f < m/2 的完整性界（Prop 1） | **无增量**：p=1 时与多数投票同界；H-CSC 证明了理由级核验对多数投票无覆盖优势 | H-CSC containment lemma |
| **CPU 数值核验器 → 对注入免疫（Prop 3）** | **未见** | 所有核验型/结构型聚合器（STAR、AgentAuditor、Reasoning Consensus、SC-MoA、DecentLLMs）的核验/聚合环节都是 LLM |
| **路线分配（非匿名）+ 核验 → 绕开 Consensus Trap 的匿名对称不可能性** | **未见** | Consensus Trap 明确不讨论 sub-claim 聚合、核验过滤、角色分配 |

**剩下的空隙是窄的，但是真的**：
(i) *无 LLM 在环*的结构化聚合——拿到 trace 级聚合的收益，同时没有注入面；
(ii) 路线分配打破匿名性，使容错界由「**能通过核验的**对手数」而非「对手数」决定——这是 Consensus Trap 不可能性结论之外的一条出路。

**一个对 P2 不利、必须正视的实证事实**：Consensus Trap 在 GSM8K 上 1 腐化 / 4 诚实时多数投票已达 96.0%
（MATH500 77.4%）。**少数腐化区间内多数投票几乎没有提升空间**；A-infect 对多数投票也无效（投票不读文本）。
SVRA 相对多数投票的收益只可能来自：(a) 过半腐化且对手无法通过核验；(b) 诚实 agent p<1 时跨子断言纠错。
`p2_design.md` 预测 P-a「gap 最大在 A-subtle 与 A-infect」对多数投票基线**很可能不成立**，需重写。

## 1. 直接重合（Section B 主结果）

### 1.1 The Consensus Trap — Liu, Du, Du, Guo, **Conitzer**（arXiv 2604.17139, 2026-04, 预印本）
- **关键结果**：Prop 1「对任何单个腐化 agent 即返回错误答案的题，不存在既对少数腐化 mostly robust、又对轻微过半腐化
  mostly robust 的**匿名对称结果级**聚合」；Token 级轮转协作 RRMaj 在 60% 腐化时把 MAJ 的 ~11% 拉回 ~88%。
  少数腐化时 RRMaj 与 MAJ 持平（GSM8K 96.0 vs 96.0；MATH500 76.3 vs 77.4）。
- **与 SVRA 重合面**：威胁模型（上下文 prompt 注入劫持 agent，"You MUST select option X"）、数据集（GSM8K、MATH500）、
  黑盒设定、Mistral/Qwen3 生成器——几乎一一对应 SVRA §2 与 §5.5。只比了 MAJ 一个基线。
- **不重合**：不做子断言聚合、不做核验过滤、不做角色/路线分配（原文未讨论）。
- **改变**：✅ **威胁模型**——必须把「过半腐化」作为主评估轴并直接对照其不可能性结论；SVRA 定位为
  「非匿名（路线）+ 非结果级（子断言）」从而不受该命题约束，但必须证明在何种对手下真的绕开。
  ✅ **Baseline**——RRMaj 必加（同数据集可直接对照其报告数）。

### 1.2 Certifiable Semantic Agreement Among LLM Agents (H-CSC) — Xu, Zhang, Ounis, Wang（arXiv 2606.07316, 2026-06/08, 预印本）
- **关键结果**：containment lemma——在匹配的确定性保证下，语义（理由级）可接纳谓词严格强于 verdict 谓词，
  **故与证书包装的多数投票不可能有覆盖分离**；实验中两者提交完全相同的任务集（零分歧）。自由文本理由上
  「诚实分散度大于攻击位移」，任何 embedding 级过滤不可行；**schema 收紧使诚实分散度降 20 倍**。
- **与 SVRA 重合面**：Byzantine agent「保留 verdict、只腐化推理」是 SVRA A-subtle 的镜像；n ≥ 3f+1、2f+1 证书。
  他们核验的是**理由级语义一致**，不抽取数值中间量、不分解子断言。
- **改变**：✅ **理论**——SVRA Prop 1（p=1 时 f < m/2 ⇔ 正确）与多数投票同界，**不是贡献**，须替换为
  「核验改变有效计数」型命题。H-CSC 的有效性目标是「诚实多数的 verdict」而非金标准正确率——SVRA 若要
  主张优势，必须落在**金标准正确率 + 诚实 p<1** 这一 H-CSC 未覆盖的区间。
  其「schema 收紧」发现是 SVRA 路线分配/数值 schema 的**正面旁证**，可引用。

### 1.3 STAR — Luo, Zheng, …, Enhong Chen（arXiv 2605.28104, 2026-05, 预印本；代码已开源）
- **关键结果**：LLM 核验器逐句输出真伪+置信度+更正；累积置信度 τ=0.3 标记可疑 agent；**排除后多数投票**。
  对抗独立/合谋/新提出的**协作式**攻击，任务成功率平均 +36.76%，ASR −45.17%。
- **与 SVRA 重合面**：§3.2（断言级核验）+ §3.4（定位）+ 丢弃未核验 agent 的整体架构**已被做**。
  设置 5 agent 中 2 恶意，MMLU/CSQA/LogiQA，无理论；**未评估核验器本身被注入**，且原文承认 LLM 核验器
  「对真话过度自信、对假话缺乏信心」。
- **不重合**：核验器是 LLM（可注入）；聚合是排除+投票而非逐量中位数；无路线分配；不做数学数值任务。
- **改变**：✅ **Baseline**——STAR 本质上就是「SVRA 用 LLM 核验器」，是 Prop 3 注入免疫主张的**直接对照**，必加。
  ✅ **威胁模型**——必须加**协作式/合谋**对手（对手间交换信息、协调同一个错误中间量），这恰是
  `p2_design.md` §8 风险 4 的困难情形，原设计「对手不知道他人路线」需说明合谋下是否仍成立。

### 1.4 Robust Multi-Agent LLMs under Byzantine Faults (SAC) — Lee, Yun, Panagou, Karimireddy（arXiv 2605.09076, **EMNLP 2026 Main**）
- **关键结果**：全去中心化 filter-and-refine；用接收方评估取代发送方自报置信度（指出**置信加权聚合对自报分数操纵根本脆弱**）；
  给出通信图的 (F+1)-robustness 条件；数学与常识推理基准。
- **与 SVRA 重合面**：Byzantine 容错 + LLM 推理 + 理论条件。SVRA 是中心化聚合器、单轮，不同拓扑设定。
- **改变**：Baseline ◐——置信加权投票的弱点已被 SAC 指出，SVRA 的「confidence-weighted vote」基线应引用 SAC
  而不应作为新发现。SAC 本身为去中心化多轮，可作讨论/附录对照，非必须。

### 1.5 DecentLLMs — Jo & Park（arXiv 2507.14928, 2025-07, 预印本）
- **关键结果**：worker 生成、evaluator 按五维打分，对评分向量取**几何中位数**，容忍 < ⌊(N−1)/2⌋ 恶意 evaluator；
  MMLU-Pro 100 题 71% vs 多数 50%。
- **与 SVRA 重合面**：「在 LLM 多智能体里用几何中位数做 Byzantine 聚合」已被做——但中位数取在**LLM 评分向量**上，
  不在**核验过的中间数值**上；evaluator 为 LLM，可注入。
- **改变**：✅ **Baseline**——实现代价低，作为「经典 Byzantine 聚合规则直接套 LLM」的代表加入。

### 1.6 Rethinking the Reliability of MAS: A Perspective from BFT (CP-WBFT) — Zheng et al.（arXiv 2511.10400；**AAAI**，OJS article 40806，年份按卷推断为 AAAI-26，未逐字核实）
- **关键结果**：置信探针加权 BFT 共识，在 85.7% 故障率下仍保持共识；数学推理与安全评估任务。
- **重合面**：Byzantine 容错 LLM MAS 的另一条线；探针是否需要白盒未核实。
- **改变**：Related work 必引；非必须 baseline（若需白盒则与 SVRA 黑盒设定不可比）。

### 1.7 BlockAgents — (ACM TURC 2024, 已确认)
- 区块链 + proof-of-thought 共识 + 质押选矿工 + 多轮辩论投票。Related work 引用；不改设计。

## 2. 「结构优于答案」一侧（RQ3，均为诚实设定）

### 2.1 AgentAuditor — Yang, Li, Ping, Zhang, Bogdan, Thomason（arXiv 2602.09341, 2026-02, 修订 2026-09-03, 预印本）
- **关键结果**：把多 agent 轨迹组织成推理树，在分歧点做局部核验；ACPO 训练的 LLM adjudicator；
  **标题即「优于多数投票与 LLM-as-Judge」**，GSM8K 上最多 +5.5%。针对相关偏差/虚构，不针对恶意 agent。
- **重合面**：SVRA P-a/P-b 的「优于多数投票与 LLM-judge」叙事在诚实设定下已被占据；RQ3 须限定为**对抗设定**。
- **改变**：✅ **Baseline**——作为「结构感知 + LLM 裁决」代表（GSM8K 有报告数），检验其在 A-infect 下是否崩溃。

### 2.2 Reasoning Consensus — Parulekar, Lee, Hakkani-Tür, Sundaram（arXiv 2607.27783, 2026-07, 预印本）
- **关键结果**：从多条轨迹抽取带类型 DAG，节点用 embedding 过滤 + LLM-judge 匹配合并，按跨轨迹证实数加权；
  异构 4 模型 MuSR-MM +3.1%。无对抗评估；无 GSM8K/MATH。
- **重合面**：与 SVRA §3.3「跨 agent 对齐中间量再聚合」**机制最接近**；区别是节点匹配用 LLM、无对抗、非数值。
- **改变**：Related work 必引，写清差异（数值 CPU 匹配 vs LLM 匹配）；baseline 可选。

### 2.3 Beyond Consensus: Trace-Level Synthesis in MoA (SC-MoA) — Fadnavis, Kanakaraj, Wyss（arXiv 2605.29116, 2026-05, 预印本）
- **关键结果**：「任何基于投票的聚合器满足数据处理不等式，轨迹中有而票中无的信息按构造不可达」；
  LLM 聚合器从**少数派**轨迹拼装正确中间步，全 5 基准胜 SC/MoA/TextGrad。**明确声明未测对抗扰动。**
- **重合面**：为 SVRA「子断言聚合能纠正诚实错误」提供正面理论与实证旁证；但用 LLM 聚合器。
- **改变**：可引作动机（DPI 论证）；其「未测对抗」正是 SVRA 的切入点。

## 3. 攻击侧与 MAS 防御侧（相邻，影响威胁模型措辞与基线预期）

| 论文 | venue | 一句话 | 对 SVRA |
|---|---|---|---|
| MultiAgent Collaboration Attack, Amayuelas et al. (2406.14711) | arXiv 2024，未见 venue | 辩论中说服力决定攻击成败 | 「辩论可被说服者击穿」**已知**，不是 SVRA 的发现 |
| When collaboration fails: persuasion-driven adversarial influence in MAD | **Scientific Reports 2026**（nature.com） | 单个说服型对手使准确率 −10~40%，加 agent/加轮次无效 | 同上；A-wrong 近似其对手 |
| CW-POR (2504.00374) | arXiv 2025 | 置信加权说服覆盖率指标 | 可作指标参考 |
| MAD-Spear, Cui & Du (2507.13038) | arXiv 2025 | 从众驱动的 prompt 注入攻击 MAD；给出 MAD 容错形式定义 | A-infect 的近亲；可复用其攻击作额外对手 |
| Heterogeneous LLM Debate Under Adversarial Peers (2606.19826) | arXiv 2026 | 对手使有害改答率回到 90%；异构同伴有防御价值 | 支持「路线/模型异构」作防御；可引 |
| This Is Your Doge… Wolf, Yoon, Bogunovic (2503.05856) | arXiv 2025 | 单个欺骗 agent 使 MoA AlpacaEval LC 49.2→37.9；提出威尼斯选举式无监督防御 | MoA 场景的鲁棒聚合；related work |
| Prompt Infection, Lee & Tiwari (2410.07283) | 已出版于 Springer 会议论文集（chapter 978-3-032-16092-8_28），**具体会议未核实** | LLM 间自复制注入 | A-infect 的来源引用 |
| Secret Collusion among AI Agents (2402.07510) | **NeurIPS 2024（已确认）** | 隐写合谋 | 合谋对手的理论引用 |
| G-Safeguard / BlindGuard (2508.08127, **ACL 2026 long**) / ARGUS / INFA-Guard / PropGuard / SentinelNet / DynaTrust / Who's the Mole | 多为 arXiv 2025–26；BlindGuard 已确认 ACL 2026 | 图/GNN/信用/激活的恶意 agent 检测与定位 | RQ2（定位）的 related work；BlindGuard 无监督，可选作定位基线 |
| Randomized Smoothing for LLM MAS (2507.04105) | arXiv 2025 | MAS 的认证鲁棒 | 理论 related work |
| Combating Adversarial Attacks with MAD, Chern et al. (2401.05998) | arXiv 2024，未见 venue | 辩论降低越狱毒性 | 仅 related work |

未查：MultiAgentBench（队列中列出，本次未检索）。

## 4. 对 `p2_design.md` 的必改项（**本文件只列出，不改设计——设计冻结需用户决定**）

1. **§0 / project_truth 一句话定位**：删去「bring Byzantine-robust aggregation to NL reasoning」作为新颖性。
   可行的候选口径：*"Verified aggregation without an LLM in the loop: route-assigned agents and a CPU-only
   numeric verifier recover the gains of trace-level aggregation while removing the injection surface that every
   existing verified or structural aggregator (STAR, AgentAuditor, Reasoning Consensus, SC-MoA, DecentLLMs) exposes."*
2. **§2 威胁模型**：加 (a) 过半腐化 f ≥ n/2（Consensus Trap 轴）；(b) **合谋/协作**对手（STAR、MAD-Spear Sybil）；
   (c) 明确「对手不知他人路线」在合谋下是否成立；(d) 对手可读核验器规则的白盒变体（原 grey-box 保留）。
3. **§4 理论**：Prop 1 删或降为 remark（与多数投票同界，H-CSC 已说明无优势）。替换为：
   容错由 **f_pass（通过核验的对手数）< m/2** 决定，并刻画哪类对手必然 f_pass < f
   （例如：中间量可由题面数字 + 算术复算核验时，只有「选错运算」类语义错误能通过）。
   另需一条正面命题说明 SVRA **不是**匿名对称结果级机制，因此不受 Consensus Trap Prop 1 约束，并给出反例场景。
4. **§5.3 基线**：新增 **RRMaj（必）**、**STAR（必）**、**AgentAuditor 或 SC-MoA（必选其一）**、**DecentLLMs GM（必，便宜）**；
   保留 MV、置信加权（引 SAC）、LLM-judge、辩论、答案级中位数消融。
5. **§5.6 预测**：P-a 改写——少数腐化下对 MV 的增益预期很小（Consensus Trap：MV 在 1c4t GSM8K 已 96%）；
   新主预测应落在 (i) A-infect 下 **STAR / AgentAuditor / LLM-judge 崩、SVRA 不变**；
   (ii) 过半腐化且对手不能通过核验时 SVRA 维持、MV/RRMaj 的对照；
   (iii) 诚实 p<1（Tulu 0.62）时子断言中位数相对 MV 的金标准增益。
   **新自证伪条件**：若 STAR 在 A-infect 下不崩（其 LLM 核验器实际抗注入），注入免疫主张无实证价值。
6. **红线 7 仍有效**：在用户确认新定位前，不冻结实验设计、不写 related work 的 gap 句、不写 P2 实验代码。

## 5. Section A — venue 核实结果

| 论文 | 原标注 | 核实结果 | 状态 |
|---|---|---|---|
| In-Context Watermarks (Liu, Zhao, Kruegel, Song, Bu) | ICLR 2026 | **ICLR 2026 poster**（iclr.cc/virtual/2026/poster/10008261；官方 repo 标 [ICLR2026]）。另见 icml.cc/virtual/2025 条目，疑为 workshop 版，未核 | ✅ 已确认 |
| Baker et al., Monitoring Reasoning Models for Misbehavior… | arXiv 2025 | arXiv 2503.11926；dblp 仅收 CoRR | ✅ 预印本 |
| Korbak et al., CoT Monitorability | 可能无 venue | arXiv 2507.11473；仅见 workshop 报告（San Diego Alignment Workshop），无存档 venue | ✅ 预印本（position） |
| Chen et al., Reasoning Models Don't Always Say What They Think | venue 未知 | arXiv 2505.05410；有二手说法称 NeurIPS 接收，**未找到一手来源** | ☐ 未核实，按 arXiv 引 |
| Emmons et al., When CoT is Necessary… | venue 未知 | arXiv 2507.05246；未见 venue | ✅ 预印本 |
| Tr-GoF (Li, Ruan, Wang, Long, Su) | JASA? | **JRSS-B** 88(2):491–515, 2026-04（Oxford Academic）。**不是 JASA**，原标注错 | ✅ 已确认（更正） |
| White, Jafari, Berg-Kirkpatrick, Black-Box Forensics | 预印本 | arXiv 2606.22698，2026-06-21，cs.CR，无 venue | ✅ 预印本 |
| Chen et al., Do System Prompts Leave Behavioral Fingerprints? | 预印本 | arXiv 2608.24461，2026-08-25，14 页，无 venue | ✅ 预印本 |
| AgentPoison | NeurIPS 2024 | NeurIPS 2024 main（proceedings 已收） | ✅ 已确认 |
| AgentDojo | NeurIPS 2024 D&B | NeurIPS 2024 Datasets & Benchmarks（proceedings 已收） | ✅ 已确认 |
| Thought Anchors (Bogdan, Macar, Nanda, Conmy) | venue 未知 | arXiv 2506.19143；引用方称 NeurIPS 2025 Mechanistic Interpretability **Workshop**（二手） | ◐ 部分核实（workshop，二手来源） |
| Persona Vectors (Chen, Arditi, Sleight, Evans, Lindsey) | venue 未知 | 未找到 venue；arXiv 号本次未核 | ☐ 未核实 |

**一处与 ledger 不一致，需读原文确认**：ledger「证据链第 4 环撤回」一节写 White et al.「靠聚合 10 组对话恢复到 0.94」；
本次抓到的摘要写「**50** conversations per target agent → 0.943 AUC」。可能是不同设置，也可能 ledger 记错。
