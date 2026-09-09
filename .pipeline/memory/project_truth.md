# Project Truth
_最后同步：2026-09-09T14:00_

## 研究主题

多智能体 LLM 系统的 CoT 触发式版权保护（CoTGuard）。

基础：预印本 arXiv:2505.19405v1 "CoTGuard: Using Chain-of-Thought Triggering for Copyright Protection
in Multi-Agent LLM Systems"（Yan Wen, Junfeng Guo, Heng Huang, UMD；用户为第一作者）。原方法把
task-specific trigger pattern τ = T(k,t) 注入 agent 的 CoT 提示，使其随推理链在 agent 间传播，再用
相似度检测器对候选轨迹打出 leakage score δ 判定未授权复用。

**目标**：不大改标题与方法框架，寻找更强论证角度；允许并鼓励理论分析；须紧扣 CoT / safety / agent。

## 当前阶段

ideation — 总体进度：2/5 任务完成（S1 survey、S2 角度收敛已完成；S3 进行中）

## 已确认决策

- [2026-09-09] 起始阶段设为 survey，采用全自动模式（四个短板中三个需先看清竞争格局）
- [2026-09-09] 现有预印本实验证据不充分，需补齐真实实验
- [2026-09-09] 目标为保留标题与方法、寻找更强角度，鼓励理论分析
- [2026-09-09] 采纳角度组合 I1（主体）+ I2/I3/I4/I5（部件）
- [2026-09-09] 理论定位选定「共形有效的来源检测」，Type I 控制建立在可交换性而非密码学耦合上
- [2026-09-09] git commit 采用 `type(scope): subject` 格式

## 论文定位（一句话）

CoTGuard 是**共形有效（conformally valid）的推理层来源检测**，并刻画该信号在多智能体串行中继下的
**可检测性视界**。保留标题、trigger-CoT 注入机制、相似度打分基本形态；重构检测端统计基础、威胁模型、
理论陈述。

## 阶段进展摘要

### Survey（已完成）

- 语料：`.pipeline/literature/cotguard-core`，**54 篇真实论文，PDF 与 OCR 均 54/54 成功**
- 覆盖四条线：A) CoT 后门/触发　B) 水印与版权审计的理论保证　C) 多智能体安全与信息流
  D) CoT 可监控性（补充检索，safety 接入点）
- 产出：`literature_bank.md`（24 篇 accepted=yes）、`.pipeline/docs/gap_matrix.md`
- 深度精读三条线共 9 篇核心论文，产出 `digest_attacks.md` / `digest_theory.md` / `digest_agents.md`
  与汇总 `paper_digests.md`

**三条关键发现**：
1. 注入机制无法把 CoTGuard 与 BadChain 分开，但**认识论地位不同**——BadChain 假设触发器对防御方未知
   并据此论证检测不可行；CoTGuard 是**持钥验证方**，问题变为带密钥的假设检验。
2. 多跳空白在攻击线上确凿（三篇论文 "agent" 出现 0 次），但 Prompt Infection Fig.4 已画过 ASR 随
   agent 数量变化，因此贡献必须表述为**被动语义信号的可检测性衰减律**。
3. Tr-GoF **证明** sum-based 规则最好只到 `q+p=1/2`（最优边界为 `q+2p=1`），而 CoTGuard 的
   Algorithm 4 正属此族——既是硬伤也是有理论依据的改进方向。

### Ideation（S2 已完成，S3 进行中）

选定方向（`.pipeline/docs/idea_board.json`）：

- **I1（主体）共形有效的多跳来源检测**：δ 重构为共形校准逐步 p 值 + Tr-GoF 聚合，证明
  `N*(n) = [((1-q)/2)·log n + log ε_0] / log(1/ρ)`，即 `N* = Θ(log n)`
- **I2** 持钥验证方视角 → I1 的形式化前提，解决与 BadChain 的区分
- **I3** 衰减律实测（N=1..8 跳测 ε_N 是否几何衰减）→ 把 I1 的假设 A2 变成实验
- **I4** 定向规避攻击（Guard-Lexicon + Do-Not-Mention + Monitor-Aware 叠加）+ 二维鲁棒性网格
- **I5** 主动探针叙事 → 仅用于 Introduction 框架

### Experiment（未开始）

无。S5 待启动。

### Publication（未开始）

无。

## 当前最佳实验结果

**无**——尚未运行任何真实实验。预印本 Table 1–4 的数值缺乏实际运行支撑，不得作为已验证结果引用。

## 方向调整记录

- [2026-09-09] 起点：预印本框架「trigger-CoT 注入 + 相似度检测 = 版权保护」
- [2026-09-09] 调整为：**共形有效的来源检测 + 多跳可检测性视界**。注入端不变，检测端与理论陈述重构。
- [2026-09-09] 弃用「可监控性作为核心主张」（威胁模型相反），降级为叙事引子
- [2026-09-09] 弃用「多跳传播无人研究」的表述（会被 Prompt Infection 反驳），改述为可检测性衰减律

## 不可违背的红线

1. **不得包装成"有密码学保证的水印"**。水印文献的 Type I 控制源自密钥与采样过程的密码学耦合（需
   token 级白盒访问）；黑盒 + 语义相似度拿不到 `Y_t ~ U(0,1)` 这类免费精确零分布，伪称一眼可辨。
2. **不得宣称"无人研究多跳传播"**。Prompt Infection (2410.07283) Fig.4 已画过 ASR 随 agent 数量变化，
   须在引言主动引用作为最近先例。
3. **不得宣称版权检测是 CoT 可监控性的一个实例**。两者威胁模型相反。
4. **必须报告 FPR**。现有主表只给单点 LDR 不可解释；须交付 ROC/AUC 与干净轨迹经验 FPR。

## 风险 / 阻塞项

| 项 | 严重度 | 说明与应对 |
|---|---|---|
| **A3 步间弱相关假设** | **高** | 推理步高度依赖（第 t+1 步条件于第 t 步），水印文献靠哈希新鲜随机性天然去相关，我们没有。应对：分块打分 + 显式 α-mixing + 坦承理想化 + 经验校准曲线佐证 |
| A1 需真实校准集 | 中 | 共形有效性要求交付未注入 trigger 的干净轨迹校准集，实验不能省 |
| `N* - N*_sum` 系数待核 | 中 | 需按两条相变边界重新代数推导（digest_theory 已标 [存疑]） |
| OCR 公式重建 | 中 | 三篇理论论文数学符号在 pdfminer 输出中错乱，公式由散文重建，投稿前须回原文核对 |
| 预印本引用错误 | 低 | 参考文献 [15] arXiv:2305.18829 实为 UniScene（自动驾驶），需全量核验 62 条引用 |

**当前无阻塞项。**
