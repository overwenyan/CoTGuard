# Project Truth
_最后同步：2026-09-10（ideation pivot to P2）_

## 研究主题

起点：预印本 arXiv:2505.19405v1 "CoTGuard"（Yan Wen, Junfeng Guo, Heng Huang, UMD；用户为第一作者）。
用户 2026-09-10 指示：**不沿用预印本的数字与内容，当作一次从基础开始的新工作；仅承接 topic 与
related works。** 随后进一步指示：参考文章以**方法文章**为主，本工作须产出**方法**而非分析；
须紧扣 LLM + agents + AI safety；应用传统 trustworthy-ML 技术；学校级算力（单卡，无预训练）。

## 当前阶段

**ideation → experiment（pivot 已定，等待两项前置）**

- 论文问题：**P2 — Structurally-Verified Robust Aggregation (SVRA) for multi-agent LLM reasoning**
  （`.pipeline/docs/p2_design.md`）
- 前置 1：复现单次机制层结果（S20，作业 20041000 `cg_repl`，排在 D1 20040940 之后 afterany）
- 前置 2：~~P2 先验文献核查（S21）~~ **已完成 2026-09-10** → `.pipeline/docs/p2_prior_work.md`。
  **结论：P2 定位被部分占据，威胁模型与 baseline 须改；新定位待用户决定。**

## 已确认决策（按时间）

- [09-09] 起始 survey；保留标题方法找更强角度；理论定位"共形有效"（**后被自己的实验废弃**）
- [09-10] 叙事口径：不做"预印本自我证伪"，`trigger_sim` 是"自然的语义匹配基线"
- [09-10] 学习式 embedding 读出**不是**贡献（对 TF-IDF 两平三负）；中心 trick 改为范式选择
- [09-10] 外部文献审阅后撤回 4 处措辞；证据链第 4 环撤回（地板效应）
- [09-10] 方向收敛 1+2（载体判决 + 维持机制）→ 当日被 EXP-C4 证伪两通道假说
- [09-10] 体裁修正：方法论文；D1（结构 vs 措辞）为中心 trick
- [09-10] 方法候选 M1+M2 选定 → **当日被 P2 pivot 取代**
- [09-10] **P2 pivot**：SVRA 为论文问题；CoTGuard 实验降为动机；先复现、先核查文献

## 论文定位（一句话）

> ⚠️ **下面这句已被 S21 否定为新颖性主张**（Byzantine 鲁棒聚合搬到 LLM 多智能体已有 SAC/DecentLLMs/CP-WBFT/
> H-CSC/Consensus Trap；Prop 1 与多数投票同界）。仅保留作历史记录。候选替代口径见 `p2_prior_work.md` §4.1：
> *无 LLM 在环的核验聚合——路线分配 + CPU 数值核验器拿到 trace 级聚合的收益，同时去掉现有核验型/结构型聚合器
> （STAR、AgentAuditor、Reasoning Consensus、SC-MoA、DecentLLMs）暴露的注入面。* **待用户确认。**

> Multi-agent reasoning pipelines are only as trustworthy as their aggregator. We bring
> Byzantine-robust aggregation to natural-language reasoning by aggregating over **structurally
> verified sub-claims** rather than final answers: agents are assigned distinct computational routes,
> a CPU-only verifier extracts and checks their intermediate quantities, and a per-quantity robust
> median assembles the answer. We prove integrity under f < m/2 verified reporters and injection
> immunity of numeric verification, and show on GSM8K/MATH × three generator families that SVRA
> withstands confident-wrong, subtle-assembly, judge-injection and framing adversaries where majority
> vote, LLM-as-judge and debate fail.

## 实验资产（可复用于 P2）

| 资产 | 状态 | 用途 |
|---|---|---|
| 8 条结构路线的诚实轨迹（`attr_families/structural_anchor`, Tulu, GSM8K） | 已有 | SVRA 诚实 agent |
| 同上 Qwen3 / Mistral 版本 | S20 复现批次 | 跨生成器 |
| `verify_redundant.py`（M1 核验器 v1，正则，已修一处假阳性 bug） | 已有，初步验证 0.81 vs 0.31–0.74 | 子断言核验 |
| `trigger_v2.ANCHORS_STRUCTURAL`（8 条路线） | 已有 | 路线分配 |
| 中继/攻击测试床（`relay_attribution.py`, `adaptive_max`） | 已有 | 对手轨迹改写 |
| TF-IDF 读出 + 按题目划分 + 置换 null（`baselines.py`, `null_control.py`） | 已有 | 路线合规读出、null |
| 长度匹配评估（`length_control.py`, `discriminative_probe.py`） | 已有 | 混淆控制 |

## 当前最佳实验结果（诚实版）

**已复现（3 模型 × 3 数据集）**：廉价词法读出 3.0–7.4× 随机且 embedding 从未胜出；无监督
trigger 相似度归因 ≈ 随机（有监督版 ≈ length_only）；指令回声 n-gram 净超出 +0.27~+0.77，
功能词净超出 ≈ 0；密钥空间可分离性非单调。

**单次（Tulu × GSM8K × Qwen3 中继 × 单次生成，S20 待复现）**：长度混淆 0.879→0.543；
persona 无信号、指令有信号（C4）；无零样本外推（R1d）；10 跳保留 88–90%；monitor-aware 攻击
−39% 后平台 ~1.9×；密钥条件化主导前缀（N2）；置换 null 在随机；M1 冗余机制 0.81（部分数据）。

**负结果清单（8 条，全部如实记录于 ledger）**：embedding 读出输给 TF-IDF；共形校准非我们首创；
第 4 环撤回；两通道假说证伪；贪心可分离性反向；persona 零信号；provenance 应用被 White et al.
占据；"接近随机"措辞错误。

## 不可违背的红线（更新）

1. 不得包装成"有密码学保证的水印"。
2. 不得宣称"无人研究多跳传播"。
3. 不得宣称版权检测是 CoT 可监控性的实例。
4. **必须报告 FPR / null control**；置换 null 是标配。
5. **不得把单次结果当已确立机制**——S20 复现前，C4/A1/N2/R0 只能写"在 Tulu×GSM8K 上观察到"。
6. **不得说"功能词 = 零内容"、"任意盆地"、"自适应攻击"**（外部审阅撤回项）。
7. **P2 的先验文献核查未完成前，不得冻结实验设计或写 related work 的 gap 句。**（S21 已完成；
   但因定位须改，**新定位经用户确认前**本条继续生效。）
8. **不得宣称"首次把 Byzantine 鲁棒聚合用于 LLM 多智能体"**，也不得把 f < m/2 完整性界当贡献（S21）。

## 风险 / 阻塞项

| 项 | 严重度 | 说明与应对 |
|---|---|---|
| ~~工具阻塞~~ | 已解除 | cotguard-2 会话 python/WebSearch/sbatch 可用；S20 已排队 |
| **P2 先验工作** | **高（选题层，已证实）** | S21 完成：框架层被占据；剩余空隙 = 注入免疫（CPU 核验）+ 路线分配打破匿名性。Consensus Trap 显示少数腐化下 MV 在 GSM8K 已 96%，SVRA 对 MV 的实证提升空间很小 |
| M1 核验器判据 | 中 | 全量 16 密钥：冗余一致**对数**显著高于全部 15 个对照；但二值"是否出现一致"与 lexical key01 不可区分 → 核验器须用计数/比例阈值 |
| 机制层单次结果 | 中 | S20 复现；若 C4/M1 在 Qwen3/Mistral 上不复现，SVRA 的路线合规读出与冗余核验需重新评估 |
| 核验器 v1 是下界抽取器 | 中 | 正则漏文字数字；LLM 核验器可修但可被注入——转为实验（regex vs LLM verifier under A-infect） |
| 路线约束的 utility 代价 | 中 | 必须报曲线，不得写"utility preserved" |
| FOLIO 无数值中间量 | 低 | P2 先做 GSM8K/MATH；谓词级核验器列为 future work |
