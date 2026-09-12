# Project Truth
_最后同步：2026-09-10（cotguard-2：S21、SVRA v2、G0/G0-v2、D1 判决）_

## 研究主题

起点：预印本 arXiv:2505.19405v1 "CoTGuard"（Yan Wen, Junfeng Guo, Heng Huang, UMD；用户为第一作者）。
用户 2026-09-10 指示：**不沿用预印本的数字与内容，当作一次从基础开始的新工作；仅承接 topic 与
related works。** 随后进一步指示：参考文章以**方法文章**为主，本工作须产出**方法**而非分析；
须紧扣 LLM + agents + AI safety；应用传统 trustworthy-ML 技术；学校级算力（单卡，无预训练）。

## 当前阶段

**experiment：M2（黑盒共谋检测）；P2/SVRA 与 M1 均已被各自预注册门槛关闭（2026-09-11）**

- 方法候选深度调研：`.pipeline/docs/refs/deep_research_2026-09-11_methods.md`（~40 次实时检索）；候选板 `idea_board_v4.json`（M1–M6）。
  首推 **M1 同伴条件化共形预测**（Hu & Su 2609.04445 留下的防御空白，代码数据公开）与 **M2 黑盒共谋检测**（NARCBench 公开，唯一强检测器是白盒）。
- **P2/SVRA 已关闭（2026-09-11，G1 预注册门槛：两格两生成器全败；LLM judge 在 f=3 仍 0.86/0.90，朴素注入只让它掉 4 点）。**
- **M1 已关闭（2026-09-11，G-c 败）**：可复现 Hu & Su 的崩塌、有效、比其 oracle 少升级 13.7 点，但在诚实同伴下从不比「先答不看同伴」更省；副产品：SOLO 基线支配其两种修复（分析结果）。
- **M2 已结束为「分析 + 基准批评」（2026-09-11，Gate B 败）**：结构匹配对照下黑盒检测全随机；Stego 可被长度单独分开；Core 主要靠公开投票；目标域校准修得了 FPR 修不了检测力。
- **当前：M3 prompt-only 签名能否被蒸馏继承**（顺序中的下一个）。

- 论文问题：**P2 — Structurally-Verified Robust Aggregation (SVRA) for multi-agent LLM reasoning**
  （`.pipeline/docs/p2_design.md`）
- 前置 1：~~S20 复现~~ **已完成**（20041000）→ 2 条复现、2 条重述、2 条撤回；SVRA 依赖的 M1 三模型复现
- 进行中：SVRA 诚实 agent 生成（20041095，Tulu + Qwen3；Mistral 因 G0-v2 未过暂不纳入，待用户决定）
- 前置 2：~~P2 先验文献核查（S21）~~ **已完成 2026-09-10** → `.pipeline/docs/p2_prior_work.md`。
  **结论：P2 定位被部分占据** → 用户接受收窄定位（v1）→ G0 未过 → 用户选方案 1 去掉路线（v2）→ G0-v2 刀刃通过（k=1）。

## 已确认决策（按时间）

- [09-09] 起始 survey；保留标题方法找更强角度；理论定位"共形有效"（**后被自己的实验废弃**）
- [09-10] 叙事口径：不做"预印本自我证伪"，`trigger_sim` 是"自然的语义匹配基线"
- [09-10] 学习式 embedding 读出**不是**贡献（对 TF-IDF 两平三负）；中心 trick 改为范式选择
- [09-10] 外部文献审阅后撤回 4 处措辞；证据链第 4 环撤回（地板效应）
- [09-10] 方向收敛 1+2（载体判决 + 维持机制）→ 当日被 EXP-C4 证伪两通道假说
- [09-10] 体裁修正：方法论文；D1（结构 vs 措辞）为中心 trick
- [09-10] 方法候选 M1+M2 选定 → **当日被 P2 pivot 取代**
- [09-10] **P2 pivot**：SVRA 为论文问题；CoTGuard 实验降为动机；先复现、先核查文献
- [09-10] **S21 后收窄定位（用户接受）**：无 LLM 在环的核验聚合；主对手从 MV 换成读轨迹的聚合器；
  威胁模型加过半腐化与合谋/路线知情对手；design v1
- [09-10] **G0 未过 → 用户选方案 1**：去掉路线分配，全体 agent 统一 compute-twice 义务；design v2；G0-v2 以 k=1 通过（余量在噪声内）
- [09-10] **D1 判决**：结构 anchor 不比词法更抗擦洗，设计主张证伪；P4（结构水印）关闭

## 论文定位（一句话）

**现行（design v2，`p2_design.md`）：**

> Trace-reading aggregators (LLM-judge, STAR, AgentAuditor, SC-MoA, DecentLLMs) beat vote counting
> but put an LLM between adversarial text and the decision. SVRA gets trace-level information with
> **no LLM in the loop**: every agent is obliged to compute each intermediate twice, and a CPU
> verifier re-computes, grounds and redundancy-checks each agent's committed arithmetic before a
> per-quantity plurality over verified reporters. It is injection-immune by construction, and because
> the rule weighs responses by verified content (neither symmetric nor outcome-level) its tolerance is
> set by the adversaries that *pass verification* (f_pass) — escaping the Consensus-Trap impossibility
> for adversaries that cannot fake a verified trace, and, stated up front, not for those that can.

**作废（S21 否定为新颖性主张，仅留历史）：**

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
| compute-twice 义务轨迹（`attr_families/structural_anchor__key01`, Tulu, GSM8K，每题 1 条） | 已有 | G0-v2；SVRA 聚合需每题 n=7 条，**须新生成** |
| 同上 Qwen3 / Mistral 版本 | S20 复现批次 | 跨生成器 |
| `svra_verifier.py`（V1 算术 / V2 落地 / V4 闭合 / V3 冗余计数，17 条自测）+ `svra_g0.py` / `svra_g0v2.py` | 已有 | SVRA 核验器 |
| 中继/攻击测试床（`relay_attribution.py`, `adaptive_max`） | 已有 | 对手轨迹改写 |
| TF-IDF 读出 + 按题目划分 + 置换 null（`baselines.py`, `null_control.py`） | 已有 | 路线合规读出、null |
| 长度匹配评估（`length_control.py`, `discriminative_probe.py`） | 已有 | 混淆控制 |

## 当前最佳实验结果（诚实版）

**已复现（3 模型 × 3 数据集）**：廉价词法读出 3.0–7.4× 随机且 embedding 从未胜出；无监督
trigger 相似度归因 ≈ 随机（有监督版 ≈ length_only）；指令回声 n-gram 净超出 +0.27~+0.77，
功能词净超出 ≈ 0；密钥空间可分离性非单调。

**S20 复现后（2026-09-11，详见 ledger「S20 总判」）**：
- ✓ 复现：**M1 compute-twice 冗余**（Tulu/Qwen3/Mistral）；**N2 密钥主导前缀**（Tulu/Qwen3）。
- ◐ 重述：C4 →「无独立文体通道，信号 = 对 prompt 原词的字面回声，回声对象因模型而异」；
  A1 →「首跳 −40% 复现，后续衰减取决于中继模型（Qwen3 平台 ~2×；Mistral 5 跳降至随机，代价 −18 点正确率）」。
- ✗ 撤回为一般结论：R0「控制长度后语义相似度 ≈ 随机」（9 对伪影）；S12/B2「10 跳保留 88–90%」（Mistral 中继 5 跳仅 35%）。
- 仍为单次：R1d、C5、D1、G0。

**负结果清单（10 条，全部如实记录于 ledger）**：**D1 结构 anchor 不比词法更抗擦洗（设计主张证伪）**；**G0 结构路线不可核验、不改变计算**；embedding 读出输给 TF-IDF；共形校准非我们首创；
第 4 环撤回；两通道假说证伪；贪心可分离性反向；persona 零信号；provenance 应用被 White et al.
占据；"接近随机"措辞错误。

## 不可违背的红线（更新）

1. 不得包装成"有密码学保证的水印"。
2. 不得宣称"无人研究多跳传播"。
3. 不得宣称版权检测是 CoT 可监控性的实例。
4. **必须报告 FPR / null control**；置换 null 是标配。
5. **不得把单次结果当已确立机制**（S20 已完成：见上方记分卡；R1d/C5/D1/G0 仍须限定）。
   **凡涉及中继的鲁棒性结论，必须把中继模型当自变量报告**；不得再写「多跳不衰减」「攻击削弱但不消除」为一般结论。
6. **不得说"功能词 = 零内容"、"任意盆地"、"自适应攻击"**（外部审阅撤回项）。
7. ~~P2 先验核查完成前不得冻结设计~~（S21 完成、新定位已确认）。**替换为**：G0 CPU 可行性门槛
   （`p2_design.md` §5.0，三条 kill 判据）通过前，不得花 GPU 在 P2 上，也不得冻结 v1 网格。
   gap 句只能按 `p2_prior_work.md` 的空隙写（注入免疫 + 按内容核验打破对称性；**不再提路线**）。
8. **不得宣称"首次把 Byzantine 鲁棒聚合用于 LLM 多智能体"**，也不得把 f < m/2 完整性界当贡献（S21）。
9. **不得宣称 SVRA 在少数腐化下优于多数投票**（P4 已预注册为无增益），也不得隐藏 A-collude 下的退化（P3）；
   过半腐化结果必须**准确率与覆盖率成对报告**（G0-v2：诚实通过率仅 0.45）。
10. **不得宣称结构 anchor 抗擦洗**（D1 证伪），也不得说结构路线「改变了计算顺序」（G0）。

## 风险 / 阻塞项

| 项 | 严重度 | 说明与应对 |
|---|---|---|
| ~~工具阻塞~~ | 已解除 | cotguard-2 会话 python/WebSearch/sbatch 可用；S20 已排队 |
| **P2 先验工作** | **高（选题层，已证实）** | S21 完成：框架层被占据；剩余空隙 = 注入免疫（CPU 核验）+ 路线分配打破匿名性。Consensus Trap 显示少数腐化下 MV 在 GSM8K 已 96%，SVRA 对 MV 的实证提升空间很小 |
| G0 未通过（K2） | 已处理 | 用户选方案 1，v2 去掉路线 |
| **G0-v2 刀刃通过** | **高（方法层）** | k=1 误拒 0.494 vs 阈值 0.5（SE≈0.056）；S20 后须在 Qwen3/Mistral 重测，可能翻转 |
| **覆盖率** | **高** | 诚实通过率 0.45 → n=5,f=3 时约 30% 题无诚实通过者；主设置考虑 n=7 |
| M1 核验器判据 | 中 | 全量 16 密钥：冗余一致**对数**显著高于全部 15 个对照；但二值"是否出现一致"与 lexical key01 不可区分 → 核验器须用计数/比例阈值 |
| 机制层单次结果 | 中 | S20 复现；若 M1 冗余在 Qwen3/Mistral 上不复现，SVRA v2 的 V3 义务失效 |
| 核验器 v1 是下界抽取器 | 中 | 正则漏文字数字；LLM 核验器可修但可被注入——转为实验（regex vs LLM verifier under A-infect） |
| 义务的 utility 代价 | 低–中 | Tulu 上 compute-twice 0.79 vs clean 0.73（单次）；跨生成器须报，不得写"utility preserved" |
| FOLIO 无数值中间量 | 低 | P2 先做 GSM8K/MATH；谓词级核验器列为 future work |
