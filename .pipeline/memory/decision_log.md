# Decision Log

## 2026-09-09 — 项目目标定位

**决策**：保留现有标题与核心方法（trigger-CoT 注入 + 相似度检测），通过"更强的论证角度"而非推翻重来的方式提升论文强度；新角度须紧扣 CoT、safety、agent，并鼓励加入理论分析。

**理由**：用户明确要求"不要改动太多标题和方法"，说明沉没成本与叙事连续性需要保留；同时用户认可当前四个短板（实验、创新性、理论、威胁模型）都需处理。

**影响**：survey 阶段的检索重心随之确定为三条主线——CoT 后门/触发、水印检测的理论保证、多智能体安全与信息流。

## 2026-09-09 — 起始阶段选择 survey

**决策**：从 survey 而非直接 experiment 开始，采用全自动模式。

**理由**：四个短板中有三个（创新性、理论、威胁模型）需要先看清竞争格局才能决定怎么修；先跑实验有返工风险。

## 2026-09-09 — 角度收敛：采纳 I1 主体 + I2/I3/I4/I5 部件

**决策**：主贡献为「共形有效的多跳来源检测」（I1）；I2 持钥验证方视角作形式化前提，I3 衰减律实测作
假设验证，I4 定向规避攻击作鲁棒性边界，I5 主动探针作叙事包装。五者构成一篇论文而非五个独立方向。

**理由**：注入机制无法把 CoTGuard 与 BadChain 分开（两者同为黑盒 prompt 层注入），唯一可站住的技术
边界在检测端——持钥验证方 + 可控 FPR + 多跳维度。三条精读线一致指向这一点。

**未采纳的替代**：把可监控性当核心主张（威胁模型相反，会被审稿人指出是两个问题）；宣称多跳传播无人
研究（会被 Prompt Infection Fig.4 反驳）。两者均降级为叙事/引用层面处理。

## 2026-09-09 — 理论定位：共形有效性而非密码学保证

**决策**：Type I 控制建立在**可交换性**（共形校准的经验 p 值）上，明确放弃密码学式的分布无关强保证。

**理由**：水印文献（2404.01245 / 2501.02441 / 2411.13868）的精确零分布全部来自密钥与采样过程的密码学
耦合，需要 token 级白盒访问；CoTGuard 是黑盒 API + 语义相似度，无法继承。伪称强保证一眼可辨，反而
致命。共形校准在黑盒设定下可辩护，且是共形预测的标准论证。

**代价**：必须真实交付一份未注入 trigger 的干净推理轨迹校准集，实验不能省。

**影响**：论文定位语从"watermark with provable guarantee"改为"conformally valid provenance detection"。

## 2026-09-10 — 方法侧收敛：M1（可验证冗余计算证书）+ M2（重复编码锚点）

**决策**：用户要求"生成新方法而非继续偏分析"后，生成 5 个方法候选（全部接在本项目自己的
实验发现上，而非 9 月 9 日纯文献阶段的旧 idea_board），用户选定 M1+M2 优先推进。

**理由**：
- M1 直接把 structural anchor 池中已有的"独立算两遍、核对一致"提升为方法中心，
  给出黑盒条件下机器可核验的证书，绕开 Tr-GoF 对聚合规则的批评，也绕开 BadChain/ShadowCoT
  "防御方拿不到超统计信号"的论证——这是新的能力类别，不是同类检测器的改良。
  且与 D1 直接接盘：structural anchor 池已含此 anchor，D1 生成时天然产出所需数据。
- M2 把 EXP-N2 发现的"前缀弱残留（1.6–1.9×）"从副作用变成设计目标：
  重复放置独立的结构约束，抗轨迹碎片化/截断，且不依赖字面复制——与 Prompt Infection 的
  payload 机制性质不同。

**未采纳（推迟，非放弃）**：
- M3（对比度量学习读出）——能直接回答 EXP-B1 里 TF-IDF 反超学习式 embedding 的疑问，
  但需要新配对训练管线，工作量中等，推迟到 M1/M2 有结果后再看是否需要
- M4（密钥驱动锚点轮换）——需要重做一遍已知非单调的可分离性分析，复杂度上升
- M5（跨 agent 一致性协议）——体裁从 ML 论文偏向系统/协议贡献，且需要"中继 agent 愿意配合
  核验"这个强假设，风险最高，暂不推进

**旧 idea_board.json（I1–I5，2026-09-09）状态**：**作废，非删除**。那套"共形多跳检测理论"
建立在 trigger-sim 相似度检测这个后来被证明近乎随机的读出上，且从未被实验验证。
保留存档供对比，但不再作为规划依据。新方案见 `idea_board_v2.json`。

## 2026-09-10 — Pivot to P2: Byzantine-robust aggregation for multi-agent reasoning

**Decision.** After a robustness audit and a method-ideation pass (`idea_board_v3.json`), the user
chose **P2 (Structurally-Verified Robust Aggregation, SVRA)** as the paper's problem. The CoTGuard
provenance experiments become *motivation* (why instruction-compliance leaves a verifiable structural
signal), not the contribution. Two conditions attached: (1) **replicate the single-shot mechanism
results first** (~1 GPU-day, `replicate_mechanism.sbatch`); (2) **verify prior work in a fresh
session** before freezing the experiment design (`lit_check_queue.md`, Sec. B).

**Why pivot.** The audit showed the readout comparison is replicated (3 models × 3 datasets) but
every mechanism-layer result — length confound, persona=0, 10-hop, attack plateau, prefix-swap —
is Tulu-3-8B × GSM8K × one relay × one generation draw. Meanwhile the provenance *application* is
occupied (White et al. 2606.22698). Adding a method to the same problem (M1/M2) stays adjacent;
changing the problem while reusing the technical assets does not.

**Why P2 over P1.** Both change the problem. P1 (instruction-echo inversion as CoT-backdoor
defense) is faster and has baselines on disk; P2 is the cleaner ICML shape (problem + method +
tolerance bound + grid), lands squarely on multi-agent + safety, and applies a classic
trustworthy-ML technique (Byzantine-robust aggregation). User preference decided.

**Deferred, not discarded.** P1 (backdoor defense), P3 (conformal CoT monitor with shift
detection), P4 (structural watermark = D1 as method), P5 (relay checkpoints). M1's redundancy
verifier and D1's structural routes are absorbed into SVRA as components. M2 (repetition-coded
anchors) is shelved.

**Superseded.** `idea_board_v2.json` (M1–M5 as provenance methods) and `publishable_angle.md`
(direction 1+2, "carrier judgment + maintenance mechanism") are superseded by `p2_design.md`.

## 2026-09-10 — P2 repositioned after S21 (user accepted)

**Decision.** Keep P2/SVRA but narrow the claim to *verified aggregation with no LLM in the loop*:
injection immunity by construction (CPU numeric verifier) + route assignment breaking anonymity
(tolerance governed by f_pass, escaping Consensus Trap's impossibility only for adversaries that
cannot fake a verified trace). Design v1 in `p2_design.md`.

**Why.** S21 (`p2_prior_work.md`) found the v0 framing occupied: Byzantine aggregation for LLM MAS
(SAC, DecentLLMs, CP-WBFT, H-CSC), same threat model + datasets (Consensus Trap), claim-level
LLM verification + exclusion (STAR), structure-over-answers in the honest case (AgentAuditor,
Reasoning Consensus, SC-MoA). Prop 1 equals the MV bound (H-CSC containment lemma). What no one
has: an aggregator whose verification has no LLM, hence no injection surface.

**Consequences.** Main opponents become trace-reading aggregators, not MV (MV ≈ 96% on GSM8K under
minority corruption). New baselines: RRMaj, STAR, AgentAuditor/SC-MoA, DecentLLMs. New adversary
A-collude (colluding, route-aware, white-box) is the pre-registered limit (P3). Route pool
restricted to routes with CPU-checkable signatures, otherwise Prop B fails. Gate G0 (CPU, data on
disk) with three kill criteria precedes any P2 GPU spend.

## 2026-09-10 — SVRA v2: drop route assignment (user chose option 1 after G0)

G0 failed K2: only compute-twice has a CPU-checkable route signature; other structural routes neither
separate (≤ 0.06) nor change what is computed (node coverage 0.925). All agents now get the single
compute-twice obligation; the Consensus-Trap escape is attributed to content-based verification
(non-symmetric, non-outcome-level), not to routes. D1 link dropped. G0-v2 criteria pre-registered in
`p2_design.md` §5.0 and committed before running.
