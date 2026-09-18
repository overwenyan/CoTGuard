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

## 2026-09-11 — P2 / SVRA stopped by its pre-registered gate G1

G1 (commit 8823973) failed on both generators in both cells: the naive aggregator-note injection moved
the LLM judge only −4.3 pp (threshold −5) and the STAR-style verifier not at all, and at f=3 the LLM judge
(0.86 / 0.90) beat SVRA (0.575 / 0.515) by 28–38 pp. Decision rule → STOP P2. Reusable assets move to
M2/M4. Work continues on M1 (peer-robust conformal decisions; job 20041292 running), per the user's
"sequence by sequence" instruction.

## 2026-09-11 — M1 stopped by its pre-registered gate (G-c); move to M2

M1 (m1_design.md, 0d08f65) reproduced Hu & Su's collapse (R0), was valid (G-a) and escalated 13.7 pp less than
their oracle fix (G-b), but under honest peers never escalated less than simply deciding before reading peers
(G-c fails: 0.39–0.44 vs 0.384). The strongest robust rule in this setting is "answer solo first"; the price of
worst-case robustness is ≈7 pp extra escalation under honest peers. Recorded as an analysis result. Next in the
user's sequence: M2 (black-box collusion detection with exact FPR control).

## 2026-09-11 — M2 ends as a benchmark-critique result; move to M3

Gate B (pre-registered before data): collusion vs script-matched open advocacy is at chance on all three
generators (comb 0.452-0.511, lex 0.494-0.518, votes 0.412-0.484, LLM judge 0.502-0.594), while collusion
vs the NARCBench-style homogeneous control passes on all three (0.754-0.981) with manipulation checks clean.
Plus: NARCBench-Stego separable by message length alone (1.000); Core largely visible from public votes;
in-domain honest calibration fixes FPR but yields zero power under domain shift. Recorded as analysis +
benchmark critique. Next in the user's sequence: M3 (does a prompt-only signature survive distillation).

## 2026-09-11 — M3 passes: passive prompt-only signatures ARE radioactive

Students LoRA-trained on (plain problem -> keyed teacher trace), never shown the key, are attributed to
their training key at 0.75 mean (chance 0.25) on both 1.5B and 0.5B, with length-only and permutation
nulls clean. Caveat found in self-audit: the untuned base model is itself classified as k0
(0.705/0.645), so under the stricter null (max of clean student and base model) k1 and k5 stand and k0
does not. Contradicts my N2-based prior. Direction: prompt-only provenance for distilled reasoning
models, complementary to ACL 2026 trace rewriting (which modifies teacher outputs).

## 2026-09-12 — M3 v2 passes; M3 becomes the paper direction

Owner-side test over 16 keys: all 6 trained keys reach p = 1/16 (the floor) on both Qwen2.5-1.5B and
Llama-3.2-1B, all beat the strict clean+base null, empirical FPR 0.016 over 304 untrained pairs.
Robust to 150 traces and to 50% mixed data. The active trigger baseline (ACL 2026 style) reaches
0.14-0.24 per query but requires rewriting teacher outputs; the two routes are complementary.
Next: write-up plus robustness (paraphrase/filter attacks, more teachers, non-math domains).

## 2026-09-12 — M3 v3 robustness pre-registered (c500e86), running 20044642→43→44
- Found while designing v3: v2's "FPR 0.016" is implied by the rank construction and validates nothing.
  Validity must come from random key assignment; v3 draws a 64-key pool from the trigger_v2 generator
  and adds a per-key conditional check (a positive must not be top-3 on clean/base students).
- Scope: second teacher (Qwen2.5-7B), non-math domain (ARC-Challenge), distiller attacks (filter / para
  / compress) with attack-agnostic and attack-aware read-outs. G-R0 failing stops M3 before writing;
  G-R1 (paraphrase) only decides the wording of the robustness claim.

## 2026-09-13 — M3 v3 complete
- G-R0 12/16, G-R2 16/16, G-R3 14/16 (manipulation check fixed to per-arm teacher length after seeing
  output, fe0b2e0; pooled reading was void — user to decide reporting), G-R1 paraphrase FAIL 8/16.
- Per pre-registration: claim scoped to distillation on unmodified or filtered traces; paraphrase is the
  stated limitation. Candidate v4 (not started, needs user): content-adding anchors survive paraphrase.

## 2026-09-14 — v5 matched replication infeasible (3 pairs < 6); stopped per pre-registration
- On a fresh 40-instruction bank, format-only (PRES) instructions again had much lower teacher
  separability (median 0.52 vs 0.99) and shorter traces (578 vs 831 chars) than content-adding (OP)
  ones. So category is not identifiable apart from separability and length using natural instructions.
- Consequence: H-OP (stage 1) is reported as a practical rule (content-adding instructions are both more
  distinctive and more rewrite-robust), not as a causal effect of category.
- Dilution failed at 1/5/10% and query scaling did not recover it. The provenance use case at realistic
  mixture levels is not supported.

## 2026-09-14 — Expert decision: close the provenance-method track; write an empirical paper; one bounded replication
- **Framing:** empirical study of behavioural inheritance vs source attribution. Target ARR (Findings as the
  realistic objective), workshop fallback. Do NOT run D3 (full-FT repair) or D4-as-planned to keep a
  method paper alive.
- **Story order:** transfer is detectable → nominal keys collide → independent sources imitate the
  signature → rewriting and dilution limit identification → stronger signatures may cost utility.
  OP/PRES is secondary: "an observed association in the tested banks", not a design rule.
- **Wording fixes adopted:**
  - "distinguishability of the tested generator is dominated by its 12 reasoning instructions" (no
    general capacity claim; the v4/v5 banks are separate codebooks);
  - "detection became substantially more reliable at higher mixture fractions in the tested four-key
    configurations" (no universal threshold);
  - "outputs exhibit a signature associated with instruction k, consistent with transfer from
    instruction-conditioned traces";
  - separate owner-specific vs instruction-family endpoints;
  - lexical dependence is supported by converging evidence, not by the ablation alone;
  - Asking Back = "a small set of related behavioural markers, without many-key attribution; response
    rewriting before training not evaluated".
- **D3** full-FT cell: excluded from substantive conclusions; disclosed in the appendix inventory.
  **D5** checklist read-out: dropped from main evidence unless human labels become available.
  **D7** utility: report prominently; cause inspected with existing data (below). **D8** ARC: void +
  sensitivity.
- **Stop list:** hybrid active watermark, searching for better banks, scaling the student for its own
  sake, full-FT repair for venue reasons.
- **Stopping rule:** completion of the frozen replication matrix (m3_design.md v7), not whether results
  look publishable. Afterwards stop expanding unless an implementation error invalidates a central
  result.

## 2026-09-14 — Second outside review adopted; teacher-identity read-out run (exploratory)
- User selected: the teacher-identity analysis, three formal propositions, restructure + wording
  fixes. New-prior-work verification (Gu et al., DITTO, Unified Attacks) deferred.
- **Result:** teacher identity is recoverable from student outputs (AUC 1.00 in all 4 family × codebook
  cells; held-out instructions). Central claim becomes: *the key-based attribution procedure is not
  source-specific even though source information is present in student outputs* — not "provenance is
  unidentifiable". Closed-set (two teachers) only.
- Access model: only **known-instruction replication** was evaluated. Accidental instruction
  collision and instruction reconstruction are not evaluated and must not be claimed.

## 2026-09-15 — Advisor decision: Hybrid A+C (unified analysis paper + targeted scaling); B only as kill-early appendix
- **Paper:** "What distillation-provenance signals identify — and what they do not: a two-level
  identifiability gap in reasoning distillation." ARR, Interpretability & Analysis track.
- **Must cite and differentiate:** the ambiguity/invertibility-attack tradition (Craver et al. 1998, IEEE
  JSAC; Fan et al. 2019, NeurIPS passports; Watermark Stealing, ICML 2024; DITTO, EACL 2026) and
  Subliminal Learning's same-base requirement. Our novelty is the cross-level empirical recurrence in
  reasoning distillation, plus a TV identifiability condition with a read-out-invariance term (stated
  honestly as a specialisation of known bounds).
- **Priority experiments (C):**
  1. same-lineage alignment-stage ladder;
  2. long-CoT replication;
  3. truly-unseen-teacher open-set protocol with leave-family-out, strict FPR as the headline.
  Appendix, kill-early: an impersonation / adaptive distiller, and a B-style stylistic-token
  discriminator.

## 2026-09-15 — CORRECTION: the "Tulu-SFT" teacher in M5 (and "Tulu-3-8B" everywhere) is the final RLVR model
- The model card of `allenai/Llama-3.1-Tulu-3-8B` lists it as **Final Model (RLVR), finetuned from
  allenai/Llama-3.1-Tulu-3-8B-DPO**. The SFT checkpoint is a separate model
  (`allenai/Llama-3.1-Tulu-3-8B-SFT`).
- M5's "same-lineage SFT vs DPO indistinguishable" is therefore **RLVR-final vs DPO (adjacent
  stages)**. The status prompt and the advisor response inherited the wrong label. Experiments are
  unaffected; labels in docs and the paper must say "Tulu-3-8B (RLVR, final)".
- **Availability checked:** Tulu-3-8B-SFT and the OLMo-3-7B Instruct and Think ladders (SFT / DPO / RL)
  are all public. meta-llama/Llama-3.1-8B (base) is gated; the OLMo-3-1025-7B base is already cached.

## 2026-09-15 — User: ladder first (+ vLLM setup in parallel); lineage verification and dissociation check done
- The base_model chains of all 9 ladder checkpoints were verified from their model cards (see the m6
  design).
- Dissociation check (EXP-M5b): the RLVR-sourced Llama student is 3.1 points less accurate than the
  DPO-sourced one (p = 0.024), while identity read-outs confuse them 100% → a capability-vs-identity
  dissociation, in the reverse direction from the prediction.

## 2026-09-15 — EXP-M6 ladder result (pre-registered P1–P3 pass on reduced sets; exploratory analysis reframes P1)
- Pre-registered verdict: P1, P2 and P3 all pass. The manipulation check voided tulu_sft, olmot_sft
  and olmot_final, so P2(ii) rests on 1 line and P3 on 1 stage. Sensitivity with no voids: P1 10/12,
  P2 2/3 lines, P3 3/3.
- **Substantive reading:**
  - Per-output distinguishability is **step-specific**. SFT→DPO moves style (AUC 0.93–0.99).
    DPO→RL/final barely does (0.72–0.76) for Tulu and OLMo-Instruct; Think-RLVR is the exception
    (0.96, part length/format).
  - Owner-test FPR = 1.0 for relatives follows from out-of-line calibration.
- **Exploratory:** a pairwise student-level test separates every pair perfectly (student AUC 1.00,
  LOO 0/10), including DPO vs RLVR.
  - The "same-lineage collapse" is therefore a reference-availability limit, not an information limit.
  - This amends the M5 wording, and puts RefDistDet (2607.09692) squarely in positioning.
- The capability-identity dissociation from M5b does **not** replicate cleanly: the RLVR-vs-DPO
  student accuracy gap is +0.029 [0.000, 0.057] for Qwen and −0.012 (n.s.) for Llama.
- Long-CoT note for the next round: 1–1.5B LoRA students fail to absorb 4–7k-char Think traces
  (accuracy below base) → the long-CoT round needs a larger student or full fine-tuning.
- Open decision → user.

## 2026-09-16 — EXP-M7: H1, H2 and H3 all pass; the same-lineage limit is a calibration limit
- On fresh students trained on fresh traces for disjoint problems, T0 (calibration on other lines only)
  flags same-line relatives at FPR 0.9–1.0 for 8 of 12 ordered pairs in every cell, while T1 (adding
  reference students of each relative) holds FPR_rel ≤ 0.2 in 10–12 of 12 with TPR 1.0 everywhere.
- The leakage diagnostic came back 0 error in both the shared-trace and the fresh condition → M6's
  exploratory separation was real, not an artefact of shared traces.
- Budgets: 3 reference students and 25 probe queries already suffice.
- Capability transfer is unreliable: 19 of 24 pairs keep the teacher's sign, but 4 significant
  reversals occur where the teacher's traces are long (OLMo-Instruct DPO/final on MATH).
- Paper consequence: the headline becomes the two-level gap **plus** its resolution — provenance tests
  answer the question their calibration defines; naming a checkpoint requires same-lineage references.
  Update `paper_plan_unified.md` to the "H2 passes" branch (title candidate 3).

## 2026-09-16 — EXP-M8 result and advisor round 4: reframe, then run the adaptive distiller
- **M8 outcome:** P1b killed (protection is per-relative; the owner must enumerate its lineage);
  P2 short of its gate (out-of-line calibration cannot be dropped). The exploratory T2g shows the
  failure is a *coverage gap*: distant unreferenced relatives are rejected (FPR 0.0), adjacent ones
  are not (1.0).
- **Advisor (accepted):**
  1. Reframe the paper as **diagnosis + scoped remedy + honest boundary**, not a method paper, and
     split the regimes explicitly: T1 is a **first-party/vendor** capability (a vendor knows its own
     ladder); A1/A2 are the **third-party/auditor** regime.
  2. Pre-empt "this is cohort normalisation rediscovered" with five citations: Auckenthaler et al.
     2000 (T-norm), Koppel & Winter 2014 (impostors method), Scheirer et al. 2013 (open-set
     recognition), Bates et al. 2023 (conformal outlier p-values, exchangeability), Maini et al. 2024
     (LLM dataset inference), plus RefDistDet. Novelty delta must be stated in two sentences.
  3. **Next experiment: the adaptive distiller** (paraphrase + imitate-a-relative). In this subfield an
     adaptive attack is near-mandatory; its absence reads as a gap, not a scoping choice.
  4. Capability dissociation → paragraph + appendix. NOTE (corrected 2026-09-16, advisor round 5): Li et al.
     2502.12143 and Xu et al. 2411.07133 support only the long-trace learnability gap behind our accuracy
     reversals, NOT a capability-vs-identity dissociation. Do not cite them for the dissociation claim.
  5. Add a coverage-dependent corollary to Proposition 1 (conformal exchangeability + Le Cam
     two-point). M8's T2g result is its empirical verification.
  6. Cross-vendor ladder (Zephyr on Mistral, SFT→DPO only) as the second experiment if budget allows;
     no non-AllenAI vendor publishes a verifiable base→SFT→DPO→RL chain.
- **Decision:** M9 = adaptive distiller, pre-registered before data, on the two least separable
  adjacent pairs. Writing stays blocked until M9's gate is known.

## 2026-09-16 — EXP-M9: the adaptive imitation attack breaks T1 (G2 and G3 fail)
- G1 passes: T1 survives neutral paraphrase (mean TPR 0.92 / 0.83).
- G2 and G3 fail on the valid corpora: targeted imitation of a relative drops the owner's detection to
  a mean of 0.5 (0.0 for olmoi_dpo) and makes the innocent relative claim the student at 1.0.
- Two AD2 corpora were voided by the pre-registered answer-preservation check and excluded from the
  gates (reported).
- **Consequences, as pre-registered:** targeted imitation becomes the paper's primary limitation, in
  the abstract; the title must foreground the non-adaptive / first-party scope; G3 is written up as an
  ambiguity attack (Craver et al. 1998; Fan et al. 2019; watermark stealing).
- The scope of the positive claim is now fully bounded by our own experiments: T1 works for a vendor
  that knows its lineage, references every relative, keeps out-of-line calibration, and faces a
  distiller that does not imitate a relative.
- Open choice → user: start writing with this evidence, or run the "adaptive owner" defence round
  (reference students trained on rewritten traces, so the owner anticipates the attack).

## 2026-09-16 — Integrity correction (answer extractor), M10 passes, M9b replicates the attack
- **Extractor bug:** `utility_check.extract_answer`'s fallback skipped numbers followed by a period, so
  accuracy was underestimated by 0.03–0.13 across the project. v1 kept for provenance; `answer_v2.py`
  (unit-tested) used from now on, with v1 shown alongside in an appendix.
- **Retracted:** (1) the M5b capability–identity dissociation (its only significant result reverses
  sign and loses significance under v2); (2) "RLVR less accurate than DPO on GSM8K"; (3) "utility cost
  is student-dependent" (Llama also loses 7–9 points under v2). **Kept:** Qwen −12 points; every
  attribution result (T0/T1/AUC never used the extractor); M6/M7 voids; M9 gates (unchanged under v2).
- **M10 passes R1–R3:** the calibration failure and the reference-aware remedy replicate on a
  non-AllenAI ladder (Zephyr on Mistral), both directions. The "AllenAI-specific" objection is closed
  for the SFT→DPO step.
- **M9b:** evasion and spoofing replicate on valid corpora (G2b/G3b fail as expected). No direction rule
  and no mechanism: both pre-registered mechanism predictions came out with the opposite (non-significant)
  sign. New observed failure mode: **laundering** (neither owner nor target claims the student).
- **Paper consequences:** drop the RL-narrowing mechanism paragraph; drop the dissociation paragraph
  entirely (not even an appendix claim); report the attack as three outcomes (evade+spoof, joint claim,
  laundering) rather than a single "spoofing" story; add an integrity appendix with the extractor audit.

## 2026-09-16 — Advisor round 6 (accepted, one factual correction)
- **Terminology:** map the attack outcomes onto established names on first use — evade = scrubbing/removal
  (obfuscation; Brennan et al. 2012; Jovanović et al. 2024); evade+frame = scrubbing + spoofing (obfuscation +
  imitation); joint claim = ambiguity/invertibility attack (Craver et al. 1998); laundering = our coinage for
  "successful scrubbing without spoofing", defined explicitly and distinguished from "data laundering"
  (Mansurov et al. 2412.15255, a different sense). Rank by adjudication damage: laundering ≥ joint claim >
  framing/partial spoof > evade only > no effect.
- **No mechanism:** report the attack as an empirical finding with the two failed pre-registered accounts (ARR H6/H13
  protect this). Do not add attacks to raise n (6 → 12 is still underpowered). Add the feature-attribution diagnostic
  (done: scaffold phrases carry the moved signal).
- **Correction paragraph** goes in the main text, in the self-correction register (Rohrer et al. 2021). **Factual fix
  to the advisor's draft:** it says the extractor bug was caught via a sentinel flagging a power inversion. Wrong — the
  sentinel caught the M8 *scorer* bug; the extractor bug surfaced when M9b's rewrite check voided 7 of 8 attacks and
  the samples were inspected; the unit tests were written afterwards. Also the bug affected rounds M3–M9b, not M3–M8.
- **Must-run before submission:** a non-lexical read-out replication of the headline T0-vs-T1 cells → M11
  (pre-registered 18210d9, running). If the remedy fails under both EMB and POS, rescope the claim to lexical read-outs.
- **Hold for rebuttal:** one 7B LoRA cell; a read-out-aware attacker; a non-math task.
- **Deployment framing:** T1 as a first-party lineage attestation a vendor can pre-compute at release (like a signed
  checksum), valid against honest distillers and bounded by the imitation attack; precedents: Hugging Face `base_model`
  lineage metadata, Ecosystem Graphs (Bommasani et al.), EU AI Act Art. 53(1)(d). Never implied robust to adaptive
  adversaries.
- **Unverified citations flagged by the advisor:** ReasMark (ACL 2026?), "Protecting LMs against unauthorized
  distillation through trace rewriting" (ACL 2026?), DITTO (preprint only) — verify against the Anthology before citing.

## 2026-09-16 — EXP-M11: the diagnosis survives non-lexical read-outs
- The reference-aware remedy (N2) holds under gte-base embeddings and POS templates in all 4 M7 cells and on
  Zephyr; the TF-IDF sentinel reproduced M7/M10 exactly. The advisor's "must-run" objection is closed.
- Nuance to report, not hide: the *strength* of the no-reference collapse is read-out- and dataset-dependent
  (EMB on MATH: only 2–3 of 12 pairs collapse), while the remedy is uniform. Phrase the diagnosis as a property of
  the null a given read-out's test controls.
- Headline table: all three read-outs. Evidence is complete; next is writing.

## 2026-09-17 — Advisor round 7 (accepted): write §5–§6; three settlements; one exploratory check, spec fixed before running
- **Non-uniform collapse is a result:** the fix works under every read-out; how badly the standard test fails is a
  property of the null each read-out's test controls. One sentence in the abstract, not a footnote.
- **MATH collapse logged honestly** in the integrity appendix (see ledger entry of this date), with the note that
  TF-IDF has been the primary read-out since M5, before any of M6–M10 existed.
- **Scaffold finding owned, two statements kept separate:** *what* the imitation moves (scaffold phrases — claimed,
  descriptive) vs *when* it succeeds (two pre-registered predictors failed — not claimed). The embedding result
  (≥ 0.98 power on a read-out that largely ignores scaffolds) is stated as the counterweight: identity is not only
  formatting.
- **Headline table:** all read-outs × cells × Zephyr in the table; TF-IDF (pre-registered primary) for the main bars,
  EMB/POS as a robustness panel; "collapse without references" gets its own column.
- **Exploratory geometry check (M11 is closed, so exploratory; a check, not a claim).** Spec, written before running:
  - For each read-out R ∈ {tfidf, pos, emb}, cell ∈ {GSM8K, MATH} × {Qwen, Llama}: feature map as in M11 (fitted on
    the 6 teachers' R300 traces for tfidf/pos; gte-base for emb). Each student = mean feature vector of its 300 probe
    outputs; each teacher's reference centroid = mean over its 10 reference students (seeds 0–9).
  - For ordered same-line pair (owner a → relative b): d_rel = cosine distance(centroid_a, centroid_b);
    d_cross = min over the 3 out-of-line teachers t of cosine distance(centroid_a, centroid_t);
    **ratio = d_rel / d_cross** (small ratio = relative sits closer to the owner than any cross-line calibration point).
  - Prediction from the coverage account: smaller ratio → more collapse. Checks: (i) within each read-out × cell,
    Spearman(ratio, T0 FPR_rel) over the 12 pairs, expected negative; (ii) across the 12 read-out × cell units,
    Spearman(mean ratio, number of pairs with T0 FPR_rel ≥ 0.6), expected negative.
  - **Decision:** report as supporting only if (ii) is negative and (i) is negative in ≥ 9 of 12 units. Otherwise
    report "the geometry does not order collapse severity" and drop it from the paper's argument.
- **Geometry check result (2026-09-17):** passes its pre-set rule (12/12 within-unit negative; across-unit ρ = −0.83).
  Paper may say the read-out geometry *orders* collapse severity as the coverage account predicts, labelled
  exploratory, with the TF-IDF/MATH exception (ratio ≈ 0.95 but full collapse) stated: ordering holds, no sharp
  threshold at ratio = 1.

## 2026-09-17 — M12 arm A fails: the geometry sentence is demoted as pre-registered
- Ratio < 1 in 8 of 12 collapsed Zephyr pairs (rule ≥ 10). Collapse occurs at ratio up to 1.15. Pooled ordering over 18
  units weakens to ρ = −0.50.
- §5.4 now: consistent with collapse severity in the AllenAI cells, not replicated on the held-out vendor; no threshold
  claim; the coverage *account* stays as the explanation of why T0 cannot reject relatives (Proposition 1), but the
  centroid-distance geometry is not offered as its quantitative evidence.

## 2026-09-17 — M12 arm B: MIXED, no causal sentence (as pre-registered)
- S1 (scaffold sufficiency) fails in both families; S2 (content matters) holds for Qwen only. Per attack, scaffold-only
  was stronger than full imitation on one (tulu_rlvr→tulu_dpo: evade + frame), weaker on another (framing vanished on
  tulu_dpo→tulu_rlvr), identical on one, and evaded on the two attacks whose full-imitation corpora had been voided.
- **Paper:** §6.3 reports the scaffold-only arm attack by attack; "what imitation moves" stays descriptive; add the
  per-attack observation that a content-preserving scaffold rewrite evaded detection in 3 of 6 valid attacks (an
  observation, not a causal account of full imitation). **Delete** the descriptive "evasion exactly when the
  discriminant shift ≥ 1.0" sentence — M12-B shows evasion at shifts of 0.32–0.35 and 0.71–0.73.
- Two experiments run, one confirmatory claim demoted (geometry), one causal claim not earned (scaffold). Holds unchanged:
  7B cell, classifier-aware attacker, non-math task.

## 2026-09-17 — Anthology verification done (last open item before related work)
- ReasMark = ACL 2026 long (2026.acl-long.2185); trace rewriting = ACL 2026 long (2026.acl-long.519); DITTO = **EACL 2026**
  long (2026.eacl-long.229). Advisor round 6's "DITTO is preprint only" was wrong; recorded in `citations_verified.md`.
- Positioning consequence: Ma et al. (ACL 2026) rewrite traces owner-side to plant an active watermark — the mirror image of
  our distiller-side rewrite attacks. Both ReasMark and trace rewriting belong to the active-mark regime that §6.4 points
  to for third-party auditing.
- The remaining classic references (Craver, Brennan, Jovanović, Zhang, T-norm, impostors, Scheirer, Bates, Barber, Vovk,
  Tsybakov, Maini, Sablayrolles, Sander, Wadhwa, Mansurov, Rohrer) are listed for the same check before related work.

## 2026-09-17（顾问第 10 轮，reviewer 视角）——三个写作决定 + 两处更正
1. **§5.3 的诊断句必须单变量成立。** 顾问建议写「同一读出、同一学生、同一探针，只换校准集，判决翻转」。
   **按我们的设计这句话不成立**：T1 同时换了两样东西——覆盖亲属 *并且* 改用所有者 vs 亲属的两两读出。
   真正单变量的实验是 §6.2 的 pooled rejector（同一读出，只扩展校准集：远亲 0.0、近亲 1.0）。
   §5.3 已按此改写，T1 定位为「补救」而非「诊断」。已回告顾问。
2. **Wadhwa et al. (2025) 是我们的闭集极限**，不只是相关工作里的区分句：他们的候选集里没有同线检查点，
   所以那个失败在他们的设置里不可能出现。§5.3 显式这么写。同时预先回应「没校准过当然拒不掉」：
   (a) 这就是已发表的协议；(b) M8 证明覆盖是**距离不是布尔量**（同一校准下远亲 0.0、近亲 1.0）。
   §4（M3）被指认为同一模式的第一个实例：同样的输出、不同的零假设、不同的答案。
3. **Corollary 1b 拆分。** 定理内容（bound 对任意可疑总体 Q 成立）留在 §3；四个结果词
   （evade / frame / joint claim / laundering）移到 §6.3 表 2 旁边——它们是分数空间区域的命名，不是定理内容，
   放在 §3 会让审稿人把理论读成「预测了攻击」。`make_read_s3_s63.py` 的检查随之改为：
   **§6.3 的 remark 与生成的表 2 必须用同样的四个词**，且 §3 仍须陈述 1b。
4. **checksum 类比加限定。** checksum 是 tamper-evident，我们的检验不是——洗白正是「被篡改却无任何异常」的情形。
   §6.4 改为「像发布 checksum，除了最关键的一点：它不检测篡改」。
5. **§6.4 增加「防御会长在哪里」**：风格检验败给模仿；内容成员检验（检索、dataset inference）败给转述但**不**败给模仿，
   因为改写风格不改变训练用过哪些题。两者失效模式互补 → 能同时扛住的检验必须把风格信号和成员信号合起来，
   目前不存在。这是带理由的 future work，而不是「我们没有防御」。
6. **准确率基线更正（我的错）。** 草稿写「−0.03 到 +0.04，相对所有者未受攻击的学生」，但早期一轮实际比较的是
   **只转述（AD1）的学生**，后者本身比未受攻击学生低 2–6 个点，那个基线会让攻击显得**提高**准确率。
   重算（v2 抽取器，同家族同探针，仅有效语料 12 个 attack × family 格）：模仿 −0.028…+0.030（均值 +0.004），
   只改脚手架 −0.027…+0.023（均值 +0.000）。数字改由 `make_tables_s56.py` 的 `acc_cost` 生成，基线在正文点名。
7. **保留项优先级**：读出泛化与厂商泛化已关闭 → 审稿人下一个会问的是**学生规模**。若重开，只开一格 7B，
   且只跑 T0 坍缩 / T1 修复这一核心，目的是删掉一句 limitation，不是改结论。**预算决定权在用户。**
