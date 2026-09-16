# Status and decision prompt — distillation provenance project (2026-09-16)

_Self-contained; paste into an advisor or deep-research session. Sources of record in the repo:
`.pipeline/memory/experiment_ledger.md` (EXP-M3 … EXP-M7), `.pipeline/memory/decision_log.md`,
pre-registrations in `.pipeline/docs/` (`m3_design.md`, `m4_design.md`, `m5_design.md`, `m6_design.md`,
`m7_design.md`, `m8_design.md`), paper plan `paper_plan_unified.md`. Every round was pre-registered in
git — gates, predictions and kill criteria — before its data existed. Failed gates, voided analyses and
label corrections are all logged._

---

## 1. Context
Small academic group, school-scale compute (up to ~7 H200 plus L40S GPUs, shared). Target: **ACL 2027
main conference**. Topic: **provenance of distilled reasoning models** — can the owner of a teacher LLM
determine, black-box, that a student model was distilled from its reasoning traces, and can it name
*which* model?

Students throughout: LoRA (r = 32, 3 epochs) fine-tunes of Qwen2.5-1.5B-Instruct and
Llama-3.2-1B-Instruct on 1,500-problem GSM8K or MATH corpora of teacher traces. Read-outs: TF-IDF
(1–2 gram) + logistic regression (primary), gte-base embeddings and POS-templates (reported).

## 2. What has been run

### M3 — prompt-implanted signatures (complete)
The owner prompts its teacher with a secret reasoning instruction (a "key"), publishes traces
unedited, and later runs a rank test against 63 decoy keys with vetoes from clean and base students.
- Signature transfers: 16/16 and 12/16 in 64-key pools, two teachers, two student families; survives a
  filtering attack (7/8).
- **Not source-specific:** an *independent* teacher given the same instruction is flagged 8/8, exactly
  like the owner's own student; a different instruction 0/8. The test identifies the instruction.
- Teacher identity is nonetheless recoverable from the same student outputs (AUC 1.00, exploratory).
- Neutral paraphrase removes about half the signatures; dilution is undetectable at ≤ 10% even with
  1,319 queries; utility cost is student-dependent (Qwen −12 points, none detected on Llama).

### M4 — secret learnable reasoning-move watermark (pilot, killed by its own gate)
A per-problem "move" chosen by secret hyperplanes in embedding space, tested against 1,000 decoy
partitions. Learnable on served problems (p = 0.001), marginal on held-out problems (p = 0.011), and
**failed specificity**: an imitator with a *different* secret split was flagged at p = 0.007 with a
larger lift than the owner's own student. Abandoned per the pre-registered rule.

### M5 — open-set teacher attribution (complete)
7 teachers × 5 corpora × 2 student families = 70 students, conformal owner test.
- Closed-set 35/35 and 34/35; open-set TPR 1.00, FPR 0.07–0.09.
- **Same-lineage teachers confused 100%** (this was later shown to be a property of the test, see M7).
- With *truly unseen* teachers, FPR 0.3–1.0.
- The composite "my instruction AND my teacher" test fails: one strong format instruction erases
  teacher style.
- 10% mixtures unreliable.

### M6 — alignment-stage ladder (complete)
Nine checkpoints in three post-training lines — Tulu-3-8B (SFT → DPO → RLVR), OLMo-3-7B-Instruct
(SFT → DPO → final), OLMo-3-7B-Think (SFT → DPO → RLVR) — every `base_model` link verified from model
cards. 90 students, 4,096-token generation with vLLM.
- **Per-output distinguishability is step-specific, not distance-monotone.** SFT→DPO moves style
  (AUC 0.93–0.99); the RL step on top of DPO barely does (Tulu 0.73, OLMo-Instruct 0.72); the Think
  line's RLVR is the exception (0.96, partly length and format).
- **The pre-registered owner test flags all same-line relatives** (FPR 1.0) when the owner is a DPO or
  final checkpoint, even where per-output AUC is 0.93–0.99. Cause: the calibration set contains only
  students of *other lines*, so a same-line relative cannot be rejected.
- Sibling lines (same base, different post-training) separate at AUC 1.00 — but the Think format alone
  does that, so this shows format detection, not style.
- The manipulation check voided 3 teachers: 1–1.5B LoRA students cannot absorb 4–7k-character Think
  traces (accuracy below base).
- Exploratory: a *pairwise* read-out with a threshold from the other students separated every pair
  perfectly at the student level, including DPO vs RLVR.

### M7 — reference-aware owner test (complete; the pivotal result)
Designed to test that exploratory finding confirmatorily on **fresh students trained on fresh traces
for disjoint problems** (no shared problems, no shared teacher traces), on **two datasets**.
6 non-Think teachers, GSM8K and MATH (Levels 1–4), 420 new LoRA students (10 reference + 10 test per
teacher per dataset per family), 300 probes each.
- **T0** (M7's baseline = the standard protocol): 6-way read-out, calibration = 30 reference students
  of the *other* line.
- **T1** (reference-aware): T0 **and**, for each same-line relative b, a one-sided t prediction-interval
  test against b's 10 reference students under a pairwise a-vs-b read-out.

| Cell | T0: pairs with relative FPR ≥ 0.6 | T1 mean TPR | T1 pairs with FPR ≤ 0.2 | T1 mean relative FPR |
|---|---|---|---|---|
| GSM8K / Qwen | 8 of 12 | 1.00 | 12/12 | 0.042 |
| GSM8K / Llama | 8 of 12 | 1.00 | 11/12 | 0.067 |
| MATH / Qwen | 8 of 12 | 1.00 | 10/12 | 0.092 |
| MATH / Llama | 8 of 12 | 1.00 | 12/12 | 0.033 |

All three pre-registered hypotheses passed in all four cells, with **no teacher voided**.
- Same students, same probes: handing the owner reference students of its relatives takes the relative
  false-positive rate from 1.0 down to ≤ 0.2 while power stays at 1.0.
- **Budgets:** 3 reference students and 25 probe queries already suffice.
- **Leakage diagnostic:** the M6-style threshold made 0 errors on both shared-trace and fresh-trace
  students → M6's exploratory separation was real, not leakage.
- Step-specific structure replicates on MATH (DPO→final AUC 0.79–0.84 vs SFT→DPO 0.88–0.98).
- **Capability transfer is unreliable:** 19 of 24 pairs keep the teacher's accuracy ordering, but 4
  significant reversals occur where teacher traces are long (OLMo-Instruct's 2,200-character MATH
  traces make students *better* than a stronger teacher's do). On MATH, distillation lowered Qwen
  student accuracy overall (0.34–0.50 against a base of 0.60).

### M8 — robustness (pre-registered, running now)
Three attacks on M7's constructive claim:
- **A1 unknown relative:** the suspect came from a relative the owner never referenced. T1-partial
  (expected to fail, documenting the assumption boundary) vs **T2**, which uses the *other* relative's
  references as a "not me, but in my line" rejector. Gate: T2 relative FPR ≤ 0.3 in ≥ 8 of 12 pairs with
  TPR ≥ 0.8, in ≥ 3 of 4 cells. Kill: the owner must enumerate its whole lineage.
- **A2 truly unseen teacher:** **T3** uses *no* out-of-line calibration — only the owner's own line —
  and requires the student not to fall below the owner's own reference students. Gate: unknown-teacher
  FPR ≤ 0.2 with TPR ≥ 0.8 in ≥ 3 of 4 cells.
- **A3 mixtures:** 24 students trained on 50/50 and 10/90 blends of the two least separable adjacent
  pairs. Gate: at 50%, the owner's T1 flags ≥ 0.8 of mixture students.

## 3. The story as it now stands
**Two-level identifiability gap, plus its resolution.**

| Level | What the test identifies | What it does not | Why |
|---|---|---|---|
| Instruction (M3, M4) | the instruction family | the owner — independent same-instruction teachers and imitating partitions are flagged | behavioural keys collide on natural structure |
| Teacher (M5, M6) | the post-training *line* | the checkpoint within the line, unseen teachers, low mixture fractions, teachers whose style a strong instruction overrides | the calibration set defines the null: relatives excluded from calibration cannot be rejected |
| Resolution (M7) | the specific checkpoint | — (subject to M8) | reference students of each relative supply the missing null |

The claim is **not** that provenance signals are weak. Every negative result is a statement about the
*null a test controls*, not about missing information: the same outputs that a standard owner test
cannot resolve are perfectly separable once the comparison set includes the relative. Three formal
propositions frame this: (1) the rank test controls a key null, not a source null; (2) attribution
requires teacher-specific information surviving the read-out's invariances (a total-variation
condition); (3) more queries cannot overturn a wrong population ordering — consistent with M7, where
25 queries suffice once the ordering is right.

## 4. Decisions already taken
1. **Paper type:** unified analysis/measurement paper (advisor's hybrid A + C), not a method paper.
   Working title candidate: *"Provenance needs a reference: same-lineage attribution is a calibration
   problem, not an information limit."*
2. **Method track closed.** Three specificity failures (M3 keys, M4 moves, M5 composite) are reported
   as negative results with their pre-registered kill criteria.
3. **Scale targeted, not broad.** Ladder first (M6), then the confirmatory reference test (M7), then
   robustness (M8). A 7B student and the long-CoT/Think line are deferred; M6 established that
   1–1.5B students cannot absorb long Think traces, which is itself reported.
4. **Label correction logged:** `allenai/Llama-3.1-Tulu-3-8B` is the RLVR *final* model, not SFT. M5's
   "SFT vs DPO indistinguishable" is really "RLVR-final vs DPO".
5. **Manipulation check loosened for M7/M8** (accuracy ≥ base − 0.03 OR length ratio ∈ [0.5, 2.0]),
   because M6's tighter band voided teachers whose students were clearly shaped by them. Applied to the
   new round only, never retroactively.
6. **Writing is on hold until M8 lands** (user decision), so that §6's framing is settled before prose.

## 5. What is planned after M8
- Write §5 (ladder) and §6 (reference-aware test and its robustness), then §3 theory, §4 (M3
  compressed), related work, and the abstract last.
- Positioning must engage the ambiguity/invertibility-attack tradition (Craver et al. 1998; Fan et al.
  2019 passports; watermark stealing; DITTO; subliminal learning) and Reference-Based Distillation
  Detection (arXiv 2607.09692), which also needs a same-lineage reference checkpoint.
- Explicitly *not* claimed: adaptive distillers that deliberately imitate a relative, long-CoT
  attribution, lineages deeper than three stages, non-LoRA distillation.

## 6. Questions for you
1. Is "the same-lineage limit is a calibration property, not an information limit" a strong enough
   **main-conference** contribution, given that the constructive test (T1) assumes the owner can
   enumerate and distil from its relatives? Does M8's A1/A2 outcome change that answer, and how should
   each branch be framed?
2. What is the closest prior work that already states the calibration point? We have RefDistDet
   (reference checkpoint), Who Taught You That? (closed-set), KD detection (NeurIPS 2025, weights).
   Is there a model-attribution or authorship-verification line we are missing where "open-set
   calibration determines the null" is already the headline?
3. Which **one** additional experiment would most raise acceptance odds: (a) a 7B student to show the
   effect is not a small-student artefact; (b) an adaptive distiller that paraphrases or imitates a
   relative — the obvious attack on T1; (c) a fourth lineage from a different vendor (for example
   Qwen's chat ladder) to show generality beyond AllenAI's release practice; (d) human evaluation of
   what the read-out uses?
4. Is the capability-vs-identity dissociation worth a section, given that it did **not** replicate
   cleanly in M7 (19 of 24 pairs keep the teacher's ordering, with 4 significant reversals driven by
   trace length)? Or should it be a paragraph plus an appendix?
5. Are the propositions strong enough as stated, or should we try for a sharper theoretical result —
   for example a lower bound on same-lineage false attribution as a function of the calibration set's
   coverage of the lineage?
