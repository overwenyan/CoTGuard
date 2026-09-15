# Status and decision prompt — distillation provenance project (2026-09-15)

_Self-contained. Paste into an advisor or deep-research session. Sources of record in the repo:
`.pipeline/memory/experiment_ledger.md` (EXP-M3 … EXP-M5), design and pre-registration files in
`.pipeline/docs/` (`m3_design.md`, `m4_design.md`, `m5_design.md`), prior-work notes
`m4_deep_research_assessment.md` and `m3_prior_work.md`._

---

## Context
We are a small academic group with school-scale compute (up to ~7 H200 GPUs, intermittently),
aiming for an **ACL 2027 main-conference paper** on **provenance of distilled reasoning models**: can
the owner of a teacher LLM tell, black-box, that a student was distilled from its reasoning traces?
Every experiment round was pre-registered in git (kill criteria, gates) before its data existed.
Failed gates, voided analyses and corrections are all logged.

## What we have done (three tracks)

### M3 — prompt-implanted signatures (completed; analysis-paper material)
- **Setting:** the owner prompts a teacher with a secret reasoning instruction (a "key") and publishes
  the traces unedited. A student is distilled from them. The owner runs a black-box rank test
  (TF-IDF read-out, random decoy keys, vetoes on clean and base students).
- **Results:**
  - The signature transfers: 16/16 and 12/16 in 64-key pools, across 2 teachers and 2 student
    families.
  - Nominal keys collide. In a 768-key template × persona × instruction generator, distinguishability
    is dominated by the 12 instructions.
  - **Detection identifies the instruction, not the owner:** an independent teacher given the same
    instruction is flagged 8/8, a different instruction 0/8. Yet teacher identity is recoverable from
    the same student outputs (AUC 1.00).
  - Neutral paraphrase removes about half the signatures. Content-adding instructions survive
    rewording (8/8, 6/6) and format-only ones mostly don't (2/8, 2/6), but category is confounded
    with distinctiveness (a matched replication was infeasible).
  - Dilution: detected at 50% keyed data, not at ≤ 10%, even with 1,319 queries.
  - Utility cost is student-dependent (Qwen −12 points; none detected on Llama).
  - Three explanatory propositions: the test controls a key null, not a source null; attribution needs
    teacher-specific information (a total-variation bound); more queries cannot overturn a wrong
    population ordering.

### M4 — secret learnable reasoning-move watermark (pilot; FAILED gates → abandoned)
- **Idea:** a per-problem move chosen by secret random hyperplanes in sentence-embedding space, with a
  per-instance agreement test against 1,000 free decoy partitions.
- **Result:**
  - The move pattern is learned on problems the owner served (p = 0.001 at 2 of 3 granularities).
  - It narrowly fails on held-out problems (p = 0.011 vs 0.01).
  - **Specificity failed:** an imitator with a different secret split was flagged (p = 0.007) with a
    larger lift than the owner's own student, and a clean student reached p = 0.011.
  - Post hoc explanation: random partitions of an anisotropic embedding space are correlated, and
    natural move tendencies align with problem type ("behavioural keys collide on natural structure",
    as in M3).
  - No utility cost.
- **Novelty check:** keyed selection among equivalent behaviours that survives imitation already
  exists for agents (AgentWM, arXiv 2602.08401; TRACE, 2607.08400; SeqWM, 2605.11036).

### M5 — open-set teacher attribution (completed)
- **Setup:** 7 teachers (Tulu-3-8B SFT and DPO, Llama-3.1-8B-Instruct, Qwen2.5-7B, Qwen3-14B,
  Mistral-7B, Gemma-2-9B) × 5 corpora × 2 student families = 70 LoRA students on GSM8K. Read-outs:
  TF-IDF, embedding, POS-template baseline. Conformal owner test with 20 calibration students and two
  held-out "unknown" teachers per owner.
- **Results:**
  - **Closed-set attribution:** 35/35 (Qwen students), 34/35 (Llama students).
  - **Open-set (pre-registered primary):** TPR 1.00, false-positive rate 0.07–0.09 on unknown
    teachers.
  - **Same-lineage teachers are indistinguishable:** Tulu-SFT vs Tulu-DPO students are confused 100% in
    both directions, under all three read-outs. Qwen3 ← Qwen2.5: 0.6–1.0.
  - **The low false-positive rate depends on the read-out having seen the unknown teachers' traces.**
    With truly unseen teachers (strict variant) it is 0.3–1.0.
  - **The combined instruction ∧ teacher test fails:** independent-teacher same-instruction students
    are flagged 4/8 (gate ≤ 1/8). All failures come from a strong format instruction ("keep every
    sentence under twelve words"), which appears to erase teacher style. Owner students pass 8/8.
  - **Mixtures:** 50% owner data detected 2/2; 10% owner data 1/4.
- **Closest prior work:**
  - Who Taught You That? (Findings of ACL 2025): closed-set, POS templates.
  - Reference-Based Distillation Detection (arXiv 2607.09692): needs a same-lineage reference
    checkpoint.
  - KD detection for open-weights models (NeurIPS 2025): student weights; vision.

## Our current reading
Closed-set teacher attribution working is expected (prior work). Across all three tracks, the new
and consistent finding is a **two-level identifiability gap**:

| Level | Identifiable | Not identifiable |
|---|---|---|
| Prompt-implanted signal (M3, M4) | instruction family | owner (independent same-instruction sources and imitating partitions are flagged) |
| Teacher signal (M5) | model family / lineage (closed set) | the specific model within a lineage (SFT vs DPO), truly unseen teachers, low mixture fractions, and teachers whose style is overridden by a strong instruction |

Three method attempts (fixed-instruction rank test, keyed reasoning moves, composite
instruction ∧ teacher test) each failed at the specificity step.

## Decision needed
Options we see:
- **A. Unified analysis / measurement paper (our current preference).** "What distillation provenance
  signals identify — and what they do not." It covers both levels with pre-registered experiments,
  the propositions (key null vs source null, total-variation requirement, query-scaling limit), and
  negative results for three method designs. Target ACL main as a rigorous measurement study.
- **B. Method attempt on the M5 failures:** a same-lineage (SFT vs DPO) discriminator, or open-set
  calibration robust to unseen teachers (a larger public-teacher pool, outlier detection). Uncertain.
- **C. Strengthen A with scale before writing:** a 7B student; MATH or a long-CoT corpus; more
  same-lineage checkpoints (base / SFT / DPO / RL); more truly unseen teachers.

## Questions for you
1. Is option A plausible for **ACL main** (not Findings) as a measurement / analysis paper? What would
   reviewers require: scale, number of teachers and lineages, datasets, theory depth, human
   evaluation?
2. Is the **two-level identifiability gap** a novel enough framing given recent work
   (distillation-detection papers, watermark spoofing — DITTO, watermark stealing — agent behavioural
   watermarks, subliminal learning)? What is the closest prior work that already makes this argument?
3. Which **2–3 additional experiments** would most increase acceptance odds per GPU-hour? Candidates:
   more same-lineage pairs; truly unseen-teacher protocols; a 7B student; long-CoT data; an adaptive
   distiller that imitates another lineage.
4. Is there a **theoretical result** worth proving that unifies both levels? For example, an
   identifiability condition: attribution at level L is possible only if the training-source
   distributions differ in total variation beyond what the read-out's invariances remove, or a
   lower bound on false attribution for same-lineage models.
5. Is any method idea from options B or C likely to succeed where three specificity failures
   occurred, and if so, how should it be pre-registered and killed early?
