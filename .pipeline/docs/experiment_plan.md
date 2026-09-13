# M3 v4 stage 1 — experiment plan (DRAFT, awaiting approval; becomes `m3_design.md` v4 §1 on approval)

_2026-09-13. Follows the outside review (`m3_expert_review.md`) and stage 0 (ledger EXP-M3S0, 7b3cd4a)._

## What stage 0 changed
- **The codebook is instruction-level.** Misattribution to a same-instruction key runs at 6.5–8.3× its
  base rate; persona carries almost nothing. Stage 1 keys are therefore **reasoning instructions only**
  (one fixed template, no persona), all of them new.
- **Paraphrase keeps a coarse signal but kills 64-key attribution**, and a non-lexical read-out did not
  fix that. So the question becomes *which kinds of instruction survive*, not *which detector*.
- **The signal is mostly stereotyped phrasing** (removing the top n-grams takes the owner test from
  12/16 to 1/16). Stealth becomes a first-class experiment.
- **Same-key imitation by another teacher is flagged ~40% of the time.** Every claim must say
  "trained on traces produced under instruction k", not "trained on this owner's model".

## Common setup
- **Teacher:** Tulu-3-8B. **Domain:** GSM8K (train → test 200). **Student:** Qwen2.5-1.5B-Instruct, LoRA r=32.
- **Key bank:** 40 new instructions, 20 per category, categorised *before* any generation:
  - **OP (operation):** the instruction makes the trace contain at least one extra claim that depends
    on the problem and is not needed to compute the answer. Examples: "Before solving, estimate the
    answer to the nearest power of ten"; "After finishing, substitute the answer back into one
    condition of the problem"; "Identify one number in the problem that is not needed, and say why".
  - **PRES (presentation):** the instruction constrains form only (numbering, ordering, word choice,
    layout) and adds no problem-dependent claim. Examples: "Number each step as (i), (ii), (iii)";
    "Write every equation as result = expression"; "Put each numeric result in square brackets".
- **Compliance screen, applied symmetrically before randomisation:** a frozen LLM rubric
  (Qwen2.5-7B-Instruct yes/no logit) checks 50 teacher traces per instruction. Instructions followed
  < 50% of the time are removed from the bank **before** owner keys are drawn.
- **Owner keys:** 8 OP + 8 PRES drawn from the eligible bank with `default_rng(20260913)`.
  The rest are decoys.
- **Owner test:** p = (1 + #{j ≠ k : s_j ≥ s_k}) / K over the eligible bank; pass iff p ≤ 0.05 and
  the key is not flagged on the clean-trained or base student.

## S1-A — factorial: instruction category × read-out × transformation (primary)
- **Transformations** (Qwen2.5-7B-Instruct), applied to the 16 owner corpora + clean.
  **Answer-preserving:** the final answer must match the original's, with up to 2 retries; fidelity
  is reported.
  - **T1 wording change:** "keep every step, every stated estimate, check and claim; change the wording".
  - **T2 canonicalise:** "a concise standard solution: only the computations needed for the answer,
    one sentence each; remove restatements, estimates, checks and commentary".
- **Students:** 16 keys × {original, T1, T2} + clean × 3 + base = 52 student samplings.
- **Read-outs,** all trained on original teacher traces only:
  - lexical (TF-IDF + LR);
  - embedding (gte-base + LR);
  - **behaviour:** a frozen LLM checklist (one yes/no question per bank instruction, Qwen2.5-7B
    logits); the per-instruction yes-rate is the score.
- **Behaviour-rubric validation:** 120 blinded (text, question) items labelled by a human (the user).
  Cohen's κ ≥ 0.6 is required before behaviour results are interpreted; they are reported either way.
- **Gate H-OP (pre-registered):** under T1 with the lexical read-out, OP owner keys pass the owner
  test more often than PRES keys, by a margin of ≥ 5 of 8 and one-sided Fisher p ≤ 0.05.
- **Prediction for T2:** both categories fail (their operations are optional commentary). If OP keys
  survive T2, those operations are content the canonical solution keeps.
- **Covariates reported per key:** teacher held-out separability, length ratio, rewrite fidelity, and
  the survival ratio (T1 score / original score). Partial rank correlation of category with survival
  controlling for separability, labelled low-power (n = 16).
- **Utility:** student accuracy for original/T1/T2 × keyed/clean, with training-token exposure.

## S1-B — mixed-source dilution
- **Corpus:** 7,000 GSM8K train problems. Clean Tulu traces for all of them. For 4 owner keys (2 OP,
  2 PRES, drawn from the 16), keyed traces replace clean ones **on the same problems** (topic-matched
  by construction) at 1% / 5% / 10%, with nested subsets.
- **Students:** 0% + 4 keys × 3 fractions = 13, LoRA, 1 epoch. Report example fraction, token
  fraction and absolute keyed exposure.
- **Independent-imitation control:** a Qwen2.5-7B teacher under the same instruction at 10%, for 2 of
  the 4 keys → 2 students, tested with the Tulu owner's read-out.
- **Predictions:** 10% detected for ≥ 3/4 keys; 5% for ≥ 2/4; 1% for ≤ 1/4. Imitation: reported.

## S1-C — stealth: can a distiller find and remove keyed traces?
- **Supervised screen:** the distiller trains a keyed-vs-clean TF-IDF classifier on traces they make
  themselves with the *non-owner* bank instructions. It is tested on the owner keys' traces for
  held-out problems; report TPR at clean FPR 1% and 5%, by category.
- **Zero-shot LLM screen:** "Does this solution follow an unusual stylistic or procedural
  instruction?" yes-logit → TPR at the same FPRs.
- **Removal:** filter the 10% dilution corpora with the supervised screen at 5% clean FPR, retrain
  the 4 students, and report the owner test, student utility and the share of clean data discarded.
- **Prediction:** TPR@5%FPR ≥ 0.8 for PRES; OP lower. If filtering removes attribution while losing
  ≤ 10% of clean data and ≤ 2 points of accuracy, stealth fails, and that is stated as a limitation.

## Compute (one GPU, chained)
| Block | GPU-h (est.) |
|---|---|
| teacher bank 40 × 300 + compliance screen | 1.0 |
| T1 / T2 rewrites with retries (17 arms × 2) | 4.0 |
| factorial students (52 samplings, 48 LoRA runs) | 2.5 |
| behaviour-rubric annotation (~9k texts × ~40 questions) | 2.0 |
| dilution generation (6,700 clean + 2,800 keyed + 1,400 Qwen-teacher) | 1.0 |
| dilution students (15) | 3.5 |
| stealth screens + 4 filtered students | 1.5 |
| **total** | **≈ 15.5** (~5 chained jobs) |

**Human task:** label 120 blinded rubric items (~30 min), needed before the behaviour read-out can be
interpreted.

## Decision after stage 1
- **H-OP passes and the dilution predictions hold** ⇒ main-conference framing: a predictive rule for
  which prompt-implanted reasoning signatures transfer and survive rewriting.
- **H-OP fails** ⇒ package M3 as a scoped empirical study: prompt-implanted signatures transfer, the
  codebook is instruction-level, and rewriting or screening removes attribution.
- **Deferred to stage 2:** full fine-tuning, a larger student, a hybrid active component.
