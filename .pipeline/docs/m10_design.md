# M10 — cross-vendor ladder: does the calibration diagnosis generalise beyond one vendor? — pre-registration

_2026-09-16. Committed before any M10 data exists. Context: advisor round 5 ranks this the highest-value
single addition, because every ladder so far is AllenAI's (Tulu-3, OLMo-3) and a reviewer will ask whether
the findings are a property of one vendor's post-training recipe._

## 1. Question
M6/M7/M8 establish, on AllenAI lineages: (i) the standard owner test flags same-line relatives because its
calibration contains only other lines; (ii) reference students of each relative restore specificity; (iii)
the SFT→DPO step moves style far more than an RL step on top of DPO. **Does (i)–(iii) hold for a different
vendor's SFT→DPO recipe, on a different base model?**

## 2. Teachers (links verified from model cards on 2026-09-16)
| Line | Stage 1 | Stage 2 |
|---|---|---|
| **Zephyr** (base `mistralai/Mistral-7B-v0.1`) | `alignment-handbook/zephyr-7b-sft-full` ← Mistral-7B-v0.1 | `alignment-handbook/zephyr-7b-dpo-full` ← zephyr-7b-sft-full |

Two stages only: **no vendor outside AllenAI publishes a verifiable base→SFT→DPO→RL chain**, so M10 can test
the SFT→DPO step cross-vendor but cannot replicate the RLVR findings. This limitation is stated in the paper.

The six AllenAI teachers of M7 remain in the read-out and serve as out-of-line calibration for a Zephyr owner
(and vice versa), giving a three-line, eight-teacher setting.

## 3. Data and students (identical protocol to M7, GSM8K)
- Same splits: R300 read-out problems, POOL_REF (1,906) for reference students, POOL_TEST (2,000, disjoint)
  for test students, the same 300 probes.
- Generation: vLLM, each model's chat template, T = 0.7, top-p 0.95, ≤ 4,096 new tokens; seeds 10,000 + i for
  reference traces and 20,000 + i for test traces.
- Corpora: 1,500 problems each, `default_rng(100·i + s)` for reference seeds 0–9 (teacher index i = 7, 8, so
  no collision with M7's i = 1–6) and `default_rng(1000·i + s)` for test seeds 10–19.
- Students: 2 teachers × (10 reference + 10 test) × 2 families = **80 new LoRA students**, r = 32, 3 epochs.

## 4. Tests
Exactly M7's, with the read-out extended to all eight teachers:
- **T0:** 8-way read-out; calibration = the reference students of the teachers **outside** the owner's line
  (60 for a Zephyr owner, 50 for an AllenAI owner).
- **T1:** T0 **and** p_rel ≤ 0.05 against the reference students of each same-line relative (for Zephyr, one
  relative).

## 5. Pre-registered predictions and kill criteria
- **R1 (the calibration failure replicates cross-vendor):** T0 relative FPR ≥ 0.6 in ≥ 1 of the 2 ordered
  Zephyr pairs, in both families. (Only ≥ 1 of 2, because M7 found the failure is asymmetric: SFT owners did
  not flag their descendants, DPO/final owners did.)
- **R2 (primary — the remedy replicates):** T1 relative FPR ≤ 0.2 in **both** ordered Zephyr pairs **and**
  mean T1 TPR ≥ 0.8, in **both** families.
- **R3 (step structure replicates):** per-output AUC for the Zephyr SFT–DPO pair ≥ 0.9 in both families
  (the AllenAI SFT→DPO pairs gave 0.93–0.99 in M7).
- **Kill / restriction rules:**
  - **R2 fails in both families** → the reference-aware remedy is recipe-specific. The T1 result must then be
    reported as established only for AllenAI-style ladders, and the paper's scope narrows accordingly.
  - **R1 fails** → the calibration failure does not reproduce for this vendor; report and investigate whether
    the Zephyr SFT and DPO checkpoints are simply far apart in style (R3 would then be high).
  - **R3 fails while R2 holds** → the step structure is recipe-dependent but the diagnosis is not; report both.
- **Manipulation check:** as M7 — teacher answer-extraction ≥ 0.7; per student, accuracy ≥ base − 0.03 OR
  length ratio ∈ [0.5, 2.0]; a teacher with > 20% failing students is void.

## 6. Reported (not gating)
- **Calibration enrichment:** M7's AllenAI cells recomputed with the eight-teacher read-out and the enlarged
  out-of-line calibration. Does adding a foreign vendor's students change the AllenAI T0/T1 rates? This is a
  direct test of whether "more calibration" helps, which the coverage account predicts it will **not** for
  same-line relatives.
- Per-output AUC for every cross-line pair (Zephyr vs Tulu, Zephyr vs OLMo), expected near 1.0.
- Reference-budget and query-budget curves for the Zephyr owners, as in M7.

## 7. Not claimed from M10
- Anything about RL stages outside AllenAI (no public verifiable chain exists).
- Gemma's Zephyr variant (`zephyr-7b-gemma-sft-v0.1` → `zephyr-7b-gemma-v0.1`) is a second candidate ladder,
  held in reserve; not part of this pre-registration.

## 8. Compute
Generation 2 × 4,300 traces (about 20 GPU-minutes each), 80 LoRA students plus probes: roughly 8 GPU-hours.
