# M14 — the second 7B cell (MATH · Qwen-7B) — pre-registration

_2026-09-19. Committed **before any M14 code, job or data**. Context: advisor round 13. **To be run during review**, so it
lands either way for camera-ready; it does not touch the submission. One cell, chosen as the one most likely to hurt:
MATH · Qwen, where non-lexical collapse was already weakest at 1.5B (EMB 3/12, POS 5/12; TF-IDF 8/12)._

---

## Sentence at stake (§6.1 / §6.4)

> "In one 7B cell (GSM8K, TF-IDF, Qwen family, LoRA) the remedy is strong — owner true-positive rate 0.94, mean
> false-positive rate on siblings 0.03 — and the standard test's collapse is present but narrow (7 of 12 pairs against a
> gate of 6). That shows neither is a small-student artefact in the cell tested."

A second cell does not change that wording's truth; it changes how the 7/12 reads.

## The cell

The AllenAI · MATH · Qwen cell of Table 1 with the student scaled from Qwen2.5-1.5B-Instruct to Qwen2.5-7B-Instruct.

| | 1.5B cell (M7) | M14 |
|---|---|---|
| Student | Qwen2.5-1.5B-Instruct | Qwen2.5-7B-Instruct |
| Teachers | 6 AllenAI (tulu_sft/dpo/rlvr, olmoi_sft/dpo/final) | same |
| Dataset | MATH Levels 1–4 (M7 splits) | same |
| Corpora | `data_m7/tulu_math/corpus_grid_<teacher>_s<k>.jsonl` | same files |
| **T0 cross-line calibration** | 10 reference students per teacher | **10 per teacher, seeds s0–s9** |
| **T1 per-relative references** | 10 | **3: s0, s1, s2** (M7: 3 ≡ 10 for this test) |
| Test students per teacher | 10 (s10–s19) | **3: s10, s11, s12** |
| Probes | 300 MATH probes | same |
| LoRA | r 32, α 64, 3 epochs, batch 4, lr 1e-4, **seq cap 4,608** | **identical, seq cap 4,608** |
| Gating read-out | TF-IDF 1–2 gram + LR (C = 4) | same |

**Total 78 students** (6 × [10 + 3]). The one recipe difference from the 1.5B cell is gradient checkpointing
(`M3C_GRADCKPT=1`), which changes memory and speed, not the objective. **The sequence cap is 4,608, M7's actual recipe** —
not M13's 1,024 (M13 correction 2). On the MATH corpora (measured on seeds s0–s2 and s10–s12 of every teacher, 54,000 examples) 16.3% exceed 1,024 tokens and 0.6% exceed 4,608.

**Attainability, computed before anything runs.** T0: 3 cross-line teachers × 10 = **30 calibration scores → smallest
attainable p = 1/31 = 0.032 < α = 0.05.** T1's per-relative test is a t prediction interval with n = 3 references (no rank
floor; its p-value is continuous). The scorer checks the floor again and refuses to issue a verdict if it exceeds α.

**Resolution.** Three test students per teacher: every false-positive rate lies in {0, ⅓, ⅔, 1}; no claim finer than that.

## Manipulation check (M6/M7 rule, applied first)

A teacher is voided if fewer than 70% of its traces contain a `\boxed{}` answer, or if more than 20% of its students
neither stay within three accuracy points of the untuned Qwen2.5-7B-Instruct base (MATH probes, `math_check`
equivalence) nor stay within a factor of two of the teacher's output length. **Fewer than 4 teachers, or fewer than 2 in
either line → inconclusive; no gate is evaluated.** Note in advance: at 1.5B, MATH distillation lowered Qwen accuracy
below base (0.34–0.50 vs 0.60), so the length clause may be what keeps teachers in; that is the rule as written in M6/M7.

## Gates (identical to M13)

- **G1 — collapse.** T0 flags a same-line relative's test students at false-positive rate ≥ ⅔ in **≥ 6 of 12** ordered
  pairs. (1.5B MATH·Qwen: 8 of 12.)
- **G2 — remedy.** T1 relative false-positive rate **≤ 0.2 in ≥ 10 of 12** pairs **and** mean owner true-positive rate
  **≥ 0.9**. (1.5B: TPR 1.00, 10 of 12, mean 0.092.)
- **Reported, not gating:** per-pair T0/T1 rates; cross-line FPR; accuracy vs base; POS and EMB re-scores (CPU only).

## Wording, fixed now

| Outcome | Sentence in §6.1 / §6.4 |
|---|---|
| G1 and G2 pass | "In two 7B cells (GSM8K and MATH, Qwen family, LoRA) the remedy holds and the standard test's collapse is present." Report both TPR/FPR pairs first, then both collapse counts. |
| **G1 fails, G2 passes** | "**The remedy holds at 7B in both cells; the standard test's collapse is present in one and weaker in the other.**" This is not a retreat: *failure severity varies, the fix does not* is already the paper's pattern, and a weaker 7B collapse on MATH is its fourth instance (after read-out, dataset, vendor). Lead with the remedy numbers. |
| G2 fails | The remedy does not carry to 7B on MATH. §6.1's 7B paragraph says so in its first sentence, and the scale limitation is restated as a failure of the remedy in that cell. |
| Inconclusive | Report the void reasons; the one-cell wording stays; the attempt goes into Appendix X. |

**No post-hoc gate changes.** Any deviation after data is a correction, logged in the form used for M8, M9b and M13.

## Compute and kill criteria

- Training on H200, `--mem=96G`, ≤ 4 concurrent; estimate ~40 min per student at 7B on MATH-length traces (M7 1.5B MATH:
  ~9 min/student) → ~50 GPU-hours, ~13 h wall-clock. **Kill:** if training is not finished 36 h after the first job
  starts, score what exists and report the cell as partial with counts per teacher.
- Probes on L40S with vLLM + LoRA, `--mem=96G`, `VLLM_USE_FLASHINFER_SAMPLER=0`; scoring on CPU, `--mem=32G`.
- Check `free -h` (the *available* column) on the target node before submitting.
- No recipe retries: a teacher whose students cannot be trained within the memory budget is dropped and reported.

## Integrity

- Student directories `student_qwen7b_grid_<teacher>_s<k>` under `data_m7/tulu_math` (no existing qwen7b files there).
- **Sentinel:** the scorer first recomputes M7's published 1.5B MATH·Qwen cell (TF-IDF: collapse 8/12, T1 10/12 ≤ 0.2) and
  refuses to write results if it moves.
- **Floor guard:** as in M13 after correction 1.
- The scorer and sbatch files are committed before submission of the first job; their commit id goes into the ledger.
