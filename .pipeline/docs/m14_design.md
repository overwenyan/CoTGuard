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

## Manipulation check — **amended 2026-09-19, before any M14 code or data** (advisor round 14)

**Why the M6/M7 rule is wrong here.** That rule voids a teacher unless its students stay within three accuracy points of
the untuned base *or* within a factor of two of the teacher's trace length. On MATH at 7B the accuracy clause measures
the wrong thing: Qwen2.5-7B-Instruct's base MATH accuracy is plausibly above every teacher on our ladders, so a student
that absorbed the traces *perfectly* would lose accuracy and fail that clause, leaving the length clause as the only
route to a pass — a weak proxy for "shaped by the teacher". At 1.5B this did not bite the same way (students scored
0.34–0.50 against base 0.60), but we should not carry a check that passes for the wrong reason.

**Amended primary check — absorption, measured directly.** A teacher's students must separate from the **untuned base
model's** outputs on the same 300 MATH probes under a teacher-vs-base read-out (TF-IDF 1–2 gram + LR, fitted on the
teacher's traces vs the base model's probe outputs): **AUC ≥ 0.90**, per teacher, over its 13 students. A teacher whose
students fall below that is voided. **Fewer than 4 teachers, or fewer than 2 in either line → inconclusive; no gate is
evaluated.**

**Accuracy and length become reported diagnostics, not gates:** student accuracy against the 7B base (`math_check`
equivalence), and the student/teacher output-length ratio, are reported per teacher either way.

**Two cautions, recorded now.**
1. *It is not circular with what M14 tests.* The check is owner-vs-**base**; the gates are owner-vs-**sibling**. Passing
   it says a student was shaped by *some* teacher on this ladder, which is a precondition for asking which one, and
   nothing about whether siblings are separable.
2. *It is not the retracted dissociation.* "Absorption, not accuracy" is a statement about whether the treatment took,
   not a claim that capability and identity dissociate (a claim this project retracted after the extractor correction,
   Appendix X.1). The design doc says so here so the distinction is on record before the run, and the paper must repeat
   it wherever the check is described.

## Gates (identical to M13)

Gates are **proportions of valid ordered pairs**, so that voiding a teacher does not silently change the bar (amendment
2). A teacher voided by the absorption check takes its line from three stages to two, i.e. from six ordered pairs to two,
so the cell falls from 12 pairs to 8.

- **G1 — collapse.** T0 flags a same-line relative's test students at false-positive rate ≥ ⅔ in **≥ 50% of valid ordered
  pairs** (6 of 12 when all are valid; 4 of 8 with one teacher voided). (1.5B MATH·Qwen: 8 of 12.)
- **G2 — remedy.** T1 relative false-positive rate ≤ 0.2 in **≥ 83% of valid ordered pairs** (10 of 12; 7 of 8) **and**
  mean owner true-positive rate **≥ 0.9**. (1.5B: TPR 1.00, 10 of 12, mean 0.092.)
- **Floor for any verdict: ≥ 8 valid ordered pairs.** Below that the cell is reported descriptively and carries no gate
  verdict (third wording branch below).
- **A line reduced to two stages still counts as a line** for the "remedy holds in both cells" sentence **iff both of its
  ordered directions are valid**, because every claim in this paper is per-pair.
- **Reported, not gating:** per-pair T0/T1 rates; cross-line FPR; accuracy vs base and length ratios (the former
  manipulation-check clauses); POS and EMB re-scores (CPU only).

## Wording, fixed now

| Outcome | Sentence in §6.1 / §6.4 |
|---|---|
| G1 and G2 pass | "In two 7B cells (GSM8K and MATH, Qwen family, LoRA) the remedy holds and the standard test's collapse is present." Report both TPR/FPR pairs first, then both collapse counts. |
| **G1 fails, G2 passes** | "**The remedy holds at 7B in both cells; the standard test's collapse is present in one and weaker in the other.**" This is not a retreat: *failure severity varies, the fix does not* is already the paper's pattern, and a weaker 7B collapse on MATH is its fourth instance (after read-out, dataset, vendor). Lead with the remedy numbers. **The sentence must also say that the two 7B cells differ in two ways, not one:** dataset (GSM8K vs MATH) *and* training sequence cap (1,024 in M13, 4,608 here; M13 correction 2). |
| G2 fails | The remedy does not carry to 7B on MATH. §6.1's 7B paragraph says so in its first sentence, and the scale limitation is restated as a failure of the remedy in that cell. |
| **Partial (< 8 valid pairs)** | "**The second cell was partial, k of 12 pairs valid, and carries no gate verdict.**" Report the per-pair numbers descriptively and the void reasons; the one-cell wording of §6.1 stays. |
| Inconclusive (manipulation check leaves < 4 teachers, or < 2 in a line) | Report the void reasons; the one-cell wording stays; the attempt goes into Appendix X. |

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

---

## Amendment 1 (2026-09-19, before any M14 code or data)

Changes, all made while no M14 artefact exists, so this remains a pre-registration: (i) the manipulation check is now
absorption (teacher-vs-base read-out, AUC ≥ 0.90) rather than accuracy-or-length, with the rationale and the two
cautions above; (ii) accuracy and length become reported diagnostics; (iii) the fixed wording for "G1 fails, G2 passes"
must state that the two 7B cells differ in dataset **and** sequence cap. Requires one extra generation pass: the base
model's outputs on the 300 MATH probes (`probe_qwen7b_base.jsonl` under `data_m7/tulu_math`), which the probe job
produces alongside the students.

---

## Amendment 2 (2026-09-20, before any M14 code or data) — advisor round 15

1. **Gates are proportions of valid pairs** (G1 ≥ 50%, G2 ≥ 83%, matching 6/12 and 10/12), with a **floor of 8 valid
   ordered pairs** for any verdict; below it the cell is *partial* and gets the third wording branch.
2. **A two-stage line still counts as a line** for the cross-cell sentence iff both its directions are valid.
3. **M13's teachers get the absorption check retroactively, as a reported diagnostic, not a gate** — M13 passed its
   manipulation check under the length clause, and the two cells must not be compared on different checks without
   knowing whether M13 would have passed the new one. Running it retroactively as a *diagnostic* does not violate the
   no-retroactive-gates rule: it cannot change M13's verdict, and the paper says so where it is reported.
