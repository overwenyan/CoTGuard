# M8 — robustness of the reference-aware owner test — pre-registration

_2026-09-16. Committed before any M8 data or scoring exists. Context: EXP-M7 (H1–H3 passed; T1 holds
relative FPR ≤ 0.2 at TPR 1.0). User decision: robustness round only, before writing._

## 1. Question
M7's T1 assumes the owner can name every same-lineage relative and train reference students for each.
M8 asks what happens when that assumption breaks in the three ways a reviewer will raise:

1. **Unknown relative** — the suspect's true source is a relative the owner did not reference.
2. **Truly unseen teacher** — the suspect's source is outside anything the owner can calibrate on,
   and the owner's read-out never saw it (M5's strict variant gave FPR 0.3–1.0).
3. **Mixtures** — the distiller blended two stages of the same line.

## 2. Setting (reuses M7; no re-training for arms A1 and A2)
Six teachers, two lines, four cells (dataset × student family): GSM8K and MATH × Qwen2.5-1.5B and
Llama-3.2-1B. Per teacher: 10 reference students (seeds 0–9) and 10 test students (seeds 10–19), all
from M7. Read-outs are TF-IDF (1–2 gram) + LR, C = 4, as in M6/M7. α = 0.05 throughout.

### Tests
- **T0** (M7 baseline): 6-way read-out; calibration = the 30 reference students of the other line;
  flag iff p_out ≤ α.
- **T1** (M7): T0 **and** p_rel(b) ≤ α for every relative b the owner has references for, where p_rel
  is the one-sided t prediction-interval p-value against b's reference students under the pairwise
  a-vs-b read-out.
- **T1-partial** (new): T1 with the references of one relative withheld. The withheld relative is not
  tested, so it is only screened by T0.
- **T2 — pooled-relative** (new): for owner a, a pairwise read-out of a's R300 traces against the
  **pooled** traces of its *available* relatives; p_pool from the pooled reference students of those
  relatives (n = 10 per available relative). Flag iff T0 **and** p_pool ≤ α. Evaluated on the
  **withheld** relative's test students. This asks whether "not me, but in my line" generalises to a
  relative the owner never referenced.
- **T3 — strict, line-internal** (new): no out-of-line calibration at all; the owner uses only its own
  line. Flag iff (i) p_rel(b) ≤ α for both relatives, and (ii) p_own ≥ α, where p_own is the one-sided
  *lower* t prediction-interval p-value of the student's score against the owner's own 10 reference
  students under the 3-way in-line read-out (a student far below the owner's own references is not
  flagged). This is the protocol available to an owner with no access to other vendors' models.

## 3. Arms and pre-registered predictions

### A1 — unknown relative (CPU only, existing data)
For each of the 12 ordered pairs (a → b) per cell, with b's references withheld:
- **P1a (assumption boundary, expected to hold):** T1-partial FPR_rel(a → b) ≥ 0.6 in ≥ 8 of 12 pairs,
  in ≥ 3 of 4 cells. This documents that T1 protects only against referenced relatives.
- **P1b (primary, the open question):** T2 FPR_rel(a → b) ≤ 0.3 in ≥ 8 of 12 pairs **and** mean T2
  TPR ≥ 0.8, in ≥ 3 of 4 cells.
- **Kill:** if P1b fails in ≥ 3 of 4 cells, report that reference-aware testing does **not** generalise
  to unreferenced relatives: the owner must enumerate its lineage, which is a real limitation because
  a lineage can be extended after the owner's test is designed.

### A2 — truly unseen teacher (CPU only, existing data)
The owner is in line L and treats the other line as unknown. Read-outs for T3 are trained only on L's
three teachers' R300 traces, so the unknown line is never a class.
- Endpoints per cell: T3 mean TPR over the 3 owners of each line; T3 FPR on the 30 test students of
  the other line; the same under T0 and T1 for comparison.
- **P2 (primary):** T3 unknown-teacher FPR ≤ 0.2 **and** mean TPR ≥ 0.8, in ≥ 3 of 4 cells.
- **Kill:** if T3 mean TPR < 0.8 in ≥ 2 cells, report that the strict protocol has no power and that
  T1 requires public-model calibration to work at all.

### A3 — mixtures (needs GPU: 24 new students)
GSM8K only, the two adjacent pairs that M6 and M7 found least separable: (`tulu_dpo`, `tulu_rlvr`) and
(`olmoi_dpo`, `olmoi_final`).
- Corpora: 1,500 POOL_TEST problems per student, a fraction λ of them answered by teacher a's traces
  and 1 − λ by relative b's, λ ∈ {0.5, 0.1}, seeds 20–22, `default_rng(5000·i + 100·round(10λ) + s)`.
- 2 pairs × 2 λ × 3 seeds × 2 families = **24 students**, trained and probed exactly as in M7.
- **P3:** at λ = 0.5, owner a's T1 flags ≥ 0.8 of the mixture students, in ≥ 3 of 4 (pair × family)
  groups. λ = 0.1 is reported without a gate.
- Also reported: the relative b's T1 flag rate on the same students. Both owners flagging the same
  student is an honest outcome (joint provenance), not a failure.
- Manipulation check for these students: as in M7 (accuracy ≥ base − 0.03 OR length ratio ∈ [0.5, 2.0]).

## 4. Reported (not gating)
- T2 with the number of available relatives = 1 vs 2 (in a 3-stage line, the owner has at most 2).
- Reference budget for T2 and T3 (n_ref ∈ {3, 5, 10}) and query budget (25, 50, 100, 300 probes).
- The same arms under the conformal (non-parametric) p-value at α = 0.1.
- Per-cell tables of all four tests side by side, so the paper can show one figure: FPR on relatives
  and on unknown teachers, by test.

## 5. Not claimed from M8
- Adaptive distillers that deliberately imitate a relative (paraphrase, style transfer). Named as
  future work.
- Long-CoT / Think lines, 7B students, or non-LoRA distillation.
- Anything about lineages with more than three stages.

## 6. Compute
- A1 and A2: scoring only, on existing M7 outputs; CPU, a few hours.
- A3: 24 LoRA students plus probes, about 3 GPU-hours.
