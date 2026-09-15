# M7 — reference-aware owner test: do same-lineage reference students restore specificity? — pre-registration

_2026-09-15. Committed before any M7 data exists. Decision context: EXP-M6 (ledger) and the decision
log entry of the same date; user choice "Confirm reference effect"._

## 1. Question and motivation
In M6 the pre-registered owner test, calibrated only on students of teachers **outside** the owner's
post-training line, flagged same-line relatives with FPR = 1.0 whenever the owner was a DPO or final
checkpoint. An exploratory analysis on the same students found that a pairwise read-out, with a
threshold set from the other students of both teachers, separated every pair perfectly (student AUC
1.00), including DPO vs RLVR.

That analysis has three weaknesses:
- it was post hoc;
- it used 5 students per side;
- its reference and evaluated students were trained on the **same** teacher traces for overlapping
  problems.

**M7 asks, on fresh students trained on fresh traces for disjoint problems:** does giving the owner
reference students of each same-line relative restore specificity (low FPR on relatives) without losing
power?

## 2. Teachers
The six non-Think checkpoints of M6, with lineage links verified from model cards (see `m6_design.md`):

| Line | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| Tulu | `tulu_sft` | `tulu_dpo` | `tulu_rlvr` |
| OLMo-Instruct | `olmoi_sft` | `olmoi_dpo` | `olmoi_final` |

The OLMo-Think line is excluded: in M6, 1–1.5B LoRA students did not absorb its 4–7k-character traces
(manipulation-check failure, below-base accuracy). Teacher index i (for corpus seeds) = M6 index:
tulu 1–3, olmoi 4–6.

## 3. Data (two datasets)
Generation for both datasets:
- vLLM, each model's chat template, T = 0.7, top-p 0.95, ≤ 4,096 new tokens, as in M6;
- per-request seed 10,000 + position for read-out and reference problems, 20,000 + position for test
  problems.

### 3.1 GSM8K
- **Prompt:** as in M3–M6.
- **R300 and POOL_REF:** M6's R300 and POOL (1,906 problems), with M6's existing teacher traces.
- **POOL_TEST:** 2,000 GSM8K train problems **not** in M6's generation set, drawn with
  `default_rng(11)`. They get new teacher generation.
- **Probes:** M6's 300 GSM8K test problems.

### 3.2 MATH (Hendrycks, EleutherAI/hendrycks_math, all 7 subjects, Levels 1–4)
- **Prompt:** "Solve the problem. Think step by step, one step per line, and put the final answer in
  \boxed{}."
- **Splits:** train problems are permuted with `default_rng(1)`, then split into R300 = first 300,
  POOL_REF = next 2,000, POOL_TEST = next 2,000.
- **Probes:** 300 test problems (Levels 1–4) with `default_rng(2)`.
- **Correctness:** normalised string or numeric match of the last `\boxed{}` in the output against the
  last `\boxed{}` in the reference solution.

### 3.3 Students (LoRA, as in M6: r = 32, 3 epochs, 1,500-problem corpora, max length 4,608)
Two families, Qwen2.5-1.5B-Instruct and Llama-3.2-1B-Instruct, as in M6.

| Dataset | Role | Seeds s | Corpus source | Corpus RNG | New students |
|---|---|---|---|---|---|
| GSM8K | reference | 0–4 (**existing M6 students**), 5–9 (new) | POOL_REF traces | `default_rng(100·i + s)` | 6 × 5 × 2 = 60 |
| GSM8K | test | 10–19 | POOL_TEST traces (fresh) | `default_rng(1000·i + s)` | 6 × 10 × 2 = 120 |
| MATH | reference | 0–9 | POOL_REF traces | `default_rng(100·i + s)` | 6 × 10 × 2 = 120 |
| MATH | test | 10–19 | POOL_TEST traces (fresh) | `default_rng(1000·i + s)` | 6 × 10 × 2 = 120 |

- Total new students: **420**.
- Reference and test students never share a problem or a teacher trace.
- Every student answers its dataset's 300 probes (≤ 4,096 new tokens, T = 0.7, seed 7 + position).

## 4. Tests (primary read-out TF-IDF 1–2 gram + LR, C = 4; per dataset × family "cell")
A student's **score** under a read-out = its mean predicted probability over its 300 probe outputs.

- **T0 — out-of-line owner test (M6 E2 replication).**
  - Read-out: 6-way, trained on the six teachers' R300 traces.
  - Owner a; calibration = the 30 **reference** students of the three teachers in the other line.
  - p_out = (1 + #{cal ≥ s}) / 31; flag iff p_out ≤ 0.05.
- **T1 — reference-aware owner test.** Flag iff T0 flags **and**, for each of the two same-line
  relatives b of a, p_rel(b) ≤ 0.05, where:
  - pairwise read-out a vs b, trained on the R300 traces of a and b;
  - m_b, sd_b = mean and SD of the scores of b's 10 reference students;
  - p_rel(b) = 1 − F_{t, 9}((s − m_b) / (sd_b · √(1 + 1/10))), a one-sided t prediction-interval p-value.
  - (Intersection–union: all three p-values must be ≤ 0.05.)
- **Endpoints** on **test** students only:
  - TPR(a) = share of a's 10 test students flagged;
  - FPR_rel(a → b) = share of b's 10 test students flagged under owner a;
  - FPR_out(a) = share of the other line's 30 test students flagged under owner a.

## 5. Pre-registered predictions and kill criteria
Each dataset × family cell has 12 ordered (owner, relative) pairs: 2 lines × 3 owners × 2 relatives.

- **H1 (no-reference collapse replicates on fresh students).**
  - A cell holds iff T0 FPR_rel ≥ 0.6 in ≥ 6 of 12 ordered pairs.
  - H1 passes iff it holds in both GSM8K cells. It is reported for MATH.
- **H2 (primary: reference students restore specificity).**
  - A cell holds iff all three conditions hold:
    - (i) mean T1 TPR over the six owners ≥ 0.8;
    - (ii) T1 FPR_rel ≤ 0.2 in ≥ 10 of 12 ordered pairs;
    - (iii) mean T1 FPR_rel ≤ 0.1.
  - **H2 passes iff it holds in both GSM8K cells and in at least one MATH cell.**
- **H3 (secondary: step-specific structure on MATH).**
  - Per MATH family, E1-style per-output AUC (pairwise read-out on R300, evaluated on test-student
    outputs). H3 holds iff the mean over the two DPO→final pairs is lower than the mean over the two
    SFT→DPO pairs by ≥ 0.05.
  - Reported in both families; not a kill criterion.
- **Kill / restriction rules:**
  - **H2 fails in both GSM8K cells** → the exploratory M6 claim is killed. Report: "reference students
    do not restore specificity on fresh students; the M6 separation did not survive fresh traces." The
    same-lineage limit then stands as stated in M5/M6.
  - **H2 holds in at least one but not both GSM8K cells** → H2 fails as pre-registered. Report the claim
    as family-dependent.
  - **H2 holds in both GSM8K cells, fails in both MATH cells** → report the claim as GSM8K-only.
  - **H1 fails in both GSM8K cells** → M6's no-reference collapse did not replicate on fresh students.
    Report it; the H2 contrast is then uninformative about "restoration".
- **Manipulation check** (loosened from M6; rationale below):
  - (a) Teacher: extractable answer ≥ 70% on its POOL_TEST traces. MATH: a `\boxed{}` is present;
    GSM8K: `extract_answer` is not None.
  - (b) Student: probe accuracy ≥ base accuracy − 0.03, OR mean probe-output characters / mean teacher
    trace characters ∈ [0.5, 2.0].
  - A teacher is void in a dataset if (a) fails or if more than 20% of its students in that dataset
    (reference + test, both families) fail (b).
  - Void teachers' pairs are reported but excluded. The "10 of 12" and "6 of 12" thresholds become the
    same proportions of the remaining pairs, rounded up.
  - Rationale: M6's [0.67, 1.5] band voided teachers whose students were clearly shaped by them (their
    identity was recovered at AUC ≥ 0.93). The band was a poor proxy for "the student learned from the
    teacher".
  - This change is made for a new round, before its data, and is not applied to M6.

## 6. Reported (not gating)
- **Reference budget:** T1 with n_ref ∈ {3, 5, 10} (the first n seeds), with the t-quantile df adjusted.
- **Query budget:** T1 with 25, 50, 100 and 300 probes (median over 20 random subsets).
- **Non-parametric variant:** p_rel as a conformal rank against the 10 reference students
  (minimum 1/11, so reported at α = 0.1).
- **Shared-trace leakage:** on GSM8K, M6-style separation between reference students s0–4 and s5–9
  (shared POOL_REF traces), compared with reference vs test (fresh traces).
- **E3 capability:** probe accuracy per stage per dataset, and Δacc for each pair.
- Embedding read-out (gte-base) for T0 and T1, if time permits.

## 7. Not claimed from M7
- Long-CoT or Think-model attribution.
- Truly unseen teachers.
- Any setting where the owner cannot train reference students on a relative. T1 assumes the relatives
  are public checkpoints the owner can distil from, as all of these are.

## 8. Compute and order
1. **Smoke test** (`M7_SMOKE=1`): tiny splits, 1 teacher per dataset, 1 student, probes, scorer end to
   end.
2. **Generation:** GSM8K POOL_TEST (6 teachers × 2,000) and MATH (6 × 4,300), on L40S.
3. **420 LoRA students** on H200 (about 30 GPU-hours, up to 7 in parallel).
4. **Probes** on L40S.
5. **Score.**

Estimate: about one day of wall-clock time.
