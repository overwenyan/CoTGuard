# M13 — does the diagnosis and its remedy hold at 7B? — pre-registration

_2026-09-17. Committed **before any M13 code, job or data**. Context: advisor round 10 — read-out generality (M11) and
vendor generality (M10) are closed, so student **scale** is the first limitation a reviewer reaches for. The user
reopened the budget for exactly one cell. Purpose: **delete a limitation sentence, not change a claim.** If it changes
a claim, that is a finding and is reported as one._

---

## Sentence at stake (§6.4 / limitations)

> "All students are 1–1.5B LoRA fine-tunes; the T0-collapse / T1-repair result is untested at larger student scale."

## The one cell

Exactly the AllenAI · GSM8K · Qwen cell of Table 1, with the student scaled up and nothing else changed.

| | 1.5B cell (M7, done) | M13 |
|---|---|---|
| Student | Qwen2.5-**1.5B**-Instruct | Qwen2.5-**7B**-Instruct |
| Teachers | 6 AllenAI (tulu_sft/dpo/rlvr, olmoi_sft/dpo/final) | same 6 |
| Dataset | GSM8K | same |
| Ordered same-line pairs | 12 | 12 |
| Reference students / teacher | 10 | **3** (M7 showed 3 ≡ 10: mean relative FPR 0.008–0.083) |
| Test students / teacher | 10 | **3** |
| Probes / student | 300 | 300 |
| Read-out | TF-IDF 1–2 gram + LR (C = 4) | same, primary and only gating read-out |

Total **36 students** (6 teachers × [3 reference + 3 test]). Teacher traces, splits, prompts, LoRA config (r = 32,
α = 64, 3 epochs, loss on trace tokens) and the 300 probe problems are **reused unchanged** from M7, so the only
changed variable is student size. Reference and test students keep M7's disjoint problems and disjoint traces.

**Resolution caveat, stated in advance.** With 3 test students, every false-positive rate lives on {0, ⅓, ⅔, 1}. The
gates below are written against that grid, and no claim finer than it may be made from this cell.

**Interpretation caveat.** Qwen2.5-7B-Instruct is also the rewriter in §6.3. No attack is run here, so this does not
confound M13, but §6.3's rewriter and M13's student must not be described as independent choices.

## Manipulation check (same rule as M6/M7, applied first)

A teacher is voided in this cell if fewer than 70% of its traces contain an extractable answer (v2 extractor) or if
more than 20% of its students neither stay within three accuracy points of the untuned Qwen2.5-7B-Instruct base nor
stay within a factor of two of the teacher's output length. **If fewer than 4 teachers survive, or fewer than 2 in
either line, the cell is reported as inconclusive and no gate is evaluated.**

## Gates, fixed now

- **G1 — the diagnosis replicates at 7B.** T0 (calibrated on reference students of the 3 teachers **outside** the
  owner's line) flags a same-line relative's test students at false-positive rate ≥ ⅔ in **≥ 6 of the 12** ordered
  same-line pairs. _(The 1.5B cell: 8 of 12.)_
- **G2 — the remedy replicates at 7B.** With T1 (adding, per relative, its 3 reference students and the pairwise
  owner-vs-relative read-out): mean relative false-positive rate **≤ 0.2 in ≥ 10 of the 12** ordered pairs **and**
  owner true-positive rate **≥ 0.9** over the 6 owners. _(The 1.5B cell: 1.00 / 12 of 12 / 0.042.)_
- **Reported, not gating:** per-pair T0 and T1 rates; owner TPR under T0; cross-line false-positive rate; student
  accuracy against the 7B base; the same numbers under POS and EMB read-outs **if** they cost no extra GPU time
  (they are CPU re-scores of the same probe outputs).

## Consequences, fixed now

| Outcome | What the paper says |
|---|---|
| **G1 and G2 pass** | The limitation sentence is deleted and replaced by: "the collapse and its repair replicate with 7B students in one cell (M13)". No claim is strengthened beyond that — one cell, one family, one dataset. |
| **G1 fails** (no collapse at 7B) | **A finding, and the more interesting one.** §5.3 must state that the collapse was not reproduced with 7B students, and the diagnosis is qualified as scale-dependent: larger students may separate same-line checkpoints well enough that cross-line calibration suffices. The paper's §5.3 claim is then explicitly about small students. We do **not** bury this in an appendix. |
| **G1 passes, G2 fails** | The remedy does not carry to 7B. This contradicts the paper's positive claim in the cell where it was tested, and §6.1 must say so in its first sentence. |
| **Inconclusive** (manipulation check) | Report the void reasons; the limitation sentence stays as written, with the attempt and its failure recorded in the integrity appendix. |

**No post-hoc gate changes.** Any deviation from the above after seeing data is a correction, logged in the decision
log with the reason, in the form already used for M8, M9 and M9b.

## Kill criteria (budget)

- Wall-clock: if training is not finished within **24 h** of the first job starting, stop, score whatever students are
  complete, and report the cell as partial with the number of students per teacher stated.
- Any teacher whose students cannot be trained within the L40S/H200 memory budget is dropped and reported, not retried
  with a different recipe (a different recipe would change more than one variable).
- **This is the last experiment.** If G1 or G2 fails, the consequence is a rewritten sentence, not another round.

## Compute plan (RAM rules from `slurm-l40s-ram-cap`)

- Training on **H200** (7B LoRA, bf16, grad checkpointing, batch 2 × accum 2, seq ≤ 1,024), `--mem=96G`, array ≤ 4
  concurrent; 36 students at an estimated 20–45 min each (M7 1.5B: 4–9 min/student) ≈ 12–27 GPU-hours.
- Probes on **L40S** with vLLM + LoRA, `--mem=96G`, `VLLM_USE_FLASHINFER_SAMPLER=0`.
- Scoring on CPU, `--mem=32G`.
- Check `free -h` on the target node before submitting; `scontrol update` of memory is blocked, so fix a wrong request
  by `scancel` + resubmit with the same `--dependency`.

## Integrity

Student keys are namespaced `("m13", teacher, role, seed)` to avoid the M9b key-collision class of bug. The scorer
re-checks M7's 1.5B numbers for the same cell as a sentinel (the M8 lesson: a weaker test cannot have higher power),
and refuses to write results if the sentinel moves.
