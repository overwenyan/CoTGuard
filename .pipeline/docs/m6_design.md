# M6 — alignment-stage ladder: distinguishability vs post-training distance — pre-registration

_2026-09-15. Committed before any M6 data exists. Decision context: `decision_log.md` (advisor:
hybrid A+C; user: ladder first, with vLLM set up in parallel, lineages verified, dissociation
check). Related results: EXP-M5 (Tulu RLVR ⇄ DPO confused 100%) and EXP-M5b (a capability gap
transfers while identity does not)._

## 1. Question
Within one post-training line, how does the distinguishability of *students* distilled from
different stages of the same model grow with the distance between those stages? Does the capability
gap between stages transfer to students even where identity is not recoverable?

## 2. Teachers: nine checkpoints in three post-training lines
Every link below was verified from the model card's `base_model` field on 2026-09-15.

| Line | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| **Tulu-3-8B** (base Llama-3.1-8B) | `allenai/Llama-3.1-Tulu-3-8B-SFT` ← Llama-3.1-8B | `allenai/Llama-3.1-Tulu-3-8B-DPO` ← SFT | `allenai/Llama-3.1-Tulu-3-8B` (RLVR final) ← DPO |
| **OLMo-3-7B-Instruct** (base Olmo-3-1025-7B) | `allenai/Olmo-3-7B-Instruct-SFT` ← Olmo-3-1025-7B | `allenai/Olmo-3-7B-Instruct-DPO` ← Instruct-SFT | `allenai/Olmo-3-7B-Instruct` (final) ← Instruct-DPO |
| **OLMo-3-7B-Think** (base Olmo-3-1025-7B) | `allenai/Olmo-3-7B-Think-SFT` ← Olmo-3-1025-7B | `allenai/Olmo-3-7B-Think-DPO` ← Think-SFT | `allenai/Olmo-3-7B-Think` (RLVR final) ← Think-DPO |

**Relations used in the analysis:**
- **d = 1:** adjacent stages in one line (SFT–DPO, DPO–final).
- **d = 2:** SFT–final in one line.
- **"sibling line":** OLMo-Instruct vs OLMo-Think at the same stage (same base, different post-training).
- **"different base":** Tulu vs OLMo.

Base models are excluded: they do not follow the chat prompt.

## 3. Data
- **Problems:** `problems("train", 7000, seed=1)`, with the same splits as M5.
  - **R300:** read-out training.
  - **POOL** = S2000 \ R300: student corpora.
  - **Probes:** 300 GSM8K test problems (`seed=2`).
- **Teacher generation (vLLM):** the M3/M5 prompt "Solve the problem. Think step by step, one step per
  line.", each model's own chat template with its default thinking behaviour (Think models reason in
  their native format), T = 0.7, top-p 0.95, **≤ 4,096 new tokens** for all nine. _(Amended before
  any data from 1,024: Think models may reason past 1,024 tokens on GSM8K, which would void the Think
  line under the manipulation check.)_
  - The truncation rate and answer-extraction failure rate are reported per teacher.
  - Truncated traces are *kept*: a distiller would see them.
- **Students:** Qwen2.5-1.5B-Instruct and Llama-3.2-1B-Instruct, LoRA as in M3, 3 epochs, max
  sequence length **4,608** tokens (prompt + trace; amended with the generation limit).
  - For each teacher, 5 corpora of 1,500 POOL problems with `default_rng(100·i + s)`, i = teacher index
    1–9, s = 0–4.
  - 9 × 5 × 2 = **90 students**.
- **Probes:** each student answers the 300 probes, ≤ 4,096 new tokens, T = 0.7, seed 7 (vLLM with
  the LoRA adapter).

## 4. Read-outs
- **Primary:** TF-IDF (1–2 gram) + logistic regression.
- **Reported:** gte-base embedding + LR, and POS-template (2–4 gram) + LR, as in M5.
- **Pairwise read-out for teachers (a, b):** a binary classifier trained on R300 traces of a vs b.

## 5. Estimands and analyses
- **E1 per-output distinguishability** (the graded curve), for every unordered pair (a, b) with
  d ∈ {1, 2} or sibling line, and each student family:
  - AUC of the pairwise read-out on student probe outputs (1,500 outputs from a's students vs 1,500
    from b's).
  - 95% CI by cluster bootstrap over students (resample students within each side; 2,000 draws).
  - Pairs: 3 lines × 3 within-line pairs + 3 sibling pairs (OLMo Instruct vs Think per stage)
    = 12 pairs × 2 families × 3 read-outs.
- **E2 student-level owner-test FPR** (the attribution endpoint), for every ordered within-line pair
  (a → b):
  - owner a; calibration = the 30 students of the teachers **outside a's line** (same family);
  - 9-way read-out on all nine teachers' R300 traces; score = mean P(a);
  - p = (1 + #{calibration ≥ s}) / 31;
  - FPR(a → b) = share of b's 5 students with p ≤ 0.05;
  - TPR(a) = share of a's own 5 students with p ≤ 0.05.
- **E3 capability transfer:** teacher accuracy on POOL traces and student accuracy on probes per
  stage. Δacc per pair with problem-level bootstrap CIs, set against E1.
- **E4 truncation control:** E1 recomputed after truncating every student output and every read-out
  training trace to its first 400 tokens. This controls for length being the only signal (Think
  models).

## 6. Pre-registered predictions and kill criteria (primary read-out = TF-IDF, both families)
- **P1 (adjacent collapse):** there are 12 ordered adjacent pairs per student family (3 lines ×
  2 adjacent pairs × 2 directions). P1 holds in a family iff owner-test FPR ≥ 0.6 for ≥ 8 of these
  12. P1 passes iff it holds in both families.
- **P2 (monotonicity):** per student family, P2 holds iff (i) the mean per-output AUC over the 3
  d = 2 pairs exceeds the mean over the 6 d = 1 pairs, AND (ii) in ≥ 2 of the 3 lines,
  AUC(SFT–final) ≥ max(AUC(SFT–DPO), AUC(DPO–final)) − 0.02. P2 passes iff it holds in both families.
- **P3 (sibling lines are distinguishable):** for each stage, owner a = the OLMo-Instruct checkpoint and
  sibling b = the OLMo-Think checkpoint at the same stage.
  - A stage *meets* P3 iff the per-output AUC(a, b) ≥ 0.9 AND the sibling owner-test FPR ≤ 0.2.
  - Sibling FPR = share of b's 5 students with p ≤ 0.05 under owner a, where calibration = the
    students of all teachers outside a's line other than b (5 teachers × 5 = 25; p = (1 + #) / 26).
  - P3 holds in a family iff ≥ 2 of 3 stages meet it; P3 passes iff it holds in both families.
  - Different post-training on a shared base is expected to be identifiable.
  - _(Precision amendment before any data: the first version did not define the sibling FPR's
    calibration set or how the families combine.)_
- **Kill / demotion rules:**
  - **Kill the "graded curve" sub-claim** if P2 fails in both families. Report "no monotone structure":
    adjacent and distant stages are equally (in)distinguishable.
  - **Kill the "adjacent collapse generalises" sub-claim** if P1 fails in both families. The M5 Tulu
    result is then reported as lineage-specific.
  - **If truncation (E4) removes the Think-line distinguishability entirely** (AUC < 0.6 where the full
    output gave ≥ 0.9), report that the Think signal is length or format, not style.
  - **Manipulation check** (precise):
    - (a) a teacher's POOL traces must have an extractable final answer (`extract_answer` not None)
      in ≥ 70% of cases;
    - (b) for each student, either GSM8K probe accuracy ≥ its untuned base model's accuracy on the same
      300 probes, OR mean probe-output characters / mean teacher POOL-trace characters is within
      [0.67, 1.5].
    - A teacher failing (a), or with more than 2 of its 10 students failing (b), is void. Its pairs are
      reported but excluded from P1–P3; void pairs do not count toward the ≥ 8 of 12 or the 2 of 3
      thresholds, which are then computed over the remaining pairs with the same proportions (rounded
      up).
- **Not claimed from M6:** anything about long-CoT *attribution across families* (a later round), or
  open-set with truly unseen teachers (a later round).

## 7. Compute
Generation: 9 teachers × 2,206 problems with vLLM, a few GPU-minutes to about 1 GPU-hour each (Think
models are longer). 90 LoRA runs at up to 2,048 tokens; 90 × 300 probe generations with vLLM.
Estimated 40–70 GPU-hours, parallel on 7–8 H200s. A smoke test on 1 teacher and 1 student comes
first.
