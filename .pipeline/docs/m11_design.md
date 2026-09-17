# M11 — does the calibration diagnosis survive a non-lexical read-out? — pre-registration

_2026-09-16. Committed before any M11 scoring exists. Context: advisor round 6 ranks this the single must-run item
before submission — every headline result since M6 used one read-out (TF-IDF 1–2 gram + LR), so "calibration, not
capability" could be a lexical artefact. No new training: M11 re-scores existing M7 and M10 students._

## 1. Question
Do (a) the no-reference collapse (T0 flags same-line relatives) and (b) the reference-aware remedy (T1 restores
specificity at full power) replicate when the read-out does not see surface n-grams?

## 2. Read-outs (definitions copied verbatim from M5's `score_m5.py`, used there before any M6–M10 data existed)
- **EMB:** `thenlper/gte-base` sentence embeddings (normalised, 512-token limit, so long outputs are truncated),
  logistic regression C = 4. Semantic, not lexical.
- **POS:** nltk `pos_tag` over whitespace tokens → TF-IDF over POS-tag 2–4-grams (max 50,000 features, no lowercasing),
  logistic regression C = 4. Syntactic template, no word identity — the "Who Taught You That?" feature family.
- **TF-IDF** (the original) is recomputed in the same run as a control; it must reproduce M7/M10.

## 3. Scope (existing students only)
- **M7 cells:** GSM8K and MATH × Qwen / Llama, 6 AllenAI teachers, 10 reference + 10 test students each.
- **M10 cell:** GSM8K, Zephyr SFT/DPO with the 8-teacher read-out, both families.
- Tests T0 and T1 exactly as in M7 (T0: calibration on reference students outside the owner's line; T1: T0 and a
  one-sided t prediction interval against each same-line relative's 10 reference students, α = 0.05).

## 4. Pre-registered predictions and decision rules (per non-lexical read-out R ∈ {EMB, POS})
- **N1_R (collapse replicates):** as M7's H1 — in a cell, T0 relative FPR ≥ 0.6 in ≥ 6 of 12 ordered pairs.
  Holds iff it holds in both GSM8K cells. Zephyr: ≥ 1 of 2 ordered pairs in both families.
- **N2_R (remedy replicates — primary):** as M7's H2 — in a cell, mean T1 TPR ≥ 0.8 **and** T1 FPR_rel ≤ 0.2 in
  ≥ 10 of 12 pairs **and** mean T1 FPR_rel ≤ 0.1. Holds iff both GSM8K cells and ≥ 1 MATH cell. Zephyr: both
  ordered pairs FPR ≤ 0.2 and TPR ≥ 0.8, both families.
- **Sentinel:** the TF-IDF control must reproduce M7's H2 verdict and M10's R2 verdict; otherwise the run is void.
- **Decision rules, fixed now:**
  - **N2 holds for at least one non-lexical read-out** → the diagnosis is read-out-general; the paper reports all
    three read-outs in the headline table.
  - **N2 fails for both EMB and POS** → the constructive half is **lexical-read-out-specific**. The claim "not an
    information limit" is rescoped to "not an information limit *for a lexical read-out*"; the abstract must say so.
  - **N1 fails while N2 holds for a read-out** → the collapse is itself read-out-dependent (that read-out's
    out-of-line calibration already rejects relatives); reported as such, and the diagnosis must then be phrased as
    a property of the read-out's null, not of provenance tests in general.
- **Expectation (reported, not gating):** EMB per-output AUC for adjacent stages will be *lower* than TF-IDF's, since
  adjacent checkpoints solve the same problems with near-identical content; if EMB still separates them, the
  identity signal is not purely surface form.

## 5. Exploratory, not gating: what the imitation attack changes (M9b diagnostic)
For each valid M9b attack (6) × family: the owner-vs-target TF-IDF logistic-regression coefficient vector; the shift
in mean feature vector of attacked students relative to the owner's unattacked M7 test students; its projection onto
the discriminative direction (negative = moved toward the target); and the top 10 n-grams driving the move in each
direction. Reported qualitatively to show *what* imitation changes, without claiming a mechanism.

## 6. Not claimed
Anything about read-outs beyond these three; learned authorship encoders (e.g. LUAR) are named as future work.

## 7. Compute
EMB: ~180,000 outputs through gte-base on one L40S (minutes). POS: nltk tagging of the same outputs on CPU
(under an hour). Scoring: CPU, as M7.
