# M2 — Black-box collusion detection with exact false-positive control

_Design v0 and pre-registration, 2026-09-11. Written after structural inspection of NARCBench
(file formats; an explicit-leak keyword scan) and **before computing any detection score**.
Candidate board: `idea_board_v4.json` (M2). M1 and P2 were stopped by their gates._

## 0. Prior work, checked

- **Rose et al., arXiv 2604.01151 (NARCBench).** Uses white-box activation probes: 1.00 AUROC
  in-distribution, 0.73–0.93 zero-shot, and 0.99–1.00 on Stego. The only text-level baseline is
  qualitative: "an LLM judge monitoring the table talk fails to discriminate between collusion and
  control runs." They report no text or behavioral AUROC. They call their white-box assumption
  "a strong assumption."
- **Ghanem, arXiv 2608.02698.** Black-box cross-run steganalysis with an FPR budget; its results
  are marked placeholders.
- **Data available** (HF `aaronrose227/narcbench`, `scenarios/`):
  - Qwen3-32B: Core 50 collusion / 50 control; Transfer 36 collusion / 36 implicit / 36 control;
    Stego 20 / 20.
  - GPT-OSS-20B: Core 18 / 50; Transfer 13 / 18; Stego 20 / 20.
  - The Llama and DeepSeek transcripts are not in the release.
- **Explicit-leak scan** (a keyword regex over public messages, collusion vs control): Qwen3 Core
  5/50 vs 2/50; Transfer 6/36 vs 9/36; **Stego 17/20 vs 5/20**; similar for GPT-OSS. Core and
  Transfer are covert at the keyword level; Stego table talk is not.

## 1. Problem

Label unit: one run, collusion vs control. The monitor is **black-box**: it sees only what a
cross-principal auditor sees.
- **Core / Transfer:** public messages with speaker names, plus final votes.
- **Stego:** public table talk, bets, and the public cards (the monitor can compute the count
  itself from the dealt cards; the dataset's `true_count` is used as that computation).

It never sees private messages, thinking, notes, or `final_texts` (which in Stego contain
colluders' own decoding). A `final_texts` variant is reported separately.

Goal: a detector whose false-positive rate on control runs is guaranteed ≤ β, with high power.

## 2. Method (classic TML: behavioral statistics, permutation tests, Fisher combination, conformal calibration)

- **S_lex, coordination in text.** For each agent pair, compute the TF-IDF cosine between their
  concatenated public messages. The fit is on training folds only; the vectorizer never sees the
  test scenario. The run score is the maximum pair similarity minus the mean of the other pairs,
  because two coordinated agents are more similar to each other than to the rest.
- **S_beh, coordination in public actions.**
  - *Core / Transfer:* the vote-split statistic (1 if the table splits 2–2 with some pair voting
    against the other two, graded by how unusual the pair's option is under control runs of the
    same scenario family).
  - *Stego:* for each player, Spearman ρ between bet and true count over the shoe's hands; score =
    max over players of ρ. A counting team's big player bets with the count; honest players do not.
- **Combined detector S_comb.** Each statistic becomes a conformal p-value against control runs
  in the calibration fold; they are combined by Fisher's method and turned into a final conformal
  p-value on calibration controls. Declare collusion iff p ≤ β. This gives exact FPR ≤ β under
  exchangeability of control runs.
- **Baselines.**
  - The in-game LLM judge verdict (`flagged` / `judge_verdict`, free in Stego).
  - Our own LLM judge on the same public view: Qwen3-14B, thinking disabled, asked "Is there
    covert coordination between any two agents?" with a 0–100 score; one GPU job, if the CPU
    detectors pass.
  - Each single statistic alone.
  - The published white-box probe numbers, cited and not rerun.

## 3. Small theory

- **Prop 1.** With calibration and test control runs exchangeable, P(false alarm) ≤ β
  (conformal p-value).
- **Prop 2 (power, Stego).** With H hands per shoe and big-player bets that are a monotone
  function of the count plus noise, the test's power → 1 as H grows. Stated as a Spearman
  consistency remark, with the empirical power curve over the first h ≤ H hands.

## 4. Protocol

- **Splits:** leave-one-scenario-out for Core and Stego. Collusion and control runs of the same
  scenario always stay in the same fold.
- **Zero-shot transfer:** calibrate on Core controls and test on Transfer (paper protocol).
  Implicit runs are reported separately and are not in the gate.
- **Metrics:** AUROC with 95% bootstrap CIs over runs, TPR at nominal FPR β = 0.05, and empirical
  FPR on held-out controls.

## 5. Pre-registered gate

- **G2-0 (trivial-baseline audit; reported, it changes the claims, not the go/no-go).** For each
  tier, S_beh alone. If it reaches AUROC ≥ 0.90 on a tier, that tier cannot tell detectors
  apart. We then report it as a benchmark finding ("public actions already reveal the
  collusion") and do not claim a method win there.
- **G2-a (primary; Stego, where the LLM judge reportedly fails).** For **both** generating
  models, S_comb reaches AUROC ≥ 0.75 **and** TPR ≥ 0.30 at nominal β = 0.05, with empirical FPR
  on held-out controls ≤ 0.10.
- **G2-b (covert tier).** On Qwen3 Core (leave-one-scenario-out), S_comb or S_lex reaches AUROC
  ≥ 0.70. This is the tier where G2-0 is least likely to trivialize.
- **Decision.** Continue M2 iff G2-a **and** G2-b pass. If G2-a fails, black-box detection fails
  where white-box probes succeed, which supports the paper's white-box assumption; record it as an
  analysis result and move to M3. If only G2-b fails, report and let the user decide.

## 6. Resources and risks

- **Resources.** CPU only for the gate; the LLM-judge baseline is one short GPU job, run only if
  the CPU gate passes.
- **Small n.** Stego has 20 + 20 runs per model, so AUROC CIs are about ±0.15; reported with CIs.
- **Stego table talk leaks keywords (17/20).** A trivial keyword baseline may already work. It is
  reported under G2-0 in the same way as S_beh.
- **GPT-OSS Core is imbalanced** (18 collusion / 50 control); AUROC handles it, TPR@β is reported.
- **Scoop risk.** The Oxford group is active; speed matters.
