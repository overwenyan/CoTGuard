# M9b — imitation attack on every adjacent pair, both directions, with an answer-preserving rewrite — pre-registration

_2026-09-16. Committed before any M9b data exists. Context: EXP-M9 (G2/G3 failed, but 2 of 4 AD2 corpora were
voided by the answer-preservation check), advisor round 5 (proposed an RL-narrowing mechanism), and my correction
of the same date: **M9 contains no stage-direction rule** — the two successful attacks went in opposite directions._

## 1. Questions
1. **Replication with valid corpora:** does targeted imitation still defeat T1 (evasion and spoofing) when the
   rewrite reliably preserves the final answer?
2. **Direction:** is imitating the *later* stage easier than imitating the *earlier* one?
3. **Mechanism:** what predicts attack success — how close owner and target already are, or how narrow either
   stage's output distribution is?

## 2. Attacks (GSM8K; rewriter `Qwen/Qwen2.5-7B-Instruct`; owner does not anticipate the attack)
All four adjacent pairs, both directions → **8 (owner → imitated target) attacks**:

| Line | Pair | Toward later | Toward earlier |
|---|---|---|---|
| Tulu | SFT–DPO | tulu_sft → tulu_dpo | tulu_dpo → tulu_sft |
| Tulu | DPO–RLVR | tulu_dpo → tulu_rlvr | tulu_rlvr → tulu_dpo |
| OLMo-Instruct | SFT–DPO | olmoi_sft → olmoi_dpo | olmoi_dpo → olmoi_sft |
| OLMo-Instruct | DPO–final | olmoi_dpo → olmoi_final | olmoi_final → olmoi_dpo |

- **Rewrite prompt (AD2s, "strict"):** M9's imitation prompt with 4 in-context traces of the target, plus two
  added constraints: *"Keep every calculation and every number exactly. The last line of your rewrite must state
  the same final answer as the original, in the form 'The answer is N.'"* Only the imitation condition is run;
  neutral paraphrase already passed (M9 G1).
- Source traces: the owner's POOL_TEST traces. Corpora: 1,500 problems, `default_rng(9000·i + 10·j + s)`
  (i = owner index, j = target index), seeds 40–42.
- Students: 8 attacks × 3 seeds × 2 families = **48 new LoRA students**, trained and probed as in M7.
- **Not reusable from M9:** M9's AD2 prompt differed, so none of its students enter M9b.

## 3. Endpoints (M7 tests, unchanged)
- **TPR(owner, attack):** share of the attack's students the owner's T1 still flags.
- **Spoof(target, attack):** share the imitated target's T1 flags.
- **Success(attack)** = 1 − TPR + Spoof, in [0, 2] (a single imitability score used only for the mechanism
  analysis).

## 4. Pre-registered predictions and analyses
- **Manipulation check (corpus level):** extracted final answer matches the original in ≥ 90% of traces, mean
  length ratio in [0.5, 2.0]. Void corpora are excluded from every gate and analysis, and reported.
- **G2b (replication of evasion):** mean TPR over valid attacks ≥ 0.8, in both families. *Expected to fail*,
  replicating M9. If it **passes**, M9's G2 failure is reported as not robust to a better-controlled rewrite.
- **G3b (replication of spoofing):** mean Spoof over valid attacks ≤ 0.3, in both families. *Expected to fail.*
  If it **passes**, M9's ambiguity-attack claim is withdrawn from the abstract and reported as prompt-dependent.
- **H-dir (direction, confirmatory):** over valid attacks, mean Success toward the later stage > mean Success
  toward the earlier stage, tested with an exact paired permutation test over the 4 pairs (both families pooled,
  one-sided, α = 0.05). With only 4 pairs the smallest attainable one-sided p is 1/16 = 0.0625, so **H-dir cannot
  reach significance at α = 0.05 by construction**; it is therefore reported as an effect size with the exact p,
  and the paper may not state a direction rule unless all 4 pairs agree in sign in both families.
- **H-mech (mechanism, correlational):** Spearman correlation between Success and each of
  (a) owner–target closeness = 1 − per-output AUC of their pairwise read-out on M7 test students;
  (b) target narrowness = mean pairwise TF-IDF cosine of the target's R300 traces;
  (c) owner narrowness, same measure.
  n = up to 8 attacks per family. Exact permutation p-values. **Prediction:** (a) is the strongest predictor
  (closer pairs are easier to spoof), because it is the coverage account's quantity. The RL-narrowing account
  predicts (b) instead. Whichever is stronger is reported; neither is claimed as causal.

## 5. Not claimed from M9b
Causality; MATH; attacks with a different rewriter; attacks optimised against the read-out.

## 6. Compute
8 rewrites × 2,000 traces on vLLM (~15 GPU-minutes each), 48 LoRA students plus probes (~6 GPU-hours).
