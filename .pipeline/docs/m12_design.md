# M12 — two checks that each earn one sentence in the draft — pre-registration

_2026-09-17. Committed before any M12 computation or data. Context: advisor round 8. The draft of §5–§6 exists
(`paper/draft_s5_s6.md`, fb04122); each arm below is tied to the sentence it can change. Capped at two experiments;
held, in order: one 7B cell, a classifier-aware attacker, a non-math task._

---

## Arm A — the geometry rule on held-out data (Zephyr ladder; no training)

**Sentence at stake (§5.4):** "Smaller ratios went with more collapse … the ordering has no sharp threshold."

**Why a threshold rule, not an ordering rule.** The in-sample check (M11 geometry, 12 units = 3 read-outs × 4 AllenAI
cells) found an *ordering*. On the held-out Zephyr units, T0 collapses on every ordered pair under every read-out
(false-positive rate 0.8–1.0; M10/M11), so collapse has no variation to order and within-unit rank correlation is
undefined. The Zephyr data can test only the threshold form of the coverage account: **a pair that collapses should
have its relative closer to the owner than the nearest cross-line teacher (ratio < 1).**

**Units.** 3 read-outs (TF-IDF, POS, EMB) × 2 student families × 2 ordered pairs (zephyr_sft → zephyr_dpo,
zephyr_dpo → zephyr_sft) = **12 ordered pairs**. Cross-line teachers for a Zephyr owner: the six AllenAI teachers.

**Measurement (identical to `geometry_m11.py`).** Feature maps fitted on the eight teachers' R300 traces (TF-IDF, POS)
or gte-base (EMB); each student = mean feature vector over its 300 probe outputs; centroid = mean over a teacher's 10
reference students; ratio = cosine d(owner, relative) / min over the six AllenAI teachers of cosine d(owner, teacher).
Collapse = T0 false-positive rate on the relative ≥ 0.6, taken from the existing M11 Zephyr results.

**Pass rule (G-A).** Among the collapsed ordered pairs (expected: all 12), ratio < 1 in **≥ 10 of 12**.

**Consequences, fixed now:**
- **Pass:** §5.4 may say the coverage geometry predicts collapse on a held-out vendor. The TF-IDF/MATH exception
  (collapse at ratio ≈ 0.95, still below 1) is then consistent with a threshold at 1 plus an ordering below it, and
  the "no sharp threshold" sentence is revised to "collapse occurred only below ratio 1 in all units examined, but its
  severity below 1 is not a fixed function of the ratio".
- **Fail:** §5.4 says the geometry is "consistent with collapse severity in the AllenAI cells, not replicated on a
  held-out vendor", and the threshold reading is dropped.
- **Reported, not gating:** the pooled Spearman over all 18 read-out × cell units (12 AllenAI + 6 Zephyr) between mean
  ratio and collapse fraction.
- **Caution stated in advance:** a pass may be close to guaranteed if a vendor's two checkpoints are simply far from
  every other vendor's. We report the ratios themselves, not only the count, so a reader can judge.

---

## Arm B — scaffold-only rewrite (48 students)

**Sentence at stake (§6.3):** "What the attack changes … is carried almost entirely by reasoning-scaffold phrases …
[claimed descriptively]" and the "formatting provenance" paragraph.

**Question.** Does rewriting **only** the scaffold — step labels, line-opening connectives, the final-answer line —
toward the imitated relative reproduce the effect of full imitation, with the reasoning content left as it was?

**Rewrite (AD2-scaf).** Rewriter `Qwen/Qwen2.5-7B-Instruct`, 4 in-context traces of the target (as M9b), instruction:
*"Change only the formatting scaffold of the solution to match the style of the examples: step labels or numbering,
the words that open each line, and the way the final answer is stated. Copy every other word, every calculation and
every number exactly, in the same order. Output only the rewritten solution."*

**Attacks.** All 8 (owner → target) directions of M9b (4 adjacent pairs × 2 directions); seeds 50–52;
8 × 3 × 2 families = **48 LoRA students**, trained and probed as M7. Owner reference students and read-outs are the
non-adaptive M7 ones. Student keys are namespaced `("ad2scaf", owner, target, seed)` (the M9b key-collision bug).

**Corpus validity (pre-registered, both required):**
1. answer preserved (corrected extractor v2, final answer equal to the original's) in ≥ 90% of traces;
2. **content preserved:** after removing a fixed scaffold lexicon (below) and punctuation, the rewritten trace keeps
   ≥ 90% of the original's numbers (multiset recall) and ≥ 80% of its remaining word tokens (multiset recall), each
   averaged over traces.
   Scaffold lexicon (fixed now): step, steps, first, firstly, second, next, then, finally, lastly, so, thus, therefore,
   hence, now, let, let's, we, calculate, compute, determine, find, state, answer, final, boxed, total, result, solution.

**Endpoints (as M9b):** owner T1 true-positive rate; imitated relative's T1 rate (misattribution); fraction of the
owner→relative discriminant gap closed (as `attrib_m9b.py`).

**Comparison set.** The attacks valid in **both** M9b (full imitation) and M12-B (scaffold-only) — at most 6.

**Pre-registered predictions, both directions:**
- **S1 — scaffold sufficiency.** Per family, scaffold-only matches full imitation within one student of three
  (|Δ owner rate| ≤ 0.34 and |Δ misattribution| ≤ 0.34) in ≥ 5 of 6 comparable attacks (proportionally fewer if fewer
  are comparable, rounded up). **Holds iff in both families.**
- **S2 — content matters.** Per family, mean owner detection under scaffold-only exceeds that under full imitation by
  ≥ 0.25 **or** mean misattribution is lower by ≥ 0.25. **Holds iff in both families.**
- S1 and S2 are mutually exclusive by construction only in the extremes; if neither holds, the outcome is **mixed** and
  reported attack by attack.

**Consequences, fixed now:**
- **S1 holds:** §6.3 may state causally that rewriting the scaffold alone reproduces the attack; "what imitation moves"
  becomes an intervention result. It remains a **diagnostic of the imitation attack**, not a second attack family, and
  is not placed above it in §6.3. It says nothing about *when* the attack succeeds, which stays open.
- **S2 holds:** §6.3 says the scaffold phrases are the most visible shift but content contributes to the attack; the
  formatting-provenance paragraph is weakened to "partly formatting provenance".
- **Mixed:** reported per attack; no causal sentence.
- **If fewer than 4 attacks are comparable** (void corpora), neither prediction is evaluated and the arm is reported as
  inconclusive.

## Process guard (advisor caution)
Tables 1–2 and the step-AUC table of §5–§6 are generated by script from result JSON **before** M12 (snapshot) and
**after**; the draft is updated only from the regenerated tables, never by hand.
