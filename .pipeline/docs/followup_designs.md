# Follow-up designs (post-submission) — recorded 2026-09-19, advisor round 13

_Status: **design notes, not pre-registrations.** Each becomes a dated `mNN_design.md` committed before any data for it
exists. Nothing here touches the ACL submission._

---

## D7 — a content (retrieval) channel beside the style test

**Why a naive retrieval channel reopens the coverage failure.** Moving from held-out probes to the owner's *published*
problems changes the null from "generalises my style" to "reproduces my traces". On a published problem x, every
competent student gives the same answer and most of the same mathematics, so "similar to the owner's trace on x" is high
under any null that solved x. Other-line students on the same problems will look far from the owner and a sibling's
student will look close — for exactly the reason T0 collapsed. **No calibration trick escapes this: any channel measured
on shared problems needs sibling references.**

**Design: paired, per-instance, per-relative.** For each published problem x and each sibling b of owner a:

  d_x(y) = sim(y_x, t^a_x) − sim(y_x, t^b_x),

the suspect's output on x against the owner's actual trace on x, minus against the sibling's actual trace on x. Reference
distributions of d_x come from b's reference students (under "trained on b") and a's (under "trained on a"). This is
T1's per-relative structure carried over unchanged: the coverage requirement is **inherited and stated**, not reopened.
The channel's contribution is robustness to style rewriting, not escape from calibration. It is also the per-instance
keyed test M4 wanted, with the owner's real trace as the key instead of a secret partition — the key is content the
sibling did not write, which is why it should have the specificity M4 lacked.

**Predictions to fix before looking.**
1. *Unattacked siblings:* the content channel is **weaker** than style on adjacent stages (M11: embedding per-output AUC
   0.60–0.66 on RL-era neighbours). Predict the ordering, with a threshold.
2. *Crossover under imitation:* style TPR fell to 0.58 while content was preserved by construction; predict retrieval
   **holds** where style failed (pre-register the attacks and the margin).
3. *Price curve:* the attack axis is rewrites that alter **step content** (not wording), which are expected to cost
   accuracy. Plot attacker accuracy vs provenance survival for both channels and their combination.
4. *Limits outcome, named in advance:* if a cheap content rewrite defeats both channels at no measurable accuracy cost,
   the paper is a limits result.

**Open design choices to settle in the pre-registration:** sim() (embedding cosine vs step-aligned edit distance vs
answer-path overlap); how to aggregate d_x over problems (mean with a t interval, as T1; or a sign test); how many
published problems per suspect; the floor for any rank/conformal gate (n_cal ≥ 19 at α = 0.05).

---

## D1 — non-length style in students that failed to learn capability

**Claim under test:** *non-length* style survives in the Think-line students that fell below base accuracy. Length itself
transferring is a form of style; D1 is about what remains once length carries no information.

**Why truncation is not the control.** Cutting outputs to 400 tokens keeps the format signal (tags, opening scaffold) and
removes late content, so its AUC (0.75–0.78) is uninterpretable as a residual.

**Two conditions, fixed in advance.**
- *Matched:* for each Think-line output, pair a comparison output from the same length bin (±10% tokens); classify on the
  matched sets only.
- *Stripped-and-matched:* additionally remove think tags, "Step N:", `\boxed{}` and answer lines before matching.

**Threshold relative to the confound.** Train a length-only baseline (token count, sentence count, mean sentence length)
on the *unmatched* data — the AUC length alone buys. **Style survives** iff, in the matched condition, the full read-out
exceeds the length-only baseline by **≥ 0.10 AUC** *and* clears a permutation null for AUC at the 95th percentile, in
**≥ 3 of 4** family × stage units.

**Capability gate (separate):** accuracy ≤ base − 0.03 on the same probes, with the unit-tested extractor (v2; boxed
extraction for MATH).

**Consequences.** Both gates pass → the positive control stands ("style without capability"). Either fails → report it,
and D1 becomes a paragraph in D3. Keep clear of the retracted capability–identity dissociation: this is a different
claim, and it must be stated as such.

---

## Order after submission
D7 first, D3 (decomposition of the inherited signal) alongside, D1 only if its gates pass. D10 (benchmark release) after a
licence table. D5 waits for the ACL decision; D8 needs new training.
