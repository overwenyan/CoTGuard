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

**Similarity (settled, advisor round 14).** sim() must measure what imitation cannot change — the solution path, not
the wording. Two functions, both pre-registered:
- **Baseline: dense cosine** (gte-base). Standard retrieval choice, survives paraphrase, but on a solved problem
  siblings sit almost on top of each other; expect M11's 0.60–0.66.
- **Content channel proper: numeric-step alignment.** Extract the ordered sequence of intermediate quantities (numbers
  and equations the trace computes) with the unit-tested extractor's machinery; score by longest common subsequence over
  that sequence, with Jaccard over intermediate values as the cheaper variant. Wording-invariant by construction — and
  both attacked corpora preserved it, which is why those attacks were free.
- **Prediction:** the second separates siblings under imitation; the first mostly does not.

**Aggregation (settled, corrected in round 15).** The **t prediction interval is the test**: per suspect,
S = mean over published problems of d_x, tested one-sided against b's reference students' S computed with the *same*
trace pair. **The sign test cannot be the headline**, and the reason is the paper's own subject: its null p = 0.5 assumes
a student trained on b aligns with a's and b's traces equally often, which nothing guarantees — if a's traces are shorter
or use fewer intermediate quantities, LCS against a's trace is higher for *every* student, b's included, and the
uncalibrated sign test flags b's students as a's. That is the uncovered-sibling failure in a new statistic. Calibrating
its null proportion on b's references would make it a discretized version of the t interval, so we drop it. **Budget
curve:** the t interval's TPR and FPR at N ∈ {25, 100, 300, N*}, as in M7.

**Relation to Proposition 3 (corrected).** An earlier note said this is where "power grows with queries, unlike
Proposition 3". Wrong: it is the case Proposition 3 *permits*. More queries help only when the population ordering is
right. Under imitation the **style** ordering is wrong — the owner's attacked students sit on b's side — so M9's extra
probes could not help; the **content** ordering stays right because the attack preserves content, so N sharpens S and
power grows. Written this way, the follow-up is the paper's theory predicting where its own remedy can be rescued.

**N (settled).** Size from the reference students' d_x as planned, but note what that sizes: power for the **unattacked**
separation, which M11 suggests is weak but present. **Attacked power at that N is a result, not a design input** — do not
size from the M9b students. Pre-register **300 as primary** (it matches every other probe count in the paper), with
**1,000 as a pre-committed escalation** if gate 1 passes and attacked power is below 0.8, and report the sized N* as a
check on both. The published corpus caps N at 1,500.

**New data, no training.** Suspects must be queried on the **published** problems; our 300 probes are held-out. D7 needs
a generation pass (cheap on the L40S node) but it is new data, and the pre-registration must order the gates so that
**gate 1 is whether retrieval survives distillation at all** — unattacked owner students separating from b's reference
students — *before* any attack is run.

**Still open:** the floor for any rank/conformal gate in this design (n_cal ≥ 19 at α = 0.05).

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

**Threshold (revised, advisor round 14 — the earlier version compared across conditions and was not clean).** Two steps,
both inside the matched condition:
1. **Validation gate.** The length-only baseline must be near chance on the matched sets: **AUC ≤ 0.55**. Otherwise
   matching failed and that unit is **void**. Restrict the baseline to length-derived features only — token, character,
   line and sentence counts, and sentence-length moments. Step counts and scaffold phrases are *style*; putting them in
   the control would leak the very thing the test isolates.
2. **Survival, absolute.** Full read-out matched **AUC ≥ 0.65** with a bootstrap 95% CI over outputs excluding 0.50, in
   **≥ 3 of 4** family × stage units. No margin over the baseline is subtracted: once matching is validated, length
   carries no information and there is nothing meaningful to subtract. **The threshold is not derived from the
   baseline's bootstrap spread**, which shrinks with sample size and would put "survives" near 0.51 on large matched
   sets — an effect nobody would call style. The CI answers significance; 0.65 answers magnitude.
3. **Stripped-and-matched is primary; matched-only is secondary.** If matched passes and stripped fails, the residual is
   **scaffold** — the paper's existing finding — and must be reported as that, not as non-length style.
4. **Named near miss, fixed in advance:** AUC in **0.60–0.65** is "weak residual, not survival".

**Capability gate (separate):** accuracy ≤ base − 0.03 on the same probes, with the unit-tested extractor (v2; boxed
extraction for MATH).

**Consequences.** Both gates pass → the positive control stands ("style without capability"). Either fails → report it,
and D1 becomes a paragraph in D3. Keep clear of the retracted capability–identity dissociation: this is a different
claim, and it must be stated as such.

---

## Order after submission
D7 first, D3 (decomposition of the inherited signal) alongside, D1 only if its gates pass. D10 (benchmark release) after a
licence table. D5 waits for the ACL decision; D8 needs new training.
