# Draft — Analysis: what the owner test controls, and what it cannot

_Working draft (2026-09-14). Intended to sit right after the test definition (§3.4). These are
explanatory propositions, not a theoretical contribution; each is paired with the experiment it
explains._

## Setup and notation
- **Keys:** a bank B = (κ_1, …, κ_K) of keys drawn i.i.d. from a generator G, and an owner index
  J ~ Unif{1, …, K} drawn independently of B.
- **Read-out:** fitted on teacher traces generated under every key in B (it depends on B, not on J).
  For a suspect student with law Q, the read-out gives bounded per-output scores
  f_j : 𝒴 → [0, 1], j = 1, …, K.
- **Student outputs:** the owner draws N outputs Y_1, …, Y_N ~ Q i.i.d. and computes
  s_j = (1/N) Σ_n f_j(Y_n).
- **p-value:** p = (1 + #{ j ≠ J : s_j ≥ s_J }) / K.

Rejecting at level α means c := #{ j ≠ J : s_j ≥ s_J } ≤ ⌊αK⌋ − 1.

We distinguish two null hypotheses:
- **H₀^key:** the student's outputs are independent of the owner index, (Y_1, …, Y_N) ⟂ J | B.
- **H₀^source:** the student was not trained on the owner's traces.

## Proposition 1 (the test controls the key null, not the source null)
**(a)** Under H₀^key, P(p ≤ r/K | B) ≤ r/K for every r ∈ {1, …, K}.

**(b)** H₀^source does not imply H₀^key. Suppose a student trained only on traces from a different
teacher that was prompted with κ_J (or with κ_J's reasoning instruction) has an output law whose score
vector has the same distribution as that of an owner-trained student. Then the rejection probability
is identical in the two cases, whatever α is.

*Proof.*
- **(a)** Condition on B and on the outputs. The score vector s is then fixed, and by H₀^key, J is
  still uniform on {1, …, K}. The rank of s_J among (s_1, …, s_K), with ties resolved against
  rejection, is dominated by a uniform rank. Hence P(c ≤ r − 1) ≤ r/K. Take expectations.
- **(b)** The test statistic is a function of s and J only. Equal score distributions give equal
  rejection probabilities. ∎

**Reading.** The randomisation guarantee is a guarantee about keys, not about data provenance. It
holds for any student that has no access to the owner's instruction. It says nothing about a student
whose training traces were produced under the same instruction by someone else, because that student
is no longer independent of J.

**Evidence.**
- Same-instruction students from an independent teacher were flagged in 8/8 evaluated cases, and
  different-instruction students in 0/8 (§6).
- This is exactly the behaviour (b) permits. It is not a failure of (a).

## Proposition 2 (source attribution requires source information beyond the instruction)
Fix an instruction I, and let S ∈ {owner, alt} denote the teacher that produced the training traces.
Suppose **instruction sufficiency** holds: the student output law satisfies Q(· | S = owner, I) =
Q(· | S = alt, I). Then any test φ : 𝒴^N → [0, 1] of "S = alt" against "S = owner" has power equal to
its size.

Conversely, for any fixed test φ,

  TV( Q^N_owner, Q^N_alt ) ≥ | E_owner φ − E_alt φ |,

so any test whose power exceeds its size certifies that the two laws differ.

*Proof.* Equal laws give equal expectations of every measurable φ. The inequality is the variational
characterisation of total variation distance. ∎

**Reading.** Owner-specific attribution against a known-instruction replication is possible only if
student outputs carry teacher-specific information beyond the instruction.

**Evidence.**
- A teacher-identity read-out was trained on teacher traces for instructions that no test student
  used. It separates Tulu-sourced from Qwen-sourced students with AUC 1.00 in both student families
  and both codebooks (§6, exploratory).
- Its per-output balanced accuracy of 0.77–0.92 implies, through the inequality with a fixed 0.5
  threshold, a per-output total-variation lower bound of roughly 0.54–0.84 (up to sampling error).
- **So instruction sufficiency is violated in our data.** The key test fails at source attribution
  because it is trained to separate keys, not because the source information is absent.
- The evidence is closed-set: two known teachers. Distinguishing the owner from an arbitrary unknown
  teacher is not addressed.

## Proposition 3 (more queries cannot overturn a wrong population ordering)
Let μ_j = E_Q f_j(Y). Suppose at least ⌊αK⌋ decoys satisfy μ_j > μ_J, with gaps Δ_j = μ_j − μ_J > 0.
Then

  P(p ≤ α) ≤ min_{𝒮} Σ_{j ∈ 𝒮} exp(−N Δ_j² / 2) → 0 as N → ∞,

where 𝒮 ranges over sets of ⌊αK⌋ such decoys.

Conversely, if at most ⌊αK⌋ − 1 decoys satisfy μ_j ≥ μ_J (all others strictly below), then without
vetoes P(p ≤ α) → 1.

*Proof.*
- Rejection requires c ≤ ⌊αK⌋ − 1. So for any set 𝒮 of ⌊αK⌋ decoys with positive gaps, some j ∈ 𝒮
  must have s_j < s_J.
- s_j − s_J is a mean of N i.i.d. terms in [−1, 1] with expectation Δ_j. Hoeffding's inequality gives
  P(s_j < s_J) ≤ exp(−N Δ_j² / 2).
- A union bound over 𝒮, minimised over 𝒮, gives the first claim. The converse follows from the strong
  law of large numbers applied to the finitely many differences. ∎

**Reading.** Adding queries only sharpens estimates of the population ordering. Under dilution, the
question is whether the owner key's population score exceeds all but ⌊αK⌋ − 1 decoys, and that is
fixed by the training mixture and the read-out, not by N.

**Evidence.**
- At 10% keyed data, detection was 0/4 at N = 200 and still 0/4 at N = 1,319. For o12, p rose from
  0.062 to 0.156 as N grew, the pattern expected when several decoys have μ_j slightly above μ_J.
- At 50%, every key ranks first at N = 200 already.

## Corollary 1 (what calibration covers bounds the false-positive rate on a relative)

**Setting (teacher level, §5–§6).** An owner *a* scores a suspect student by a scalar statistic S, a function of the
student's N probe outputs (for T0, the mean read-out probability of class *a*). Randomness is over the distillation run
and the probes. Write P_a, P_b and P_C for the laws of S when the suspect is a student of the owner *a*, of a relative
*b*, and of a calibration population C (for T0, the cross-line teachers). The owner draws calibration scores
S_1, …, S_n i.i.d. from P_C, independently of the suspect, and rejects when the conformal p-value
p = (1 + #{i : S_i ≥ S}) / (1 + n) is at most α. Let TPR_a and FPR_b be the rejection probabilities when the suspect is
drawn from P_a and from P_b.

**Corollary 1.**

  TPR_a − TV(P_a, P_b)  ≤  FPR_b  ≤  α + TV(P_C, P_b).

*Proof.* The rejection event is a measurable function of (S, S_1, …, S_n), and the calibration scores have the same law
whichever population the suspect comes from. Changing only the law of S from Q to Q′ therefore changes the probability of
the event by at most TV(Q, Q′), by the variational characterisation of total variation distance. Take Q = P_a for the
lower bound. For the upper bound take Q = P_C: then S is exchangeable with the calibration scores, so P(p ≤ α) ≤ α by
conformal validity (Vovk et al., 2005; Bates et al., 2023). ∎

**Tightness.** Rearranged, the lower bound reads TPR_a − FPR_b ≤ TV(P_a, P_b), and it is attained by the test that
rejects exactly on the set achieving the total variation distance. That statement *is* Le Cam's two-point identity
inf over tests of [FPR_b + (1 − TPR_a)] = 1 − TV(P_a, P_b) (Tsybakov, 2009, Ch. 2) — the same inequality, not a second
argument for it, so we cite Le Cam for the identity rather than as an extra step. The upper bound is the simplest
instance of the coverage-gap bounds for conformal prediction under distribution shift (Barber et al., 2023).

**Reading.** The false-positive rate on a relative is squeezed between two distances measured *in the law of the
test's own statistic*:
- **Covered relative.** If the calibration population produces scores distributed like the relative's, the test
  flags the relative's students at a rate of at most about α (rejecting the null wrongly), however close the relative is to
  the owner.
- **Uncovered, nearby relative — the coverage gap.** If the relative's scores are far from every calibration
  population but close to the owner's, the upper bound is vacuous and the lower bound forces FPR_b ≥ TPR_a − TV(P_a, P_b):
  a test powerful enough to catch the owner's own students must also flag the relative.
- **The distances belong to the statistic, not to the teachers.** TV(P_a, P_b) is the distance between score laws after
  the read-out and pooling, so two read-outs applied to the same students can sit on different sides of the gap
  (Proposition 2's invariance point).

This is a corollary of standard results, and we claim no novelty for it. Its role is to state precisely which quantity
the experiments of §5–§6 move.

**Corollary 1b (the bound holds for any suspect population, including one an adversary produced).** Nothing in the proof
used the suspect's law, so replacing P_b by an arbitrary law Q gives, for each party separately and under that party's
own statistic,

  owner's detection rate on Q  ≥  TPR_a − TV(P_a, Q),
  relative b's claim rate on Q  ≥  TPR_b − TV(P_b, Q).

In §6.3 the distiller rewrites the owner's traces before training on them, so the attacked students are such a Q, and
the outcomes of that attack are positions of Q relative to the two laws. We name those positions in §6.3, beside the
table that reports them, rather than here: they are names for regions of score space, not theorem content, and stating
them here would let the corollary be read as *predicting* the attack when it only locates it.

**Evidence.**
- **An uncovered relative (§5.3).** T0 calibrates on cross-line students only, whose scores lie far below the relative's,
  so TV(P_C, P_b) is near 1 and the upper bound gives no protection. The conformal threshold sits at the cross-line
  tail, below both the owner's and the relative's scores. Owners detect their own students at 1.00 and flag same-line
  relatives at 0.9–1.0 in 8 of 12 ordered pairs per cell, in both directions on the Zephyr ladder. Note what this does
  *not* show: TPR_a − FPR_b ≈ 0–0.1 gives only the vacuous lower bound TV(P_a, P_b) ≥ 0–0.1. The collapse is a failure of
  coverage, not evidence that the owner's and the relative's score laws are close, and a threshold between them —
  which T1's pairwise test effectively supplies — separates them.
- **Covering the relative (§6.1).** T1 adds, for each relative, a test whose reference population is that relative's
  own students, so its calibration covers the relative by construction. Its relative false-positive rates fall to
  0.00–0.09, across two vendors, two datasets and three read-outs. T1's reference test uses a t prediction interval
  rather than a conformal rank, so its guarantee is approximate; the conformal variant (α = 0.1) gives the same
  conclusion [EXP-M7].
- **Removing coverage for one relative (§6.2).** Withholding one relative's reference students restores its false-positive
  rate to ≥ 0.6 in 6–8 of the 8 affected pairs per cell. A pooled rejector whose calibration includes the *other*
  relatives rejects an unreferenced *distant* relative (0.0) but not an *adjacent* one (1.0) [EXP-M8]. The distant relative
  is covered by teachers it resembles; the adjacent one is close only to the owner.
- **Where the lower bound bites: imitation (§6.3).** An attack that rewrites the owner's traces toward a relative moves
  the attacked students' score law Q away from P_a and toward P_b under the owner's pairwise statistic. Corollary 1b
  then leaves the test only two options on those students: miss them, or flag the relative too. All four observed
  outcomes — evasion, framing, joint claims and laundering — are positions of Q [EXP-M9, EXP-M9b, EXP-M12], and the two
  rewrites of Table 2 evade on disjoint sets of attacks. The corollary does not predict *which* attacks get there, and
  our pre-registered predictors of that failed.
- **The distances depend on the read-out (§5.4).** The same students, calibration set and owners give 8 collapsed pairs
  per MATH cell under TF-IDF but 2–3 under sentence embeddings [EXP-M11]: under the embedding statistic the relatives'
  score laws are covered by the cross-line calibration population.
- **Geometric corroboration (exploratory, §5.4).** A centroid-distance proxy for "closer to the owner than to any
  calibration teacher" orders collapse severity in all 12 AllenAI read-out × cell units. Its threshold form is
  contradicted on the Zephyr ladder, where four pairs collapse with the relative farther from the owner than the nearest
  cross-line teacher. This is what Corollary 1 leads one to expect of such a proxy: the bound is stated in terms of the
  statistic's score laws, and centroid distance in feature space is not that quantity, so it can order severity without
  locating a threshold.

## Placement and scope notes
- Proposition 1 goes in §3.4, immediately after the test. It defines the two endpoints used in §6.
- Proposition 2 opens §6 and frames the teacher-identity read-out.
- Proposition 3 goes in §7 (dilution), next to the query-scaling result.
- Corollary 1 goes at the end of §3, after Proposition 2; §5.3, §5.4, §6.1–§6.3 cite it. Corollary 1b is one further
  paragraph there, and Table 2's outcome column cites it for the four labels (evade / frame / joint claim / laundering).
- The evaluated access model is **known-instruction replication** only. Accidental instruction
  collision and instruction reconstruction from published traces are not evaluated, and none of the
  propositions quantify them.
