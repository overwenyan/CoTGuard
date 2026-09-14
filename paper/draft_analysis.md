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

## Placement and scope notes
- Proposition 1 goes in §3.4, immediately after the test. It defines the two endpoints used in §6.
- Proposition 2 opens §6 and frames the teacher-identity read-out.
- Proposition 3 goes in §7 (dilution), next to the query-scaling result.
- The evaluated access model is **known-instruction replication** only. Accidental instruction
  collision and instruction reconstruction from published traces are not evaluated, and none of the
  propositions quantify them.
