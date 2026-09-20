# Draft — Appendix X: Integrity record

_Working draft (2026-09-18). Every number here is either generated (`make_appendix_x.py`, `make_tables_s56.py`) or traced
to a ledger entry in brackets. Tables X.1–X.3 are inserted from `paper/generated/`._

---

## Appendix X  Integrity record

Every experiment in this paper was pre-registered in a version-controlled design document before its data existed; the
commit identifiers are in the supplementary material. Pre-registration makes errors visible rather than preventing them,
so we list every error we found, how we found it, what it touched and what the paper now says. We follow the view that a
record of one's own corrections belongs next to the results rather than in a later erratum (Rohrer et al., 2021). Entries
are ordered by how much of the paper they could have affected.

### X.1  Answer extractor (affects accuracy figures only)

**What.** From the first instruction-level round (§4) through the first attack replication (§6.3), GSM8K answers were
extracted by a fallback that skipped any number immediately followed by a period, so "The answer is 8." was read as an
earlier number such as the "3" of "Step 3".

**How found.** A pre-registered answer-preservation check voided seven of eight rewritten corpora in the attack
replication. We inspected the rejected samples rather than accepting the void, and found correct answers being misread.
Unit tests were written afterwards; they did not exist before.

**Scope.** No attribution statistic consumes the extractor: every T0 and T1 decision, AUC and false- or true-positive
rate is unaffected. Accuracy was underestimated by +0.005 to +0.133 for teachers and +0.016 to +0.091 for student means
(Table X.1). Under the corrected extractor 2 of 16 rewritten corpora are void rather than 9 (Table X.2); §6.3 reports the
corrected voids, and its conclusions do not change. The manipulation-check verdicts of every ladder round are unchanged.

**Retracted.** Three secondary claims from earlier drafts: a capability-versus-identity dissociation (its only
significant effect reverses sign and loses significance); that the RL stage of Tulu-3 is less accurate than its DPO
parent on GSM8K (0.854 vs 0.849 after correction); and that the utility cost of prompt-implanted keys depends on the
student family (both families pay it).

[Table X.1 — `generated/appx_extractor.md`, first table]
[Table X.2 — `generated/appx_extractor.md`, second table]

### X.2  The 7B cell: a gate that could not pass (§6.1)

**What.** The pre-registered 7B cell used three reference students per teacher for T0's cross-line calibration as well as
for T1's per-relative test. With three cross-line teachers that is 9 calibration scores, so the smallest attainable
conformal p-value was 1/10 = 0.10, above α = 0.05: the gate could not pass whatever the data. The reuse came from M7's
finding that three references suffice for T1's per-relative test, which says nothing about the calibration pool.

**How found.** The scorer reported "collapse gate fails"; the same run showed the owner detecting 0.00 of its own
students, which no working test does. The p-values were all at their floor.

**What we did.** Withdrew the verdict; restored the calibration size M7 had pre-registered (ten reference students per
teacher, 30 scores, floor 0.032) — a size fixed by the attainability arithmetic, not chosen after looking; trained the 42
missing students; and scored again under the unchanged gates. Both gates pass (§6.1). A description computed on the void
run at α = 0.10 suggested 8 of 12 collapsed pairs; the correct test gives 7. We never used the former and report it only
to show why a post-hoc number is not a result. The scorer now refuses to issue a verdict when the floor exceeds α, and
§5.3 states the floor rule (n_cal ≥ 19 for α = 0.05) as method.

**A second deviation in the same cell, found later.** While preparing a second 7B cell we found that the 7B students were
trained with sequences capped at 1,024 tokens, whereas every 1–1.5B student in §5–§6 was trained with a cap of 4,608; the
cell's design document had wrongly recorded 1,024 as the earlier recipe. We measured the consequence rather than assume
it: 1,075 of the 117,000 GSM8K training examples (0.92%) exceed 1,024 tokens and lose their ends, against 73 (0.06%) that
exceed 4,608. We did not retrain the cell; §6.1 states the difference. The second 7B cell, on MATH, where 16.3% of
examples exceed 1,024 tokens, uses 4,608.

**A check applied afterwards, as a diagnostic.** The second 7B cell (in preparation) replaces the accuracy-or-length
manipulation check with absorption — a teacher's students must separate from the untuned base model under a
teacher-vs-base read-out at AUC ≥ 0.90 — because on MATH at 7B the base model plausibly outscores every teacher, so a
perfectly absorbed student would fail an accuracy clause. This cell passed the older check partly through its length
clause, so we ran the new one here retroactively **as a reported diagnostic, not a gate**: it cannot change this cell's
verdict, and all six teachers pass it (AUC 0.96–1.00). The two 7B cells are therefore comparable on that criterion.

### X.3  Robustness round: inverted class order in the scorer (§6.2)

**What.** The pairwise scorer returned the probability of the relative instead of the owner, so every test in its first
run had true- and false-positive rates of zero, and the run printed a spurious kill verdict.

**How found.** One of its tests is by construction weaker than T1, whose true-positive rate on the same students was
1.00; a weaker test cannot have lower power. The run was voided, the index fixed, and a permanent sentinel added that
recomputes the §6.1 true-positive rate on every scoring run and halts if it moves. Every later scorer, including the 7B
cell's, carries a sentinel of this kind.

### X.4  Robustness round: two planned tests were the same test (§6.2)

**What.** The robustness round defined a "partial references" test and a pooled test that, in a line of three
checkpoints with one relative withheld, both reduce to one remaining relative — the same test, with identical numbers in
every row. Its two predictions were thresholds on one quantity.

**Consequence.** The pooled "not me" rejector reported in §6.2 and §5.3 was added **after** this was found and is
labelled exploratory wherever it appears. The pre-registered results that bracket it — collapse without references (§5.3)
and its return when one relative's references are withheld (§6.2) — are unaffected.

### X.5  Attack replication: student-key collision (§6.3)

**What.** The scorer keyed attacked students as (owner, target, seed), which collided with the mixture students'
(a, b, λ, seed); one attack pair would have been scored with mixture students mixed in. The first scoring crashed on the
collision and was voided. Keys are now namespaced per experiment.

### X.6  First attack round: gates computed over voided corpora (§6.3)

**What.** The first attack round's gates averaged over all rewritten corpora, including those its own answer check had
voided. Recomputed on valid corpora only; the conclusions (paraphrase does not defeat T1; imitation does) are unchanged.

### X.7  Attack accuracy baseline (§6.3)

**What.** An earlier draft reported the attacked students' accuracy relative to students trained on *paraphrased* traces,
which are themselves 2–6 points below unattacked students, so the attack appeared to raise accuracy. §6.3 now compares
with the owner's own unattacked students (imitation −0.028 to +0.030; scaffold-only −0.027 to +0.023), generated by
`make_tables_s56.py`.

### X.8  An expectation not met: collapse on MATH under two read-outs (§5.4)

**What.** We expected T0 to collapse on same-line relatives in every cell. On MATH it does under TF-IDF (8 of 12 pairs in
both families) but not under sentence embeddings (3 and 2 of 12) or, for Qwen, POS templates (5 of 12). The formal gates
covered only the GSM8K cells, so no gate failed; we report this as an expectation not met rather than as a pass. TF-IDF
has been the pre-registered primary read-out since before any ladder data existed, and the other read-outs' definitions
were copied verbatim from that earlier round, so the pattern was not selected after the fact. §5.4 treats read-out
dependence as a result.

### X.9  A geometric account contradicted on held-out data (§5.4)

**What.** A pre-registered rule predicted that a pair collapses only when the relative is closer to the owner than any
cross-line teacher (ratio < 1). On the held-out Zephyr ladder four collapsed pairs have ratio 1.04–1.15: the threshold
form is **contradicted**, not merely unreplicated. The ordering form was untestable there because every pair collapses.
§5.4 reports both separately.

### X.10  Instruction-level ARC result: reported with both readings, verdict void (§4, Appendix Y)

**What.** On ARC-Challenge the instruction-level signature passes in 14 of 16 pairs if student length is compared with
each student's own teacher arm, and the round is void if it is compared with the pooled mean over arms (736 characters;
one arm's traces average 1,215, putting its students at 1.62–1.72× and outside the 0.67–1.5 band). The design text says
"length ratio to the teacher", which fits the per-arm reading, but the per-arm computation was adopted after the output
was seen, and it decides the outcome directly. We therefore take the reading as run — **void** — as the verdict, and give
both in Appendix Y. This is the same discipline as X.2: a post-hoc number is not a result.

### X.11  A near-tautological false-positive rate (§4)

**What.** An early instruction-level round reported an empirical false-positive rate of 0.016 for its rank test. Because
a student's 16 key scores form one ranking, at most one key can reach the minimum p-value, and in a trained student the
owner's key occupies it; the rate was close to fixed by construction and tested nothing. It is not reported as a result.

### X.12  A mislabelled checkpoint

**What.** `allenai/Llama-3.1-Tulu-3-8B` is the final RL model, fine-tuned from the DPO checkpoint; the SFT checkpoint is
a separate release (`allenai/Llama-3.1-Tulu-3-8B-SFT`). In our earlier open-set round the teacher labelled "Tulu SFT" was
this RL model, so that round's finding that "SFT and DPO are indistinguishable" concerns the adjacent RL-final and DPO
stages instead. The finding is not used in this paper. The ladder rounds (§5–§6) load every stage by its explicit
model-card name, and all labels here follow the model cards; we flag the naming because the unsuffixed repository name
invites the same mistake.

### X.13  Hand-typed numbers the generators caught

Three numbers typed into drafts by hand disagreed with the generated values and were corrected: the extractor's effect
on accuracy (typed as 0.03–0.13; generated +0.005 to +0.133, X.1), the attack's accuracy change (typed as −0.03 to +0.04
against the wrong baseline, X.7), and the outcome vocabulary shared by §3 and §6.3 (typed "evasion/framing" against
Table 2's "evade/frame"). Every table and every quoted range in the body is now generated from result files, and a build
check fails if §6.3 and Table 2 stop using the same outcome words.

### X.14  Reference and query budgets

[Table X.3 — `generated/s56_budget.md`. T1 true-positive rate / mean false-positive rate on relatives, varying T1's
per-relative references (3, 5, 10) or the number of probe queries (25–300). T0's cross-line calibration is ten students
per teacher in every column.]

---

## Notes for integration (remove before submission)
- Commit identifiers for every pre-registration and correction go in the supplementary material, not the appendix body.
- Appendix Y (instruction-level key banks, ARC both readings) is still to be assembled from `draft_s3_s5.md` §4.x.
- Numbers in X.2 come from `m13_result.json` and the ledger's M13 entry; X.8 from `s56_numbers.json` (table1); X.9 from
  `m12a_result.json`; X.10 from the ledger's M3C entry.
