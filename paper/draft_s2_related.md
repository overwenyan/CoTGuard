# Draft — §2 Related work

_Working draft (2026-09-17). Every venue in this section is verified in `.pipeline/docs/citations_verified.md`;
nothing unverified may be cited with a venue. Written after §3, §5 and §6 so that each paragraph ends at the
boundary our experiments actually measure._

---

## 2 Related work

### 2.1 Telling teachers apart from student outputs

Distilled students carry lexical and syntactic habits of the model that produced their training traces. Wadhwa et
al. (2025) make this a task: given a student's outputs and a set of candidate teachers treated as black boxes,
identify the teacher. They report that n-gram overlap alone is unreliable and that part-of-speech templates
preferred by the student track those of its teacher — the observation our POS read-out (§5.4) is built on. Their
candidate teachers are different models from different vendors. Ours are checkpoints of the same model, produced
by consecutive stages of one post-training pipeline, and that change of population is the subject of this paper:
the read-out still separates them (§5.2), while the *test* built on the read-out no longer means what it did
(§5.3).

Two contemporaneous lines share our title words without making the same claim. Rawat et al. (2026, preprint)
detect distillation by comparing a suspect against an earlier checkpoint of the teacher, which assumes access to
that checkpoint's weights; we assume only sampled outputs, and the vendor's own reference students. Liu et al.
(2025, preprint) classify each *sentence* of a distilled model's reasoning as teacher-, student- or jointly originated
by comparing the probabilities that the teacher, the original student and the distilled model assign to it; this
needs logits from all three models, and it attributes sentences within an output rather than a model to one of
several candidate checkpoints. An appendix of theirs shows the same probabilities can, to some extent, pick the true
teacher between two candidates from different vendors, by threshold rather than by a calibrated test. Our setting
differs on all three axes: black-box sampled outputs, candidates from the owner's own line, and a test with a
controlled false-positive rate.

### 2.2 Marks planted before release, and the rewriting that plants them

A second tradition does not read a signal out of the data but puts one in. Radioactive data (Sablayrolles et al.,
2020) perturbs training images so that any model trained on them is detectable; watermarking a language model's
outputs makes downstream students radioactive in the same sense, at as little as 5% of the fine-tuning data
(Sander et al., 2024), and ReasMark (Lv et al., 2026) carries the idea into reasoning traces specifically, binding
the mark to target-domain inputs so that it survives black-box distillation. These are *active* marks: they
require the owner to alter what it releases, before it releases it, and they buy a guarantee our setting does not
have — detection from a small share of the training data (§6.2 measures the passive test's opposite behaviour: it
reports the majority contributor and offers a minority one no claim).

The closest work to ours sits exactly on this line. Ma et al. (2026) protect a model against unauthorised
distillation by **rewriting its reasoning traces before release**, so that traces stay useful to honest readers
while a student distilled from them carries the owner's mark. That is the same operation we study, run from the
other side of the trace: in §6.3 it is the **distiller** who rewrites the owner's traces, with a 7B instruction
model and four in-context examples of a sibling checkpoint, in order to remove or redirect the provenance signal
before training on them. Owner-side rewriting plants; distiller-side rewriting scrubs and spoofs. The symmetry is
not decoration — it is why the attack is cheap. A rewriter good enough to install a mark is good enough to move
one, and the same pipeline that Ma et al. run once at release the distiller can run once at collection. What our
§6.3 adds is the measurement on the other side: against a reference-aware passive test, a rewrite that costs no
accuracy evades the owner, frames a sibling checkpoint, or leaves the student attributable to nobody.

### 2.3 Attacks that make an ownership test say the wrong thing

That an ownership test can be attacked rather than merely evaded is old, and the vocabulary is worth keeping.
Craver et al. (1998) showed that invisible watermarks admit **ambiguity attacks**, in which a second party
produces an equally valid claim and the scheme cannot adjudicate; our "joint claim" outcome (Table 2) is that
situation, reached without a counterfeit scheme — two honest checkpoints of one lineage both pass. Brennan et al.
(2012) separate **obfuscation**, where an author hides, from **imitation**, where an author frames someone else,
and find imitation the more effective of the two against stylometry; we recover both as separate outcomes of one
rewriting attack, and a third, which we call **laundering**: the owner's signal is gone and no other party is
implicated. In the watermarking literature the same pair appears as **scrubbing** and **spoofing**, and Jovanović
et al. (2024) show that querying a watermarked model is enough to approximate its watermark and do either;
DITTO (An et al., 2026) obtains spoofing specifically by distillation, using the radioactivity of watermarks as
the attack vector. Zhang et al. (2024) prove that a sufficiently good quality-preserving rewriter defeats *any*
strong watermark; our results are the passive-test analogue of that statement, measured rather than proved, and
they arrive at a weaker rewriter: an off-the-shelf 7B model with no knowledge of the test. (Mansurov et al. (2024)
use "data laundering" for an unrelated phenomenon, benchmark contamination through distillation; we keep the term
for provenance erasure and flag the collision.)

Our point of departure from all of these is the adversary's necessity. Every attack above requires one. The
central failure of §5.3 requires none: an *honest* sibling checkpoint, doing nothing, is indistinguishable from
the owner because the test's null hypothesis never contained it.

### 2.4 Whom the test normalises against

That failure is a calibration failure, and the question "against which population do you score a suspect?" has
been asked before ours, in fields that had to answer it. Speaker verification normalises a score against a cohort
of impostor models chosen to resemble the claimed speaker (Auckenthaler et al., 2000); authorship verification
decides whether two documents share an author by checking whether one can select the other from a background set
of **impostors** (Koppel & Winter, 2014). Our reference-aware test (§6.1) is the same move — the owner supplies,
for each sibling checkpoint, its own small set of reference students, and the suspect is scored against them —
and §6.2 is the measurement of what that move costs when the impostor set is incomplete. Open-set recognition
(Scheirer et al., 2013) names the general condition under which a closed-set classifier's confidence stops being
evidence; §5.3 is a case of it with a specific structure, in that the unknown class is not an arbitrary outsider
but the owner's own neighbour. We state the test's guarantee with conformal p-values (Vovk et al., 2005; Bates
et al., 2023), and Corollary 1 uses the coverage-gap bound of Barber et al. (2023) to say precisely which
distances control the false-positive rate on a relative.

### 2.5 Membership and dataset inference

A different question asks whether a *dataset* was trained on, rather than which model produced the traces. Maini
et al. (2024) aggregate weak per-example signals into a dataset-level test and show that per-example membership
inference is largely confounded by distribution shift between members and non-members. Such tests need the
candidate training data, and in our setting the owner has it: it published the traces. Retrieval against a provider's
own past generations survives paraphrase at the text level (Krishna et al., 2023), which is why §6.4 names a content
signal over the published traces as where a defence against imitation would have to live — and what about it is
untested for distilled students.

---

## Notes for integration (remove before submission)
- **Pivot to keep:** §2.2's Ma et al. paragraph is the hinge of the section — active/passive is not a taxonomy
  slot here but the reason the attack of §6.3 is cheap. §6.3 keeps only a one-clause back-reference; the argument
  lives here.
- **Do not claim** priority over Wadhwa et al. (2025) for teacher identification. Our contribution is the
  within-lineage population, the calibration semantics, and the attack.
- **Liu et al. (arXiv 2512.20908) read in full 2026-09-18.** Sentence-level provenance, white-box (logits of teacher,
  original student and distilled model), teachers DeepSeek-R1 / QwQ-32B / GPT-OSS-120B, math and GPQA benchmarks; main
  use is teacher-guided data selection (+1.7–2.5 points). Its App. A.4.3 weakly separates the true teacher from a
  spurious cross-vendor one, with no statistical test. No adversarial rewriting. Not a competing claim; §2.1 now
  characterises it at that level of detail.
- **Venues:** ReasMark and Ma et al. are ACL 2026; DITTO is EACL 2026 (not a preprint); Rawat et al. and Liu
  et al. and Mansurov et al. are preprints and are cited as such.
