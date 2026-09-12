# M3 — Is a prompt-only signature inherited by a distilled student?

_Design v0 and pre-registration, 2026-09-11, written before any M3 data exists. Candidate board:
`idea_board_v4.json` (M3). P2, M1 and M2 were each closed by their own pre-registered gates._

## 0. Prior work, checked today

- **In-Context Watermarks (Liu, Zhao, Kruegel, Song, Bu; ICLR 2026, arXiv 2505.16934).** Prompt-only
  watermarking. Its four strategies are explicitly instructed surface patterns: zero-width Unicode,
  word initials, a lexical green list, and acrostics; detection is by counting those artifacts
  (ROC-AUC 0.995–1.000 with a strong model, much weaker with a small one). **It contains no
  fine-tuning, distillation, radioactivity or student-model experiment** (checked the full HTML for
  "fine-tun", "distill", "radioactiv", "student").
- **Trace rewriting (Ma, Yeoh, Zhang, Vorobeychik; ACL 2026, arXiv 2602.15143, code released).**
  The active competitor: the teacher *rewrites* its traces to implant a trigger → target
  association, then verifies it by querying the student (TPR 1.00, FPR ≤ 0.02, surviving
  paraphrase, filtering and distillation scrubbing). It changes the teacher's outputs.
- **Token-level radioactivity** (TextSeal 2605.12456; Sander et al.; ACL 2025 scrubbing study)
  needs decoding access.
- **The untested question:** whether a *passive, trigger-free* signature — one induced only by
  prompting the teacher, leaving its outputs otherwise unmodified — is inherited by a student
  trained on those outputs. Nothing found tests it in either direction.

## 1. Setting

The teacher is prompted with a secret key (a persona plus a reasoning instruction) and publishes
only its reasoning traces. A distiller collects (problem, trace) pairs and fine-tunes a student on
them. The distiller never sees the key and never prompts the student with it. The owner later has
black-box access to the student and asks: was this student trained on *my* key's traces?

This is exactly our project's own instruction-echo signal, carried one step further. Our prior is
weak: N2 showed a prefix stripped of its key retains only 1.5–1.9× chance attribution, and the
key, not the text, drove continuation.

## 2. Method

- **Teacher:** Llama-3.1-Tulu-3-8B (the generator whose echo we have characterised).
- **Keys:** three from `v2_diverse`, the space with our strongest measured attribution (learned
  read-out 0.475 top-1 vs 0.125 chance): v2d-0 (accountant / track units), v2d-1 (pharmacist /
  restate the target), v2d-5 (cartographer / verify direction of comparisons). Plus a **clean**
  arm with no key.
- **Data:** 600 GSM8K *train* problems per arm (disjoint from the 200 test problems used later),
  generated once per arm.
- **Students:** Qwen2.5-1.5B-Instruct (primary) and Qwen2.5-0.5B-Instruct (secondary), LoRA SFT on
  (plain problem → teacher trace) pairs. The prompt carries **no key**, matching a real distiller.
  One student per arm: three keyed students and one clean student.
- **Student sampling:** 200 held-out GSM8K test problems, plain prompt, no key.
- **Detector:** the read-out that works in our setting — TF-IDF (1–2 grams) plus logistic
  regression, trained on **teacher** traces over 4 classes (3 keys + clean), problem-disjoint from
  the student's test problems, then applied to student outputs. Chance is 0.25.
- **Nulls:** (a) the clean student's rate of being assigned to key k; (b) a label-permutation null
  over the teacher classifier (our standard C5-style control).

## 3. Pre-registered gate

- **M3-0 (manipulation check; the students must have learned something).** Each keyed student's
  GSM8K accuracy must exceed the untuned base model's by ≥ 5 points, or its trace length/format
  must visibly match the teacher. If training did not take, a null result is uninformative and the
  experiment is void rather than negative.
- **M3-a (primary).** Let p_k be the fraction of student k's 200 outputs that the teacher-trained
  classifier assigns to key k. Pass iff mean_k p_k ≥ 0.50 (2× chance) **and** p_k exceeds the
  clean student's rate for the same key on ≥ 2 of 3 keys (binomial p ≤ 0.05).
- **M3-b (reported, not gating).** The same on the 0.5B student; the 3-way confusion matrix; and
  attribution measured on the *teacher's* held-out traces as the ceiling.
- **Decision.** Continue M3 iff M3-0 and M3-a pass. If M3-0 passes and M3-a fails, the result is
  "a passive prompt-only signature is **not** radioactive: owners cannot trace a distilled student
  without actively modifying traces, which is what the ACL 2026 method does" — an honest negative
  that completes the provenance thread; record it and stop.

## 4. Resources

Teacher generation 4 × 600 ≈ 2400 traces (~1 GPU-h); 4 LoRA runs on 1.5B (~40 min); student
sampling 4 × 200 (~15 min); the 0.5B replicate (~30 min). Total ≈ 2.5 GPU-h, single GPU, chained.

## 5. Risks

1. **Likely negative** (our own N2 prior). That is an acceptable, publishable-as-analysis outcome
   and the gate is written to distinguish it from a failed experiment (M3-0).
2. **600 examples may be too few to imprint style.** M3-0 catches "the student learned nothing";
   if the student learns the task but not the style, that is the finding, not a bug.
3. **LoRA may be too weak a channel.** Reported as a limitation; a full fine-tune of the 0.5B model
   is the fallback if M3-0 fails on both students.

---

## v2 (2026-09-11, after M3-a passed) — pre-registered before any v2 data exists

v1 result (ledger EXP-M3): students attributed to their training key at 0.757 (1.5B) / 0.747 (0.5B)
vs 0.25 chance; length-only and permutation nulls clean; **but** the untuned base model is itself read
as k0 (0.705 / 0.645), so under the strict null only k1 and k5 stand. v1 had 3 keys, one student
family, and answered "which key" rather than the owner's real question.

### What v2 adds
1. **16 keys** spanning the three types we have characterised: 8 persona+anchor (`v2_diverse`),
   4 anchor-only (structural: prescribe what to compute), 4 lexical (prescribe wording).
   300 teacher traces each (Tulu-3-8B), plus a 300-trace clean arm.
2. **Students on 6 of the 16 keys** (2 instruction-type, 2 persona-type, 2 lexical), on **two
   families**: Qwen2.5-1.5B-Instruct and Llama-3.2-1B-Instruct. The other 10 keys are never used for
   training and exist only to calibrate the test.
3. **Owner-side hypothesis test with false-positive control (the real question).** For a suspect
   student and the owner's key k: score every one of the 16 keys on the same student outputs and
   take the rank of k. p = (1 + #{keys with score ≥ score_k}) / 16. This is our wrong-key
   calibration (EXP-005) transplanted to distillation: the null comes from other keys, not from
   other students, so no extra training runs are needed.
4. **Ablations (reported):** training-set size 150 / 300 / 600 for one key; a mixture arm
   (50% keyed + 50% clean traces); key type as a factor.
5. **Active-watermark baseline** in the spirit of ACL 2026 trace rewriting: inject a
   trigger → target association into 10% of one arm's traces, train a student, then query the
   student with the trigger. This quantifies the trade-off: modifying the teacher's outputs buys
   near-perfect verification; our passive route does not modify them at all.

### Pre-registered gate (v2)
- **G-M3b (owner-side test).** For the 6 trained keys, on **both** student families: p ≤ 1/16 for
  ≥ 4 of 6 keys. And the empirical false-positive check: across all (untrained key, student) pairs
  plus the clean and base students, the share with p ≤ 1/16 must be ≤ 0.10.
- **G-M3c (base-model confound, now a gate).** Every key counted in G-M3b must also beat the strict
  null: its score on the student must exceed its score on both the clean student and the untuned
  base model of the same family.
- **Decision.** If G-M3b and G-M3c pass, M3 becomes the paper direction and we move to writing:
  "passive prompt-only provenance for distilled reasoning models". If G-M3b passes on one family
  only, report and let the user decide. If both fail, v1 stands as a single-family curiosity and we
  stop.
- **Reported, not gating:** the size curve, the mixture arm, key-type differences, and the active
  baseline.
