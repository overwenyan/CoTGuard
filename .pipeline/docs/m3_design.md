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

---

## v3 (2026-09-12, after v2 passed) — robustness; pre-registered before any v3 data exists

### 0. A correction to how v2 was read
v2 reported an "empirical false-positive share 0.016". **That number is close to automatic.** For a
single student the 16 key scores form a ranking, so at most one key can reach p = 1/16. A trained
student's own key takes that slot, so its 15 untrained keys almost never can. The v2 number checks
nothing about validity. What makes the owner-side test valid is **random key assignment**. If the
owner's key and the decoys are i.i.d. draws from one key generator, and the student is independent of
the draw, then P(rank of owner's key ≤ r) = r/K exactly under H0, for any innocent student.
v3 therefore (a) draws every key from an explicit generator, (b) replaces the vacuous FPR with a
conditional check that can fail: a **hot key** is one that ranks in the top r on innocent students.
v2's 16 keys mixed three hand-written types, so they were not exchangeable. v2's p-values are
therefore descriptive; the valid test starts here.

### 1. Key generator and pool
`trigger_v2.py`: 4 templates × 16 personas × 12 anchors = 768 keys. The pool is **K = 64**: the 8 v2
persona keys (key_00..07, already generator outputs; their Tulu/GSM8K traces are reused unchanged)
plus 56 drawn without replacement from the remaining 760 with `numpy.random.default_rng(3)`. Decoys
may share persona and anchor with a trained key and differ only in template; this is what the
generator produces and makes the test harder, not easier. Test: p = (1 + #{j ≠ k : s_j ≥ s_k}) / 64;
**α = 0.05 ⇔ rank ≤ 3**. Trained keys: g00..g07 (= v2 key_00..07), chosen before any distillation data
existed. Read-out as in v2 (TF-IDF 1–2 grams + LR on teacher traces, K + 1 classes, score = mean class
probability over the student's 200 outputs).

### 2. Settings
| setting | teacher | domain (train → test) | students |
|---|---|---|---|
| `tulu_gsm` | Tulu-3-8B | GSM8K train 300 → test 200 | Qwen2.5-1.5B, Llama-3.2-1B |
| `qwen_gsm` | **Qwen2.5-7B-Instruct** | GSM8K | both |
| `tulu_arc` | Tulu-3-8B | **ARC-Challenge** train 300 → test 200 (MC, "Answer: <letter>") | both |

In each setting: 64 key arms + clean, 300 teacher traces each. Students are trained on 8 keys plus clean,
and the untuned base model is also sampled. The 16 (key, family) pairs per setting are the positives.

### 3. Distiller attacks (on `tulu_gsm`), applied to the training corpus of every trained key and of clean
- **filter**: keep only traces whose final answer is correct (rejection sampling).
- **para**: Qwen2.5-7B-Instruct rewrites every trace in its own words, keeping all calculations and
  the final answer.
- **compress**: Qwen2.5-7B-Instruct rewrites every trace into minimal equation lines + "Answer: X".
Students: both families for `para`, Qwen2.5-1.5B for `filter` and `compress`. Two read-outs:
**agnostic** (trained on raw teacher traces; the owner does not know the attack) and **aware** (trained
on 100 traces per arm, paraphrased/compressed by the owner with a *different* model,
Llama-3.1-8B-Instruct, and a different seed. The owner simulates the attack class, not the attacker's
exact pipeline). Attack cost is reported as the student's accuracy relative to the unattacked student.

### 4. Pre-registered gates
A positive pair counts only if **p ≤ 3/64 on its student AND the same key is not in the top 3 on
that family's clean student or base model** (the conditional check; it replaces v2's G-M3c).
- **G-R0 (finer test, `tulu_gsm`, no attack):** ≥ 12 of 16 pairs. *Fail ⇒ the v2 result does not survive
  a 64-key test; stop and report before any writing.*
- **Hot-key diagnostic (every setting; reported, with prediction):** the share of the 64 keys ranking
  top 3 on ≥ 2 of the 4 innocent students (clean ×2, base ×2). Under independence ≈ 0.01;
  predicted ≤ 0.10. Above 0.10 means the per-key guarantee is weak and the paper must state the test
  is valid only marginally over the key draw.
- **G-R1 (paraphrase):** ≥ 12 of 16 pairs with the **agnostic** read-out ⇒ "survives paraphrase".
  Aware-only pass ⇒ "survives paraphrase when the owner simulates it". Neither ⇒ paraphrase is a
  removal attack: the claim is scoped to distillation on unmodified traces, and the paper presents the
  passive route as complementary to active trace rewriting (which claims paraphrase robustness), with
  paraphrase as the stated limitation.
- **filter / compress (reported, predictions written now):** filter keeps ≥ 6 of 8 (it removes whole
  traces, not style); compress breaks ≥ 4 of 8 under the agnostic read-out (it deletes most prose, and
  the persona anchors live in prose).
- **G-R2 (second teacher, `qwen_gsm`):** ≥ 12 of 16 pairs.
- **G-R3 (non-math, `tulu_arc`):** manipulation check first. Keyed students' mean output length must be
  within 0.67–1.5× the teacher's, or their accuracy ≥ base + 5 pts; otherwise void, not negative. Then
  ≥ 12 of 16 pairs.
- **Query budget (reported):** for `tulu_gsm` positives, the share with p ≤ 3/64 when the owner uses
  n ∈ {10, 20, 50, 100, 200} student outputs (50 random subsamples each).

**Decision.** G-R0 + G-R2 + G-R3 pass ⇒ the claim generalises across teachers and domains. G-R1
decides only how the paper words robustness; it does not decide whether M3 continues.

### 5. Compute
Teacher generation: 56 + 65 + 65 arms × 300 traces. Attack rewrites: ≈ 5.4k attacker and 13k owner
generations. 69 LoRA runs; ~75 student samplings. Roughly 8–10 GPU-h, split into three chained jobs
with checkpoint-resume, one GPU.
