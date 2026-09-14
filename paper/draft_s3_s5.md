# Draft — §3 Setup and Test, §5 Nominal Keys Collide

_Working draft (2026-09-14), following `.pipeline/docs/m3_paper_plan.md`. Numbers are traceable to
`.pipeline/memory/experiment_ledger.md` (entry IDs in brackets; remove before submission)._

---

## 3 Setup and Test

### 3.1 Threat model
A model owner serves a teacher LLM whose system or user prompt carries a secret **key**: a short
natural-language reasoning instruction, optionally wrapped in a persona and a sentence template (e.g.,
*"Adopt the mindset of a pharmacist checking doses. As you reason, restate the target before each
computation."*). The owner publishes, or exposes through an API, only the teacher's reasoning traces,
without editing them after generation. A distiller collects (problem, trace) pairs and fine-tunes a
student on them. The distiller never sees the key and never prompts the student with it. Later, the
owner can query the student with plain prompts and observe only its text outputs.

We ask two questions that the literature often merges:
- **Instruction-family detection:** do the student's outputs exhibit a signature associated with
  instruction *k*?
- **Owner-specific attribution:** can the owner conclude that the student was trained on *this
  owner's* keyed traces, as opposed to traces produced under the same instruction by someone else?

A detector can succeed at the first and fail at the second. We keep the two endpoints separate
throughout.

### 3.2 Models, data and training
- **Teachers:** Llama-3.1-Tulu-3-8B (main) and Qwen2.5-7B-Instruct.
- **Students:** Qwen2.5-1.5B-Instruct and Llama-3.2-1B-Instruct, fine-tuned with LoRA (rank 32,
  α = 64, all attention and MLP projections; lr 1e-4, batch 4, sequences ≤ 1,024 tokens, loss on
  trace tokens only).
- **Data:** GSM8K. For each key the teacher answers 300 training problems (T = 0.7, top-p 0.95,
  ≤ 400 new tokens); students train for 3 epochs and answer 200 held-out test problems with a plain
  prompt. ARC-Challenge is used as a second domain (§4).
- **Mixtures (§7):** keyed traces replace clean traces on the same problems inside a 7,000-problem
  corpus.

### 3.3 Read-out
The owner trains a multinomial logistic-regression classifier on TF-IDF features (word 1–2 grams) of
**teacher** traces only: one class per key in a candidate bank of *K* keys, plus a no-key class. For a
suspect student, the owner scores key *j* as the mean predicted probability of class *j* over the
student's outputs, giving a score vector *s*. We also report a sentence-embedding read-out (gte-base +
logistic regression) where noted.

### 3.4 Owner-side rank test
Given the owner's key *k* among *K* candidate keys,

  p = (1 + #{ j ≠ k : s_j ≥ s_k }) / K,

and the owner rejects "not trained on key-*k* traces" when p ≤ α = 0.05. When the owner's key and the
*K − 1* decoys are drawn i.i.d. from the same key generator, independently of the student,
P(p ≤ r/K) ≤ r/K under the null for any innocent student (ties counted conservatively).

Two qualifications matter:
1. **The guarantee averages over the random key assignment.** It does not bound the false-positive
   rate of a *fixed* deployed key. A key that resembles a model's default style can be flagged on
   innocent students.
2. **Any screening of keys must be applied symmetrically before the owner key is drawn.**

We therefore add a **veto**: a pair counts as detected only if key *k* is *not* flagged (p > α) on two
innocent controls trained or sampled the same way: a student trained on clean traces from the same
teacher, and the untuned base model. We also report a **hot-key share**: the fraction of keys ranking
in the top ⌊αK⌋ on at least two of four innocent students.

### 3.5 Key banks
- **Generator bank** (§4–§6): 4 sentence templates × 16 personas × 12 reasoning instructions = 768
  keys. A test pool of K = 64 keys: 8 trained keys chosen before any distillation data existed, plus
  56 decoys drawn uniformly without replacement.
- **Instruction banks** (§7–§9): two banks of 40 instructions each, with no persona or template, split
  a priori into 20 **content-adding** instructions (the trace must contain an extra problem-dependent
  claim not needed for the answer, e.g., *"State one assumption that the problem leaves implicit"*)
  and 20 **format-only** instructions (constraints on numbering, layout or wording, e.g., *"Write every
  equation as result = expression"*).
- **Compliance screen:** a frozen LLM rubric (Qwen2.5-7B-Instruct) removes instructions the teacher
  follows in < 50% of 50 traces. This leaves K = 32 (first bank) and 34 (second bank). It is applied
  before owner keys are drawn.

These banks are different codebooks and are analysed separately.

### 3.6 Pre-registration
Every experiment was specified in a versioned design document, committed to git before its data were
generated. Each specification fixes the analysis, the pass criteria, and predictions or stopping
rules. We report all pre-registered outcomes, including failed gates and one voided analysis. Appendix
X lists every deviation and correction, with the commit at which it was made.

---

## 5 Nominal Keys Collide

A generator with 768 keys suggests a large attribution codebook. We test whether the student-side
signal actually distinguishes the combinations, or only some of their components. [EXP-M3S0 D1]

**Held-out attribution among teacher traces.** With problem-grouped 5-fold cross-validation over the
64-key pool:
- Exact-key top-1 accuracy is modest: 0.28 (Tulu, GSM8K), 0.33 (Qwen, GSM8K) and 0.26 (Tulu, ARC),
  against a chance rate of 0.016.
- Collapsing the same predictions to the component classes gives 0.65–0.72 for the reasoning
  instruction (12 classes), 0.31–0.40 for the persona (16 classes) and 0.47–0.53 for the template
  (4 classes).

**Where the errors go.** Among misattributed traces, the predicted key shares the true key's
instruction 48–61% of the time. That is **6.5–8.3× the base rate** of same-instruction decoys (≈ 7%).
Shared personas occur at 1.05–1.8× their base rate, and shared templates at 1.1–1.3×.

**Consequence for the owner test.** On students trained on a trained key, a same-instruction decoy
occupies one of the top three ranks (excluding the owner key) for 14/16, 16/16 and 16/16 key–student
pairs in the three settings. The owner key usually still ranks first, and removing same-instruction
decoys leaves the pass counts nearly unchanged (12/16, 16/16, 14/16). So the test still detects a
signature, but the runners-up it must beat are almost always keys that share its instruction.

**Interpretation.**
- In the tested generator, distinguishability is dominated by the 12 reasoning instructions. Personas
  and templates add little distinguishing signal even in the teacher traces (the D1 analysis is on
  teacher traces; the student-side evidence is the rank pattern above).
- This is a statement about this generator, not a general capacity limit for prompt-implanted
  signatures. It does imply that the nominal key count overstates the effective codebook.
- It also changes the natural null. Collapsing the bank to 12 instruction classes would change the
  hypothesis being tested, and with a minimum attainable p of 1/12, a non-randomised rank test at
  that granularity could never reach α = 0.05.

The instruction banks used later (§7–§9) are built directly over distinct instructions for this
reason.
