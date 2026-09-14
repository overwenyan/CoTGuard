# Deep research prompt — upgrading M3 to an ACL main-conference paper (2026-09-15)

---

## Context: who we are and what we have
We are a small academic group (school-scale compute: typically one GPU, occasionally up to ~7 H200s)
working on **provenance of distilled reasoning models**. We want to turn a completed, pre-registered
empirical study into an **ACL 2027 main-conference paper**, ideally with a **method contribution**, not
only analysis. Please do a thorough literature and feasibility investigation. Use sources up to
September 2026 and cite every claim with a link.

### Setting studied so far
- **Owner:** prompts a teacher LLM with a secret natural-language **reasoning instruction** (a "key",
  e.g., "state one assumption the problem leaves implicit"). The owner publishes the reasoning traces
  **unedited**.
- **Distiller:** fine-tunes a student on (problem, trace) pairs and never sees the key.
- **Owner's test:** black-box queries to the student. A TF-IDF + logistic-regression read-out trained
  only on teacher traces scores every key in a bank of K randomly drawn keys, and a rank test gives
  p = (1 + #{decoys scoring ≥ owner key}) / K, with vetoes on clean-trained and base students.
- **Models:** teachers Tulu-3-8B and Qwen2.5-7B-Instruct; students Qwen2.5-1.5B-Instruct and
  Llama-3.2-1B-Instruct, LoRA; tasks GSM8K (main) and ARC-Challenge.

### Results
1. **Transfer is detectable.** 16/16 pairs with a Qwen teacher; 12/16 with a Tulu teacher in a 64-key
   pool (α = 0.05). Survives correctness filtering (7/8). About 150 traces suffice in one setting.
2. **Nominal keys collide.** In a 768-key generator (template × persona × instruction),
   distinguishability is dominated by the 12 reasoning instructions: misattribution goes to
   same-instruction keys at 6.5–8.3× base rate, while persona and template carry little signal.
3. **Detection identifies the instruction, not the source.**
   - Students trained on an **independent teacher's traces under the same instruction** are flagged
     8/8 (Llama + Qwen students × 2 seeds). Different-instruction students: 0/8.
   - A **teacher-identity read-out** trained on held-out instructions separates the two teachers'
     students with AUC 1.00 (both student families, both key banks). So source information exists in
     student outputs, but the key-based test ignores it.
4. **Rewriting.**
   - A neutral paraphrase of the traces removes attribution for ~half the keys (8/16).
   - Under claim-preserving rewording, **content-adding** instructions survive and **format-only**
     ones mostly do not: 8/8 vs 2/8 on Qwen (Fisher p = 0.0035), 6/6 vs 2/6 on Llama.
   - A matched replication was infeasible: format-only instructions are intrinsically less
     distinctive (median separability 0.52 vs 0.99) and shorter.
   - Rewriting to standard solutions or bare equations removes signatures but wrecks the student.
5. **Dilution.** Detected at 50% keyed data (4/4 Qwen, 6/6 Llama), partially at 25% (2/4), not at
   ≤ 10% (0/4, 0/6). Increasing queries from 200 to 1,319 does not help.
6. **Utility.** Keyed training costs the Qwen student about 12 accuracy points [−17.8, −6.6]. No loss
   was detected on Llama, whose students barely improve over base. The cost is associated with the
   instruction's effect on teacher accuracy (ρ = 0.50).
7. **Stealth.** Distiller-side screens detect keyed traces at only 17–25% TPR at 5% FPR (not adaptive).
8. **Theory (explanatory).**
   - The rank test controls a *key* null, not a *source* null.
   - Source attribution needs teacher-specific information beyond the instruction (a total-variation
     argument).
   - More queries cannot overturn a wrong population ordering, which explains the dilution result.

### Prior work already known (please go beyond these)
Asking Back (arXiv 2605.16462), Trace Rewriting (Ma et al., ACL 2026), ReasMark (ACL 2026), PROSE /
The Shape of Ownership (arXiv 2609.02553), In-Context Watermarks (ICLR 2026), Watermarking Makes
Language Models Radioactive (NeurIPS 2024), On the Learnability of Watermarks (ICLR 2024), DITTO
(EACL 2026), Unified Attacks / CDG-KD (arXiv 2504.17480), Subliminal Learning (arXiv 2507.14805), LLM
Dataset Inference (arXiv 2406.06443).

---

## What I want from this research

### A. Candidate method directions: novelty, closest prior work, feasibility
For each direction, find the closest existing work (2024–2026), say precisely what is and is not
already done, estimate novelty (high / medium / low), and outline a minimal convincing experiment
under our compute.

1. **Source-bound keyed reasoning signatures (prompt-only).** Instead of one fixed instruction, the
   owner computes a *per-problem* instruction from a secret seed. For example, `hash(problem, secret)`
   selects which of several equally valid reasoning moves (a sanity check, an estimate, a restatement,
   a unit check…) the teacher must perform. An imitator who knows the instruction family but not the
   seed cannot reproduce the per-problem pattern.
   - Detection becomes a **per-instance keyed test**: agreement between the student's move and the
     secret selection on owner-chosen probe problems. Its power grows with the number of queries,
     unlike our rank test.
   - Is anything like a "semantic / reasoning-move-level keyed watermark implemented purely by
     prompting" already published? How does it relate to PROSE, ReasMark, SemStamp/k-SemStamp, and
     semantic watermarks?
   - Would content-level moves survive paraphrase, as our content-adding result suggests?
2. **Composite attribution test.** Stage 1: detect the instruction. Stage 2: verify the owner's
   *teacher identity* against a reference pool of public teachers (open-set), calibrated with
   conformal or rank-based p-values over teacher decoys. What exists on model / teacher attribution of
   distilled students, open-set model fingerprinting from outputs, and "model lineage" or
   "distillation detection" (e.g., identifying which model a student was distilled from)?
3. **Dilution-robust detection.** Likelihood-ratio or per-instance paired tests, active query
   selection (probing problems where the keyed behaviour is most distinguishable), and sequential
   testing. What does the radioactivity / membership-inference / dataset-inference literature offer
   for detecting a small keyed fraction (1–10%) of a training mix?
4. **Paraphrase-robust read-outs.** Claim- or step-level representations (extracting reasoning moves
   with an LLM, reasoning-graph features) instead of lexical features. What evidence exists on
   stylometry vs semantic features for attributing LLM outputs after paraphrase (DIPPER, Parrot)?
5. **Utility-aware key design.** Selecting or optimising instructions that are distinctive and
   rewrite-robust but do not reduce teacher accuracy. Any prior work on optimising
   watermark/fingerprint prompts under utility constraints?
6. **Adaptive attackers.** Instruction reconstruction from published traces (can an attacker infer the
   key?), accidental instruction collision in real-world prompts, and targeted scrubbing that
   preserves utility. What attacks exist, and what do strong papers in this area evaluate?

### B. Theory that would strengthen a main-conference submission
- Identifiability conditions for source attribution from student outputs. Links to hypothesis
  testing, total variation, and data-processing inequalities under distillation.
- Sample complexity of detecting a keyed fraction α in the training mix with N queries: rank test vs
  per-instance keyed test (e.g., N ∝ 1/(αΔ)² scaling).
- Capacity of prompt-implanted codebooks: how many distinguishable keys a behavioural channel
  supports, and collision bounds.
- Point to comparable theory in watermarking or radioactivity papers that we could adapt.

### C. What ACL main-conference reviewers expect here
- For a security/provenance paper at ACL or EMNLP main (not Findings): typical baselines, attack
  suites, datasets (e.g., larger reasoning corpora such as OpenThoughts or s1K-style data, MATH,
  code), student scales (7B?), full fine-tuning vs LoRA, long-CoT reasoning models, number of keys and
  seeds, statistical reporting.
- Which of our current results would count as strong contributions, and which are likely to be seen
  as expected (e.g., "someone who knows the key can reproduce the signature" as analogous to key
  compromise)?
- Recent accepted examples (2024–2026) combining an analysis finding with a method, and how they
  framed the contribution.

### D. Concrete recommendation
- Rank 2–3 paper framings for ACL main, e.g.:
  - (i) "a source-bound prompt-only reasoning watermark that survives paraphrase and dilution, with
    the collision/source-ambiguity analysis as motivation";
  - (ii) "a composite attribution test";
  - (iii) an analysis paper with a lighter method.
- For each: expected novelty, main risks, a minimal experiment plan (models, data, baselines, attacks,
  metrics), rough GPU-hours, and the stop / kill criteria we should pre-register.

Please be explicit about uncertainty and cite primary sources (arXiv / ACL Anthology / OpenReview
pages), not blog summaries.
