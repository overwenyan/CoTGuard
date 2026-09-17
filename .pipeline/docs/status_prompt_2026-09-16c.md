# Research prompt — distillation provenance, evidence complete (2026-09-16, v3)

_Supersedes v2 (`status_prompt_2026-09-16b.md`). Self-contained; paste into an advisor or deep-research session.
Sources of record: `.pipeline/memory/experiment_ledger.md` (EXP-M3 … EXP-M10), `.pipeline/memory/decision_log.md`,
pre-registrations `.pipeline/docs/m3…m10_design.md` and `m9b_design.md`. Every round was pre-registered in git before
its data existed. Failed gates, voided analyses, corrections, three scorer bugs and one extractor bug are all logged._

---

## 1. Context
Small academic group, school-scale compute (~7 H200 + L40S, shared). Target **ACL 2027 main**. Question: can the owner
of a teacher LLM tell, black-box, that a student was distilled from its reasoning traces — and name *which* checkpoint?
Students: LoRA (r = 32, 3 epochs) fine-tunes of Qwen2.5-1.5B-Instruct and Llama-3.2-1B-Instruct on 1,500-problem GSM8K
or MATH corpora. Read-out: TF-IDF (1–2 gram) + logistic regression.

**Tests used throughout.** *T0* (standard owner test): multi-class read-out, calibrated on reference students of
teachers **outside** the owner's post-training line. *T1* (reference-aware): T0 **and** a one-sided t prediction
interval against reference students of **each same-line relative**, under a pairwise owner-vs-relative read-out.

## 2. Experiments, in order
| Round | What | Outcome |
|---|---|---|
| M3 | Prompt-implanted secret instruction ("key") + rank test | Signature transfers, but identifies **the instruction, not the owner** (independent teacher with same instruction flagged 8/8). Teacher identity still recoverable (AUC 1.00). |
| M4 | Secret learnable reasoning-move watermark | **Killed by specificity gate**: an imitator with a different secret split was flagged more strongly than the owner's own student. |
| M5 | Open-set teacher attribution, 7 teachers, 70 students | Closed-set 35/35, open-set TPR 1.00 / FPR 0.07; same-lineage confused 100%, unseen teachers FPR 0.3–1.0, 10% mixtures unreliable. |
| M6 | Alignment-stage ladder: 9 checkpoints, 3 AllenAI lines (Tulu-3, OLMo-3-Instruct, OLMo-3-Think), 90 students | Distinguishability is **step-specific**: SFT→DPO AUC 0.93–0.99, RL-on-DPO 0.72–0.73. **T0 flags all same-line relatives (FPR 1.0).** 1–1.5B students cannot absorb long Think traces. |
| M7 | Reference-aware test on 420 fresh students, GSM8K + MATH, disjoint problems and traces | **T1 cuts relative FPR from 1.0 to ≤ 0.2 at TPR 1.00** in all 4 cells. 3 references and 25 queries suffice. No leakage. |
| M8 | Robustness: unknown relative, unseen teacher, mixtures | Protection is **per-relative**; a **coverage gap** (distant unreferenced relative rejected at 0.0, adjacent one flagged at 1.0); out-of-line calibration **cannot be dropped**; 50% mixtures detected, 10% not — the test reports the **majority contributor**. |
| M9 | Adaptive distiller: paraphrase (AD1) vs imitate-a-relative (AD2), owner does not anticipate | T1 **survives paraphrase** (TPR 0.92/0.83) but **imitation evades and spoofs** (TPR 0.58, spoof 0.5 on both families). |
| M10 | Cross-vendor ladder: Mistral-7B → zephyr-7b-sft-full → zephyr-7b-dpo-full, 80 students | **All predictions pass.** T0 relative FPR 1.0 in both directions; T1 → 0.0–0.1 at TPR 1.00; SFT→DPO AUC 0.97. Closes "AllenAI-specific" for the SFT→DPO step. |
| M9b | Imitation on all 4 adjacent pairs × both directions, answer-preserving rewrite, 48 students | **Attack replicates** (mean TPR 0.67, spoof 0.61/0.50; 6 of 8 attacks valid). **No direction rule, no mechanism** identified. Three outcome types (below). |

### M9b outcome types (Qwen / Llama)
| Attack (owner → imitated target) | Owner still detects | Target wrongly claims | Type |
|---|---|---|---|
| tulu_sft → tulu_dpo | 0.0 / 0.0 | 1.0 / 0.67 | **evade + frame** |
| tulu_dpo → tulu_sft | 1.0 / 1.0 | 1.0 / 1.0 | **joint claim** |
| tulu_dpo → tulu_rlvr | 1.0 / 1.0 | 0.67 / 0.33 | partial frame |
| tulu_rlvr → tulu_dpo | 1.0 / 1.0 | 1.0 / 1.0 | joint claim |
| olmoi_sft → olmoi_dpo | 0.0 / 0.0 | 0.0 / 0.0 | **laundering** (nobody can claim it) |
| olmoi_final → olmoi_dpo | 1.0 / 1.0 | 0.0 / 0.0 | no effect |

- **Direction:** only two pairs had both directions valid, with opposite signs (+0.83, −0.5) → no rule.
- **Mechanism:** both pre-registered predictors came out with the *opposite* sign and non-significant — pair
  closeness ρ = −0.72 (p = 0.13), target narrowness ρ = −0.63 (p = 0.27), n = 6. No mechanism is claimed.

## 3. Corrections and retractions (all logged; these change earlier claims)
1. **Answer-extractor bug (largest).** The GSM8K extractor used since M3 skipped any number followed by a period
   ("The answer is 8." read as the "3" in "Step 3"), underestimating accuracy by 0.03–0.13. Kept for provenance; a
   unit-tested replacement is used from now on, with both shown.
   - **Retracted:** the capability-vs-identity dissociation (M5b — its only significant result reverses sign and
     loses significance); "RLVR is less accurate than DPO on GSM8K" (0.854 vs 0.849 corrected); "utility cost is
     student-dependent" (Llama also loses 7–9 points, significantly).
   - **Unaffected:** every attribution result (T0/T1/AUC/FPR/TPR never used the extractor); M6/M7 voids; M9 gates.
2. **Misreported direction claim.** I previously said imitation "succeeded toward the later/stronger stage"; the
   two M9 successes went in opposite directions. An advisor built an RL-narrowing mechanism on that summary; M9b
   found no direction rule.
3. **Scorer bugs:** an inverted pairwise class order produced a spurious KILL in M8 (caught because a weaker test
   cannot have lower power; a sentinel now recomputes M7's T1 TPR every run); M9b student keys collided with M8
   mixture students; M9's gates initially included voided corpora.
4. **Pre-registration flaw:** in a three-stage line, M8's "pooled relatives" test was identical to its
   "partial" test, so two predictions measured one quantity.
5. **Citation misuse:** Li et al. 2502.12143 and Xu et al. 2411.07133 support only the long-trace learnability gap,
   not a capability–identity dissociation (moot now that the dissociation is retracted).

## 4. The claim as it now stands
**Diagnosis.** Every negative provenance result in M3–M8 is a statement about the **null the test controls**, not
about missing information: the same student outputs that a standard owner test cannot resolve separate perfectly once
the calibration set contains the relative. This replicates across two vendors (AllenAI, HuggingFace/Mistral), two
datasets and two student families.

**Scoped remedy.** Reference-aware testing (T1) names the specific checkpoint when the owner **(i)** is the vendor of
its own lineage, **(ii)** references every relative, **(iii)** keeps out-of-line calibration, **(iv)** faces a
distiller that does not imitate a relative, and **(v)** supplied ≥ ~50% of the corpus.

**Boundary.** Each condition was broken experimentally with a measured failure rate. The sharpest is (iv): an
in-context imitation rewrite — free in accuracy — evades, frames an innocent relative, or launders the student so
nobody can claim it.

**Theory.** Three propositions (key null vs source null; a total-variation condition on what the read-out
preserves; query scaling cannot fix a wrong population ordering), plus a coverage-dependent FPR floor stated as a
corollary of TV + conformal validity (Tsybakov 2009; Vovk et al. 2005; Barber et al. 2023; Bates et al. 2023;
Mondrian conformal prediction). Novelty is claimed only for the empirical verification, not the bound.

## 5. Decisions taken
1. **Diagnosis-forward measurement paper**, with the attack as load-bearing evidence and the remedy as setup. Working
   title: *"Calibration, Not Capability: What a Black-Box Test Can and Cannot Prove About Distilled Reasoning
   Provenance."*
2. **Regimes split explicitly:** first-party/vendor (remedy works) vs third-party/auditor (it does not).
3. **Positioning:** cite and differentiate from RefDistDet (Rawat et al., arXiv 2607.09692, preprint) — it needs
   earlier-checkpoint *weights* and has no calibration analysis, attack, ladder or mixture study. Pre-empt "cohort
   normalisation rediscovered" with T-norm (Auckenthaler 2000), impostors (Koppel & Winter 2014), open-set recognition
   (Scheirer 2013), conformal exchangeability (Bates 2023), dataset inference (Maini 2024). Frame the attack in the
   ambiguity/imitation tradition (Craver 1998; Brennan et al. 2012; Watermarks in the Sand, Zhang et al. 2024;
   Watermark Stealing, Jovanović et al. 2024).
4. **Dropped from the paper:** the capability–identity dissociation (retracted); any RL-narrowing or closeness
   mechanism for the attack (not supported).
5. **Mixture finding** framed as passive (~50% floor) vs active marks (1–5%; Sablayrolles 2020, Sander 2024).
6. **Integrity reporting:** the extractor correction goes in the **main text** as a short paragraph (it retracts a
   prior result), with the full v1-vs-v2 audit in an appendix.
7. **Held for rebuttal, not run:** a read-out-optimising attacker; a 7B student; long-CoT with a larger student.

## 6. What happens next
1. Write §5 (ladder, M6/M10) and §6 (remedy, boundary, attack: M7/M8/M9/M9b) first; then §3 theory with the
   corollary; §4 instruction level (M3, compressed); related work; abstract last.
2. Regenerate every accuracy number with the corrected extractor; build figures: T0-vs-T1 bars across vendors and
   datasets, the coverage-gap plot, the attack-outcome matrix, reference- and query-budget curves.
3. Integrity appendix: extractor audit, scorer bugs, pre-registration flaw, void corpora.

## 7. Questions for you
1. **Laundering vs framing.** The attack produced three outcome types. Is "laundering" (no party can claim the
   student) a recognised category in the watermark/authorship literature, and is it a stronger or weaker threat to
   report than spoofing?
2. **No mechanism.** Both mechanism predictions reversed sign (n = 6, non-significant). Is it acceptable at ACL to
   report the attack without a mechanism, stating explicitly that two pre-registered accounts failed? Or does that
   invite "why does it work?" rejections that we should pre-empt with more attacks (larger n)?
3. **The retraction paragraph.** How should a main-text correction of our own earlier result be worded so it reads
   as rigour rather than unreliability? Any precedent of papers doing this well?
4. **Is the evidence enough for main track?** Two vendors, two datasets, two small student families, ~900 LoRA
   students, one attack family. What would a reviewer still ask for first?
5. **Scope of "vendor regime."** Condition (ii) requires referencing every relative. Is there a realistic deployment
   story (e.g. a vendor pre-computing reference students for every public checkpoint it releases) that makes the
   positive half practically meaningful, and is there prior work on such provenance registries?
