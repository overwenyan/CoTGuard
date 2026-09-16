# Research prompt — distillation provenance after the robustness and attack rounds (2026-09-16, v2)

_Supersedes `status_prompt_2026-09-16.md`, which was written before M8 and M9 finished. Self-contained;
paste into an advisor or deep-research session. Repo sources of record: `.pipeline/memory/experiment_ledger.md`
(EXP-M3 … EXP-M9), `.pipeline/memory/decision_log.md`, pre-registrations `m3/m4/m5/m6/m7/m8/m9_design.md`,
paper plan `paper_plan_unified.md`. Every round was pre-registered in git — predictions, gates, kill
criteria — before its data existed; failed gates, voided analyses, corrections and one scorer bug are
all logged._

---

## 1. Context
Small academic group, school-scale compute (~7 H200 + L40S, shared). Target **ACL 2027 main**. Question:
can the owner of a teacher LLM tell, black-box, that a student was distilled from its reasoning traces —
and name *which* checkpoint? Students throughout: LoRA (r = 32, 3 epochs) fine-tunes of Qwen2.5-1.5B-Instruct
and Llama-3.2-1B-Instruct on 1,500-problem GSM8K or MATH corpora. Read-outs: TF-IDF (1–2 gram) + logistic
regression (primary); gte-base embeddings and POS-templates (secondary).

## 2. The finished arc, in one line each
- **M3** (prompt-implanted keys): the signature transfers, but the test identifies **the instruction, not the
  owner** — an independent teacher given the same instruction is flagged 8/8. Teacher identity is nonetheless
  recoverable from the same outputs (AUC 1.00).
- **M4** (secret learnable reasoning-move watermark): **killed by its own specificity gate** — an imitator with
  a different secret split was flagged more strongly than the owner's own student.
- **M5** (open-set teacher attribution, 7 teachers, 70 students): closed-set 35/35; open-set TPR 1.00 / FPR 0.07;
  but same-lineage checkpoints confused 100%, unseen-teacher FPR 0.3–1.0, composite instruction∧teacher test fails,
  10% mixtures unreliable.
- **M6** (alignment-stage ladder, 9 checkpoints in 3 lines, 90 students): distinguishability is **step-specific,
  not distance-monotone** (SFT→DPO AUC 0.93–0.99; RL-on-DPO 0.72–0.73). The standard owner test flags **all**
  same-line relatives (FPR 1.0) because calibration contains only other lines.
- **M7** (reference-aware test, 420 fresh students, GSM8K + MATH, disjoint problems and traces): handing the owner
  10 reference students per relative takes relative FPR from 1.0 to **≤ 0.2 at TPR 1.00**, in all four
  dataset × family cells. 3 reference students and 25 queries suffice. Leakage diagnostic: 0 errors.
- **M8** (robustness): protection is **per-relative**; **coverage gap** demonstrated; out-of-line calibration
  **cannot** be dropped; 50% mixtures detected, 10% not.
- **M9** (adaptive distiller): T1 **survives neutral paraphrase** but **targeted imitation of a relative both
  evades and spoofs**.

## 3. The two decisive new rounds, in detail

### M8 — robustness of the reference-aware test (pre-registered c7e3076)
Three arms on the M7 material; A1/A2 needed no new training.
- **A1 unknown relative.** Withholding a relative's references restores the failure *for that relative*: in 6 of
  the 8 pairs where the naive test failed, the unreferenced relative is still flagged at 1.0. **P1b killed:** the
  owner must enumerate its lineage.
- **Exploratory T2g** (a generic "not me" rejector trained against the four other teachers pooled, excluding the
  withheld relative) revealed the structure: on GSM8K, an unreferenced **distant** relative is rejected (FPR 0.0)
  while an unreferenced **adjacent** relative is not (FPR 1.0); mean FPR 0.525 → 0.333 at TPR 0.975.
  **This is a coverage gap:** a source closer to the owner than anything in the calibration set cannot be rejected.
- **A2 truly unseen teacher.** A strict, line-internal test (no out-of-line calibration at all) keeps power
  (mean TPR 0.85–0.92) but leaks: on GSM8K one owner per line flags unknown-line students at 1.0 (mean unknown FPR
  0.33); MATH is clean (0.0 / 0.02). Passes 2 of 4 cells, gate was 3 of 4 → **out-of-line calibration is required**.
- **A3 mixtures.** At 50% owner share the owner detects its data 4/4 groups (gate met); at 10% detection falls to
  0.33–0.67 — but **the relative claims those students at 1.0**, correctly, since most of their data came from it.
  Reading: the test reports the **majority contributor**; a minority contributor below ~50% has no usable claim.

### M9 — adaptive distiller (pre-registered c5775cc)
The distiller trains on the owner's traces but rewrites them first; **the owner does not anticipate the attack**
(its references and read-outs are the non-adaptive M7 ones). Rewriter: Qwen2.5-7B-Instruct. 48 students, GSM8K.
- **AD1 neutral paraphrase** (M3's wording). **AD2 imitate a relative**: same rewrite, conditioned on 4 in-context
  traces of the relative.
- Manipulation check: AD1 preserved the final answer in 94–97% of traces (all valid); AD2 in 83–93%, which
  **voided two of four AD2 corpora** (excluded from gates, reported).

| Attacked owner (Qwen / Llama) | AD1 TPR | AD1 spoof | AD2 TPR | AD2 spoof |
|---|---|---|---|---|
| tulu_dpo | 1.0 / 1.0 | 0.0 / 0.0 | *(void)* | *(void)* |
| tulu_rlvr | 1.0 / 1.0 | **0.67 / 0.67** | *(void)* | *(void)* |
| olmoi_dpo | 0.67 / 0.33 | **1.0 / 1.0** | **0.0 / 0.0** | **1.0 / 1.0** |
| olmoi_final | 1.0 / 1.0 | 0.0 / 0.0 | 1.0 / 1.0 | 0.0 / 0.0 |

- **G1 passes** (mean TPR 0.92 / 0.83): the test survives generic rewriting — notably better than M3's
  prompt-implanted signatures, which the same rewrite halved.
- **G2 fails** (mean TPR 0.5 / 0.5 on valid corpora): targeted imitation makes the owner miss students trained on
  its own traces, down to 0.0 for one owner.
- **G3 fails** (mean spoof 0.5 / 0.5, gate ≤ 0.3): the **innocent relative** claims those students at 1.0 — a
  working ambiguity attack in the sense of Craver et al. (1998).
- **The attack is free:** attacked students are *more* accurate than paraphrase-only ones (0.66–0.70 vs 0.52–0.55).
- **Direction asymmetry, unexplained:** imitation succeeded toward the later/stronger stage (rlvr→dpo, dpo→final)
  and failed in the reverse direction.

## 4. Where the claim now stands, fully bounded by our own experiments
**Positive:** an owner can name the specific checkpoint a student was distilled from — *if* it (i) is the vendor of
its own lineage, (ii) trains reference students for **every** relative, (iii) retains out-of-line calibration
against other vendors, (iv) faces a distiller that does not imitate a relative, and (v) supplied ≥ ~50% of the
distillation corpus.
**Negative, everywhere else:** each of (i)–(v) was broken experimentally and each break is a measured failure rate,
not a speculation.
**Mechanism:** every negative result is a statement about the **null the test controls**, not about missing
information. The same outputs a naive test cannot resolve separate perfectly once the comparison set includes the
relative. Three propositions frame this (key null vs source null; a total-variation condition on what the read-out
preserves; query scaling cannot fix a wrong population ordering — corroborated by 25 queries sufficing in M7).

## 5. Positioning already settled (advisor round 4, accepted)
- Paper is **diagnosis + scoped remedy + honest boundary**, not a method paper; the regimes are split explicitly
  into **first-party/vendor** (where the remedy works) and **third-party/auditor** (where it does not).
- Pre-empt "this is cohort normalisation rediscovered" with: Auckenthaler et al. 2000 (T-norm), Koppel & Winter 2014
  (impostors method), Scheirer et al. 2013 (open-set recognition), Bates et al. 2023 (conformal exchangeability),
  Maini et al. 2024 (LLM dataset inference), plus RefDistDet (arXiv 2607.09692, preprint, unverified venue).
- Capability-vs-identity dissociation → paragraph + appendix (mechanism already published: Li et al. 2502.12143;
  Xu et al. 2411.07133).
- Add a **coverage-dependent corollary** to Proposition 1 (conformal exchangeability + a Le Cam two-point argument):
  relatives inside the coverage gap cannot be rejected, and the achievable FPR degrades with the distance to the
  nearest calibration point. M8's T2g result is its qualitative verification.

## 6. Corrections logged this round (for transparency)
1. A scorer bug inverted the pairwise classifier's class order, producing an all-zeros run and a **spurious KILL
   verdict**; caught because the weaker test cannot have lower power than one with TPR 1.00. A permanent sentinel
   now recomputes M7's T1 TPR in every cell.
2. **A pre-registration flaw:** M8's T2 ("pooled available relatives") is identical to T1-partial in a three-stage
   line, so two predictions measured one quantity. The genuinely distinct version (T2g) was added as exploratory.
3. M9's gates initially included two corpora that the pre-registered answer-preservation check had voided;
   recomputed on the valid subset (conclusions unchanged).

## 7. Questions for you — what further research is worth doing?
1. **Is there a defence?** The obvious one: the owner anticipates the attack and trains reference students on
   *rewritten* traces, so calibration covers the attack distribution. Is that a real defence or circular (the owner
   must guess the attacker's rewriter and prompt)? Is there a principled alternative — invariant read-outs,
   randomised probing, or a test whose null explicitly includes "any rewriting of my traces"?
2. **The direction asymmetry.** Imitation toward the later stage worked; toward the earlier stage did not. Is there
   a known account of this (RL narrowing the output distribution, so imitating a narrower target is easier)? Would a
   measurement — for example read-out distance or entropy per stage — be worth including as mechanism?
3. **Does the coverage bound exist in the literature already?** We plan to state it via conformal exchangeability
   plus Le Cam. Is there a stronger existing result — Mondrian / class-conditional conformal prediction, open-set
   risk bounds — that already gives a coverage-dependent lower bound we should cite instead of re-deriving?
4. **Scope of the mixture finding.** "The test reports the majority contributor, and a minority contributor below
   ~50% cannot claim" is practically important (it says a vendor whose data made up 10% of a corpus has no case).
   Is there prior work on minority-contributor attribution in training corpora — data provenance, influence
   functions, dataset inference — that sets a better threshold or a better test?
5. **Given all of this, what is the strongest honest framing for ACL main?** Candidates: (a) a measurement paper
   whose headline is the calibration diagnosis, with the remedy and its attack as evidence; (b) an
   attack-forward paper — "same-lineage provenance claims are spoofable" — with the remedy as the setup;
   (c) a scoped-protocol paper for vendors, stating the five conditions as the contribution. Which survives review
   best, and what is missing from each?
6. **Is anything still missing experimentally?** Candidates we did not run: a 7B student; long-CoT/Think attribution
   with a bigger student; a cross-vendor ladder (Zephyr on Mistral, SFT→DPO only — no vendor outside AllenAI
   publishes a verifiable base→SFT→DPO→RL chain); an attacker that optimises directly against a known read-out.
