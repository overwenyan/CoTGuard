# Research prompt — distillation provenance: experiments closed (M3–M13), paper in drafting (2026-09-18, v5)

_Supersedes v4 (`status_prompt_2026-09-17.md`). Self-contained; paste into an advisor, reviewer or deep-research
session. Sources of record: `.pipeline/memory/experiment_ledger.md` (EXP-M3 … EXP-M13), `.pipeline/memory/decision_log.md`,
pre-registrations `.pipeline/docs/m3…m13_design.md`, verified citations `.pipeline/docs/citations_verified.md`.
Every round was pre-registered in git before its data existed; failed gates, voided runs, corrections, three scorer bugs,
one extractor bug and one pre-registration design flaw (M13) are all logged. **No experiment is running or queued.**_

---

## 1. Context

Small academic group, school-scale shared compute (8×H200 node, 8×L40S node). Target **ACL 2027 main**.

**Question.** Can the owner of a teacher LLM tell, black-box, that a student was distilled from its reasoning traces —
and name *which checkpoint* of its own release line?

**Setup.** Teachers are public post-training ladders: Tulu-3-8B (SFT→DPO→RLVR), OLMo-3-7B-Instruct (SFT→DPO→final),
OLMo-3-7B-Think (voided: small students cannot absorb its long traces), Zephyr-7B (Mistral base, SFT→DPO). Students are
LoRA (r = 32, α = 64, 3 epochs) fine-tunes of Qwen2.5-1.5B-Instruct and Llama-3.2-1B-Instruct on 1,500-trace GSM8K or
MATH corpora — about 700 students — plus 78 Qwen2.5-7B-Instruct students for one scale check. Each student answers the
same 300 held-out probes. Read-outs: TF-IDF 1–2 gram + LR (pre-registered primary), gte-base embeddings, POS 2–4-gram
templates.

**Tests.**
- **T0** (standard owner test): multi-class read-out; conformal p = (1 + #{cal ≥ s})/(1 + n), α = 0.05; calibration =
  reference students of teachers **outside** the owner's post-training line.
- **T1** (reference-aware): T0 **and**, for each same-line relative *b*, a one-sided t prediction interval against
  *b*'s own reference students under a pairwise owner-vs-*b* read-out.

---

## 2. Experiments run (all closed)

| Round | What | Outcome |
|---|---|---|
| M3 | Prompt-implanted secret instruction ("key") + rank test | Transfers, but identifies **the instruction, not the owner**: an independent teacher given the same instruction is flagged 8/8. A teacher-identity read-out separates the sources at AUC 1.00 **from the same outputs**. Paraphrase leaves 8/16; 10% dilution 0/6. |
| M4 | Secret learnable reasoning-move watermark | **Killed by its specificity gate.** |
| M5 | Open-set teacher attribution, 7 teachers | Closed-set 35/35; same-lineage confused 100%. |
| M6 | Alignment ladder, 9 checkpoints | Distinguishability **step-specific** (SFT→DPO AUC 0.93–0.99, RL-on-DPO 0.72–0.76). **T0 flags same-line relatives at 1.0.** |
| M7 | Reference-aware test, 420 fresh students, GSM8K + MATH | **T1 cuts relative FPR from 0.9–1.0 to ≤ 0.2 at TPR 1.00** in all 4 cells; 3 per-relative references and 25 queries suffice. |
| M8 | Robustness | Per-relative protection; **coverage gap** (unreferenced distant relative rejected 0.0, adjacent flagged 1.0 under one calibration); cross-line calibration cannot be dropped; test reports the **majority** contributor. |
| M9 / M9b | Adaptive distiller | Survives paraphrase; **imitating a sibling evades and frames** (replicated on all 4 adjacent pairs × 2 directions; 6/8 valid). No direction rule, no mechanism. |
| M10 | Cross-vendor ladder (Zephyr) | All predictions pass. |
| M11 | Non-lexical read-outs | Diagnosis and fix read-out-independent; **severity** read-out-dependent (TF-IDF 8 collapsed pairs per MATH cell, embeddings 2–3). |
| M12-A | Geometry rule on held-out Zephyr | **Failed.** Threshold form **contradicted** (4 collapsed pairs at ratio 1.04–1.15); ordering form **untestable** there. |
| M12-B | Scaffold-only rewrite | **Mixed.** Evades on pairs **disjoint** from imitation's (imitation 2/8, scaffold-only 3/8, union 5/8). |
| **M13** | **7B student**, the AllenAI · GSM8K · Qwen cell, all else reused | **First run void by my pre-registration error; after the correction, G1 and G2 both pass** (details §5). |

**The attack's four outcomes (Table 2):** **evade** (owner loses the student) · **frame** (a sibling claims it) ·
**joint claim** (both pass — Craver's ambiguity attack) · **laundering** (signal gone, nobody implicated — silent, so the
worst for an auditor). Accuracy cost against the owner's **own unattacked** students: imitation −0.028 to +0.030,
scaffold-only −0.027 to +0.023 (12 valid cells each).

---

## 3. The paper's claim

A standard owner test cannot reject a vendor's own earlier or later checkpoints because its calibration set never
contains them — a **semantics failure, not a power failure**, needing no adversary. The single-variable demonstration is
M8's pooled rejector (one read-out; only the calibration population changes; distant relative 0.0, adjacent 1.0). T1 is
the **remedy** (it changes both coverage and read-out), and it works across two vendors, two datasets, three read-outs
and, in one cell, 7B students. It is bounded: per-relative, needs cross-line calibration, dies to a distiller that
imitates a sibling. So it is a **first-party lineage attestation** — like a published checksum **except that it is not
tamper-evident**, because laundering leaves no trace.

**Five conditions for the positive claim** (each removed experimentally and costed): tester is the **vendor**;
references for **every** relative; **cross-line** calibration kept; distiller does **not imitate**; owner supplied the
**majority** of the data.

**Pattern the paper opens and closes on:** *same outputs, different null, different answer* — first at the instruction
level (§4, M3), then at the checkpoint level (§5.3).

**Theory (§3, no novelty claimed).** Corollary 1: TPR_a − TV(P_a, P_b) ≤ FPR_b ≤ α + TV(P_C, P_b), in the law of the test's
own statistic; T0's collapse is the **upper** bound going vacuous. Tightness cites Le Cam's two-point identity (the same
inequality, not an extra step). Corollary 1b: the bound holds for any suspect law Q, including an adversary's. The four
outcome **names** live in §6.3 beside Table 2, not in §3, so the theory is not read as predicting the attack.

---

## 4. Decisions in force

1. **Experiment budget closed.** M13 was the one reopened cell and is done. Classifier-aware attacker and non-math task
   remain held.
2. **Pre-register gates and kill criteria in git before data**; negative results in the first sentence; log every
   correction; **compute the smallest attainable p-value, 1/(1+n_cal), for every rank/conformal gate before committing**
   (new rule after M13), and report an unattainable test as *void*, never *failed*.
3. **Contradicted ≠ not replicated.** State each failure mode separately.
4. **Scaffold-only sits beside imitation, not above it.**
5. **No mechanism claims.** Post-hoc stories (e.g. "SFT traces carry less scaffold") stay in the ledger, labelled, for
   future work only.
6. **Numbers come from scripts, never typed by hand.** `make_tables_s56.py` (Tables 1–2, step AUC, accuracy cost, 7B
   cell), `make_appendix_x.py` (both extractors), `make_read_s3_s63.py` (fails the build if §6.3's remark and Table 2
   stop using the same four outcome words). Three hand-typed numbers have already been caught wrong this way.
7. **Only primary-verified venues are cited with a venue.**
8. **Compute:** account `ihc`; ≤ 96 GB RAM per GPU on L40S; check `free -h` (look at *available*) before memory-heavy
   jobs; training on H200.

---

## 5. Corrections (all go in the integrity appendix; the first two are also in the body)

- **M13 void first run (my error).** I reused M7's "3 references ≡ 10" (a T1 result) for T0's calibration too: 3 teachers ×
  3 students = 9 scores, floor p = 0.1 > α, TPR 0 by construction. The scorer printed "G1 fails — scale-dependent"; the
  verdict was **withdrawn**, calibration restored to M7's 10 per teacher (floor 0.032) **after seeing the void result**,
  gates unchanged, 42 more students trained. Corrected result: **G1 7/12** (need 6; 1.5B had 8), **G2 11/12** (need 10),
  mean relative FPR 0.03, owner TPR 0.94. Resolution is thirds (3 test students). The void run's post-hoc picture (8/12)
  differed from the correct test (7/12) — kept only to show why post-hoc numbers are not results.
- **Extractor bug.** Skipped numbers followed by a period. Underestimated accuracy by up to 0.13 (teachers +0.005 to
  +0.133, student means +0.016 to +0.091 — an earlier "0.03–0.13" was hand-typed and wrong). Retracts three secondary
  claims. No attribution statistic consumes it.
- **Accuracy baseline for the attack.** An earlier round compared attacked students with paraphrase-only students
  (themselves 2–6 points lower), flattering the attack; now compared with unattacked students.
- **M8 scorer inverted class order** (caught by a weaker-test-cannot-out-power check; permanent sentinel added). **M8
  pre-registration flaw** (two tests identical in a 3-stage line). **M9b key collision.** **M9 gates on voided corpora.**
  **Tulu-3 checkpoint mislabel.**
- **Advisor-proposed sentence that did not hold:** "only the calibration set changes, and the verdict flips" is false for
  T0→T1 (T1 also swaps the read-out); §5.3 now uses M8's pooled rejector for the single-variable claim.
- **Analysis errors of mine, fixed:** a vacuous-TV inference in Corollary 1's evidence; "evasion/framing" vs Table 2's
  "evade/frame" (now enforced by script).

---

## 6. Writing status

| Section | State |
|---|---|
| §2 related work | **Drafted.** Hinge: owner-side trace rewriting (Ma et al., ACL 2026) vs our distiller-side rewrites. Wadhwa et al. (Findings ACL 2025) credited as the closed-set limit in which our failure cannot occur. |
| §3 setup, test, analysis | **Drafted** (Propositions 1–3, Corollaries 1 and 1b). |
| §4 instruction level (M3) | **Drafted** as the opening demonstration of the pattern. |
| §5 ladders, diagnosis, read-out dependence | **Drafted**; §5.3 sharpened (pooled rejector, Wadhwa as the limit, "you can't reject what you didn't calibrate on" answered, M3 as the first instance). |
| §6 remedy, 7B cell, boundary, attacks, scope | **Drafted**; §6.4 answers "why no defence" (common regime; Zhang et al. impossibility; where a defence would have to live: style + membership signals over the same probes). |
| Appendix X (integrity) | **Tables generated**; prose not written. |
| Figure 2, budget figures | **Not made** (script from result JSON). |
| §1 abstract + intro | **Last.** |

Citations: 17 classics + 4 recent verified against primary records; caveats: Tsybakov section number unverified (cite
"Ch. 2"); Mansurov et al. still a preprint.

---

## 7. Next steps, in order

1. **Read arXiv 2512.20908 in full** before anything touches the abstract (title-level collision; abstract suggests it
   attributes spans within one teacher's output, not models to checkpoints — unconfirmed).
2. **Figure 2** (T0 vs T1 paired bars per cell) and the budget curves, by script.
3. **Appendix X prose** around the generated tables, including the M13 void run.
4. **§1 abstract and intro**, using the carried sentences in `draft_s5_s6.md`.
5. Full-paper consistency pass: every number traced to a generated file or a ledger entry.

## 8. Open decisions for the user

- **ARC-Challenge instruction-level result (M3 G-R3, 14/16):** the per-arm length rule that makes it a pass was adopted
  after seeing the output. Report it as a pass, or only in the appendix with both readings? (Currently appendix only.)
- Whether to re-open either held experiment (classifier-aware attacker, non-math task) — default is no.

## 9. What would most help from a reader

- Does §6.1's disclosure of the M13 void run read as integrity or as a red flag, and is one 7B cell with thirds
  resolution enough to drop the scale limitation?
- Does §5.3 now carry the diagnosis sharply enough to be the contribution, given Wadhwa et al.?
- Does "attestation, not a guarantee — and not tamper-evident" survive a reviewer who wants a defence?
