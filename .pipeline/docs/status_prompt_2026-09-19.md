# Research prompt — distillation provenance: paper drafted, M14 pre-registered, follow-ups designed (2026-09-19, v7)

_Supersedes v6 (`status_prompt_2026-09-18b.md`). Self-contained; paste into an advisor, reviewer or planning session to
continue analysis, method design or experiments. Sources of record: `.pipeline/memory/experiment_ledger.md` (EXP-M3 …
EXP-M13), `.pipeline/memory/decision_log.md`, pre-registrations `.pipeline/docs/m3…m14_design.md`, follow-up designs
`.pipeline/docs/followup_designs.md`, verified citations `.pipeline/docs/citations_verified.md`.
**No experiment is running or queued.**_

---

## 1. Context

Small academic group; shared compute: 8×H200 node (training), 8×L40S node (vLLM generation/probes; 768 GB RAM).
Target **ACL 2027 main**. Question: can the owner of a teacher LLM tell, black-box, that a student was distilled from its
reasoning traces — and name **which checkpoint** of its own release line?

**Setup.** Teachers: Tulu-3-8B (SFT→DPO→RLVR), OLMo-3-7B-Instruct (SFT→DPO→final), OLMo-3-7B-Think (voided), Zephyr-7B
(SFT→DPO). Students: LoRA (r 32, α 64, 3 epochs, batch 4, **seq cap 4,608**) on 1,500-trace GSM8K or MATH L1–4 corpora;
Qwen2.5-1.5B-Instruct and Llama-3.2-1B-Instruct (~700), plus 78 Qwen2.5-7B-Instruct (one cell, **trained at cap 1,024 —
a disclosed deviation**). 300 held-out probes each. Read-outs: TF-IDF 1–2 gram + LR (pre-registered primary), gte-base
embeddings, POS 2–4-gram templates.

**Tests.** **T0**: multi-class read-out; conformal p = (1 + #{cal ≥ s})/(1 + n_cal), α = 0.05; calibration = 10 reference
students per teacher **outside** the owner's line (≥ 30 scores; floor ≤ 0.032). **T1**: T0 **and**, per same-line
relative *b*, a one-sided t prediction interval against *b*'s reference students under a pairwise owner-vs-*b* read-out.

---

## 2. Experiments (all closed)

| Round | What | Outcome |
|---|---|---|
| M3 | Prompt-implanted instruction ("key") + rank test | Names **the instruction, not the owner** (independent teacher, same instruction: flagged 8/8; different instruction: 0/8). Teacher-identity read-out separates sources at AUC 1.00 from the same outputs. Paraphrase → 8/16; 10% dilution → 0/6. ARC: **void** as run. |
| M4 | Learnable reasoning-move watermark | Killed by its specificity gate (imitator flagged above the owner's student). |
| M5 | Open-set teacher attribution | Closed-set 35/35; same-lineage confused. (Its "Tulu SFT" was really the RL-final model — mislabel.) |
| M6 | Alignment ladder, 9 checkpoints | Step-specific separability (SFT→DPO AUC 0.93–0.99; RL-on-DPO 0.72–0.76). T0 flags relatives at 1.0. |
| M7 | Reference-aware test, 420 students, GSM8K + MATH | T1 cuts relative FPR 0.9–1.0 → ≤ 0.2 at TPR 1.00 in 4/4 cells. Budgets flat (3/5/10 refs; 25–300 probes). |
| M8 | Robustness | Protection is per-relative (A1, pre-registered). **Exploratory, GSM8K:** pooled rejector — unreferenced *distant* relative 0.0, *adjacent* 1.0. Cross-line calibration can't be dropped; mixtures → majority contributor. |
| M9 / M9b | Adaptive distiller | Survives paraphrase; imitation evades + frames (6/8 valid). No direction rule; both mechanism predictors wrong-signed, n.s. |
| M10 | Cross-vendor (Zephyr) | All predictions pass. |
| M11 | POS + embedding read-outs | Diagnosis and fix read-out-independent; **severity** read-out-dependent (MATH: TF-IDF 8/12 vs EMB 2–3/12). |
| M12-A | Geometry rule, held-out Zephyr | Threshold form **contradicted** (4 collapsed pairs at ratio 1.04–1.15); ordering untestable there. |
| M12-B | Scaffold-only rewrite | Mixed; evades on pairs **disjoint** from imitation's (2/8 vs 3/8; union 5/8). |
| M13 | **7B cell** (AllenAI·GSM8K·Qwen) | First run **void by construction** (9 calibration scores → floor 0.10 > α). After restoring M7's 10-per-teacher calibration, gates unchanged: **G1 7/12** (need 6), **G2 11/12**, mean FPR 0.03, **TPR 0.94**. |
| **M14** | **Second 7B cell (MATH·Qwen-7B)** | **Pre-registered 2026-09-19 (`m14_design.md`); to run during review.** Not implemented; no code or data exists. |

**Attack outcomes (Table 2):** evade · frame · joint claim · laundering. Accuracy vs the owner's *unattacked* students:
imitation −0.028…+0.030; scaffold-only −0.027…+0.023.

---

## 3. Claims as drafted

- **Diagnosis:** an owner test controls the null its calibration population defines; the standard population omits the
  vendor's own siblings. Semantics, not power; more probes don't help (Prop. 3). Pattern: *same outputs, different null,
  different answer* — §4 (instruction level) and §5.3 (checkpoint level).
- **Independent corroboration (Rawat et al., 2026, arXiv 2607.09692):** calibrating their detection threshold without one
  teacher's students gives false detections on that teacher (4/6 and 1/6); with every teacher represented, none. Their
  statistic needs student + reference log-probs; candidates cross-vendor; no siblings; no rewriting attacks.
- **Remedy:** reference students per sibling → mean sibling FPR < 0.10 in every cell (2 vendors, 2 datasets, 3 read-outs,
  one 7B cell), at unchanged power; 3 references or 25 probes suffice (varied separately).
- **Five conditions:** vendor is the tester; references for every relative; cross-line calibration kept; distiller does
  not imitate; owner supplied the majority.
- **Attack:** distiller-side rewriting (mirror of Ma et al.'s owner-side rewriting) evades/frames/launders at no
  measurable accuracy cost. *What* it moves: scaffold phrases (descriptive). *When* it succeeds: unexplained.
- **Theory (§3):** Corollary 1 (TPR_a − TV(P_a,P_b) ≤ FPR_b ≤ α + TV(P_C,P_b)); 1b (holds for any suspect law Q). Outcome
  names live in §6.3 beside Table 2.
- **Framing:** first-party lineage attestation — like a checksum **except not tamper-evident** (laundering is silent).

---

## 4. Rules in force

1. Pre-register in git before data; negatives first; log every correction.
2. **Floor rule:** compute 1/(1 + n_cal) < α before committing any rank/conformal gate (n_cal ≥ 19 at α = 0.05). An
   unattainable gate is **void**, never *failed*. (M13.)
3. **Diff the actual job settings** before writing "reused unchanged from Mx" in a design doc. (M13 correction 2: M7
   trained at seq 4,608; M13 at 1,024 — 0.92% of GSM8K examples truncated; disclosed, not retrained.)
4. Exploratory labels go **in the sentence that makes the claim**, bracketed by the pre-registered results.
5. Post-hoc numbers are not results (M13's void-run 8/12; ARC's per-arm pass).
6. Contradicted ≠ not replicated. No mechanism claims; post-hoc stories live in the ledger.
7. Numbers come from scripts only (`make_tables_s56.py`, `make_appendix_x.py`, `make_fig2.py`, `make_read_s3_s63.py`);
   every scorer carries a sentinel recomputing a known result, and refuses to write if it moves.
8. Citations: only primary-verified venues. Liu et al. (2512.20908) = preprint (OpenReview blocked; ICLR 2026 only per a
   secondary index). ADFP (2602.03812) = ICML 2026 poster (confirmed on icml.cc).
9. Compute: account `ihc`; L40S ≤ 96 GB RAM per GPU (check *available*, not *free*); scoring 32 GB; `scontrol update`
   blocked → cancel + resubmit.

---

## 5. Paper status

Drafted in `paper/`: abstract + §1 (`draft_s1_abstract_intro.md`, **206 words — trim to ≤ 200**), §2
(`draft_s2_related.md`), §3 + Corollaries (`draft_s3_s5.md`, `draft_analysis.md`), §4 (`draft_s4.md`), §5–§6
(`draft_s5_s6.md`), Appendix X (`draft_appx_x.md`, 14 entries + the seq-cap disclosure). Generated: Figure 2
(`generated/fig2.pdf`), Tables 1, 2, X.1–X.3, `s56_numbers.json`, `fig2_values.json`.

**Remaining for submission, in order:**
1. **Consistency pass** — ~440 numeric tokens across the drafts (293 in §5–§6), each to be traced to a generated file or
   a ledger entry. Three hand-typed numbers have already been caught this way.
2. **Appendix Y** — instruction-level key banks (from `draft_s3_s5.md` §4.x) and the ARC result with both readings,
   verdict void.
3. **ACL LaTeX assembly**, then trim the abstract.

---

## 6. Assets (no new training needed)

- `experiments/radioactive/data_m7/{tulu_gsm,tulu_math}`, `data_m6`: ~780 LoRA adapters + 300 probe outputs each;
  teacher traces (R300 / POOL_REF / POOL_TEST); attacked corpora (paraphrase, imitation, scaffold-only); mixture
  students; voided Think-line students; the 7B cell.
- Code: `run_m7.py` (gen/probe, vLLM 0.21 venv), `run_m3c.py sft` (LoRA; `M3C_MAXLEN`, `M3C_GRADCKPT`), `score_m7.py`
  (`Cell`: T0/T1/rates), `score_m13.py` (7B cell + floor guard + sentinel), `readouts.py`, `answer_v2.py`.

---

## 7. Next work

**Now (submission):** consistency pass → Appendix Y → LaTeX → abstract trim.

**During review:** **M14**, the second 7B cell (MATH·Qwen-7B), pre-registered: 78 students, T0 calibration 10/teacher
(floor 0.032), T1 references s0–s2, tests s10–s12, seq cap 4,608, same gates as M13, ~50 GPU-hours. Wording for every
outcome is fixed in advance, including **G1 fails / G2 passes** → "the remedy holds at 7B in both cells; the standard
test's collapse is present in one and weaker in the other" (the pattern's fourth instance, after read-out, dataset,
vendor — not a retreat).

**After submission (designs recorded, pre-registrations still to write):**
1. **D7 — content channel + cost-of-attack frontier (first).** A naive retrieval channel on published problems
   reopens the coverage failure (every competent student matches the owner's trace on a solved problem). Design that
   avoids it: paired, per-instance, per-relative — d_x = sim(y_x, t^a_x) − sim(y_x, t^b_x), with reference distributions
   from a's and b's reference students, i.e. **T1's structure carried over, coverage requirement inherited**. It is the
   per-instance keyed test M4 wanted, with the owner's real trace as the key. Pre-register: content weaker than style on
   unattacked adjacent stages (M11 EMB AUC 0.60–0.66); retrieval **holds** where style failed under imitation (style TPR
   0.58); attack axis = rewrites that change **step content**, expected to cost accuracy. If a cheap content rewrite
   defeats both channels at no cost → limits result.
2. **D3 — decompose the inherited signal** (scaffold / lexical / syntactic / semantic) from existing outputs.
3. **D1 — non-length style in capability-failed students**, only if its gates pass: length-binned matching (±10%) and a
   stripped-and-matched condition; "survives" = full read-out ≥ length-only baseline + 0.10 AUC **and** past a
   permutation null at the 95th percentile in ≥ 3 of 4 units; separate capability gate (accuracy ≤ base − 0.03).
   Truncation is **not** length matching.
4. **D10 — benchmark release** after a licence table (Tulu-3 and the Llama students are Meta community licence; Qwen
   adapters per model card).
5. **Waiting / out:** D5 (synthesis paper) until the ACL decision — same data, same framing, concurrent-submission risk.
   D8 (subliminal-channel separation) needs same-base teacher/student pairs; none in the bank.

## 8. Open questions

- D7: what similarity function and aggregation, and how many published problems per suspect?
- D1: is the length-only baseline strong enough that a +0.10 AUC margin is meaningful, or should the margin be set from
  its variance?
- M14: if it comes out inconclusive on the manipulation check (MATH students fell below base accuracy at 1.5B), is the
  length clause the right escape, or should the check be re-thought for MATH before the run?
