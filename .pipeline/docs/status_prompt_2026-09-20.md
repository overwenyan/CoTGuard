# Research prompt — distillation provenance: drafted + number-checked; M14 pre-registered; D7/D1 designed (2026-09-20, v8)

_Supersedes v7 (`status_prompt_2026-09-19.md`). Self-contained; paste into an advisor, reviewer or planning session to
continue analysis, method design or experiments. Records of truth: `.pipeline/memory/experiment_ledger.md` (EXP-M3 …
EXP-M13), `.pipeline/memory/decision_log.md`, pre-registrations `.pipeline/docs/m3…m14_design.md`, follow-up designs
`.pipeline/docs/followup_designs.md`, citations `.pipeline/docs/citations_verified.md`.
**Nothing is running or queued.**_

---

## 1. Context

Small academic group; 8×H200 node (training) + 8×L40S node (vLLM probes, 768 GB RAM). Target **ACL 2027 main**.
Question: can the owner of a teacher LLM tell, black-box, that a student was distilled from its reasoning traces — and
name **which checkpoint** of its own line?

**Setup.** Teachers: Tulu-3-8B (SFT→DPO→RLVR), OLMo-3-7B-Instruct (SFT→DPO→final), OLMo-3-7B-Think (voided), Zephyr-7B
(SFT→DPO). Students: LoRA (r 32, α 64, 3 epochs, batch 4, seq cap 4,608) on 1,500-trace GSM8K or MATH L1–4 corpora;
Qwen2.5-1.5B-Instruct + Llama-3.2-1B-Instruct (~700 students), plus 78 Qwen2.5-7B-Instruct (one cell, trained at cap
1,024 — a disclosed deviation). 300 held-out probes each. Read-outs: TF-IDF 1–2 gram + LR (pre-registered primary),
gte-base embeddings, POS 2–4-gram templates.

**Tests.** **T0**: multi-class read-out, conformal p = (1 + #{cal ≥ s})/(1 + n_cal), α = 0.05, calibration = 10 reference
students per teacher **outside** the owner's line (≥ 30 scores → floor ≤ 0.032). **T1**: T0 **and**, per same-line
relative *b*, a one-sided t prediction interval against *b*'s reference students under a pairwise owner-vs-*b* read-out.

---

## 2. Experiments (M3–M13 closed; M14 pre-registered, not run)

| Round | What | Outcome |
|---|---|---|
| M3 | Prompt-implanted instruction ("key") + rank test | Names **the instruction, not the owner** (independent teacher, same instruction: 8/8 flagged; different instruction: 0/8). Teacher-identity read-out separates sources at AUC 1.00 from the same outputs. Paraphrase → 8/16. 10% dilution → 0/6. ARC: **void** as run. |
| M4 | Learnable reasoning-move watermark | Killed by its specificity gate. |
| M5 | Open-set teacher attribution | Closed-set 35/35; same-lineage confused. ("Tulu SFT" was the RL-final model — mislabel.) |
| M6 | Alignment ladder, 9 checkpoints | Step-specific separability (SFT→DPO 0.93–0.99; RL-on-DPO 0.72–0.76). T0 flags relatives at 1.0. |
| M7 | Reference-aware test, 420 students, GSM8K + MATH | T1: relative FPR 0.9–1.0 → ≤ 0.2 at TPR 1.00, 4/4 cells. Budgets flat (3/5/10 refs; 25–300 probes). |
| M8 | Robustness | Protection is per-relative (A1, pre-registered). **Exploratory, GSM8K:** pooled rejector — unreferenced *distant* relative 0.0, *adjacent* 1.0. Cross-line calibration can't be dropped; mixtures → majority contributor. |
| M9 / M9b | Adaptive distiller | Survives paraphrase; imitation evades + frames (6/8 valid). No direction rule; both mechanism predictors wrong-signed, n.s. |
| M10 | Cross-vendor (Zephyr) | All predictions pass. |
| M11 | POS + embedding read-outs | Diagnosis and fix read-out-independent; **severity** read-out-dependent (MATH: TF-IDF 8/12 vs EMB 2–3/12). |
| M12-A | Geometry rule, held-out Zephyr | Threshold form **contradicted** (4 collapsed pairs at ratio 1.04–1.15); ordering untestable there. |
| M12-B | Scaffold-only rewrite | Mixed; evades on pairs **disjoint** from imitation's (2/8 vs 3/8; union 5/8). |
| M13 | **7B cell** (AllenAI·GSM8K·Qwen) | First run **void by construction** (9 calibration scores → floor 0.10 > α). Corrected (M7's 10/teacher calibration), gates unchanged: **G1 7/12** (need 6), **G2 11/12**, mean FPR 0.03, **TPR 0.94**. |
| **M14** | **MATH·Qwen-7B**, second 7B cell | **Pre-registered (`m14_design.md` + amendment 1); runs during review.** No code or data yet. |

**Attack outcomes (Table 2):** evade · frame · joint claim · laundering. Accuracy vs the owner's *unattacked* students:
imitation −0.028…+0.030; scaffold-only −0.027…+0.023.

---

## 3. Claims as drafted

- **Diagnosis:** an owner test controls the null its calibration population defines; the standard population omits the
  vendor's own siblings. Semantics, not power (Prop. 3: more probes don't help). Pattern: *same outputs, different null,
  different answer* — §4 and §5.3.
- **Independent corroboration (Rawat et al. 2026, arXiv 2607.09692):** calibrating their threshold **without** one
  teacher's students → false detections on that teacher (4/6, 1/6); with every teacher represented → none.
- **Remedy:** reference students per sibling → mean sibling FPR < 0.10 in every cell (2 vendors, 2 datasets, 3 read-outs,
  one 7B cell); 3 references *or* 25 probes suffice (varied separately).
- **Five conditions:** vendor is tester; references for every relative; cross-line calibration kept; distiller doesn't
  imitate; owner supplied the majority.
- **Attack:** distiller-side rewriting (mirror of Ma et al.'s owner-side rewriting) evades/frames/launders at no
  measurable accuracy cost. *What* it moves: scaffold phrases (descriptive). *When* it succeeds: unexplained.
- **Theory (§3):** Corollary 1 (TPR_a − TV(P_a,P_b) ≤ FPR_b ≤ α + TV(P_C,P_b)); 1b (any suspect law Q). Outcome names
  live in §6.3 beside Table 2, not in §3.
- **Framing:** first-party lineage attestation — like a checksum **except not tamper-evident**.

---

## 4. Rules in force

1. Pre-register in git before data; negatives first; log every correction.
2. **Floor rule:** compute 1/(1 + n_cal) < α before committing a rank/conformal gate (n_cal ≥ 19 at α = 0.05). An
   unattainable gate is **void**, never *failed*.
3. **Diff the actual job settings** before writing "reused unchanged from Mx" (M13 correction 2: 1,024 vs 4,608 cap;
   0.92% of GSM8K examples truncated; disclosed, not retrained).
4. Exploratory labels go in the sentence that makes the claim, bracketed by the pre-registered results.
5. Post-hoc numbers are not results (M13's void-run 8/12; ARC's per-arm pass).
6. Contradicted ≠ not replicated. No mechanism claims.
7. **Numbers come from scripts, and a checker enforces it:** `paper/check_numbers.py` re-derives 29 quoted values from
   result files and fails if the drafts disagree (29/29 pass). Generators: `make_tables_s56.py`, `make_appendix_x.py`,
   `make_fig2.py`, `make_read_s3_s63.py`. Every scorer carries a sentinel recomputing a known result.
8. Citations: only primary-verified venues. Liu et al. (2512.20908) = preprint; ADFP (2602.03812) = ICML 2026 poster.
9. Compute: account `ihc`; L40S ≤ 96 GB/GPU (check *available*); scoring 32 GB; `scontrol update` blocked → cancel +
   resubmit.

---

## 5. Paper status

Drafted in `paper/`: abstract + §1 (**200 words**), §2, §3 + Corollaries 1/1b, §4, §5–§6, Appendix X (14 entries + the
seq-cap disclosure). Generated: Figure 2, Tables 1, 2, X.1–X.3, `s56_numbers.json`, `fig2_values.json`,
`appx_extractor.json`. **Consistency pass done** (29/29), with four claims listed as MANUAL because no file in the repo
can settle them: §4's M3 numbers, §6.2's M8 numbers, §5.2's truncation figure, and the Rawat quotation.

**Remaining for submission:** (1) **Appendix Y** — instruction-level key banks (from `draft_s3_s5.md` §4.x) + the ARC
result with both readings, verdict void; (2) **ACL LaTeX assembly** and a PDF build.

---

## 6. Assets (no new training needed)

`experiments/radioactive/data_m7/{tulu_gsm,tulu_math}`, `data_m6`: ~780 LoRA adapters + 300 probe outputs each; teacher
traces (R300 / POOL_REF / POOL_TEST); attacked corpora (paraphrase, imitation, scaffold-only); mixture students; voided
Think-line students; the 7B cell. Code: `run_m7.py` (gen/probe, vLLM venv), `run_m3c.py sft` (LoRA; `M3C_MAXLEN`,
`M3C_GRADCKPT`), `score_m7.py` (`Cell`: T0/T1/rates), `score_m13.py` (floor guard + sentinel), `readouts.py`,
`answer_v2.py`, `paper/check_numbers.py`.

---

## 7. Next work

**Now:** Appendix Y → LaTeX → build → advisor read.

**During review — M14** (pre-registered, MATH·Qwen-7B): 78 students; T0 calibration 10/teacher (floor 0.032); T1
references s0–s2; tests s10–s12; **seq cap 4,608**; gates as M13; ~50 GPU-hours.
**Amendment 1 (before any code/data):** the manipulation check is now **absorption** — students must separate from the
untuned base under a teacher-vs-base read-out, **AUC ≥ 0.90** — because on MATH at 7B the base model plausibly outscores
every teacher, so a perfectly absorbed student would fail an accuracy clause and pass only on length. Accuracy and length
become reported diagnostics. Recorded cautions: the check is owner-vs-**base** (not circular with owner-vs-sibling), and
"absorption, not accuracy" is **not** the retracted capability–identity dissociation. Fixed wording for **G1 fails /
G2 passes**: "the remedy holds at 7B in both cells; the standard test's collapse is present in one and weaker in the
other" — and that sentence must say the two cells differ in **dataset and sequence cap**.

**After submission (designs settled; pre-registrations still to write):**
1. **D7 — content channel + cost-of-attack frontier (first).** Paired, per-instance, per-relative:
   d_x = sim(y_x, t^a_x) − sim(y_x, t^b_x), reference distributions from a's and b's reference students — T1's structure,
   so the coverage requirement is inherited, not reopened. **sim():** dense cosine (baseline) *and* numeric-step
   alignment (LCS over the ordered intermediate quantities; Jaccard variant) as the content channel, predicting only the
   latter separates siblings under imitation. **Aggregation:** per-suspect mean of d_x + one-sided t prediction interval
   (T1's shape); a sign test over d_x as the budget curve — where "power grows with queries" finally holds, unlike
   Prop. 3. **N:** sized from reference students for 90% power at the sign test; if it lands at ~1,000 that is a finding.
   **Gate 1 (before any attack):** does retrieval survive distillation at all — unattacked owner students vs b's
   references? Needs a generation pass on the **published** problems (our probes are held-out): new data, no training.
2. **D3 — decompose the inherited signal** (scaffold / lexical / syntactic / semantic) from existing outputs.
3. **D1 — non-length style in capability-failed students**, gated: matched (±10% length bins) and stripped-and-matched
   conditions; **validation gate** length-only AUC ≤ 0.55 (else the unit is void), baseline restricted to length-derived
   features so style cannot leak into the control; **survival** full read-out AUC ≥ 0.65 with a bootstrap 95% CI
   excluding 0.50, in ≥ 3 of 4 units; separate capability gate (accuracy ≤ base − 0.03). Truncation is not matching.
4. **D10 — benchmark release** after a licence table (Tulu-3 and the Llama students: Meta community licence; Qwen
   adapters per model card).
5. **Waiting / out:** D5 (synthesis paper) until the ACL decision — same data and framing, concurrent-submission risk.
   D8 needs same-base teacher/student pairs; none exist in the bank.

## 8. Open questions for the next session

- D7: how many published problems per suspect once N is sized, and does the sign-test budget curve replace or accompany
  the t interval in the headline?
- D1: is the ≥ 0.65 survival threshold right, or should it be set from the length-only baseline's bootstrap spread?
- M14: if absorption (AUC ≥ 0.90) voids a teacher, does the cell still have ≥ 2 teachers per line — and if not, is a
  partial cell worth reporting?
