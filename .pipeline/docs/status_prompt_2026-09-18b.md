# Research prompt — distillation provenance: paper drafted, next analyses / methods / experiments (2026-09-18, v6)

_Supersedes v5 (`status_prompt_2026-09-18.md`). Self-contained; paste into an advisor, reviewer or planning session to
continue analysis, method design or experiments. Sources of record: `.pipeline/memory/experiment_ledger.md` (EXP-M3 …
EXP-M13), `.pipeline/memory/decision_log.md`, pre-registrations `.pipeline/docs/m3…m13_design.md`, verified citations
`.pipeline/docs/citations_verified.md`. **No experiment is running or queued.**_

---

## 1. Context

Small academic group; shared compute: one 8×H200 node (training), one 8×L40S node (vLLM generation/probes, 768 GB RAM).
Target **ACL 2027 main**. Question: can the owner of a teacher LLM tell, black-box, that a student was distilled from its
reasoning traces — and name *which checkpoint* of its own release line?

**Setup.** Teachers are public post-training ladders — Tulu-3-8B (SFT→DPO→RLVR), OLMo-3-7B-Instruct (SFT→DPO→final),
OLMo-3-7B-Think (voided: small students can't absorb 4–7k-char traces), Zephyr-7B (Mistral base, SFT→DPO). Students:
LoRA (r = 32, α = 64, 3 epochs, batch 4, seq ≤ 1024) on 1,500-trace GSM8K or MATH (L1–4) corpora; Qwen2.5-1.5B-Instruct
and Llama-3.2-1B-Instruct (~700 students), plus 78 Qwen2.5-7B-Instruct students (one cell). Every student answers the
same 300 held-out probes. Read-outs: TF-IDF 1–2 gram + LR (C = 4, pre-registered primary since M5), gte-base embeddings,
POS 2–4-gram templates.

**Tests.** **T0** (standard owner test): multi-class read-out; conformal p = (1 + #{cal ≥ s})/(1 + n_cal), α = 0.05;
calibration = reference students (10 per teacher) of teachers **outside** the owner's line. **T1** (reference-aware):
T0 **and**, per same-line relative *b*, a one-sided t prediction interval against *b*'s reference students under a
pairwise owner-vs-*b* read-out.

---

## 2. Experiments done (all closed)

| Round | What | Outcome |
|---|---|---|
| M3 | Prompt-implanted secret instruction ("key") + rank test | Transfers, but names **the instruction, not the owner** (independent teacher with same instruction flagged 8/8); teacher-identity read-out separates sources at AUC 1.00 **from the same outputs**. Paraphrase → 8/16; 10% dilution → 0/6. ARC result: **void** as run (post-hoc per-arm rule would pass it; not accepted). |
| M4 | Secret learnable reasoning-move watermark | Killed by its specificity gate. |
| M5 | Open-set teacher attribution, 7 teachers | Closed-set 35/35; same-lineage confused. (Its "Tulu SFT" teacher was actually the RL-final model — mislabel.) |
| M6 | Alignment ladder, 9 checkpoints | Distinguishability step-specific (SFT→DPO AUC 0.93–0.99; RL-on-DPO 0.72–0.76). T0 flags relatives at 1.0. |
| M7 | Reference-aware test, 420 students, GSM8K + MATH | T1 cuts relative FPR 0.9–1.0 → ≤ 0.2 at TPR 1.00. Budgets flat: 3/5/10 references and 25–300 probes all give TPR 0.98–1.00, FPR 0.008–0.092 (Table X.3). |
| M8 | Robustness | Per-relative protection (pre-registered A1: withholding one relative's references restores its collapse in 6–8/8 pairs). **Exploratory, GSM8K only** pooled rejector: unreferenced distant relative 0.0, adjacent 1.0 — the single-variable coverage contrast. Cross-line calibration can't be dropped; mixtures → majority contributor. |
| M9 / M9b | Adaptive distiller | Survives paraphrase; imitating a sibling evades and frames (6/8 valid attacks). No direction rule; both pre-registered mechanism predictors wrong-signed, n.s. |
| M10 | Cross-vendor (Zephyr) | All pass. |
| M11 | Non-lexical read-outs | Diagnosis and fix read-out-independent; **severity** read-out-dependent (MATH: TF-IDF 8/12 collapsed, EMB 2–3/12). |
| M12-A | Geometry rule on Zephyr | Threshold form **contradicted** (4 collapsed pairs at ratio 1.04–1.15); ordering untestable. |
| M12-B | Scaffold-only rewrite | Mixed; evades on pairs **disjoint** from imitation's (2/8 vs 3/8, union 5/8). |
| M13 | **7B student**, AllenAI·GSM8K·Qwen cell | First run **void by construction** (9 calibration scores → floor p = 0.10 > α). After restoring M7's 10-per-teacher calibration (floor 0.032), gates unchanged: **G1 7/12** (need 6; 1.5B: 8), **G2 11/12**, mean FPR 0.03, TPR 0.94. |

**Attack outcomes (Table 2):** evade · frame · joint claim · laundering (silent). Accuracy vs owner's *unattacked*
students: imitation −0.028…+0.030, scaffold-only −0.027…+0.023.

---

## 3. Current claims (as drafted)

- **Diagnosis:** owner tests control a null defined by their calibration population; the standard population omits the
  vendor's own siblings. Semantics failure, not power; more probes don't fix it (Prop. 3). Pattern: *same outputs,
  different null, different answer* — instruction level (§4) and checkpoint level (§5.3).
- **Independent corroboration:** Rawat et al. (2026, arXiv 2607.09692) — calibrating their threshold without one
  teacher's students gives false detections on that teacher (4/6, 1/6); with every teacher represented, none. Their
  statistic needs student + reference log-probs; candidates cross-vendor; no siblings, no rewriting.
- **Remedy:** reference students per sibling; mean sibling FPR < 0.10 in every cell (2 vendors, 2 datasets, 3 read-outs,
  one 7B cell). Failure severity varies (read-out, dataset, student size); the fix does not — seen three times.
- **Bounds (five conditions):** vendor is the tester; references for every relative; cross-line calibration kept;
  distiller doesn't imitate; owner supplied the majority.
- **Attack:** distiller-side rewriting (mirror of Ma et al.'s owner-side rewriting) evades/frames/launders at no
  measurable accuracy cost. What it moves is scaffold phrases (descriptive); when it succeeds is unexplained.
- **Theory (§3, no novelty claimed):** Corollary 1: TPR_a − TV(P_a, P_b) ≤ FPR_b ≤ α + TV(P_C, P_b); 1b: holds for any
  suspect law Q. Outcome names live in §6.3, not §3.
- **Framing:** first-party lineage attestation — like a checksum *except not tamper-evident*.

---

## 4. Decisions and rules in force

1. **Pre-register in git before data**; negative results first; every correction logged.
2. **Floor rule:** for any rank/conformal gate, compute 1/(1 + n_cal) and require it < α before committing
   (n_cal ≥ 19 for α = 0.05). An unattainable gate is *void*, never *failed*. (Learned from M13.)
3. **Exploratory labels go in the sentence that makes the claim**, bracketed by the pre-registered results.
4. **Post-hoc numbers are not results** (M13 void-run 8/12; ARC per-arm pass).
5. **Contradicted ≠ not replicated.** No mechanism claims; post-hoc stories stay in the ledger for future work.
6. **Numbers come from scripts only** (`make_tables_s56.py`, `make_appendix_x.py`, `make_fig2.py`,
   `make_read_s3_s63.py`); every scorer carries a sentinel that recomputes a known result.
7. **Citations:** only primary-verified venues. Liu et al. (2512.20908) stays a preprint (OpenReview blocked; ICLR 2026
   only per a secondary index). ADFP (2602.03812) = ICML 2026 poster (confirmed on icml.cc).
8. **Budget:** experiments for the ACL paper are closed. Classifier-aware attacker and non-math task held; limitation
   sentences written instead.
9. **Compute:** account `ihc`; L40S ≤ 96 GB RAM per GPU (check *available*, not *free*); scoring jobs 32 GB; `scontrol
   update` blocked → cancel + resubmit.

---

## 5. Paper status

Drafted (markdown, `paper/`): abstract + §1 (`draft_s1_abstract_intro.md`, 206 words — trim to ≤ 200), §2
(`draft_s2_related.md`), §3 + Corollaries (`draft_s3_s5.md`, `draft_analysis.md`), §4 (`draft_s4.md`), §5–§6
(`draft_s5_s6.md`), Appendix X integrity record (`draft_appx_x.md`, 14 entries). Figure 2 (`generated/fig2.pdf`), Tables
1, 2, X.1–X.3 generated. **Remaining:** full consistency pass (every number → generated file or ledger); Appendix Y
(key banks, ARC both readings); ACL LaTeX assembly; abstract trim.

---

## 6. Assets available for further work (no new training needed)

- **Student bank** (`experiments/radioactive/data_m7/tulu_gsm`, `tulu_math`, `data_m6`): ~780 LoRA adapters + 300 probe
  outputs each; teachers' traces (R300, POOL_REF, POOL_TEST); attacked corpora (paraphrase, imitation, scaffold-only);
  mixture students; voided Think-line students (M6); 7B cell.
- **Code:** `run_m7.py` (gen/probe via vLLM 0.21 venv), `run_m3c.py sft` (LoRA training; `M3C_GRADCKPT` for 7B),
  `score_m7.py` (Cell class: T0/T1/rates), `readouts.py` (EMB/POS with caches), `answer_v2.py` (extractor, unit-tested).

---

## 7. What comes next

**For the ACL paper (now):** consistency pass → Appendix Y → LaTeX → trim abstract → send to advisor.

**After submission (candidate follow-ups, each needs its own pre-registration):**
1. **D7 — joint style + membership provenance with an attacker-cost frontier (first).** Add a content channel:
   query suspects on the *published* problems and retrieve their answers against the owner's traces (Krishna et al.,
   NeurIPS 2023 show retrieval survives paraphrase for text); combine with T1; plot attacker accuracy vs provenance
   survival across paraphrase, imitation and scaffold-only rewrites. Open questions to pre-register: does retrieval
   survive distillation (student outputs are generated, not copied)? Can it tell sibling checkpoints apart (they answer
   the same problems)? It abandons the held-out-probe design, so the null changes — state it. Risk: some cheap attack
   defeats both channels (then the paper is a limits result).
2. **D3 — decompose the inherited signal** (scaffold vs lexical vs syntactic vs semantic) across teachers, students and
   trace lengths, reusing existing outputs and the M11 read-outs plus the M12-B scaffold-only students. Low risk.
3. **D1 — "style before capability"** using voided Think-line students as positive controls — **only if a pre-registered
   length-matched check passes.** Truncation to 400 tokens drops the Think-line AUC to 0.75–0.78 (not chance), so the
   question is the length-matched residual; define "survives" and predict before looking. Student-size ordering is *not*
   in the bank (needs training). Beware: close to the retracted capability–identity dissociation.
4. **D10 — release the bank as a benchmark.** First a licence table: Tulu-3 (Llama-3.1 derivative) and Llama-3.2
   students fall under Meta's community licence (redistributable with naming/notice requirements); Qwen adapters per
   their model cards (expected Apache — verify).
5. **Waiting / out of scope:** D5 (a "calibration defines the null" synthesis paper) waits for the ACL decision
   (concurrent-submission risk: same data, same framing). D8 (subliminal-channel separation) needs same-base
   teacher/student pairs — none exist in the bank; new training. D2/D4/D9/D11/D12 dropped or folded in.

**Competitive landscape to watch:** Wadhwa et al. (Findings ACL 2025, teacher ID via POS templates); Rawat et al. 2026
(reference-normalised NLL); Liu et al. (sentence-level provenance, white-box); Ma et al. ACL 2026 (owner-side trace
rewriting); ReasMark ACL 2026; ADFP ICML 2026; Antidistillation Sampling. Re-run a literature check before any
follow-up submission; the D7 "open gap" is absence of evidence, not proof.

## 8. Questions for the next session

- For D7: what is the right null for a retrieval channel on published problems, and how is it calibrated without
  reopening the coverage failure the paper diagnoses?
- For D1: what length-matched read-out and what threshold count as "style survives" — fixed before looking?
- Is one 7B cell (thirds resolution, narrow G1 pass) enough for the scale sentence as written, or should a second cell be
  pre-registered for camera-ready?
