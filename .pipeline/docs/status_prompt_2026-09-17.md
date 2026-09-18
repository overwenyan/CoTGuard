# Research prompt — distillation provenance, experiments closed, paper in drafting (2026-09-17, v4)

_Supersedes v3 (`status_prompt_2026-09-16c.md`). Self-contained; paste into an advisor, reviewer or deep-research
session. Sources of record: `.pipeline/memory/experiment_ledger.md` (EXP-M3 … EXP-M12), `.pipeline/memory/decision_log.md`,
pre-registrations `.pipeline/docs/m3…m12_design.md`, verified citations `.pipeline/docs/citations_verified.md`.
Every round was pre-registered in git before its data existed; failed gates, voided analyses, corrections, three
scorer bugs and one extractor bug are all logged. **No experiment is running and none is queued.**_

---

## 1. Context

Small academic group, school-scale shared compute (~7 H200 + one 8×L40S node). Target **ACL 2027 main**.

**Question.** Can the owner of a teacher LLM tell, black-box, that a student was distilled from its reasoning
traces — and name *which checkpoint* of its own release line?

**Setup.** Teachers are public post-training ladders: Tulu-3-8B (SFT→DPO→RLVR), OLMo-3-7B-Instruct (SFT→DPO→final),
OLMo-3-7B-Think (voided, below), Zephyr-7B (Mistral base, SFT→DPO). Students are LoRA (r = 32, α = 64, 3 epochs)
fine-tunes of Qwen2.5-1.5B-Instruct and Llama-3.2-1B-Instruct on 1,500-trace GSM8K or MATH corpora; each answers the
same 300 held-out probes. Read-outs: TF-IDF 1–2 gram + LR (pre-registered primary since M5), gte-base sentence
embeddings, POS 2–4-gram templates. About 700 students in §5–§6.

**Tests.**
- **T0** (standard owner test): multi-class read-out; conformal p = (1 + #{cal ≥ s})/(1 + n), α = 0.05; calibration =
  reference students of teachers **outside** the owner's post-training line.
- **T1** (reference-aware): T0 **and**, for each same-line relative *b*, a one-sided t prediction interval against
  *b*'s own 10 reference students under a pairwise owner-vs-*b* read-out.

---

## 2. What has been run (M3 → M12; all closed)

| Round | What | Outcome |
|---|---|---|
| M3 | Prompt-implanted secret instruction ("key") + rank test | Signature transfers but identifies **the instruction, not the owner** (independent teacher with the same instruction flagged 8/8). |
| M4 | Secret learnable reasoning-move watermark | **Killed by its specificity gate**: an imitator with a different secret split scored above the owner's own student. |
| M5 | Open-set teacher attribution, 7 teachers, 70 students | Closed-set 35/35; same-lineage confused 100%; 10% mixtures unreliable. |
| M6 | Alignment ladder, 9 checkpoints, 90 students | Distinguishability is **step-specific** (SFT→DPO AUC 0.93–0.99, RL-on-DPO 0.72–0.73). **T0 flags same-line relatives at FPR 1.0.** 1–1.5B students cannot absorb 4–7k-char Think traces → Think line voided. |
| M7 | Reference-aware test, 420 fresh students, GSM8K + MATH, disjoint problems *and* traces | **T1 cuts relative FPR from 1.0 to ≤ 0.2 at TPR 1.00** in all 4 cells. 3 references and 25 queries suffice. |
| M8 | Robustness: unknown relative, unseen teacher, mixtures | Protection is **per-relative**; **coverage gap** shown directly (unreferenced *distant* relative rejected 0.0, *adjacent* one flagged 1.0); cross-line calibration cannot be dropped; 50% mixtures detected, 10% not — the test reports the **majority contributor**. |
| M9 | Adaptive distiller: paraphrase vs imitate-a-relative | T1 **survives paraphrase** (TPR 0.92/0.83); **imitation evades and spoofs** (TPR 0.58, spoof 0.5). |
| M10 | Cross-vendor ladder (Zephyr), 80 students | **All predictions pass**; closes "AllenAI-specific" for the SFT→DPO step. |
| M9b | Imitation on all 4 adjacent pairs × both directions, answer-preserving | **Attack replicates** (TPR 0.67, spoof 0.61/0.50; 6 of 8 valid). **No direction rule, no mechanism.** |
| M11 | Non-lexical read-outs (EMB, POS) | Diagnosis is **read-out-independent** (T0 collapses, T1 fixes it) but **how badly T0 fails is read-out-dependent**: 8 collapsed pairs per MATH cell under TF-IDF vs 2–3 under embeddings. |
| **M12-A** | Geometry rule on held-out Zephyr (no training) | **G-A failed.** Threshold form **contradicted** (4 of 12 collapsed pairs at ratio 1.04–1.15, rule needed ≥ 10/12 below 1); ordering form **untestable there** (every pair collapses under every read-out). Pooled ρ over 18 units −0.50 (was −0.83 in-sample). |
| **M12-B** | Scaffold-only rewrite, 48 students, content verifiably preserved | **Mixed** (neither pre-registered prediction held in both families). Evades on **different** pairs than imitation: imitation 2 of 8 attacks (both SFT owners), scaffold-only 3 of 8 (post-SFT owners), **disjoint, union 5 of 8**. |

### The attack's four outcomes (Table 2 of the draft)
**evade** (owner loses the student) · **frame** (a sibling checkpoint claims it) · **joint claim** (both pass —
Craver's ambiguity attack) · **laundering** (signal gone, nobody implicated; our coinage, and the worst for an
auditor because it is silent). No accuracy cost: −0.03 to +0.04 on GSM8K.

---

## 3. The paper's claim, in one paragraph

A standard owner test cannot reject the vendor's own earlier or later checkpoints, **because its calibration set never
contains them** — a semantics failure, not a power failure, and it needs no adversary. Supplying reference students for
each relative fixes it uniformly (two vendors, two datasets, three read-outs) at TPR 1.00. The remedy is bounded: it
is per-relative, it needs cross-line calibration kept, and it does not survive a distiller that rewrites traces toward
a sibling. Reference-aware testing is therefore a **first-party lineage attestation** — closer to publishing a checksum
than to a guarantee against an adaptive adversary.

**Five conditions for the positive claim** (each removed experimentally, each removal costed): tester is the **vendor**
of the lineage; references for **every** relative; **cross-line** calibration retained; distiller does **not imitate**;
owner supplied the **majority** of the data.

**Theory (§3, no novelty claimed).** *Corollary 1*: TPR_a − TV(P_a, P_b) ≤ FPR_b ≤ α + TV(P_C, P_b), where P_a, P_b, P_C
are the laws of the test's own statistic for owner, relative and calibration population. T0's collapse is the **upper**
bound going vacuous (the relative is uncovered); T1 covers it by construction. *Corollary 1b*: replacing P_b by the
attacked students' law Q gives two statements, each under the claiming party's statistic, and the four outcomes above
are positions of Q — evade = far from P_a, frame = close to P_b, joint = both, laundering = neither. It says what the
outcomes **are**, not why a rewriter lands there.

---

## 4. Decisions in force

1. **Experiment budget is closed.** Capped at two pre-registered rounds after M11; both (M12-A, M12-B) are spent.
   **Held, not to be run without an explicit decision:** one 7B student cell, a classifier-aware attacker, a non-math task.
2. **Pre-register gates and kill criteria in git before data exists**; report negative results in the first sentence;
   log every correction.
3. **Never write "not replicated" for a contradicted rule.** M12-A's threshold form was contradicted by counterexamples;
   its ordering form was untestable. These are different failures and are stated separately.
4. **Scaffold-only is not promoted above imitation.** Same distiller, same rewriter, one different instruction — its own
   rows in Table 2, not a second attack family.
5. **No mechanism claims.** Both pre-registered predictors of *when* the attack succeeds came out with the wrong sign and
   non-significant (ρ = −0.72, p = 0.13; ρ = −0.63, p = 0.27; n = 6). Post-hoc stories (e.g. "SFT traces carry less
   scaffold") live in the ledger labelled as such, for future work only — this is the failure mode that already cost us
   once, when a mis-summarised direction asymmetry led the advisor to build a mechanism on it.
6. **Tables are generated by script** (`paper/make_tables_s56.py`) before and after each round; the draft is updated only
   from regenerated tables, never by hand. `paper/make_read_s3_s63.py` additionally **fails the build** if the theory and
   the attack stop using the same four outcome words.
7. **Citations:** only venues verified against a primary record may be cited with a venue (`citations_verified.md`).
8. **Compute:** SLURM account `ihc`; on the 8×L40S node (768 GB) request **≤ 96 GB per GPU**, scoring jobs sized to
   measured MaxRSS (32 G); `scontrol update` of memory is blocked — cancel and resubmit.

---

## 5. Corrections already published in the draft (integrity appendix)

- **Extractor bug (largest).** A regex fallback skipped numbers followed by a period, so "The answer is 8." read as an
  earlier "Step 3". Underestimated accuracy by 0.03–0.13 across M3–M9b. Found when a pre-registered answer-preservation
  check voided 7 of 8 rewritten corpora and the samples were inspected. **Retracts three secondary claims**
  (capability–identity dissociation; "RLVR less accurate than DPO on GSM8K"; "utility cost is student-dependent").
  **No attribution statistic consumes the extractor.**
- **M8 scorer inverted class order** → all-zeros run and a spurious kill verdict; caught because a weaker test cannot have
  lower power than a stronger one. A permanent sentinel now recomputes M7's T1 TPR each run.
- **M8 pre-registration flaw**: two tests are identical in a 3-stage line. **M9b key collision** with mixture keys; first
  scoring voided. **M9 gates** included voided corpora; recomputed. **Tulu-3 checkpoint mislabel** (`Llama-3.1-Tulu-3-8B`
  is the RLVR final, not SFT).
- **Two of my own analysis errors, corrected in place**: a vacuous-TV inference in Corollary 1's evidence, and
  "evasion/framing" vs Table 2's "evade/frame" (now enforced by script).

---

## 6. Writing status

| Section | State |
|---|---|
| §2 related work | **Drafted** (`paper/draft_s2_related.md`). Hinge: owner-side trace rewriting (Ma et al., ACL 2026) vs our distiller-side rewrites — same operation, opposite sides, which is why the attack is cheap. |
| §3 setup + test + analysis | **Drafted** (`draft_s3_s5.md` + `draft_analysis.md`: Propositions 1–3, Corollaries 1 and 1b). Reading copy: `paper/generated/read_s3_s63.md`. |
| §4 instruction-level (M3, compressed) | **Not written.** |
| §5 ladders + read-out dependence | **Drafted** (`draft_s5_s6.md`). |
| §6 reference-aware test, boundary, attacks, scope | **Drafted** (same file). |
| §1 abstract + intro | **Last**, by decision. |
| Integrity appendix, Figure 2, budget figures | **Not written** (figures via script). |

**Verified citations** (17 classics + 4 recent, `citations_verified.md`): journals read back from Crossref by DOI,
proceedings from the publisher's page. Two caveats recorded: Tsybakov's section number inside Ch. 2 is unverified (cite
"Ch. 2"); Mansurov et al. is still a preprint. Two findings from the check: **Wadhwa et al. (Findings ACL 2025,
"Who Taught You That?")** is a direct ancestor — teacher identification from student outputs, PoS templates carrying the
signal — and **arXiv 2512.20908** collides at the title level but traces spans within one known teacher's output rather
than attributing a model to a checkpoint (**abstract read only; full read pending before submission**).

---

## 7. Next steps, in order

1. **§4** — compress M3 (instruction-family detection vs owner attribution) into one short section.
2. **Integrity appendix** — every correction in §5 above, with both extractor versions for each accuracy number.
3. **Figures** — Figure 2 (T0 vs T1) and the budget curves, by script from result JSON.
4. **§1 abstract and intro** — last, using the three carried sentences in `draft_s5_s6.md`.
5. **Before submission** — read arXiv 2512.20908 in full; decide whether any held experiment is worth reopening.

## 8. What would most help from a reader

- Whether the §5.3 diagnosis (calibration defines the null) is stated sharply enough to be the paper's contribution,
  given that closed-set teacher attribution is already published (Wadhwa et al., 2025).
- Whether §6.4's "first-party attestation, not a guarantee" framing survives a reviewer who wants a defence.
- Whether Corollary 1b earns its place in §3 or should be a remark in §6.3.
