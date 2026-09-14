# M3 paper plan — empirical study (2026-09-14)

_Decision: `decision_log.md` 2026-09-14 (expert). Target ARR, with Findings as the realistic outcome and
a workshop as fallback. This plan fixes the argument, the allowed wording and the evidence for every
claim before drafting. The v7 replication (m3_design.md v7) fills the cells marked **[v7]**._

## Working title
**Inherited, Not Identified: Prompt-Implanted Reasoning Signatures Under Distillation**
(alternative: *When Distilled Students Inherit a Teacher's Reasoning Habits but Not Its Identity*)

## One-paragraph thesis
A teacher prompted with a secret reasoning instruction leaves a habit that students distilled from
its unmodified traces reliably inherit, and a black-box read-out can detect it. But inheritance is not
attribution:
- nominally distinct keys collapse onto their reasoning instruction;
- an independent teacher given the same instruction produces a signature the owner's test also
  flags;
- rewriting the traces and mixing them into larger corpora erode identification;
- the instructions that leave the strongest signatures can cost student accuracy.

We characterise these limits with pre-registered experiments, and report the failures alongside the
successes.

## Story order (sections 4–8 follow it)
1. Transfer is detectable → 2. nominal keys collide → 3. independent sources imitate the signature →
4. rewriting and dilution limit identification → 5. stronger signatures may cost utility.
Secondary: content-adding vs format-only instructions (an observed association in the tested banks).

## Outline
1. **Introduction.** Motivation: distillation from published traces; why a prompt-only mark is
   attractive (no decoding access, no output editing). Contributions, stated as findings, not a
   method. Summary of limits.
2. **Related work.** Behavioural and distillation watermarks: Asking Back (a small set of related
   behavioural markers, without many-key attribution; response rewriting before training not
   evaluated), Trace Rewriting (active; its attack set is not comparable to ours), ReasMark, PROSE
   (fine-tuned semantic structures), In-Context Watermarks, Radioactivity. Also subliminal learning
   (why ours is *not* subliminal), dataset inference, and authorship attribution.
3. **Setup and test.**
   - Threat model; keys; teachers and students; read-out trained on teacher traces only.
   - The rank test and its validity (random key assignment; averages over the key draw, not a per-key
     guarantee); vetoes; hot-key diagnostic.
   - Endpoints: **instruction-family detection** vs **owner-specific attribution**.
4. **Transfer is detectable.** v2/v3 raw results across 2 teachers × 2 student families; query budget
   (exploratory); filter attack; ARC reported as a VOID pre-registered outcome plus a per-teacher
   sensitivity analysis.
5. **Nominal keys collide.** Stage 0 D1: misattribution to same-instruction decoys at 6.5–8.3× base
   rate, persona ≈ nothing; owner key rank vs same-instruction siblings.
   *Wording:* "the tested generator's distinguishability is dominated by its 12 reasoning
   instructions". No capacity claim. The v4/v5 banks are separate codebooks. Note that collapsing to
   12 classes would change the test (minimum p = 1/12).
6. **Independent sources imitate the signature.**
   - Stage 0 D4: same key, other teacher, flagged 6/16 and 7/16. Other-key students 0–0.9%.
   - **[v7 E1]** own / imit / neg on two families × 2 seeds.
   - *Wording:* "outputs exhibit a signature associated with instruction k, consistent with transfer
     from instruction-conditioned traces". Report owner-specific false attribution and
     instruction-family detection as separate endpoints.
7. **Rewriting and dilution limit identification.**
   - *Rewriting:* v3 neutral paraphrase 8/16; owner-aware read-out worse. Stage 0 D2: coarse 9-way
     signal survives, 64-key attribution does not; embedding read-out no better; lexical dependence
     from converging evidence (n-gram ablation + embedding failure), with the ablation's distribution
     shift acknowledged. Stage 1: T1 OP 8/8 vs PRES 2/8, T2 kills both. **[v7 E2]** Llama × 2 seeds.
   - *Dilution:* S1-B 0/4 at 1/5/10%; query scaling 0/4; v6 curve 50% 4/4, 25% 2/4, 10% × 3 epochs
     1/4. **[v7 E3]** Llama × 2 seeds.
     *Wording:* "detection became substantially more reliable at higher mixture fractions in the
     tested four-key configurations"; report example, character and (if available) token shares;
     training exposure matters.
   - *Removal cost:* T2/compress students fall to or below base, but a weak student from an attack
     does not show that removal must cost utility (no competent utility-preserving remover was
     tested). Clean-rewrite controls: stage 0 D3, stage 1.
8. **Stronger signatures may cost utility.** Stage 1: keyed 0.563 vs clean 0.660. Inspection
   (EXP-M3U): not extraction, truncation or exposure; tracks teacher accuracy under the instruction
   (ρ = 0.50); residual ~6.6 points among teacher-accurate keys, with a single-seed clean baseline.
   **[v7 E4]** seeds and bootstrap intervals. v3 persona keys showed ~no cost.
9. **Secondary: content-adding vs format-only.** Stage 1 H-OP (with confounds); v5 matching
   infeasible (PRES separability median 0.52 vs 0.99; shorter traces).
   *Wording:* an observed association in the tested banks. Distinctiveness may be part of the
   mechanism, not only a nuisance. Not a prospective design rule.
10. **Stealth (bounded).** Screens caught 17–25% at 5% FPR. Not evidence against an adaptive
    distiller; the removal effect could not be assessed because 10% dilution was undetectable to
    begin with.
11. **Limitations and ethics.** Scale (1–1.5B LoRA students, GSM8K-scale corpora); lexical read-out;
    four-key curves; the undertrained full-FT cell (excluded from conclusions); the checklist read-out
    (not validated, excluded unless human labels are added); dual use.
12. **Reproducibility statement.** Pre-registration commits per round, the integrity log (the
    corrections listed in m3_status_2026-09-14 §5), code, data-generation seeds.

## Figures and tables
- **T1:** main raw-transfer results (v2/v3 settings, families).
- **F1:** stage-0 confusion collapsed by instruction / persona / template (misattribution ratios).
- **T2:** source ambiguity, own / imit / neg × family × seed, with exact intervals **[v7]**.
- **F2:** rewriting — pass rates by corpus (raw/T1/T2, v3 paraphrase) and category, Qwen and Llama **[v7]**.
- **F3:** dilution — detection vs example share and character share, per key, with training regime
  marked **[v7]**.
- **T3:** utility — implantation vs removal cost with bootstrap intervals **[v7]**.
- **Appendix:** instruction banks v4/v5 with compliance rates; the pre-registration and deviation log;
  the full-FT cell; the ARC void analysis; the query-scaling curve; screens.

## Claims NOT to make
- Any "watermark that works", "ownership proof", or universal dilution threshold.
- "12 identities" as a capacity limit.
- Content-adding as a causal design rule.
- Stealth against adaptive distillers.
- That removal necessarily costs utility.
- That the limits hold for all behavioural watermarking methods.
- Direct numeric comparison with Trace Rewriting / PROSE headline rates.

## Next writing steps
1. Draft §3 (setup, test, endpoints) and §5–6 from finished evidence while v7 runs.
2. Fill the [v7] cells when the matrix completes (stopping rule: matrix completion).
3. Then `/omp:write` section passes and an internal review.
