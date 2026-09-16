# Unified paper plan (hybrid A + C) — draft 2026-09-15, written while M7 runs

_Supersedes `m3_paper_plan.md` as the plan of record for the full paper; that file stays as the M3
section plan. Decision context: advisor response (hybrid A + C), user choice "ladder first", then
"confirm reference effect". Claims below are tagged with their evidence status; the M7 branches are
written before M7 data exists._

## Working title (candidates)
1. **"Identified by what? Two levels of the distillation-provenance gap"**
2. "What distillation-provenance signals identify — and what they do not"
3. (if H2 passes) "Provenance needs a reference: same-lineage attribution is a calibration problem,
   not an information limit"

## The claim, in one paragraph
Black-box provenance tests on distilled reasoning models answer a narrower question than their
framing suggests. At the **instruction level**, a key-based rank test identifies the reasoning
instruction, not the owner: an independent teacher given the same instruction is flagged as often as
the owner's own student (M3, 8/8 vs 8/8). At the **teacher level**, an open-set attribution test
identifies the post-training *line*, not the checkpoint: a standard owner test, calibrated on
students of unrelated vendors' models, flags every same-line relative (M6, FPR 1.0 at 12 of 12
ordered pairs in the no-void sensitivity, 10 of 12 as pre-registered). In both cases the information
needed for the finer decision is **present in the same student outputs** — teacher identity is
recoverable at AUC 1.00 (M3), and adjacent checkpoints separate perfectly at the student level once
the read-out is pairwise (M6 exploratory, M7 confirmatory). What fails is not detectability but the
**null the test controls**: the calibration set defines what "not the owner" means.

## Three questions and the sections that answer them
1. **What does a provenance test identify?** — M3 (instruction level), M5 and M6 (teacher level).
2. **Why does it identify that?** — Propositions 1–3 (key vs source null; total-variation
   requirement; query scaling cannot fix a wrong population ordering), plus the calibration analysis
   that makes M6's FPR = 1.0 predictable rather than surprising.
3. **What would close the gap, and what does that cost the owner?** — M7: reference-aware testing,
   with its reference-budget and query-budget curves; the residual limits (unseen teachers, strong
   instructions that erase style, low mixture fractions).

## Evidence map (status tags: **C** confirmatory pre-registered, **E** estimation pre-registered,
**X** exploratory, **P** post hoc)
| Result | Source | Status |
|---|---|---|
| Signature transfers to students | M3 v2/v3 | C |
| Key-based test is not source-specific (8/8 independent teacher) | M3 stage 1 | C |
| Teacher identity recoverable from the same outputs (AUC 1.00) | M3 v7b | X |
| Closed-set teacher attribution 35/35, open-set TPR 1.0 / FPR 0.07 | M5 | C |
| Strict unseen-teacher FPR 0.3–1.0 | M5 | C |
| Composite instruction ∧ teacher test fails (strong format instruction erases style) | M5 | C |
| Per-output distinguishability is step-specific, not distance-monotone | M6 E1 | E |
| Owner test flags all same-line relatives | M6 E2, P1 | C |
| Sibling lines (same base, different post-training) are trivially separable — format | M6 P3 | C |
| Capability gap transfers weakly and its sign is dataset-dependent | M5b, M6 E3, M7 | X / E |
| Pairwise student-level test separates adjacent stages | M6 explore | X |
| **Reference-aware test restores specificity on fresh students** | **M7 H2** | **C (pending)** |

## M7 branches (written before the data)
- **H2 passes (both GSM8K cells, ≥ 1 MATH cell).** The paper's second half becomes constructive: the
  same-lineage limit is a *reference-availability* limit. Title candidate 3. Position against
  RefDistDet (2607.09692), which also needs a same-lineage reference: our contribution is the
  measurement that says *why* the reference is needed (the calibration null, Proposition 1) and *how
  much* reference is needed (the n_ref ∈ {3, 5, 10} curve).
- **H2 is family-dependent or GSM8K-only.** Report as scoped: the restoration works where it works,
  and the paper keeps the two-level gap as the headline with a partial remedy.
- **H2 killed (both GSM8K cells fail).** The M6 exploratory separation was an artefact of shared
  traces between reference and evaluated students. This is the strongest version of the negative
  result: even with same-lineage reference students, adjacent checkpoints are not separable on fresh
  traces. Report the leakage diagnostic (M7 "leakage" block) as the explanation, and state the M6
  exploratory claim as refuted by our own confirmatory test.
- In every branch, the leakage diagnostic (reference-internal vs fresh evaluation) is reported. It is
  the methodological lesson for anyone building same-lineage detectors.

## Theory (§3, already drafted in `paper/draft_analysis.md`)
- **Prop 1** key null vs source null — now also the explanation of M6's FPR = 1.0 and of why T1 differs
  from T0. Restate the corollary explicitly: *an owner test's false-positive rate is only defined
  relative to its calibration population; relatives excluded from calibration cannot be rejected.*
- **Prop 2** total-variation requirement with the read-out's invariances — explains the composite-test
  failure (a strong format instruction collapses the TV distance the read-out can see).
- **Prop 3** query scaling cannot overturn a wrong population ordering — matched against the M7 query
  budget curve (25 / 50 / 100 / 300 probes), which shows the ordering is right under T1 and wrong
  under T0.

## Figures
- **F1 two-level gap.** Grid: rows = level (instruction, teacher), columns = "identified" /
  "not identified", cells carry the numbers.
- **F2 ladder curve (M6).** Per-output AUC by step type (SFT→DPO, DPO→RL, SFT→final, sibling), two
  student families, with cluster-bootstrap CIs. This is the figure that shows structure is
  step-specific, not distance-monotone.
- **F3 T0 vs T1 (M7).** Paired bars of FPR on relatives and TPR, per dataset and family.
- **F4 budgets (M7).** Reference budget (n_ref) and query budget curves.
- **F5 dilution / mixture (M3, M5)** as the existing `fig_dilution`.
- A radar chart was considered for the read-out comparison; only include one if it carries numbers a
  table cannot (currently it does not — drop it unless a reviewer asks).

## Reviewer-facing risks and the planned answer
- *"Closed-set attribution is known."* Agreed, and cited (Who Taught You That?, Findings ACL 2025).
  Our results are about the limits and the calibration semantics, not the positive result.
- *"Only 1–1.5B students, GSM8K and MATH."* Stated as a limitation, with the concrete failure we
  observed: 1–1.5B LoRA students cannot absorb 4–7k-character Think traces (M6 manipulation check),
  which is itself a finding about long-CoT distillation at small scale.
- *"Why not a defence?"* Three method attempts failed at specificity (M3 keys, M4 keyed reasoning
  moves, M5 composite test); each is reported with its pre-registered kill criterion. M7 is the
  constructive part.
- *"Ambiguity attacks are old news."* Cite Craver et al. 1998 (invertibility), Fan et al. 2019
  (passports), watermark stealing, DITTO, subliminal learning; state the difference: those attacks
  need an adversary, whereas here an *honest* relative checkpoint is indistinguishable by construction
  of the test's null.

## Writing order once M7 lands
1. §3 theory (drafted) + calibration corollary.
2. §5 ladder (M6) and §6 reference-aware test (M7) — the new material.
3. §4 instruction level (M3) compressed from the existing plan.
4. §2 related work with the ambiguity-attack tradition.
5. §1 abstract and intro last, using the wording table in `m3_paper_plan.md`.
