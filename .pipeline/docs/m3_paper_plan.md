# M3 paper plan — empirical study (revised 2026-09-14 after the second outside review)

_Decisions: `decision_log.md` (2026-09-14, both entries). Target ARR, with Findings as the realistic
objective and a workshop as fallback. Experiments are closed (v7 stopping rule). The only
post-matrix analysis is the exploratory teacher-identity read-out (v7b), which used existing outputs._

## Working title
**Inherited, Not Identified: What Prompt-Implanted Reasoning Signatures Reveal About Distillation Sources**

## Central claim (scoped)
Students distilled from the traces of a teacher prompted with a reasoning instruction inherit a
detectable signature of that instruction. **The tested key-based attribution procedure is not
source-specific:** it accepts students trained on an independent teacher's traces under the same
instruction, even though those students' outputs still carry recoverable teacher information. The
evidence weakens further under rewriting and dilution, and implanting the signature can cost student
accuracy, depending on the student.

**Not claimed:** that teacher provenance is generally unidentifiable, or that behavioural fingerprints
cannot establish provenance.

## Structure: three questions

### Q1 — What is inherited?  (§4)
- **Transfer is detectable**, reported as separate results, not pooled:
  - v2: 16 keys × 2 student families;
  - v3 Tulu/GSM8K (12/16) and Qwen teacher (16/16);
  - the 150-trace ablation, shown only for its setting (Qwen student, key_01).
- ARC is reported as a **VOID pre-registered outcome** plus a per-teacher sensitivity analysis.
- The filter attack does not remove the signature (7/8).
- **Key collisions:** in the tested template/persona/instruction generator, distinguishability is
  dominated by the 12 reasoning instructions (misattribution to same-instruction decoys at 6.5–8.3×
  base rate, measured on teacher traces). No claim about prompt-based key spaces in general.

### Q2 — What does detection identify?  (§3.4 analysis + §5, the central section)
- **Proposition 1** (key null vs source null) stated next to the test definition.
- **Main figure: source × instruction controls** (instruction-only keys o12 and p07; 2 families ×
  2 seeds):

  | Student training source | Instruction | Result | Establishes |
  |---|---|---|---|
  | Owner's teacher (Tulu) | owner's | 8/8 | intended detection |
  | Independent teacher (Qwen) | same | 8/8 | source specificity *conditional on a shared instruction*: absent |
  | Independent teacher (Qwen) | different | 0/8 evaluated | specificity to the instruction |
  | Clean training / other keys | none / different | hot-key share 0.031–0.047 (v3); other-key students flagged 0–0.9% (stage 0) | baseline false alarms |

- **Access model, stated explicitly:** only **known-instruction replication** is evaluated (the
  alternative teacher was given the instruction). The alternative-source pipeline never used
  owner-generated answers: different teacher, different problems. **Accidental instruction
  collision** and **instruction reconstruction** are not evaluated and are not claimed. The
  generator-bank result (~40% for persona keys under another teacher, stage 0) is likewise
  conditional on the shared key.
- **Proposition 2** + **teacher-identity read-out** (exploratory): teacher identity is recoverable
  from student outputs (AUC 1.00, held-out instructions, both families, both codebooks). The failure
  belongs to the key-based procedure, not to the absence of source information. Closed-set (two
  teachers) only.
- *Wording for unknown models:* "outputs exhibit a signature associated with instruction k"; the
  controlled experiments establish association with that training intervention, and other routes to
  the same behaviour remain possible.

### Q3 — When does the evidence weaken?  (§6)
- **Neutral paraphrase** (v3): attribution survives for 8/16 keys; an owner-aware read-out is worse.
  Stage 0: the coarse instruction signal survives but 64-key attribution does not; lexical dependence
  is supported by converging evidence (n-gram ablation, which induces distribution shift, and the
  embedding read-out's failure).
- **Deliberate standardisation** (T2, compress), kept separate from ordinary curation: removes most
  signatures, and the resulting students are weak. That does not show removal must cost utility.
- **Claim-preserving rewording** (T1) and the OP/PRES association: Qwen 8/8 vs 2/8; Llama 6/6 vs 2/6.
  - Caveat: T1 explicitly preserves claims, which are what define OP, and imposes no analogous
    requirement for formatting.
  - v5 matched replication infeasible (separability and length entangled with category).
  - Secondary; an observed association in the tested banks.
- **Dilution:** Qwen 50% 4/4, 25% 2/4, 10% 0/4 (the 10% × 3-epoch cell detected 1/4, preserved).
  Llama 50% 6/6, 10% 0/6.
  - *Wording:* "not detected under the stated training recipe and query budget"; detection became
    more reliable at higher mixture fractions in the tested configurations.
  - Proposition 3 explains why 1,319 queries did not help. Example and character shares reported.
- **Utility:**
  - Qwen keyed − clean: −12.3 points [−17.8, −6.6] on the v7 subset (−9.7 over all 16 stage-1 keys).
  - Llama: CIs include 0, but those students barely improve over base, so the result is "no loss
    detected", not "no loss".
  - Association with teacher accuracy under the instruction: ρ = 0.50 over 16 instructions (p = 0.05).
    No decomposition claimed.
- **Stealth (bounded):** screens caught 17–25% at 5% FPR; not evidence against adaptive distillers.

## Evidence status labels (used in every table caption)
- **Confirmatory (pre-registered gate):** v3 G-R0, G-R1, G-R2; stage 1 H-OP, S1-B; v6 predictions.
- **Estimation (pre-registered, no gate):** v7 E1–E4.
- **Exploratory (specified before running, not gated):** stage 0 D1–D4; query scaling; v7b
  teacher-identity.
- **Post hoc / corrected:** ARC per-teacher sensitivity; the utility inspection; every item in the
  integrity log.
- Later pre-registration does not make earlier retrospective explanations confirmatory.

## Wording table (applies to abstract and intro)
| Avoid | Use |
|---|---|
| across teachers, families and tasks, from ~150 traces | separate results per setting; ARC as sensitivity only |
| the key space is smaller than it looks | the tested generator's distinguishability is dominated by its 12 instructions |
| a different instruction is never flagged | 0/8 evaluated cases |
| absent at 10% | not detected under the stated recipe and query budget (3-epoch exception noted) |
| ordinary data handling erodes it | neutral paraphrase vs deliberate standardisation, reported separately |
| cost follows teacher accuracy | associated with teacher accuracy (ρ = 0.50, 16 instructions) |
| not detected on Llama → no cost | no loss detected on Llama |
| every experiment was pre-registered | confirmatory / estimation / exploratory / post hoc labels |
| provenance is unidentifiable | the key-based procedure is not source-specific; teacher information remains recoverable |

## Figures and tables
- **F1 (main):** source × instruction control grid, with the teacher-identity AUC panel.
- **F2:** key-confusion by instruction / persona / template.
- **F3:** detection vs mixture fraction (example and character share), both families, 3-epoch cell marked.
- **T1:** transfer results per setting. **T2:** rewriting (neutral paraphrase, T1, T2) by category and
  family. **T3:** utility (implantation vs removal cost, CIs).
- **Appendix:** instruction banks and compliance; evidence-status table; integrity log; ARC void
  analysis; full-FT cell (excluded); checklist read-out (excluded).

## Still open
- Verify and position the new prior work: Gu et al. 2312.04469 (watermark learnability and
  spoofing), DITTO 2510.10987, Unified Attacks 2504.17480. The distinction to state: an independent
  teacher reproduces the instruction-associated signal without the owner's traces, given the
  instruction.
- `paper/make_tables.py`: every number in the text generated from result files.
