# M9 — adaptive distiller: does T1 survive paraphrase and relative-imitation? — pre-registration

_2026-09-16. Committed before any M9 data exists. Context: EXP-M7 (T1 works), EXP-M8 (protection is
per-relative; coverage gap), advisor round 4: "an adaptive attack is near-mandatory in this subfield;
its absence reads as a gap, not a scoping choice."_

## 1. Threat model
The distiller trains on the **owner's** traces but knows a provenance test may follow and deliberately
obscures the source. The owner does **not** anticipate the attack: its reference students are the
non-adaptive M7 ones (seeds 0–9), and its read-outs are trained on raw teacher traces. This asymmetry
is the point — an owner who already had attacked reference students would be testing a different null.

**Attacker access:** the owner's published traces, any public model for rewriting, and public
knowledge of the owner's lineage (so it can name a relative to imitate). It does **not** have the
owner's read-out, reference students, or probe set.

## 2. Conditions (GSM8K; rewriter = `Qwen/Qwen2.5-7B-Instruct`, the M3 attacker model)
For each attacked owner a, the 1,500-problem POOL_TEST corpus of a's traces is rewritten before
training:
- **AD0 — control:** no rewriting (the M7 test students, seeds 10–19, reused; no new training).
- **AD1 — neutral paraphrase:** "Rewrite this step-by-step solution in your own words. Keep every
  calculation, every number and the final answer exactly, but do not reuse the original phrasing or
  sentence structure." (verbatim from M3's `para` attack, where it removed about half of the
  prompt-implanted signatures).
- **AD2 — imitate a relative:** the same rewrite, but conditioned on **4 in-context example traces of
  the relative b** drawn from b's R300 traces, with the instruction to match that style while keeping
  every calculation and the final answer. This is the spoofing attack: owner content, relative style.

**Attacked owners and their relatives** (the two adjacent pairs M6/M7 found least separable, in both
directions): a = `tulu_dpo` (b = `tulu_rlvr`), a = `tulu_rlvr` (b = `tulu_dpo`),
a = `olmoi_dpo` (b = `olmoi_final`), a = `olmoi_final` (b = `olmoi_dpo`).

**Students:** 4 owners × 2 conditions × 3 seeds (30–32) × 2 families = **48 new LoRA students**,
trained and probed exactly as in M7 (r = 32, 3 epochs, 1,500 traces, 300 probes).

## 3. Endpoints (owner a's tests from M7, unchanged)
- **TPR_att(a, cond):** share of a's 6 attacked students (3 seeds × 1 family, per family reported
  separately) that T1 still flags. Evasion succeeds when this drops.
- **Spoof(a, cond):** share of those students that **relative b's** T1 flags. Spoofing succeeds when
  this rises — the attacker makes an innocent relative look like the source.
- Reported alongside: T0 rates, per-output AUC against a's read-out, and the same endpoints under the
  embedding and POS-template read-outs (M5's secondary read-outs), which is where a lexical paraphrase
  attack should behave differently.

## 4. Pre-registered predictions and decision rules
- **G1 (paraphrase robustness):** mean TPR_att under **AD1** ≥ 0.8 in both families.
- **G2 (imitation robustness):** mean TPR_att under **AD2** ≥ 0.8 in both families.
- **G3 (no successful spoofing):** mean Spoof under **AD2** ≤ 0.3 in both families.
- **Attack-success rule (advisor's threshold):** the attack is reported as successful in a condition if
  TPR_att < 0.8 **or** Spoof > 0.3.
- **Consequences, fixed in advance:**
  - **G1 and G2 and G3 all hold** → T1 is robust to non-adaptive-owner rewriting; state it as a
    strength, with the caveat that stronger attacks (fine-tuned style transfer, mixture-of-relatives)
    remain untested.
  - **G1 fails** → paraphrase defeats the test. This becomes the **primary limitation**, stated in the
    abstract, and the paper's scope is explicitly narrowed to non-adaptive distillers. Title must
    foreground that scope.
  - **G1 holds but G2 fails** → the test survives generic rewriting but not targeted imitation; report
    as the boundary of the vendor regime.
  - **G3 fails** → report as an *ambiguity attack* in the Craver et al. (1998) sense: the attacker can
    shift attribution to an innocent relative. This is a strong negative result and must be reported
    even though it weakens the constructive claim.
- **Manipulation check:** a rewritten corpus is valid only if (a) the rewritten trace's extracted final
  answer matches the original trace's in ≥ 90% of cases, and (b) mean rewritten length is within
  [0.5, 2.0] × the original. A corpus failing either is void and reported, not silently dropped.
  Student-level check as in M7.

## 5. Not claimed from M9
- Attacks that fine-tune a style-transfer model on the relative, mix several relatives, or optimise
  directly against a known read-out (the owner's read-out is secret here, but a determined attacker
  could approximate it).
- Attacks on the instruction level (M3 already reports paraphrase there).
- MATH or long-CoT.

## 6. Compute
Rewriting: 4 owners × 2 conditions × 1,500 traces with a 7B model on vLLM, about 20 GPU-minutes each.
48 LoRA students plus probes: about 6 GPU-hours. Smoke test on one owner and one condition first.
