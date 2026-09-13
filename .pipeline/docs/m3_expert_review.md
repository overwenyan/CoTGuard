# M3 — outside expert review (2026-09-13) and our response

Brief sent: `m3_expert_brief.html` (4bfa8bb). The review is summarised here in our words; the
literature claims were verified by us on 2026-09-13 (links below).

## 1. Verdict
Continue for **one bounded round**; do not submit the current framing as a finished main-conference
paper. The inheritance result is credible, but novelty is narrower than the brief said, and the
attribution guarantee needs more qualification than the paraphrase limitation alone.

## 2. Prior work we had missed (verified)
| Work | What it already has | Our possible distinction |
|---|---|---|
| **Asking Back: Interaction-Layer Antidistillation Watermarks** (Yang et al., arXiv 2605.16462, May 2026) | System prompt induces behavioural markers (including a declarative restatement); Llama-3.3-70B teacher; 63 LoRA students (Qwen3.5-0.8B, Gemma-3-1B, OLMo-2-1B; 3,009 pairs, r=16, 1 epoch); black-box detection by an LLM judge vs a per-family baseline. DIPPER paraphrases **user prompts only**; the limitations section says "response-side rewriting and adaptive attacks are not evaluated". One shared marker policy, **no multi-key attribution**. | Large randomised key space with competing-key attribution; response rewriting before training. |
| **The Shape of Ownership / PROSE** (Sun et al., arXiv 2609.02553, Sep 2026) | Private semantic structures implanted by **fine-tuning**, detected in responses to natural queries. | Prompt-only implantation. "Use semantic structure instead of wording" is already occupied. |
| **ReasMark** (Lv et al., ACL 2026 long) | Training-based reasoning watermark, black-box attribution after distillation. | No teacher training; different carrier and read-out. |
| In-Context Watermarks (ICLR 2026); Trace Rewriting (Ma et al., ACL 2026); Radioactivity (Sander et al., NeurIPS 2024) | Already in the brief. | — |
| Subliminal learning (Cloud et al., 2507.14805) | Trait transfer through semantically *unrelated* data. | Our channel is overt reasoning behaviour, so we must **not** call it subliminal. |
| LLM Dataset Inference (2406.06443), model authorship attribution | Different questions (was this data used? which model wrote this?). | Needs distribution-matched negative controls. |

**Candidate framing (not a verified gap):** characterising how distinguishable many prompt-induced
reasoning signatures are, whether they survive distillation, and when response rewriting destroys
their attribution value. Terminology: "prompt-implanted" and "no post-generation editing", not an
unqualified "passive". Detection supports association with a keyed source under stated assumptions,
not exclusive ownership.

## 3. Statistical qualifications we accept
1. **Rank test validity.** p ≤ 3/64 holds under H0 only if the owner index is uniform and independent
   of the score vector, **given the whole scoring and selection procedure**. Any screening (e.g.
   dropping keys that look like the teacher's default style) must be applied symmetrically to owner
   keys and decoys, *before* randomisation.
   - Our 8 trained keys are the v2 persona keys: generator outputs chosen before any distillation data
     existed, but not drawn with the same RNG as the 56 decoys. This must be stated.
2. **The guarantee averages over the random key assignment.** It is not a 4.7% FPR for each fixed key.
   The hot-key diagnostic is the per-key view. The clean/base veto does not cover every unrelated
   student: add students trained on other keys, other teachers, and independently generated traces
   with similar habits.
3. **Collisions.** Only 12 reasoning instructions sit behind the 768 keys. The number of behaviourally
   distinguishable identities may be far smaller; 768 keys is at most ~9.6 bits of nominal entropy.
   This is an attribution codebook, not a secret. **Test it before any new training.**
4. **Reporting.** 42/48 is descriptive (keys, teachers and families are reused; not 48 replications).
   The query-budget curve came from exploratory resampling and must be labelled that way.

## 4. Corrections to our own reporting
- **ARC gate G-R3.** The design text ("within 0.67–1.5× the teacher's") does not unambiguously
  specify the per-teacher denominator. So the **pre-registered outcome is VOID**, and the per-teacher
  pass (14/16) is a **post hoc sensitivity analysis**. Also, length matching does not show the task
  was learned. On ARC the Qwen-1.5B keyed students score 0.645–0.70 against base 0.655 and clean 0.74,
  i.e. no accuracy gain over base. Llama-1B keyed students score 0.49–0.575 against base 0.44.
- **Attack cost is not a trade-off result.** The accuracy drops show these attacks cost utility, not
  that signature removal must. The matched control (rewritten *clean* traces) already exists for
  paraphrase: see stage 0.
- **The v2 active-watermark comparison** used our passive read-out, not the active method's native
  detector. It shows detector specificity only.
- **The owner-simulated rewrites** kept the correct answer ~70% of the time. Their failure is evidence
  against that augmentation pipeline, not against non-lexical read-outs in general.

## 5. Plan adopted (ordering from the review)
**Stage 0** — diagnostics on existing data, no new generation. Pre-registered in `m3_design.md` v4 §0.
**Stage 1** — frozen after stage 0, before any new data:
- a fresh-instruction factorial (behavioural spec × read-out × original/rewritten traces, with a
  wording-change rewrite and a canonicalising rewrite);
- a utility-matched removal attack (answer-preserving rewrite with consistency checks and bounded
  retries, clean-rewritten controls, token exposure reported);
- a mixed-source dilution study (~10k examples at 0/1/5/10% keyed, topic-matched, several
  prospectively chosen keys, plus an independent-imitation control);
- a stealth screen (LLM screen plus a supervised detector on held-out keys and problems; TPR at a
  fixed clean FPR; retrain on the retained corpus).

Defer hybrid active/passive work. If stage 1 yields only stronger TF-IDF results on unmodified
traces, package M3 as a scoped empirical study. If it yields a rule that predicts transfer and removal
on fresh keys, aim for a main-conference paper.
