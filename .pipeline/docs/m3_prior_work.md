# M3 prior work — verified positioning notes (2026-09-14)

Each entry was opened and checked on the date above. **Claim** = what we attribute to the work in the
paper.

| Work | Verified content | Claim we make | Our distinction |
|---|---|---|---|
| **Gu, Li, Liang, Hashimoto. On the Learnability of Watermarks for Language Models. ICLR 2024** (arXiv 2312.04469) | Watermark distillation: students trained on outputs of teachers using three decoding-based watermarks learn to generate detectable watermarked text. Limits: the watermark is lost under further fine-tuning on normal text, and low-distortion watermarks need many samples. Spoofing is discussed as a provenance risk. | Watermark signals can be learned by distillation, and learnability enables spoofing. | Token-level decoding watermarks, learned **from the watermarked model's outputs**. Ours: prompt-implanted reasoning signatures. |
| **An, Park, Woo, Han. DITTO: A Spoofing Attack Framework on Watermarked LLMs via Knowledge Distillation. EACL 2026 (oral)** (arXiv 2510.10987) | Spoofing via distillation from a watermarked teacher ("watermark radioactivity"), aimed at misattributing content to reputable sources; argues a watermark does not prove authorship. | A detectable watermark need not identify its generator; distillation-based spoofing is demonstrated. | DITTO distils **from the owner's model**. Our alternative source never uses owner outputs: it re-generates the signature from the instruction. |
| **Yi, Li, Zheng, Wang, Wang, He. Unified Attacks to LLM Watermarks: Spoofing and Scrubbing in Unauthorized Knowledge Distillation** (arXiv 2504.17480, rev. Aug 2025) | CDG-KD: contrastive-decoding-guided bidirectional distillation for scrubbing or forging watermarks while preserving student performance. | Inheritance of watermarks through distillation can be exploited for both removal and forgery. | Also relies on the watermarked model's outputs; ours involves no owner outputs. |
| Asking Back (2605.16462) | System-prompted behavioural markers transferred to LoRA students; a small set of related markers; no many-key attribution; response rewriting before training not evaluated (Appendix O). | As stated. | Many-key collisions, controlled alternative-source comparison, response rewriting. |
| Trace Rewriting (Ma et al., ACL 2026), ReasMark (ACL 2026), PROSE (2609.02553), ICW (ICLR 2026), Radioactivity (NeurIPS 2024), Subliminal learning (2507.14805), Dataset inference (2406.06443) | Verified earlier (m3_design v0 §0, m3_expert_review §2). | — | Headline rates not directly comparable (different attacks, carriers and access). |

## Positioning risk found during verification (must be addressed in the paper)
**Known-instruction replication is the analogue of key compromise.** For decoding-based watermarks,
anyone who holds the secret key can also generate watermarked text. So "a source given the owner's
instruction reproduces the signature" is, taken alone, the expected behaviour of any keyed scheme. A
reviewer may say it is unsurprising.

What makes our result more than that, and how the paper must frame it:
1. **The key is natural language and the effective codebook is small.** In the tested generator,
   distinguishability is dominated by 12 reasoning instructions, and the instruction banks are made of
   common reasoning habits. Knowing or *coinciding with* an instruction is therefore far more
   plausible than guessing a cryptographic key. **But accidental collision and reconstruction were not
   evaluated**, so this can motivate future work, not support a claim.
2. **The formally valid test hides the issue.** Proposition 1: the rank test's guarantee concerns the
   key null, so a correct p-value is compatible with false source attribution. That is a
   methodological point about how such tests are reported.
3. **Source information is present anyway.** The teacher-identity read-out recovers the teacher
   (closed-set), so the failure lies in the key-based procedure, not in the absence of signal. This
   distinguishes our finding from "watermarks can be spoofed".
4. **No owner outputs were used by the alternative source.** Unlike DITTO and the Unified Attacks
   (distillation from the watermarked model), the alternative teacher re-generates the behaviour from
   the instruction alone.

Framing sentence for related work: *Prior work shows that distillation lets watermark signals be
learned and spoofed from a watermarked model's outputs. We study a behavioural carrier implanted by
prompting, and show a different failure: an independent teacher given the same instruction
reproduces the signal without any owner outputs, and a formally valid key-level test cannot separate
the two sources, although the student outputs still carry teacher-identifying information.*
