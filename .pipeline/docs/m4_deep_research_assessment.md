# Deep research result (2026-09-15) — our verification and assessment

_Source: the user's deep-research report, answering `.pipeline/docs/deep_research_prompt_2026-09-15.md`.
We checked the claims that decide the direction on 2026-09-15._

## Verified
- **R-CoT** (Zhang et al., arXiv 2604.25247, Apr 2026): reasoning-layer watermark embedded by
  GRPO fine-tuning; TPR > 95% under fine-tuning; distillation not mentioned. Model-side control.
- **Echoes within the Reasoning / BiCoT** (Lu et al., arXiv 2605.28890, May 2026): fine-tuning-based,
  private signature subspace; verification uses top-logprobs + sentinel tokens. Model-side control;
  distillation not mentioned.
- **Reference-Based Distillation Detection** (Rawat, Chen, Anand, Duan, Rotsted, Min, arXiv
  2607.09692, Jun 2026): teacher attribution and distillation detection using an earlier same-lineage
  reference checkpoint; statistical tests with an open-world extension; near-perfect single-teacher
  accuracy. **This makes a composite teacher-attribution paper (framing ii) crowded.**

## Found by us, missing from the report — changes the novelty of framing (i)
- **AgentWM: On Protecting Agentic Systems' IP via Watermarking** (Wang et al., arXiv 2602.08401,
  Feb 2026). Biases the distribution over *semantically equivalent tool-execution paths*, and is
  evaluated against **imitation attacks** (models trained on the victim's outputs), with statistical
  hypothesis-testing verification.
  → "Keyed selection among functionally equivalent behaviours that survives into imitating models"
  is already published, for agent tool paths.
- **TRACE** (Gao et al., arXiv 2607.08400, Jul 2026): a two-channel watermark for agent trajectories.
  Its selection channel chooses actions keyed on local content with a distortion-free sampler.
- **SeqWM: Sequential Behavioral Watermarking for LLM Agents** (An et al., arXiv 2605.11036, May 2026):
  history-conditioned transition patterns, verified against random-key baselines.
- Also AgentMark, Agent Guide and ActHook (named in the same literature; not yet opened).
- SimKey (arXiv 2510.12828): a semantic-hashing key module for token-level watermarks.

**Revised novelty of framing (i): MEDIUM, not HIGH.** Differentiators that remain:
1. **Prompt-only** implantation. The owner needs no decoding or sampling control; this works with a
   black-box teacher API. The agent watermarks bias sampling.
2. **Reasoning-move** carrier inside CoT traces for distillation, not tool paths.
3. **Rewrite robustness** of content-level moves (our H-OP evidence).
4. The **key-null vs source-null** analysis and the collision results as motivation.
5. **Learnable secret partitions** (below) plus dilution-scaling theory.

## Design flaw in the report's framing (i), and the fix
If `hash(problem, secret)` selects the move pseudorandomly, the mapping cannot be learned by a
student: it is a pseudorandom function of the problem text. On **held-out** problems the student
reproduces only the marginal mix of moves, so per-instance agreement ≈ chance. Two fixes, both
testable:
- **(a) Probe on served problems (memorisation channel).** The owner logs which problems it answered
  and probes the student with those. Agreement then tests memorisation of keyed choices. It is
  realistic when distillers query the owner's API with their own prompts, but it needs access to the
  distiller's query set (which the owner has, via its logs).
- **(b) A learnable secret partition (generalisation channel).** The move is chosen by a secret
  low-complexity function of *semantic* problem features, e.g. random hyperplanes in a sentence
  embedding space (SemStamp-style LSH) mapped to moves. Coarse partitions are learnable but leak
  more; fine partitions are more secret but less learnable. This **learnability–secrecy trade-off**
  is itself an analysis axis and a capacity question.

**A decisive practical advantage:** decoy keys are free. The null distribution comes from thousands
of random partitions / seeds applied to the same student outputs, with no teacher generation per
decoy. p-values can reach 1e-3–1e-4, where the M3 rank test was stuck at K = 32–64.

## Other corrections to the report
- "Dilution failure below 10% is a read-out weakness, not a fundamental limit": **not established**.
  Radioactivity's 1% result uses token-level keyed watermarks with a per-token statistic (and
  open-model access for the strongest numbers). Our carrier is behavioural and black-box. It is a
  hypothesis to test in a pilot, not a finding.
- The 200–400 H200-hour estimate for the full grid is plausible but must be preceded by a cheap pilot.
