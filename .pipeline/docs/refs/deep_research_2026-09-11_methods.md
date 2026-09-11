# Deep research: method candidates on this project (2026-09-11)

_Manual fallback for `gemini-deep-research` (no `GEMINI_API_KEY` in this environment): ~40 live
WebSearch/WebFetch queries, cutoff 2026-09-11. Venues marked **confirmed** come from the arXiv
comments field, a conference/journal page, or the official repo; everything else is a preprint.
Numbers quoted from fetched abstracts/HTML have not been re-derived. Read §3 before choosing._

## 1. Executive summary

The two framings this project has tried are both occupied at the level of the *claim*: "prompts
leave attributable fingerprints in reasoning traces" (White et al. 2026, Chen et al. 2026, ICW at
ICLR 2026) and "Byzantine-robust aggregation for LLM agents" (SAC at EMNLP 2026, DecentLLMs,
CP-WBFT, H-CSC, Consensus Trap, STAR). What the project *has* that is not occupied is
infrastructure plus three replicated facts: instruction echo is the carrier of prompt signatures;
relay fidelity, not the signature, decides multi-hop survival; and a CPU-checkable redundancy
obligation is executed by all three generators. Six method candidates reuse these. Two have a
verified, narrow, recent gap with a public baseline and single-GPU cost:

- **M1 — Conformity-robust conformal prediction for multi-agent LLM decisions.** Hu & Su
  (arXiv 2609.04445, 2026-09-03) show clean-calibrated conformal coverage collapses 90% → 74%
  when peers unanimously assert a wrong answer, prove weighted conformal reduces to the identity
  here, and explicitly leave "a practical defense: detecting pressured items, selective
  abstention, or running the conformity stress test online as a detector" to future work [1].
  Their code and data are public. A solo-anchored, peer-conditional conformal procedure is a
  small, provable fix nobody has published.
- **M2 — Black-box, trace-only collusion detection with exact FPR control.** The only strong
  detector (Rose et al., arXiv 2604.01151) is white-box (activation probes, 0.73–0.93 AUROC
  zero-shot) and ships an open benchmark, NARCBench [5]; the only black-box proposal (Ghanem,
  arXiv 2608.02698) reports "placeholder results pending measurement" [6]. Cross-principal
  deployments cannot read activations. A permutation-test detector on released transcripts is CPU
  work with our null-control machinery.

The other four (M3 prompt-only radioactivity, M4 conclusion-blind fault localization, M5 prompt-only
hop attribution, M6 obligation watermark) are viable but either risk a negative first result or
return to the occupied provenance turf; details in §3.

## 2. Key findings

- **Conformal prediction is the freshest, sharpest gap.** Three preprints in the last five months
  (Conformal Social Choice [2], Not All Flips [3], Conformity Breaks [1]) establish the problem
  and the failure; none repairs validity under peer pressure. [1] tried weighted CP (identity),
  per-model Mondrian (3/4 models still fail at α=0.10), and condition-aware recalibration (needs
  pressure labels; "escalates almost everything").
- **Collusion detection has a public benchmark and no black-box result.** NARCBench: 4 agents,
  2 colluding, 50 scenarios × 17 domains, 3 tiers, transcripts on HuggingFace [5]. Black-box
  detection from traces is argued necessary ("no principal can look inside another principal's
  models") but unmeasured [6]; tool use makes steganography undetectable in principle [7], so the
  achievable claim is a detection-capacity frontier, not universal detection.
- **Distillation protection of reasoning traces is occupied at the *active* end.** Ma et al. 2026
  rewrite 10% of teacher traces to implant a trigger→target behavioral watermark; black-box
  verification on the student reaches TPR 1.00 / FPR ≤ 0.02 and survives paraphrase, filtering
  and CDG-KD [9]. Token-level radioactivity (TextSeal [10], Sander et al.) needs decoding access.
  Whether a **prompt-only, trigger-free** signature (ICW [11], or our instruction echo) is
  inherited by a student is not reported anywhere found.
- **Accountability layers that relay conclusions are a named, open problem.** Arslan (arXiv
  2609.07680, 2026-09-07): auditors recover the true fault origin 4.1% of the time when upstream
  agents miss it; removing the conclusion field raises accuracy by +41 pp; remedy left open
  ("evidence sufficiently independent of the conclusions it verifies") [14]. This is the same
  mechanism as our Prop B (decision as a function of extracted content only), pointed at fault
  localization instead of accuracy.
- **Multi-hop attribution from final text exists only with decoding access.** IET [17] and
  Chronology [18] embed key-conditioned statistical signals at generation time; TRACE [19] needs a
  distortion-free sampler. Prompt-only multi-writer attribution is open, but S20 showed a lossy
  relay erases upstream prompt signatures (benign Mistral paraphrase keeps 35% by hop 5), so the
  honest claim is last-writer attribution plus a fidelity test.
- **Areas to avoid.** PRM robustness (Reward Under Attack, ICML 2026 [26]; VPRM, Sci-PRM,
  VeriBound [27]); agent-memory poisoning (SMSR certified, MemSAD, Cordon-MAS, TRIS [32]);
  AI-control monitor defenses (adaptive injection attacks subvert all protocols [28]; evaluations
  need frontier models and control benchmarks); CoT-backdoor defense via prompt inversion
  (output2prompt/RPE occupy inversion [31]; MirageBackdoor defeats CoT-based detection [30];
  Critical-CoT is white-box [30]).

## 3. Detailed analysis

### M1. Peer-conditional conformal prediction (PCCP) for multi-agent decisions

**Problem.** A model calibrated answering alone, then deployed reading peers, loses coverage when
peers agree on a wrong answer; the input is unchanged, only the score mechanism moves [1].
Decision layers that act on conformal singletons then act on wrong consensus (12.1% of flipped
items pooled; 71.4% for Qwen2.5-32B) [1]. Conformal Social Choice [2] pools verbalized
probabilities and intercepts 81.9% of wrong-consensus cases at α=0.05, but calibrates on the
deliberation distribution it deploys in, so an adversary who controls peer messages controls the
score distribution.

**Exact gap.** No procedure keeps a coverage guarantee when peer context is adversarial. [1]
lists three untried candidates: pressure detection, selective abstention, mixture/augmented
calibration over synthetic peer contexts.

**Method sketch (classic TML: split conformal + Mondrian conditioning + selective prediction).**
Two queries per item: a *solo* query (no peers) and a *social* query (with peers).
1. Calibrate nonconformity on solo scores. Exchangeability holds because the solo query never
   sees peers, so the solo set C_solo(x) has valid marginal coverage regardless of what peers say.
2. Let the social query only *act within* C_solo(x): act on a singleton iff the social answer lies
   in C_solo(x) and |C_solo(x)| = 1; otherwise escalate. This inherits P(act on wrong) ≤ α.
3. Pressure detector: the divergence between solo and social score vectors is an observable
   statistic; calibrate its null on clean peer contexts and flag items whose divergence exceeds
   the (1−β) quantile (an online conformity stress test, as [1] suggests).
4. Optional single-query variant for deployments that cannot re-query: Mondrian calibration
   conditioned on the *observable* peer statistic (peer unanimity, agreement with the model's
   prior answer) over an augmented calibration set with synthetic peer contexts at several pressure
   levels — this needs no ground-truth pressure labels, which was the blocker in [1] §4.4.

**Small theory.** Prop: coverage of C_solo is exactly the split-conformal guarantee; the act-gate
has false-action rate ≤ α for any peer-message distribution (the adversary cannot touch the solo
score). Prop: for the Mondrian variant, coverage holds conditionally on each observable-pressure
bin under exchangeability within bins. Cost statement: abstention rate as a function of pressure.

**Minimal experiment (single GPU, ≤1 GPU-day).** Reproduce [1] with its released code on
Qwen2.5-7B / Llama-3.1-8B / Mistral-7B-v0.3 / Gemma-2-9B on ARC-C and TruthfulQA; add PCCP and
the Mondrian variant; baselines = clean CP, per-model Mondrian, condition-aware recalibration
(oracle labels), Conformal Social Choice [2]. **Pre-registered gate:** under unanimous-wrong
peers at α=0.10, act-on-wrong ≤ 10% and coverage ≥ 90% with abstention strictly below
condition-aware recalibration; if abstention is not lower, the method is a triviality.

**Asset reuse.** Conformal calibration code from EXP-001/003/005 (per-pattern rank calibration,
wrong-key null), permutation null, generation pipeline; Mistral-7B-v0.3 and Qwen3-14B on disk.

**Risks.** Scoop risk is the highest of the six: [1] is 8 days old and its authors will follow up;
the solo-anchor idea is simple. Mitigation is speed — data and code are public, the experiment is
small. The reviewer objection "solo-anchoring discards deliberation's benefit" must be answered
with the gate's abstention/accuracy curve versus [2].

### M2. Black-box collusion detection from transcripts with exact FPR control

**Problem.** Detecting covert coordination among agents when the monitor sees only messages,
tool calls and timing. White-box probes reach 1.00 in-distribution and 0.73–0.93 zero-shot on
NARCBench (Llama-3.1-70B) [5]; cross-principal deployments cannot use them [6].

**Exact gap.** No measured black-box detector on a public collusion benchmark. [6] proposes
cross-run mutual information + permutation tests with an FPR budget and a detection-capacity
frontier, but its results are placeholders. [5] should be checked for any text-only baseline in
its tables before claiming this (not visible in the abstract).

**Method sketch (classic TML: permutation tests, conditional dependence, anomaly detection).**
For each agent pair, estimate dependence between their messages *beyond what the shared
transcript explains* (residual dependence after conditioning on the public context), using
cheap features (TF-IDF/stylometry, stance trajectories, timing of position changes) and a
permutation null that reassigns agent identities across scenarios (our C5 machinery). Aggregate
pairwise statistics into a group-level test with a Bonferroni/closed-testing correction; the
permutation test gives exact FPR control at any nominal level. Report the capacity frontier
empirically: detection power versus the number of covert bits per message.

**Minimal experiment (CPU first).** Run on the released NARCBench transcripts (3 tiers).
**Pre-registered gate:** zero-shot AUROC ≥ 0.75 on the hardest tier at FPR ≤ 0.05 by construction;
below 0.60, stop. Second testbed with our generators: 5 agents, 2 instructed to covertly
coordinate on a target (our G1 adversary generator already produces coordinated agents; add a
covert-channel instruction), ≤ 2 GPU-h. Baselines: LLM-judge collusion classifier (Qwen3-14B),
the white-box probes' published numbers, [6]'s MI estimator if code appears.

**Asset reuse.** Permutation null (C5), stylometric/TF-IDF readouts with problem-split CV,
adversary generation (targets + coordinated variants), LLM-judge baseline code.

**Risks.** The Oxford group (Schroeder de Witt) is active and could add black-box baselines; [6]
claims the framing. Tool-enabled steganography is undetectable in the limit [7], so the paper must
be positioned as a capacity frontier, not a guarantee. If NARCBench collusion is mostly explicit
(agents openly coordinate), text features will look trivially good — the hardest tier decides.

### M3. Prompt-only radioactivity: is an instruction signature inherited by a distilled student?

**Problem.** A competitor fine-tunes a student on your agent's reasoning traces. Token-level
watermarks are radioactive [10,12] but need decoding access; prompt-only watermarks (ICW [11],
our echo) are deployable through an API but their inheritance is unreported.

**Exact gap.** [9] occupies the *active* end (rewrite traces to implant trigger→target; black-box
student verification, TPR 1.00 / FPR ≤ 0.02; survives paraphrase/filtering/CDG-KD). Nothing
found tests whether a trigger-free, prompt-induced signature survives SFT, nor gives a black-box
student test with FPR control for it.

**Method sketch (classic TML: two-sample test / radioactivity test).** Detector = our TF-IDF
readout trained on teacher traces per key, applied to student outputs on held-out problems, with a
permutation null over keys (wrong-key calibration) for FPR. Small theory: test power as a
function of the number of watermarked training traces and echo strength (a Hoeffding-style
bound on the per-key excess). Design side: which instruction families are *learned* by students
(formatting habits, fixed phrases) versus not (persona) — our C4/S20 already say persona
carries nothing on Tulu/Mistral.

**Minimal experiment.** LoRA-SFT Qwen2.5-1.5B/3B on 700–2,000 traces from one key (Tulu or
Qwen3 traces on disk), 4 keys × 2 students, ≤ 4 GPU-h. **Gate:** student-output attribution ≥ 2×
chance at permutation-null FPR ≤ 0.05; N2's no-key residual (1.5–1.9×) is the prior, so a
negative is likely. A negative here is an analysis result, not a method paper.

**Risks.** Adjacent to [9] (stronger mechanism, same benchmarks GSM8K/MATH); if the passive
signature does not survive, the method collapses to "use [9]". Ranked third for that reason.

### M4. Conclusion-blind fault localization in agent pipelines with independent verified evidence

**Problem.** Auditors in multi-agent pipelines relay upstream conclusions instead of checking;
with conclusions hidden, fault-origin accuracy rises +41 pp [14]. Injected errors persist across
83–100% of relay chains without spillover, so they are localizable in principle [15].

**Exact gap.** [14] names the remedy ("evidence sufficiently independent of the conclusions")
and leaves it open. Automated failure attribution exists as a task (Who&When-style logs) but
with LLM judges reading conclusions.

**Method sketch (classic TML: verification + change-point localization).** Each hop must emit
machine-checkable evidence (numeric equations, structured facts) alongside prose; the auditor
never sees conclusions, only the CPU-verified evidence multiset of each hop; the fault is the
first hop whose evidence fails re-computation or contradicts verified upstream evidence.
Guarantee: localization within one hop under an honest checker; false-flag rate = checker
false-reject on honest hops (our verifier: 0.44–0.48 on GSM8K, which must drop — evidence
*diffs* between hops are far more stable than absolute pass/fail).

**Minimal experiment.** Relay chains from our testbed with one corrupted hop (numeric corruption,
semantic corruption, injected "AUDITOR NOTE"); Tulu/Qwen3 generators, Qwen3/Mistral relays,
≤ 3 GPU-h. Baselines: LLM auditor with and without conclusion field (replicating [14]'s
manipulation), STAR-style verifier. **Gate:** localization accuracy ≥ LLM-auditor-without-
conclusions + 10 pp under injected auditor notes.

**Asset reuse.** Highest of all six: relay testbed, CPU verifier, adversary generator, STAR-style
and judge baselines, SVRA honest data.

**Risks.** Numeric corruption may be too easy (a diff finds it), semantic corruption too hard
(verifier passes 44% of honest-wrong traces). The interesting middle — corrupted hop that keeps
arithmetic consistent — is exactly where our verifier is weak. Medium.

### M5. Prompt-only hop attribution from final text (last-writer + fidelity test)

Decoding-time schemes (IET [17], Chronology [18], TRACE [19]) do multi-writer attribution;
prompt-only is open, but S20 shows upstream signatures die under a lossy relay. Achievable
method: attribute the last writer (its instruction echo survives), and use the *presence* of
upstream echo as a relay-fidelity test. Reuses everything; low scoop risk; but it re-enters the
provenance area White et al. occupy at the phenomenon level, and the claim is modest. Low-medium.

### M6. Prompt-only computational-obligation watermark (P4 with executed obligations)

ReasonMark (ICLR 2026) and SWAN are decoding-time; ICW is prompt-only but lexical. A prompt-only
watermark carried by *which quantities are computed* would be paraphrase-invariant by
construction — but D1/G0 showed only compute-twice is executed (capacity ≈ 1 bit) and
monitor-aware rewriting cuts its redundancy by two thirds. Needs an obligation-screening
experiment first; low priority.

## 4. Areas of consensus

- Prompt-level signals in reasoning traces are literal echo of the instruction (our C3/C4/S20;
  consistent with ICW's design and White et al.'s lexical classifiers) [11,25].
- Multi-hop survival of any text signal depends on relay fidelity, not on the signal ([15], our
  S20 C1/C2; TRACE's answer is to move the signal into log skeleton structure [19]).
- LLM-in-the-loop verifiers and monitors are an injection surface (STAR, judges, control
  monitors [28,33]); PRMs behave as fluency detectors [26].
- Exchangeability, not model quality, is the binding constraint on conformal guarantees for
  LLMs in multi-agent settings [1,2].

## 5. Areas of debate

- Whether black-box collusion detection is achievable at useful rates or only a capacity
  frontier [6,7].
- Whether deliberation helps or hurts: [2] reports +22 pp over consensus with conformal
  singletons; [1,3] report harmful conformity of 12–29%; the reconciliation is selection
  (abstain on pressured items), which is what M1 formalizes.
- Whether passive prompt-only watermarks can be radioactive at all (no evidence either way).

## 6. Sources

[1] Hu, Su. Conformity Breaks Conformal Prediction. arXiv 2609.04445, 2026-09-03. Preprint;
    code+data github.com/yibo-hu-lab/conformity-breaks-conformal. (High relevance; recent.)
[2] From Debate to Decision: Conformal Social Choice for Safe Multi-Agent Deliberation.
    arXiv 2604.07667, 2026-04. Preprint.
[3] Hao et al. Not All Flips Are Conformity. arXiv 2606.00820, 2026-05. Preprint.
[4] Liu. Robust Conformal Consensus (multi-judge intervals). arXiv 2609.06367, 2026-09. Preprint,
    single author, no numbers in abstract.
[5] Rose, Cullen, Abdelnabi, Torr, Kaplowitz, Schroeder de Witt. Detecting Multi-Agent Collusion
    Through Multi-Agent Interpretability. arXiv 2604.01151, 2026-04. Preprint; NARCBench at
    github.com/aaronrose227/narcbench and HF datasets/aaronrose227/narcbench.
[6] Ghanem. Steganalysis of Adaptive Covert Collusion in Tool-Using Agent Populations.
    arXiv 2608.02698, 2026-08. Preprint, single author, results marked placeholders.
[7] Tool Use Enables Undetectable Steganography in Multi-Agent LLM Systems. arXiv 2606.28425.
[8] Secret Collusion among AI Agents. NeurIPS 2024 (confirmed).
[9] Ma, Yeoh, Zhang, Vorobeychik. Protecting LMs Against Unauthorized Distillation through Trace
    Rewriting. arXiv 2602.15143, 2026-02/04. Preprint.
[10] Sander, Chang et al. TextSeal. arXiv 2605.12456, 2026-05. Preprint (token-level, radioactive).
[11] Liu, Zhao, Kruegel, Song, Bu. In-Context Watermarks. ICLR 2026 (confirmed).
[12] Sander et al. Watermarking Makes Language Models Radioactive. ICML 2024 — from memory, not
     re-verified this session.
[13] Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation? ACL 2025 Main (per
     official repo THU-BPM/Watermark-Radioactivity-Attack).
[14] Arslan. Audit Without Verification. arXiv 2609.07680, 2026-09-07. Preprint, single author,
     pre-registered design.
[15] Shawn. Faithful, Not Corrective: Message-Format Effects in Multi-Hop Agent Relays Are
     Tier-Dependent. arXiv 2607.09678, 2026-06. Preprint, single author, analysis only.
[16] LLM as a Broken Telephone. arXiv 2502.20258, 2025. Preprint.
[17] Nian et al. When Only the Final Text Survives: Implicit Execution Tracing. arXiv 2603.17445,
     2026-03/08. Preprint (decoding-time signals).
[18] Chronology of Multi-Agent Interactions for Provenance of Evolving Information. Royal Society
     Open Science 13(7), 2026 (journal); arXiv 2504.12612. Decoding-time steganography.
[19] Gao et al. TRACE two-channel trajectory watermark. arXiv 2607.08400, 2026-07. Preprint.
[20] Watermarking LLM Agent Trajectories (ActHook). ICML 2026 poster (confirmed).
[21] AgentMark. ACL 2026 Main (per official repo README).
[22] SeqWM. arXiv 2605.11036. Preprint.
[23] MemMark. EMNLP 2026 Findings (per arXiv page).
[24] RedAct. arXiv 2606.10813, 2026-06. Preprint.
[25] White, Jafari, Berg-Kirkpatrick. Black-Box Forensics for Conversational LLM Agents.
     arXiv 2606.22698. Preprint.
[26] Reward Under Attack: Robustness and Hackability of PRMs. ICML 2026 poster (confirmed).
[27] VPRM arXiv 2601.17223; Sci-PRM 2606.04579; VeriBound 2606.20740. Preprints.
[28] Terekhov et al. Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols.
     arXiv 2510.09462, rev. 2026-03. Preprint.
[29] Adaptive Evaluation of Out-of-Band Defenses (2606.26479); Adaptive Attacks Break Defenses
     Against IPI (2503.00061). Preprints.
[30] Critical-CoT 2604.10681 (white-box); MirageBackdoor 2604.06840; DarkMind 2501.18617;
     ShadowCoT 2504.05605; BadChain ICLR 2024 (confirmed).
[31] Extracting Prompts by Inverting LLM Outputs (output2prompt), EMNLP 2024 (confirmed);
     Reverse Prompt Engineering 2411.06729.
[32] SMSR 2606.12703 (certified); MemSAD 2605.03482; Cordon-MAS 2605.26754; TRIS 2609.00470.
[33] Consensus Trap 2604.17139; STAR 2605.28104; SAC EMNLP 2026; H-CSC 2606.07316 (see
     `p2_prior_work.md`).
[34] Trajectory Guard 2601.00516; AgentTether 2607.06273; Content-Aware Attack Detection in
     Tool-Call Traffic 2605.11053. Preprints.

## 7. Gaps and further checks before committing

- M1: read [1] §4 in full for any solo-query control they may have run; read [2]'s calibration
  protocol to confirm it calibrates on deliberation outputs.
- M2: inspect [5]'s tables for a text-only baseline; download NARCBench and check whether tier-3
  collusion is covert or explicit.
- M3: check ICW's paper for any fine-tuning/distillation experiment; check whether [9]'s
  code is released (its trigger watermark would be the natural baseline).
- M4: read [14]'s appendix for the report schema, to align the "evidence field" design.
- SVRA/G1 (job 20041130) verdict is still pending and does not change the ranking above; its
  adversary traces and baselines are reusable by M2 and M4.
