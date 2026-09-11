# Literature verification queue (run in a fresh session where WebSearch works)

> **DONE 2026-09-10 (cotguard-2)** — results in `p2_prior_work.md` (Section B verdict in §0–§4, Section A table in §5).
> Not searched: MultiAgentBench.

_Created 2026-09-10. WebSearch/WebFetch were blocked by the safety classifier in the ideation session;
venue labels in `idea_board_v3.json` are from memory (cutoff 2026-06)._

## A. Venue confirmation (ICML / NeurIPS / ICLR 2025–2026)
| Paper | Claimed | Confidence | Verify |
|---|---|---|---|
| In-Context Watermarks (Liu, Zhao, Kruegel, Song, Bu) | ICLR 2026 | from deep-research report | ☐ |
| Baker et al., Monitoring Reasoning Models for Misbehavior | arXiv 2025 | venue unknown | ☐ |
| Korbak et al., CoT Monitorability position paper | arXiv 2025 | likely no venue | ☐ |
| Chen et al., Reasoning Models Don't Always Say What They Think | arXiv 2025 | venue unknown | ☐ |
| Emmons et al., When CoT is Necessary… | arXiv 2025 | venue unknown | ☐ |
| Tr-GoF (Li, Ruan, Wang, Long, Su) | arXiv 2024/25 | JASA? | ☐ |
| White, Jafari, Berg-Kirkpatrick, Black-Box Forensics | arXiv 2606.22698 | preprint | ☐ |
| Chen et al., Do System Prompts Leave Behavioral Fingerprints? | arXiv 2608.24461 | preprint | ☐ |
| AgentPoison | NeurIPS 2024 | fairly confident | ☐ |
| AgentDojo | NeurIPS 2024 D&B | fairly confident | ☐ |
| Thought Anchors (Bogdan et al.) | arXiv 2025 | venue unknown | ☐ |
| Persona Vectors | arXiv 2025 | venue unknown | ☐ |

## B. Prior-work search for the P2 pivot (must run BEFORE experiment design is frozen)
- "Byzantine" + "LLM agents" / "multi-agent LLM" (2024–2026)
- "adversarial agent" + "multi-agent debate" (robustness of MAD to a malicious debater)
- "robust aggregation" + "LLM ensemble" / "mixture of agents" + adversarial
- "multi-agent debate" + "prompt injection" / "persuasion attack"
- "trust" / "reputation" + "multi-agent LLM" aggregation
- "verifiable reasoning" + "multi-agent" / "step verification" + "aggregation"
- Known adjacent: Prompt Infection (2410.07283), Secret Collusion (NeurIPS 2024), AgentPoison,
  Combating Adversarial Attacks with Multi-Agent Debate (2024, venue?), MultiAgentBench (2025?)

## C. What to record
For each hit: venue + year, the one result that matters, exact overlap with SVRA (Sec. 3 of
`p2_design.md`), and whether it changes the threat model or the baselines.
