# M4 — secret learnable reasoning-move watermark (pilot) — pre-registration

_2026-09-15. Written and committed before any M4 data exists. Motivation and verified prior work:
`m4_deep_research_assessment.md`. Pilot approved by the user. This is a go/no-go pilot for framing
(i), not the paper's main experiment._

## Idea
The owner prompts the teacher with a *per-problem* reasoning move. The move is chosen by a secret,
learnable function of the problem's meaning:
- embed the problem with a public sentence encoder;
- b secret random hyperplanes split embedding space into 2^b regions;
- a secret assignment maps each region to one of 4 moves.

A student distilled on these traces may learn to perform the move its region calls for. The owner
tests this per instance, comparing the student's detected move with the secret assignment. Decoy
keys are free: thousands of random partitions applied to the same student outputs, with no teacher
generation. An imitator who knows the four moves but not the secret draws a different partition.

## Fixed components
- **Teacher:** Tulu-3-8B. **Student:** Qwen2.5-1.5B-Instruct, LoRA as in M3.
- **Problems:** `problems("train", 7000, seed=1)` (the M3 dilution corpus); positions
  `pos = default_rng(7).permutation(7000)`.
  - S2000 = pos[:2000]; keyed positions for dilution = pos[:700] (⊂ S2000);
  - served probes = pos[:300];
  - held-out probes = `problems("test", 300, seed=2)`.
- **Moves** (all from the v4 bank; compliant ≥ 0.96, teacher accuracy ≥ 0.76, separability ≥ 0.98):
  - m0 = o07 "After finishing, say whether the answer should be a whole number and why that makes sense."
  - m1 = o12 "After finishing, solve the problem a second, shorter way and compare the two results."
  - m2 = o18 "Before solving, state one assumption that the problem leaves implicit."
  - m3 = o20 "Before the final answer, state how many arithmetic operations were used in total."
  - Prompt = `"Solve the problem. Think step by step, one step per line. " + move`.
- **Teacher generation:** all 4 moves × S2000 (8,000 traces). Clean traces for all 7,000 problems are
  reused from `data4/tulu_gsm/dil_clean_all.jsonl`.
- **Embedding:** `thenlper/gte-base`, L2-normalised, centred by the mean over the 7,000 train
  problems (public).
- **Partition P_s:** seed s draws b standard-normal hyperplanes; region = bits of sign((e − μ)·h_i).
  Region → move:
  - b = 1: 2 distinct moves drawn without replacement;
  - b = 2: a random permutation of the 4 moves;
  - b = 3: each move assigned to 2 of the 8 regions, at random.
- **Keys:** owner seed = 20260915 + b; imitator seed = 777 + b; decoy seeds = 1,000,000 + i,
  i = 0…999 (same b).
- **Move detector** (per output, for the owner): TF-IDF (1–2 gram) + logistic regression over
  5 classes (m0–m3 + clean). Trained on teacher traces of S2000 minus the served-probe problems. Every
  problem appears in all classes, so problem content is balanced across classes.
- **Statistic (primary, soft):** A(P) = mean over probe outputs of the detector probability assigned
  to the move P prescribes for that problem.
  - p = (1 + #{decoys with A ≥ A_owner}) / 1001.
  - Secondary: hard agreement (argmax = prescribed move).
  - Lift = A_owner − mean decoy A.

## Students (10 LoRA runs)
| id | corpus | epochs |
|---|---|---|
| key100_b{1,2,3} | S2000, each trace = move prescribed by the owner partition | 3 |
| imit100_b{1,2,3} | S2000, moves prescribed by the imitator partition | 3 |
| clean2k | S2000 clean traces | 3 |
| key10_b{1,2,3} | 7,000 clean corpus with pos[:700] replaced by owner-partition traces | 1 |

The 10% students use the same regime as M3 S1-B, whose rank test detected 0/4. The existing
`dil0_clean` Qwen student is the 10% null control. Total: 3 + 3 + 1 + 3 = 10 LoRA runs.

Each student answers 300 served probes and 300 held-out probes (plain prompt, T = 0.7, seed 7).

## Gates (pre-registered)
- **G1 held-out learnability (key100):** at ≥ 1 granularity b, p ≤ 0.01 on held-out probes AND
  soft lift ≥ 0.05.
- **G2 served channel (key100):** p ≤ 0.01 on served probes at all three b.
- **G3 dilution (key10, N = 300):** at ≥ 1 b, p ≤ 0.05 on served OR held-out probes. For comparison,
  the M3 rank test detected 0/4 at 10%.
- **G4 specificity** (amended before any real pilot data — see the note below): the null tests are
  the 3 imitator students × 2 probe sets with their matching owner key (6 tests), plus clean2k and
  dil0_clean × 2 probe sets × 3 owner keys (12 tests), 18 in total. Pass iff ≤ 3 of the 18 have
  p ≤ 0.05 (at a true null rate of 0.05, P(≥ 4 of 18) ≈ 0.01) AND none has p ≤ 0.01.
  - _Amendment note:_ the original text said "10 tests, one failure tolerated". It miscounted:
    each clean student is tested against all three owner keys. With 18 tests, "≤ 1 failure" would
    reject a valid null about 23% of the time. Corrected before any non-smoke M4 data existed.
- **Reported, not gated:**
  - p and lift vs number of probes N ∈ {25, 50, 100, 200, 300} (20 subsamples) — the query-scaling
    check;
  - learnability vs b;
  - GSM8K accuracy of every student vs clean2k and dil0_clean;
  - compliance of teacher traces with the prescribed move (detector agreement on teacher traces).

## Decision
- **(G1 or G2) and G3 and G4 pass** ⇒ framing (i): design and pre-register the full grid (2 teachers,
  2–3 students including 7B, GSM8K + MATH + a long-CoT set, attacks: paraphrase, claim-preserving
  rewrite, CDG-KD-style scrub, prompt inversion).
- **(G1 or G2) and G4 pass, G3 fails** ⇒ the keyed-move signal needs teacher-dominated data. Consider
  framing (iii), or a per-instance test combined with active probe selection, before any full grid.
- **G1 and G2 both fail, or G4 fails** ⇒ abandon framing (i); report and decide between (ii) and (iii).

## Budget
8,000 teacher generations (~30 min over 4 GPUs); 10 LoRA runs; 20 × 300 samplings; detector and
decoys on CPU. About 6–8 GPU-hours, run in parallel.
