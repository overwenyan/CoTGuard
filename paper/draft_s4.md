# Draft — §4 A signature that names the instruction, not the owner

_Working draft (2026-09-17). Compressed from the M3 rounds (`.pipeline/docs/m3_design.md` v2–v7; ledger EXP-M3B …
EXP-M3G-b). This section is the paper's first instance of the pattern §5–§6 repeat at the checkpoint level: **same
outputs, different null, different answer.** Status labels as in §3.6. The earlier, longer key-bank analysis
(`draft_s3_s5.md`, "§4.x Nominal keys collide") becomes Appendix Y; one sentence of it survives below._

---

## 4 A signature that names the instruction, not the owner

The most direct way for an owner to make its traces attributable is to shape them. The owner adds a secret **key** — a
short reasoning instruction, e.g. *"restate the target before each computation"* — to its teacher's prompt, releases
only the resulting traces, and later tests a suspect student with plain prompts, ranking its own key against a bank of
decoy keys under a read-out trained on teacher traces (§3.4). The student never sees the key.

**The signature transfers** [confirmatory; EXP-M3B, EXP-M3C]. Students of Tulu-3-8B place the owner's key first among
16 hand-written keys in 6 of 6 cases in both student families, and among 64 generated keys the owner test passes a
strict criterion — rank ≤ 3 and the key not in the top three for undistilled or clean-distilled students — for 12 of 16
key–student pairs; 16 of 16 with Qwen2.5-7B-Instruct as the teacher. Removing traces with wrong answers does not remove
it (7 of 8).

**But it identifies the instruction, not the owner** [confirmatory; EXP-M3G E1]. We gave an *independent* teacher
(Qwen2.5-7B-Instruct) the owner's instruction and distilled students from its traces. The owner's test flagged them in
**8 of 8** cases, every one at the minimum attainable p-value, while students of the same independent teacher under a
*different* instruction were flagged in 0 of 8. As a detector of the instruction family the test is sensitive and
specific; as a test of *who produced the traces* it fails completely. Proposition 1 says why: the rank test controls
the null "the student is independent of the owner's key", and a student taught under the same instruction by anyone is
not.

**The information needed is in the same outputs** [exploratory, specification committed before running; EXP-M3G-b].
A read-out trained to tell the two *teachers* apart — on teacher traces for instructions no test student used — separates
Tulu-sourced from Qwen-sourced students at **AUC 1.00** in both student families and both key codebooks, with 0.77–0.92
per-output accuracy. By Proposition 2, instruction sufficiency is violated in our data: the outputs carry source
information beyond the instruction, and the key test does not look for it. The student outputs are identical in the two
analyses; only the hypothesis being tested differs, and the verdict flips.

**Where it breaks** [confirmatory]. The boundaries §6 later meets at the checkpoint level already appear here.
- **Paraphrase.** A distiller who rewrites traces with a 7B instruction model before training leaves the owner's key
  detectable in only 8 of 16 pairs [EXP-M3C]; §6.3 reuses this attack verbatim.
- **Dilution.** At 50% keyed traces in a 7,000-problem corpus the key is detected in 6 of 6 students; at 10%, in 0 of 6,
  and more probe queries do not recover it (Proposition 3) [EXP-M3F, EXP-M3G E3].
- **Codebook size.** The nominal key count overstates what the read-out distinguishes: misattributed traces land on a
  key sharing the true key's instruction 6.5–8.3 times more often than chance [EXP-M3S0; Appendix Y].

Implanting the key has a utility cost in both student families under the corrected extractor (§5.1, Appendix X); an
earlier draft's claim that the cost depended on the student is retracted.

**What §4 leaves open.** An owner who wants to know *whether its traces* were used needs the teacher-identity signal,
not the key. That turns the question from "which instruction?" into "which model?" — and for a vendor, the hardest
version of "which model?" is not a competitor but its own adjacent checkpoint. §5 drops keys entirely and asks it.

---

## Notes for integration (remove before submission)
- **Numbers to re-verify against the ledger before camera-ready:** 6/6 (M3B), 12/16 and 16/16 (M3C G-R0, G-R2), 7/8
  filter, 8/8 vs 0/8 (M3G E1), AUC 1.00 and 0.77–0.92 (M3G-b), 8/16 para, 6/6 vs 0/6 dilution (M3G E3), 6.5–8.3×
  (M3S0 D1).
- **Deliberately not claimed:**
  - M3B's "empirical FPR 0.016" — near-tautological by construction of the rank (ledger correction, 2026-09-12).
  - G-R3 (ARC, 14/16) — the per-arm length rule was adopted after seeing the output; **reporting it as a pass is still
    an open decision for the user.** Until decided, ARC appears only in Appendix Y with both readings.
  - Any mechanism for which keys survive paraphrase (the "propositional content" story in the ledger is post hoc).
  - Any utility magnitude in the body: the pre-correction figures (e.g. Qwen −12 points, Llama "not detected") are
    superseded; Appendix X reports both extractors.
- **Why this section exists:** it is the paper's opening demonstration of the pattern (advisor round 10), not a
  leftover. §5.3 cites it as the first instance.
