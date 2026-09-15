# M5 — open-set teacher attribution of distilled students (framing ii)

_Chosen by the user on 2026-09-15 after the M4 pilot failed its gates. The full pre-registration
(students, open-set protocol, composite test, gates) is written in §1 **before any student is
trained**. §0 is data generation only: no analysis, no gates._

## §0 Teacher trace generation (2026-09-15)
- **Problems:** `problems("train", 7000, seed=1)`. The union of
  - the v3 bank set (first 300; used for read-out training in M3), and
  - S2000 = `default_rng(7).permutation(7000)[:2000]` (student corpora, as in M4).
- **Prompt:** the plain M3 prompt "Solve the problem. Think step by step, one step per line." — no key.
  T = 0.7, top-p 0.95, ≤ 400 new tokens; thinking disabled for Qwen3.
- **New teachers:** Llama-3.1-8B-Instruct, Llama-3.1-Tulu-3-8B-DPO, Mistral-7B-Instruct-v0.3,
  Gemma-2-9B-it, Qwen3-14B. Also Qwen2.5-7B-Instruct on S2000 (its 300 bank traces already exist).
- **Existing:** Tulu-3-8B clean traces for all 7,000 problems (`data4/tulu_gsm/dil_clean_all.jsonl`).
- **Reason for the choice of teachers:** near-lineage hard negatives (Tulu-SFT vs Tulu-DPO vs
  Llama-3.1-Instruct; Qwen2.5 vs Qwen3) and unrelated families (Mistral, Gemma).

## §1 Pre-registration (2026-09-15; committed before any M5 student is trained)

### Teachers and data
- **Seven teachers:**
  - T1 Tulu-3-8B (SFT) and T2 Tulu-3-8B-DPO — same lineage;
  - T3 Llama-3.1-8B-Instruct — T1/T2's base lineage;
  - T4 Qwen2.5-7B-Instruct and T5 Qwen3-14B — same family;
  - T6 Mistral-7B-Instruct-v0.3 and T7 Gemma-2-9B-it — unrelated families.
- **Problem sets** (all from `problems("train", 7000, seed=1)`):
  - **R300** (first 300): teacher-identity read-out training only.
  - **POOL** = S2000 \ R300: student corpora.
  - **Probes:** 300 GSM8K test problems (`seed=2`).
- A trace whose thinking block is unclosed (Qwen3 failure flag) or whose text is empty is dropped.
  The drop rate is reported per teacher.

### Read-out
- **Primary:** 7-way TF-IDF (1–2 gram) + logistic regression trained on the 7 teachers' R300 traces.
  Score of teacher t for a student = mean P(t) over its probe outputs.
- **Reported:** a gte-base embedding + LR read-out (an "idiosyncrasies"-style baseline).

### Students
- **Calibration / attribution grid:** Qwen2.5-1.5B-Instruct, LoRA (as M3), 3 epochs. For each teacher,
  5 corpora of 1,500 problems drawn from POOL with `default_rng(100·t + s)`, s = 0…4 → 35 students.
- **Family replicate:** Llama-3.2-1B-Instruct, one corpus (s = 0) per teacher → 7 students.
- **Composite (instruction + teacher):**
  - New Tulu-DPO traces under instructions o12 and p07 on R300 ("same instruction, same-lineage
    independent teacher");
  - students: Qwen-1.5B and Llama-1B × seeds {0, 1} → 8.
  - Reused from v7 E1: own (Tulu + k), imit (Qwen2.5-7B + k), neg (Qwen2.5-7B + other instruction).
- **Mixtures** (Qwen-1.5B, seeds {0, 1}), 1,500-problem corpora from POOL:
  - 50% T1 + 50% T4;
  - 10% T1 + 90% T4;
  - 10% T1 + 90% T3 (hard: base lineage) → 6 students.

### Tests
- **Closed-set attribution:** a student is attributed to the argmax teacher score.
- **Owner test (open-set, conformal):**
  - For owner O, with U = two held-out "unknown" teachers (not used for calibration or read-out
    classes beyond their fixed role), the calibration students are the grid students of the
    remaining reference teachers: 4 teachers × 5 = 20.
  - p = (1 + #{calibration students with score_O ≥ test score_O}) / 21.
  - Every ordered choice of owner O with a fixed rotation of U is evaluated:
    U = {T(O+1), T(O+3)} mod 7, so each owner faces one near and one far unknown where possible.
    The owner's own grid students are the positives (5 per owner).
  - The read-out keeps all 7 classes: the owner can train on public teachers' traces, including
    the unknowns, since generating traces is cheap. What is *unknown* is only that no students of
    U are used in calibration. **A stricter variant is reported:** read-out retrained without the U
    classes.
- **Composite owner test:** p_comp = max(p_instruction, p_teacher) (an intersection–union test: it
  rejects only if both reject, so level α holds for the union null).
  - p_instruction is the v7 rank test (K = 32).
  - p_teacher is the owner test above, with owner T1 and calibration students of T3–T7.

### Gates
- **G-A closed-set:** top-1 correct for ≥ 30/35 Qwen grid students and ≥ 6/7 Llama students.
- **G-B open-set (primary):** averaged over the 7 owners, TPR at p ≤ 0.05 on the owner's own
  students ≥ 0.80 AND FPR on unknown-teacher students ≤ 0.10.
  - Same-lineage FPR (owner T1 vs unknown T2 students, and the reverse) is reported separately.
    **Prediction:** it is the hardest case, possibly above 0.10.
- **G-C composite:**
  - owner-own students (Tulu + k) pass in ≥ 7/8;
  - independent-teacher same-instruction students (Qwen2.5-7B + k) pass in ≤ 1/8 (v7 E1: 8/8 under
    the instruction test alone);
  - Tulu-DPO + k reported, with no gate.
- **Mixtures (reported):** p_teacher for T1 at 50% and 10%, both mixture partners.
- **Kill criteria:** G-B TPR < 0.60 or FPR > 0.20 ⇒ framing (ii) is not viable ⇒ framing (iii).
  G-B passes but G-C fails ⇒ report the teacher-attribution result, not the composite claim.

### Positioning (verified prior work; see m4_deep_research_assessment.md)
- Reference-Based Distillation Detection (arXiv 2607.09692): needs an earlier same-lineage checkpoint.
- Knowledge Distillation Detection for Open-weights Models (arXiv 2510.02302): open-weight student.
- "Who Taught You That?" (arXiv 2502.06659): linguistic features.

Our distinctions to test: black-box student, conformal open-set calibration against unknown
teachers, the composite instruction ∧ teacher test that removes M3's source ambiguity, and
near-lineage hard negatives. **These three papers must be opened and their protocols compared before
writing.**
