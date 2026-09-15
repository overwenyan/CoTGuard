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
