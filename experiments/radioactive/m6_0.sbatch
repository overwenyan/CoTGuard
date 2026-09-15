#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m6s
#SBATCH --time=1:30:00
#SBATCH --output=experiments/radioactive/logs/m6s_%j.out
#SBATCH --error=experiments/radioactive/logs/m6s_%j.err
# M6 smoke (pre-registered 437dcd8; amended 6db9a32, 7c07b8d): 2 teachers (Tulu-SFT, OLMo-3-Think final), tiny data.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
V=/local/projects-t3/primelab/yan.wen/venvs/vllm/bin/python
export PATH=/local/projects-t3/primelab/yan.wen/venvs/vllm/bin:$PATH   # ninja for any JIT kernels
export VLLM_USE_FLASHINFER_SAMPLER=0                                   # avoid JIT-compiling the FlashInfer sampler
export M6_SMOKE=1 M6_OUT=$PWD/data_m6_smoke M3C_OUT=$PWD/data_m6_smoke M3C_MAXLEN=1400
rm -rf data_m6_smoke
$P run_m6.py prep
for T in tulu_sft olmot_final; do $V run_m6.py gen --teacher $T; done
$P run_m6.py corpora
$P run_m3c.py sft --setting tulu_gsm --student qwen15 --epochs 1 --corpus grid --arms tulu_sft_s0,olmot_final_s0
$V run_m6.py probe --student qwen15 --names base,grid_tulu_sft_s0,grid_olmot_final_s0
$P - <<'PY'
import json
for f in ["teacher_olmot_final.jsonl", "tulu_gsm/probe_qwen15_grid_olmot_final_s0.jsonl"]:
    r = [json.loads(l) for l in open("data_m6_smoke/" + f)]
    print(f, "n", len(r), "truncated", sum(x["truncated"] for x in r), "\n--- sample ---\n", r[0]["text"][:600])
PY
$P score_m6.py
rm -rf data_m6_smoke
echo "=== smoke ok ==="
