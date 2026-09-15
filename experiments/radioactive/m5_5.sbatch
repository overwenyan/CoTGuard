#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m5S
#SBATCH --time=3:00:00
#SBATCH --output=experiments/radioactive/logs/m5S_%j.out
#SBATCH --error=experiments/radioactive/logs/m5S_%j.err
# M5 §1: scoring and gates.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
M="$P run_m5.py"; C="$P run_m3c.py"
ORDER=(tulu_sft tulu_dpo llama31_8b qwen25_7b qwen3_14b mistral_7b gemma2_9b)
export M5_OUT=$PWD/data_m5 M3C_OUT=$PWD/data_m5
$P score_m5.py
echo "=== done ==="
