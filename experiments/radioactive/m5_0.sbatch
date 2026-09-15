#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m5g
#SBATCH --array=0-5
#SBATCH --time=3:00:00
#SBATCH --output=experiments/radioactive/logs/m5g_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m5g_%A_%a.err
# M5 §0: clean traces from 6 teachers (one per GPU). No analysis.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
T=(qwen25_7b llama31_8b tulu_dpo mistral_7b gemma2_9b qwen3_14b)
/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python run_m5.py --teacher ${T[$SLURM_ARRAY_TASK_ID]}
echo "=== done ==="
