#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m4g
#SBATCH --array=0-3
#SBATCH --time=2:00:00
#SBATCH --output=experiments/radioactive/logs/m4g_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m4g_%A_%a.err
# M4 pilot: teacher traces, one move per GPU.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
M="$P run_m4.py"; C="$P run_m3c.py"; S="--setting tulu_gsm --student qwen15"
export M3C_OUT=$PWD/data_m4
$M gen --shard $SLURM_ARRAY_TASK_ID
echo "=== done ==="
