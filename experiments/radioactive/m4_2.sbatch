#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m4b
#SBATCH --time=1:00:00
#SBATCH --output=experiments/radioactive/logs/m4b_%j.out
#SBATCH --error=experiments/radioactive/logs/m4b_%j.err
# M4 pilot: embeddings, partitions, corpora.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
M="$P run_m4.py"; C="$P run_m3c.py"; S="--setting tulu_gsm --student qwen15"
export M3C_OUT=$PWD/data_m4
$M build
echo "=== done ==="
