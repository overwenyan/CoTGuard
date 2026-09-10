#!/bin/bash
#SBATCH --job-name=cg_base
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/relay/logs/base_%j.out
#SBATCH --error=experiments/relay/logs/base_%j.err

# 补最大缺口: baseline 对比. 回答"更简单的方法是不是也行".
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python

echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
for RUN in attribution attr_qwen3_v2 attr_math500 attr_folio; do
  echo "########## $RUN ##########"
  $P experiments/relay/baselines.py --run-dir "experiments/relay/runs/$RUN" --space v2_diverse
done
echo "=== done ==="
