#!/bin/bash
#SBATCH --job-name=cg_m2bj
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=experiments/collusion/logs/m2bj_%j.out
#SBATCH --error=experiments/collusion/logs/m2bj_%j.err

# M2 Part B, second half: the LLM-judge baseline is re-run after the votes were re-parsed
# (its prompts include the final votes; the first attempt would have shown "Option ?" on ~48% of
# them, handicapping the baseline), then the pre-registered Gate B analysis.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/collusion
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
$P judge_collusion.py
echo "########## gate B ##########"
$P analyze_clean.py
echo "=== done ==="
