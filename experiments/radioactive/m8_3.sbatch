#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=cg_m8S
#SBATCH --time=6:00:00
#SBATCH --output=experiments/radioactive/logs/m8S_%j.out
#SBATCH --error=experiments/radioactive/logs/m8S_%j.err
# M8: scoring, A1-A3.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export M7_OUT=$PWD/data_m7 M3C_OUT=$PWD/data_m7
/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python score_m8.py
echo "=== done ==="
