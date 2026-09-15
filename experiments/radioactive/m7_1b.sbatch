#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=cg_m7c
#SBATCH --time=1:00:00
#SBATCH --output=experiments/radioactive/logs/m7c_%j.out
#SBATCH --error=experiments/radioactive/logs/m7c_%j.err
# M7: build all reference and test corpora once, after generation.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export M7_OUT=$PWD/data_m7 M3C_OUT=$PWD/data_m7
/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python run_m7.py corpora
echo "=== done ==="
