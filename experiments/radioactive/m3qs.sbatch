#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m3qs
#SBATCH --time=4:00:00
#SBATCH --output=experiments/radioactive/logs/m3qs_%j.out
#SBATCH --error=experiments/radioactive/logs/m3qs_%j.err
# M3 v4 stage 1b (exploratory, pre-registered 52c34c3): query scaling on the dilution students.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export M3C_OUT=$PWD/data4
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
$P query_scale.py sample
$P query_scale.py score
echo "=== done ==="
