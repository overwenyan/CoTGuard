#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m3f4
#SBATCH --time=1:00:00
#SBATCH --output=experiments/radioactive/logs/m3f4_%j.out
#SBATCH --error=experiments/radioactive/logs/m3f4_%j.err
# M3 v6: scoring.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
export M3C_OUT=$PWD/data4
D="$P run_m3d.py"; C="$P run_m3c.py"
KEYS=(o12 o17 p07 p13)
$P score_m3f.py
echo "=== done ==="
