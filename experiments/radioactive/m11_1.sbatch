#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=cg_m11
#SBATCH --array=0-2
#SBATCH --time=12:00:00
#SBATCH --output=experiments/radioactive/logs/m11_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m11_%A_%a.err
# M11 (pre-registered 18210d9): tasks 0 tfidf control, 1 emb, 2 pos; each runs the M7 scope then the M10 scope.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export M7_OUT=$PWD/data_m7
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
R=(tfidf emb pos)
$P score_m11.py --readout ${R[$SLURM_ARRAY_TASK_ID]} --scope m7
$P score_m11.py --readout ${R[$SLURM_ARRAY_TASK_ID]} --scope m10
echo "=== done ==="
