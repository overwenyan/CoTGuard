#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=cg_m13s
#SBATCH --time=2:00:00
#SBATCH --output=experiments/radioactive/logs/m13s_%j.out
#SBATCH --error=experiments/radioactive/logs/m13s_%j.err
# M13 scoring: manipulation check, G1 (T0 collapse), G2 (T1 repair), with the M7 sentinel.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export M7_OUT=$PWD/data_m7 M3C_OUT=$PWD/data_m7
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
$P score_m13.py
echo "=== done ==="
