#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m3gS
#SBATCH --time=1:00:00
#SBATCH --output=experiments/radioactive/logs/m3gS_%j.out
#SBATCH --error=experiments/radioactive/logs/m3gS_%j.err
# M3 v7 scoring (estimates with uncertainty; no gates).
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export M3C_OUT=$PWD/data4
/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python score_m3g.py
