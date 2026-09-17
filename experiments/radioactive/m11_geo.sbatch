#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=cg_m11g
#SBATCH --time=2:00:00
#SBATCH --output=experiments/radioactive/logs/m11g_%j.out
#SBATCH --error=experiments/radioactive/logs/m11g_%j.err
# M11 exploratory geometry check (spec 9a640d5).
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache M7_OUT=$PWD/data_m7
/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python geometry_m11.py
