#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=cg_m12a
#SBATCH --time=2:00:00
#SBATCH --output=experiments/radioactive/logs/m12a_%j.out
#SBATCH --error=experiments/radioactive/logs/m12a_%j.err
# M12 arm A (pre-registered 09790f8).
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache M7_OUT=$PWD/data_m7
/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python geometry_m12a.py
