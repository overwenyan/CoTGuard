#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --job-name=cg_m3s0
#SBATCH --time=5:00:00
#SBATCH --output=experiments/radioactive/logs/m3s0_%j.out
#SBATCH --error=experiments/radioactive/logs/m3s0_%j.err

# M3 v4 stage 0 diagnostics (m3_design.md v4 §0, pre-registered d0fe896). Existing data only.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
$P stage0_m3.py d3 d2 d4 d1
