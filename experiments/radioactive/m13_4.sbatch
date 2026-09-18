#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m13t2
#SBATCH --array=0-11%4
#SBATCH --time=8:00:00
#SBATCH --output=experiments/radioactive/logs/m13t2_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m13t2_%A_%a.err
# M13 correction 1 (744a6a2): 42 more 7B reference students so T0 calibrates on 10 per teacher, as in M7.
# Task layout: teacher (6) x {s0-s3, s4+s8+s9} = 12 tasks. s0-4 corpora are the M6 corpora (symlinked).
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
export M7_OUT=$PWD/data_m7 M3C_OUT=$PWD/data_m7 M3C_MAXLEN=1024 M3C_GRADCKPT=1
T=(tulu_sft tulu_dpo tulu_rlvr olmoi_sft olmoi_dpo olmoi_final)
I=$SLURM_ARRAY_TASK_ID
TT=${T[$((I / 2))]}
if [ $((I % 2)) = 0 ]; then SEEDS="0 1 2 3"; else SEEDS="4 8 9"; fi
ARMS=$(for S in $SEEDS; do printf "%s_s%d," $TT $S; done)
$P run_m3c.py sft --setting tulu_gsm --student qwen7b --epochs 3 --corpus grid --arms ${ARMS%,}
echo "=== done ==="
