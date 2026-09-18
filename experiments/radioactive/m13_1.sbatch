#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m13t
#SBATCH --array=0-11%4
#SBATCH --time=8:00:00
#SBATCH --output=experiments/radioactive/logs/m13t_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m13t_%A_%a.err
# M13 (pre-registered 68cfc6a): 36 Qwen2.5-7B-Instruct students, 3 per task.
# Task layout: teacher (6) x role (ref s5-7 / test s10-12) = 12 tasks. Corpora reused from M7.
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
if [ $((I % 2)) = 0 ]; then S0=5; else S0=10; fi
ARMS=$(for S in $(seq $S0 $((S0 + 2))); do printf "%s_s%d," $TT $S; done)
$P run_m3c.py sft --setting tulu_gsm --student qwen7b --epochs 3 --corpus grid --arms ${ARMS%,}
echo "=== done ==="
