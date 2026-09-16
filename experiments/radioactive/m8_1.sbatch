#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m8m
#SBATCH --array=0-1
#SBATCH --time=6:00:00
#SBATCH --output=experiments/radioactive/logs/m8m_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m8m_%A_%a.err
# M8 A3 (pre-registered c7e3076): 12 mixture students per family on GSM8K.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
export M7_OUT=$PWD/data_m7 M3C_OUT=$PWD/data_m7 M3C_MAXLEN=4608
F=$([ $SLURM_ARRAY_TASK_ID = 0 ] && echo qwen15 || echo llama1b)
ARMS=""
for PR in tulu_dpo_tulu_rlvr olmoi_dpo_olmoi_final; do for L in 5 1; do for S in 20 21 22; do
  ARMS="$ARMS,mix_${PR}_l${L}_s${S}"; done; done; done
$P run_m3c.py sft --setting tulu_gsm --student $F --epochs 3 --corpus grid --arms ${ARMS#,}
echo "=== done ==="
