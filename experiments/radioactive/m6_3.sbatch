#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m6p
#SBATCH --array=0-1
#SBATCH --time=6:00:00
#SBATCH --output=experiments/radioactive/logs/m6p_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m6p_%A_%a.err
# M6: probes for all students of one family (+ base) with vLLM LoRA on L40S.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
V=/local/projects-t3/primelab/yan.wen/venvs/vllm/bin/python
export PATH=/local/projects-t3/primelab/yan.wen/venvs/vllm/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export M6_OUT=$PWD/data_m6 M3C_OUT=$PWD/data_m6 M3C_MAXLEN=4608
T=(tulu_sft tulu_dpo tulu_rlvr olmoi_sft olmoi_dpo olmoi_final olmot_sft olmot_dpo olmot_final)
F=$([ $SLURM_ARRAY_TASK_ID = 0 ] && echo qwen15 || echo llama1b)
NAMES=base
for TT in "${T[@]}"; do for S in 0 1 2 3 4; do NAMES=$NAMES,grid_${TT}_s$S; done; done
$V run_m6.py probe --student $F --names $NAMES
echo "=== done ==="
