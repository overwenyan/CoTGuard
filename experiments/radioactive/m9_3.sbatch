#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m9p
#SBATCH --array=0-1
#SBATCH --time=4:00:00
#SBATCH --output=experiments/radioactive/logs/m9p_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m9p_%A_%a.err
# M9: probes for the 24 attacked students of one family.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
V=/local/projects-t3/primelab/yan.wen/venvs/vllm/bin/python
export PATH=/local/projects-t3/primelab/yan.wen/venvs/vllm/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export M7_OUT=$PWD/data_m7 M3C_OUT=$PWD/data_m7 M3C_MAXLEN=4608
O=(tulu_dpo tulu_rlvr olmoi_dpo olmoi_final)
F=$([ $SLURM_ARRAY_TASK_ID = 0 ] && echo qwen15 || echo llama1b)
NAMES=""
for OW in "${O[@]}"; do for C in ad1 ad2; do for S in 30 31 32; do NAMES="$NAMES,grid_${C}_${OW}_s$S"; done; done; done
$V run_m7.py probe --dataset gsm --student $F --names ${NAMES#,}
echo "=== done ==="
