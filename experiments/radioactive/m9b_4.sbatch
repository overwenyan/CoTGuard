#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m9bp
#SBATCH --array=0-1
#SBATCH --time=4:00:00
#SBATCH --output=experiments/radioactive/logs/m9bp_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m9bp_%A_%a.err
# M9b: probes for the 24 attacked students of one family.
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
A=(tulu_sft:tulu_dpo tulu_dpo:tulu_sft tulu_dpo:tulu_rlvr tulu_rlvr:tulu_dpo olmoi_sft:olmoi_dpo olmoi_dpo:olmoi_sft olmoi_dpo:olmoi_final olmoi_final:olmoi_dpo)
F=$([ $SLURM_ARRAY_TASK_ID = 0 ] && echo qwen15 || echo llama1b)
NAMES=""
for X in "${A[@]}"; do for S in 40 41 42; do NAMES="$NAMES,grid_ad2s_${X%%:*}_to_${X##*:}_s$S"; done; done
$V run_m7.py probe --dataset gsm --student $F --names ${NAMES#,}
echo "=== done ==="
