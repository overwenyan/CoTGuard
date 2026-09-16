#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m9t
#SBATCH --array=0-15%7
#SBATCH --time=6:00:00
#SBATCH --output=experiments/radioactive/logs/m9t_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m9t_%A_%a.err
# M9: 3 students per (owner, attack, family); corpora built by every task if missing.
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
I=$SLURM_ARRAY_TASK_ID
F=$([ $I -lt 8 ] && echo qwen15 || echo llama1b); J=$((I % 8))
C=$([ $((J % 2)) = 0 ] && echo ad1 || echo ad2); OW=${O[$((J / 2))]}
[ -f data_m7/tulu_gsm/corpus_grid_ad2_olmoi_final_s32.jsonl ] || $P run_m9.py corpora
$P run_m3c.py sft --setting tulu_gsm --student $F --epochs 3 --corpus grid --arms ${C}_${OW}_s30,${C}_${OW}_s31,${C}_${OW}_s32
echo "=== done ==="
