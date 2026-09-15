#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m6t
#SBATCH --array=0-17%7
#SBATCH --time=8:00:00
#SBATCH --output=experiments/radioactive/logs/m6t_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m6t_%A_%a.err
# M6: 5 LoRA students per (teacher, family); tasks 0-8 qwen15, 9-17 llama1b. Corpora built once by task 0.
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
I=$SLURM_ARRAY_TASK_ID
F=$([ $I -lt 9 ] && echo qwen15 || echo llama1b)
TT=${T[$((I % 9))]}
[ -f data_m6/tulu_gsm/corpus_grid_olmot_final_s4.jsonl ] || $P run_m6.py corpora
$P run_m3c.py sft --setting tulu_gsm --student $F --epochs 3 --corpus grid --arms ${TT}_s0,${TT}_s1,${TT}_s2,${TT}_s3,${TT}_s4
echo "=== done ==="
