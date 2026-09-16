#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m10t
#SBATCH --array=0-7%7
#SBATCH --time=6:00:00
#SBATCH --output=experiments/radioactive/logs/m10t_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m10t_%A_%a.err
# M10: 80 LoRA students, 5 per task (2 teachers x 2 families x 2 groups of 10).
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
V=/local/projects-t3/primelab/yan.wen/venvs/vllm/bin/python
export PATH=/local/projects-t3/primelab/yan.wen/venvs/vllm/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export M7_OUT=$PWD/data_m7 M3C_OUT=$PWD/data_m7 M3C_MAXLEN=4608 M10_ZEPHYR=1
T=(zephyr_sft zephyr_dpo)
I=$SLURM_ARRAY_TASK_ID
F=$([ $I -lt 4 ] && echo qwen15 || echo llama1b); J=$((I % 4))
TT=${T[$((J / 2))]}; G=$((J % 2))
[ -f data_m7/tulu_gsm/corpus_grid_zephyr_dpo_s19.jsonl ] || $P run_m10.py corpora
ARMS=$(for S in $(seq $((10 * G)) $((10 * G + 9))); do printf "%s_s%d," $TT $S; done)
$P run_m3c.py sft --setting tulu_gsm --student $F --epochs 3 --corpus grid --arms ${ARMS%,}
echo "=== done ==="
