#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m7t
#SBATCH --array=0-83%7
#SBATCH --time=10:00:00
#SBATCH --output=experiments/radioactive/logs/m7t_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m7t_%A_%a.err
# M7: 420 LoRA students, 5 per task. Task layout: fam (2) x teacher (6) x groups; gsm groups s5-9,s10-14,s15-19; math s0-4,s5-9,s10-14,s15-19 (7 groups) -> 84 tasks.
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
T=(tulu_sft tulu_dpo tulu_rlvr olmoi_sft olmoi_dpo olmoi_final)
I=$SLURM_ARRAY_TASK_ID
FAMS=(qwen15 llama1b)
F=${FAMS[$((I / 42))]}; J=$((I % 42)); TT=${T[$((J / 7))]}; G=$((J % 7))
if [ $G -lt 3 ]; then SET=tulu_gsm; S0=$((5 + 5 * G)); else SET=tulu_math; S0=$((5 * (G - 3))); fi
[ -f data_m7/tulu_math/corpus_grid_olmoi_final_s19.jsonl ] || { echo "corpora missing"; exit 1; }
ARMS=$(for S in $(seq $S0 $((S0 + 4))); do printf "%s_s%d," $TT $S; done)
$P run_m3c.py sft --setting $SET --student $F --epochs 3 --corpus grid --arms ${ARMS%,}
echo "=== done ==="
