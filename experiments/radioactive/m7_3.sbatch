#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m7p
#SBATCH --array=0-23%4
#SBATCH --time=8:00:00
#SBATCH --output=experiments/radioactive/logs/m7p_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m7p_%A_%a.err
# M7: probes per dataset x family x teacher (+ base on the first teacher of each dataset x family).
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
DS=( gsm math ); FAMS=(qwen15 llama1b)
D=${DS[$((I / 12))]}; F=${FAMS[$(((I % 12) / 6))]}; TT=${T[$((I % 6))]}
NAMES=$([ $((I % 6)) = 0 ] && echo base, || true)
NAMES=$NAMES$(for S in $(seq 0 19); do printf "grid_%s_s%d," $TT $S; done)
$V run_m7.py probe --dataset $D --student $F --names ${NAMES%,}
echo "=== done ==="
