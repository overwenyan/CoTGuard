#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m13p2
#SBATCH --array=0-5%3
#SBATCH --time=6:00:00
#SBATCH --output=experiments/radioactive/logs/m13p2_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m13p2_%A_%a.err
# M13 correction 1: probes for the 42 added reference students (7 per teacher).
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
V=/local/projects-t3/primelab/yan.wen/venvs/vllm/bin/python
export PATH=/local/projects-t3/primelab/yan.wen/venvs/vllm/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export M7_OUT=$PWD/data_m7 M3C_OUT=$PWD/data_m7
T=(tulu_sft tulu_dpo tulu_rlvr olmoi_sft olmoi_dpo olmoi_final)
I=$SLURM_ARRAY_TASK_ID; TT=${T[$I]}
NAMES=$(for S in 0 1 2 3 4 8 9; do printf "grid_%s_s%d," $TT $S; done)
$V run_m7.py probe --dataset gsm --student qwen7b --names ${NAMES%,}
echo "=== done ==="
