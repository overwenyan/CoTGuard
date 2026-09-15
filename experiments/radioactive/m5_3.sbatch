#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m5t
#SBATCH --array=0-6
#SBATCH --time=4:00:00
#SBATCH --output=experiments/radioactive/logs/m5t_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m5t_%A_%a.err
# M5 §1: grid students for one teacher (Qwen-1.5B and Llama-1B, 5 corpora each) + probes.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
M="$P run_m5.py"; C="$P run_m3c.py"
ORDER=(tulu_sft tulu_dpo llama31_8b qwen25_7b qwen3_14b mistral_7b gemma2_9b)
export M5_OUT=$PWD/data_m5 M3C_OUT=$PWD/data_m5
T=${ORDER[$SLURM_ARRAY_TASK_ID]}
ARMS=${T}_s0,${T}_s1,${T}_s2,${T}_s3,${T}_s4
for F in qwen15 llama1b; do
  $C sft --setting tulu_gsm --student $F --epochs 3 --corpus grid --arms $ARMS
  $M probe --names $(for S in 0 1 2 3 4; do printf "%s:grid_%s_s%s," $F $T $S; done | sed 's/,$//')
done
echo "=== done ==="
