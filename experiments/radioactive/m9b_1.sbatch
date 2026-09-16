#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m9br
#SBATCH --array=0-7%3
#SBATCH --time=3:00:00
#SBATCH --output=experiments/radioactive/logs/m9br_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m9br_%A_%a.err
# M9b (pre-registered 3eed3e4): strict imitation rewrite for one (owner, target) attack.
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
$V run_m9b.py rewrite --attack $SLURM_ARRAY_TASK_ID
echo "=== done ==="
