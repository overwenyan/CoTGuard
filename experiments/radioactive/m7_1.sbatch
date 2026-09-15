#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m7g
#SBATCH --array=0-11%3
#SBATCH --time=4:00:00
#SBATCH --output=experiments/radioactive/logs/m7g_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m7g_%A_%a.err
# M7 (pre-registered 4fa5040): teacher traces; tasks 0-5 gsm POOL_TEST, 6-11 math R300+POOL_REF+POOL_TEST. prep run once beforehand.
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
if [ $I -lt 6 ]; then $V run_m7.py gen --dataset gsm --teacher ${T[$I]} --parts test
else $V run_m7.py gen --dataset math --teacher ${T[$((I - 6))]} --parts ref,test; fi
echo "=== done ==="
