#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m10g
#SBATCH --array=0-1
#SBATCH --time=4:00:00
#SBATCH --output=experiments/radioactive/logs/m10g_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m10g_%A_%a.err
# M10 (pre-registered): Zephyr teacher traces, ref + test.
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
$V run_m10.py gen --teacher ${T[$SLURM_ARRAY_TASK_ID]} --parts ref,test
echo "=== done ==="
