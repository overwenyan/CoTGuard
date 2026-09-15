#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m4s
#SBATCH --time=1:30:00
#SBATCH --output=experiments/radioactive/logs/m4s_%j.out
#SBATCH --error=experiments/radioactive/logs/m4s_%j.err
# M4 pilot (pre-registered 4578207): end-to-end smoke on 40 problems.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
M="$P run_m4.py"; C="$P run_m3c.py"; S="--setting tulu_gsm --student qwen15"
export M4_SMOKE=1 M3C_OUT=$PWD/data_m4_smoke
rm -rf data_m4_smoke
for I in 0 1 2 3; do $M gen --shard $I; done
$M build
$C sft $S --epochs 1 --corpus key100 --arms b1,b3
$C sft $S --epochs 1 --corpus imit100 --arms b1
$C sft $S --epochs 1 --corpus key10 --arms b1
$C sft $S --epochs 1 --corpus clean2k --arms clean
$M probe --names key100_b1,key100_b3,imit100_b1,key10_b1,clean2k_clean,dil0_clean
$P score_m4.py
rm -rf data_m4_smoke
echo "=== smoke ok ==="
