#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m4t
#SBATCH --array=0-3
#SBATCH --time=4:00:00
#SBATCH --output=experiments/radioactive/logs/m4t_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m4t_%A_%a.err
# M4 pilot: students + probes. 0 = key100 b1-3; 1 = imit100 b1-3; 2 = key10 b1-3; 3 = clean2k + dil0_clean probes.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
M="$P run_m4.py"; C="$P run_m3c.py"; S="--setting tulu_gsm --student qwen15"
export M3C_OUT=$PWD/data_m4
case $SLURM_ARRAY_TASK_ID in
0) $C sft $S --epochs 3 --corpus key100 --arms b1,b2,b3; $M probe --names key100_b1,key100_b2,key100_b3 ;;
1) $C sft $S --epochs 3 --corpus imit100 --arms b1,b2,b3; $M probe --names imit100_b1,imit100_b2,imit100_b3 ;;
2) $C sft $S --epochs 1 --corpus key10 --arms b1,b2,b3; $M probe --names key10_b1,key10_b2,key10_b3 ;;
3) $C sft $S --epochs 3 --corpus clean2k --arms clean; $M probe --names clean2k_clean,dil0_clean ;;
esac
echo "=== done ==="
