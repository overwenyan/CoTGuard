#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m3f3
#SBATCH --array=0-4
#SBATCH --time=5:00:00
#SBATCH --output=experiments/radioactive/logs/m3f3_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m3f3_%A_%a.err
# M3 v6: students. Tasks 0-3 = one key each (25%, 50%, 10% x3 epochs, 10% full FT); task 4 = regime-matched 0% controls.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
export M3C_OUT=$PWD/data4
D="$P run_m3d.py"; C="$P run_m3c.py"
KEYS=(o12 o17 p07 p13)
I=$SLURM_ARRAY_TASK_ID
S="--setting tulu_gsm --student qwen15"
if [ $I = 4 ]; then
  $C sft $S --epochs 3 --corpus dil0e3 --arms clean;        $C student $S --n-test 1319 --corpus dil0e3 --arms clean
  $C sft $S --epochs 1 --full --corpus dil0ft --arms clean;  $C student $S --n-test 1319 --corpus dil0ft --arms clean
else
  K=${KEYS[$I]}
  $C sft $S --epochs 1 --corpus dil25 --arms $K;            $C student $S --n-test 1319 --corpus dil25 --arms $K
  $C sft $S --epochs 1 --corpus dil50 --arms $K;            $C student $S --n-test 1319 --corpus dil50 --arms $K
  $C sft $S --epochs 3 --corpus dil10e3 --arms $K;          $C student $S --n-test 1319 --corpus dil10e3 --arms $K
  $C sft $S --epochs 1 --full --corpus dil10ft --arms $K;   $C student $S --n-test 1319 --corpus dil10ft --arms $K
fi
echo "=== done ==="
