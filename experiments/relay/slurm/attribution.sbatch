#!/bin/bash
#SBATCH --job-name=cg_attr
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/relay/logs/attr_%j.out
#SBATCH --error=experiments/relay/logs/attr_%j.err

# EXP-R1: 密钥空间可分离性 -> attribution 能力
# 每个 (space, key) 落一个 checkpoint, 中断后重投即续跑.

set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python

RUN=${CG_RUN:-experiments/relay/runs/attribution}
N_PROB=${CG_N_PROBLEMS:-100}
N_KEYS=${CG_N_KEYS:-8}

echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== RUN=$RUN N_PROB=$N_PROB N_KEYS=$N_KEYS ==="

$P experiments/relay/gen_attribution.py \
  --out-dir "$RUN" --n-problems "$N_PROB" --n-keys "$N_KEYS" \
  --batch-size 16 --max-new-tokens 400

echo "=== 打分 ==="
$P experiments/relay/score_attribution.py --run-dir "$RUN"

echo "=== attribution done ==="
