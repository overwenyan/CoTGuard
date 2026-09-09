#!/bin/bash
#SBATCH --job-name=cg_relay
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=experiments/relay/logs/relay_%j.out
#SBATCH --error=experiments/relay/logs/relay_%j.err

# EXP-R3: per-key 签名能否撑过多跳语义改写(决定论文前提)
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python

echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
$P experiments/relay/relay_attribution.py \
  --out-dir experiments/relay/runs/relay_attr \
  --space v2_diverse \
  --relay-model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --styles paraphrase,guard_lexicon \
  --max-hops 4 --n-problems 60 --batch-size 16

echo "=== 打分 ==="
$P experiments/relay/score_relay_attribution.py --run-dir experiments/relay/runs/relay_attr
echo "=== done ==="
