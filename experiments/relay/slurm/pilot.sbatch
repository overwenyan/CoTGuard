#!/bin/bash
#SBATCH --job-name=cg_pilot
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/relay/logs/pilot_%j.out
#SBATCH --error=experiments/relay/logs/pilot_%j.err

set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python

echo "=== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Pilot: 小规模验证管线跑通 + 测单位耗时
$P experiments/relay/gen_and_relay.py \
  --out-dir experiments/relay/runs/pilot \
  --generator allenai/Llama-3.1-Tulu-3-8B \
  --relay-model Qwen/Qwen3-14B \
  --n-problems 24 --max-hops 3 \
  --styles paraphrase,guard_lexicon \
  --batch-size 8 --max-new-tokens 320

echo "=== 生成完成, 开始打分 ==="
$P experiments/relay/score_and_measure.py \
  --run-dir experiments/relay/runs/pilot \
  --n-calib-problems 800

echo "=== pilot done ==="
