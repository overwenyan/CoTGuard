#!/bin/bash
#SBATCH --job-name=cg_smoke
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/relay/logs/smoke_%j.out
#SBATCH --error=experiments/relay/logs/smoke_%j.err

# 验证 think-mode 修复: 小规模跑通, 确认 <think> 截断问题已解决.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python

echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

echo "--- 冒烟1: gen_attribution.py 用 Qwen3-14B (原 EXP-R2 污染源) ---"
$P experiments/relay/gen_attribution.py \
  --out-dir experiments/relay/runs/smoke_gen \
  --generator Qwen/Qwen3-14B \
  --n-problems 8 --n-keys 2 --batch-size 8 --max-new-tokens 600

echo "--- 冒烟2: relay_attribution.py 用 Qwen3-14B (替换原 DeepSeek-R1 污染源) ---"
$P experiments/relay/relay_attribution.py \
  --out-dir experiments/relay/runs/smoke_relay \
  --space v2_diverse --relay-model Qwen/Qwen3-14B \
  --styles paraphrase --max-hops 2 --n-problems 8 --batch-size 8 --max-new-tokens 600

echo "=== smoke done ==="
