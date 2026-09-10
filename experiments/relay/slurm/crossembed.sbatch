#!/bin/bash
#SBATCH --job-name=cg_xemb
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/relay/logs/xemb_%j.out
#SBATCH --error=experiments/relay/logs/xemb_%j.err

# 队列项4: 第二 embedding (gte-base) 交叉验证, 确认核心结论不依赖单一打分器.
# 之前在登录节点 CPU 上跑超时(50min 未完成 MATH-500), 改用 GPU 节点.
# key_identifiability.py 的 embedding 缓存按密钥空间增量保存, 会自动复用已完成部分.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python

echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

for RUN in attr_math500 attr_qwen3_v2; do
  echo "--- gte-base on $RUN ---"
  $P experiments/relay/key_identifiability.py \
    --run-dir "experiments/relay/runs/$RUN" \
    --model thenlper/gte-base
done

echo "=== xemb done ==="
