#!/bin/bash
#SBATCH --job-name=cg_math
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/relay/logs/math_%j.out
#SBATCH --error=experiments/relay/logs/math_%j.err

# 队列项3: 第二数据集 MATH-500, 检验核心发现是否跨数据集稳健.
# 核心发现 = 朴素文本相似度读出≈随机, 学习式读出显著恢复密钥身份.
# 生成方用 Tulu-3-8B(已验证无 think 污染), 与 EXP-R1 保持一致以便直接对照.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python

echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

$P experiments/relay/gen_attribution.py \
  --out-dir experiments/relay/runs/attr_math500 \
  --generator allenai/Llama-3.1-Tulu-3-8B \
  --dataset math500 \
  --n-problems 100 --n-keys 8 --batch-size 16 --max-new-tokens 600

echo "=== 打分 ==="
$P experiments/relay/score_attribution.py --run-dir experiments/relay/runs/attr_math500
$P experiments/relay/key_identifiability.py --run-dir experiments/relay/runs/attr_math500
echo "=== done ==="
