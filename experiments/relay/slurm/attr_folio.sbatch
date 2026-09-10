#!/bin/bash
#SBATCH --job-name=cg_folio
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/relay/logs/folio_%j.out
#SBATCH --error=experiments/relay/logs/folio_%j.err

# 第三数据集 FOLIO (一阶逻辑推理). 选逻辑而非又一数学集, 用于检验 MATH-500 上
# 观察到的"效应量域依赖"是数学/LaTeX 特有, 还是"形式化程度"的普遍规律.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python

echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
$P experiments/relay/gen_attribution.py \
  --out-dir experiments/relay/runs/attr_folio \
  --generator allenai/Llama-3.1-Tulu-3-8B \
  --dataset folio \
  --n-problems 100 --n-keys 8 --batch-size 16 --max-new-tokens 600

echo "=== 打分 ==="
$P experiments/relay/score_attribution.py --run-dir experiments/relay/runs/attr_folio
$P experiments/relay/key_identifiability.py --run-dir experiments/relay/runs/attr_folio
echo "=== gte-base 交叉验证 ==="
$P experiments/relay/key_identifiability.py --run-dir experiments/relay/runs/attr_folio --model thenlper/gte-base
echo "=== done ==="
