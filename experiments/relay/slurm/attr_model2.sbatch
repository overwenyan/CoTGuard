#!/bin/bash
#SBATCH --job-name=cg_attr2
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/relay/logs/attr2_%j.out
#SBATCH --error=experiments/relay/logs/attr2_%j.err

# EXP-R2: 第二生成模型家族. 用于 (a) 跨模型迁移检验 trigger 遵循度是否语义性,
# (b) 补齐"3 个模型"的经验门槛, (c) 检验主结论对生成模型的稳健性.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python

echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
$P experiments/relay/gen_attribution.py \
  --out-dir experiments/relay/runs/attr_qwen3_v2 \
  --generator Qwen/Qwen3-14B \
  --n-problems 100 --n-keys 8 --batch-size 12 --max-new-tokens 600

echo "=== 打分 ==="
$P experiments/relay/score_attribution.py --run-dir experiments/relay/runs/attr_qwen3_v2
$P experiments/relay/key_identifiability.py --run-dir experiments/relay/runs/attr_qwen3_v2
echo "=== done ==="
