#!/bin/bash
#SBATCH --job-name=cg_attr3
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/relay/logs/attr3_%j.out
#SBATCH --error=experiments/relay/logs/attr3_%j.err

# EXP-R5: **第三生成模型家族** —— 补齐经验门槛里唯一剩下的硬缺口.
#
# 选 Mistral-7B-Instruct-v0.3 而非 OLMo-2-7B-Instruct, 理由是避免后训练配方混淆:
# 现有两个模型是 Tulu-3-8B(Llama-3.1 架构) 与 Qwen3-14B(Qwen 架构), 而
# OLMo-2-*-Instruct 与 Tulu-3 共用 allenai 的 Tulu 3 后训练配方 —— 若 per-key 签名
# 部分来自后训练风格, 用 OLMo 就无法把"第三个家族"与"第三个配方"分开.
# Mistral 在**预训练与后训练两方面都独立**, 才是干净的第三家族.
# 同时它是非推理模型, 不涉及 <think> 截断污染(见 EXP-R2/R3 事故复盘).
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python

echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
RUN=experiments/relay/runs/attr_mistral
$P experiments/relay/gen_attribution.py \
  --out-dir "$RUN" \
  --generator mistralai/Mistral-7B-Instruct-v0.3 \
  --n-problems 100 --n-keys 8 --batch-size 12 --max-new-tokens 600

echo "=== 打分 ==="
$P experiments/relay/score_attribution.py --run-dir "$RUN"
$P experiments/relay/key_identifiability.py --run-dir "$RUN"
echo "=== baseline 对比(v2 协议) ==="
$P experiments/relay/baselines.py --run-dir "$RUN" --space v2_diverse
echo "=== done ==="
