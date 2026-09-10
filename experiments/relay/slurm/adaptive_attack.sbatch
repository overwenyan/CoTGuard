#!/bin/bash
#SBATCH --job-name=cg_adapt
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/relay/logs/adapt_%j.out
#SBATCH --error=experiments/relay/logs/adapt_%j.err

# 队列项5: 叠加压力版对抗攻击 (Guard-Lexicon + Do-Not-Mention + Monitor-Aware).
# 单项 guard_lexicon 已证无效(10跳后仍0.4111), 升级攻击强度检验真实攻防边界.
# 两种结果都有价值: 仍无效 -> 鲁棒性主张更强; 能破坏 -> 得到真实边界.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python

echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

$P experiments/relay/relay_attribution.py \
  --out-dir experiments/relay/runs/relay_adaptive \
  --space v2_diverse --relay-model Qwen/Qwen3-14B \
  --styles adaptive_max \
  --max-hops 5 --n-problems 60 --batch-size 16 --max-new-tokens 600

echo "=== 打分 ==="
$P experiments/relay/score_relay_attribution.py --run-dir experiments/relay/runs/relay_adaptive
echo "=== 跨跳迁移(部署现实) ==="
$P experiments/relay/cross_hop_transfer.py --run-dir experiments/relay/runs/relay_adaptive --max-hops 5
echo "=== done ==="
