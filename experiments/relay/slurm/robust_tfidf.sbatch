#!/bin/bash
#SBATCH --job-name=cg_rob
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/relay/logs/rob_%j.out
#SBATCH --error=experiments/relay/logs/rob_%j.err

# EXP-B2: 用**新主方法(TF-IDF 读出)**重测多跳鲁棒性与跨跳迁移.
#
# 必要性: EXP-B1 判定学习式 embedding 读出不优于朴素 TF-IDF(四个 run 两平两负),
# 主方法已改. 而证据链第 9 环(10 跳不衰减)与第 11 环(部署现实跨跳迁移)当初都是
# 用 embedding 读出做的 —— 不重测就等于用一个已被降级的读出器支撑鲁棒性主张.
# 同时跑 embed_meanstd 作对照, 以便报告"结论是否依赖读出器选择".
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python

echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
for RUN in relay_attr_v2 relay_adaptive; do
  for RO in tfidf_word embed_meanstd; do
    echo "########## $RUN / $RO ##########"
    $P experiments/relay/cross_hop_transfer.py \
      --run-dir "experiments/relay/runs/$RUN" --readout "$RO"
  done
done
echo "=== done ==="
