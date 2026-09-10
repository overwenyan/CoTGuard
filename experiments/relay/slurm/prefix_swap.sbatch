#!/bin/bash
#SBATCH --job-name=cg_n2
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/relay/logs/n2_%j.out
#SBATCH --error=experiments/relay/logs/n2_%j.err

# EXP-N2: 指纹由密钥持续条件化维持, 还是由已生成前缀自维持?
#
# EXP-C4 判定载体是**被诱导的行为**(指令合规)而非表层文体. 本实验问该行为靠什么维持:
#   持续条件化 -> 去掉密钥重新生成即可清除
#   前缀自维持 -> 指纹会随轨迹被截断/摘录而**继续传播**, 溯源含义完全不同
#
# 三条件: matched(key_i+prefix_i) / swapped(key_i+prefix_j) / nokey(无密钥+prefix_j).
# 切换密钥时**重新拼完整 prompt 整体重算**, 不复用任何 KV cache.
# 续写单独打分不含前缀; 读出器训练在原始轨迹尾部(第 m 步之后), 按题目划分.
#
# 用 anchor_only 空间 —— EXP-C4 已证明 persona_only 无信号可测, 在其上做交换无意义.
#
# 可证伪预测(先写下, 免得事后合理化):
#   持续条件化: swapped 的 acc_key >> acc_prefix, 且 nokey 塌到随机
#   前缀自维持: swapped 的 acc_prefix >= acc_key, 且 nokey 仍显著高于随机
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python

echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
for M in 2 4; do
  RUN=experiments/relay/runs/prefix_swap_m$M
  echo "########## prefix-steps=$M ##########"
  $P experiments/relay/prefix_swap.py \
    --src-dir experiments/relay/runs/attr_channels --space anchor_only \
    --out-dir "$RUN" --prefix-steps $M \
    --n-problems 80 --batch-size 16 --max-new-tokens 500
  $P experiments/relay/score_prefix_swap.py --run-dir "$RUN"
done
echo "=== done ==="
