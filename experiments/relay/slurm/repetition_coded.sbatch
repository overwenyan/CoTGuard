#!/bin/bash
#SBATCH --job-name=cg_m2
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=experiments/relay/logs/m2_%j.out
#SBATCH --error=experiments/relay/logs/m2_%j.err

# EXP-M2: 重复编码的结构化锚点 —— 把 N2 的前缀弱残留从副作用变成设计目标.
#
# 分块生成, 每块之间在文本流中重新插入同一 anchor 的提醒(而非只在开头系统提示出现一次).
# 可证伪预测: 若重复编码有效, 中段片段(去掉开头/结尾)上的归因应显著高于 D1 的单次注入
# (structural_anchor). 若无显著差异, M2 的设计动机不成立.
#
# 注: gen_repetition_coded.py 的分块拼接逻辑在提交前做过一次静态复核并修了一个真实 bug
# (旧版本会把已生成内容通过滚动累加 running 变量重复拼接两次, 链条越长重复越严重).
# 由于工具环境阻塞导致无法本地试跑(python 执行被安全分类器拦截), 此次提交前**没有**
# 动态测试确认, 提交后应尽早人工检查前几条落盘轨迹是否有异常重复文本.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
RUN=experiments/relay/runs/repetition_coded

echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
$P experiments/relay/gen_repetition_coded.py \
  --out-dir "$RUN" --generator allenai/Llama-3.1-Tulu-3-8B \
  --n-problems 80 --n-keys 8 --n-chunks 3 --chunk-new-tokens 200 --batch-size 16

echo "########## 打分: 单次注入 vs 重复编码, 中段片段对比 ##########"
$P experiments/relay/score_repetition_coded.py \
  --single-run experiments/relay/runs/attr_families --single-space structural_anchor \
  --repeat-run "$RUN" --repeat-space repetition_structural

echo "=== done ==="
