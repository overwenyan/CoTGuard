#!/bin/bash
#SBATCH --job-name=cg_chan
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=experiments/relay/logs/chan_%j.out
#SBATCH --error=experiments/relay/logs/chan_%j.err

# EXP-C4: 两通道假说的直接检验.
#
# EXP-C3 发现指纹有两条来源不同的通道, 而 v2 的 trigger 恰由对应两段构成:
#   C1 词法通道 <- ANCHOR("restate the target"): 字面漏出, 判别 n-gram 与密钥文本
#      重叠 0.27-0.77, 且随指令遵循能力增强(Qwen3 0.771 >> Tulu 0.312)
#   C2 特异文体通道 <- PERSONA("chess coach"): 无字面痕迹, 判别功能词与密钥文本
#      重叠恒定 0.11-0.16(四个 run), 即基本为零
#
# 本实验把两段拆开各自成空间, 其余设置完全一致, 然后施加 adaptive_max(其规则 2
# 正是"不得复述任何指令/角色", 即针对 C1 的词汇擦洗).
#
# **可证伪预测(先写下, 免得事后合理化)**:
#   anchor_only : hop0 归因更高, 但在 adaptive_max 下**塌陷**
#   persona_only: hop0 归因更低, 但在同样攻击下**基本不动**
#   若两者衰减幅度无差异 -> 两通道假说被证伪, 须如实记录.
#
# 顺带检验一条推论: Qwen3 的 7.44x 领先若主要来自 C1, 则它在该攻击下应塌得比 Tulu 更狠.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
RUN=experiments/relay/runs/attr_channels

echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "########## 生成 hop0: persona_only / anchor_only ##########"
$P experiments/relay/gen_attribution.py \
  --out-dir "$RUN" --spaces channels \
  --generator allenai/Llama-3.1-Tulu-3-8B \
  --n-problems 100 --n-keys 8 --batch-size 16 --max-new-tokens 600

echo "########## 施加 adaptive_max 攻击 ##########"
for SP in persona_only anchor_only; do
  $P experiments/relay/relay_attribution.py \
    --src-dir "$RUN" --out-dir "experiments/relay/runs/chan_relay_$SP" \
    --space "$SP" --styles adaptive_max --max-hops 5 \
    --n-problems 60 --batch-size 16 --max-new-tokens 600
done

echo "########## 打分 ##########"
$P experiments/relay/score_attribution.py --run-dir "$RUN"
$P experiments/relay/key_identifiability.py --run-dir "$RUN"   # 无 --space, 一次跑全部
for SP in persona_only anchor_only; do
  echo "--- $SP ---"
  $P experiments/relay/baselines.py --run-dir "$RUN" --space "$SP"
  $P experiments/relay/fingerprint_anatomy.py --run-dir "$RUN" --space "$SP"
  $P experiments/relay/cross_hop_transfer.py \
    --run-dir "experiments/relay/runs/chan_relay_$SP" --readout tfidf_word --max-hops 5
done
echo "=== done ==="
