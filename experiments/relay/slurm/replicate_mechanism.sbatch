#!/bin/bash
#SBATCH --job-name=cg_repl
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/relay/logs/repl_%j.out
#SBATCH --error=experiments/relay/logs/repl_%j.err

# EXP-REP: 把"单模型单数据集单中继"的机制层结果复现到其他生成器与第二个中继模型.
#
# 审计结论(2026-09-10): 读出对比(B1)已在 3 模型 x 3 数据集复现; 但现在撑起故事的机制层
# —— 长度混淆(R0)、persona 无信号(C4)、10 跳鲁棒(S12)、攻击平台(A1)、前缀交换(N2)、
# M1 冗余机制 —— 全部只在 Tulu-3-8B x GSM8K x Qwen3-14B 中继 x 单次生成上做过.
# 用户决定: 在任何新方法依赖这些结果之前先复现. 本脚本 ~5 GPU 小时.
#
# 提交方式(一卡纪律, 挂在 D1 之后):
#   sbatch --dependency=afterany:20040940 experiments/relay/slurm/replicate_mechanism.sbatch
#
# 顺序按"最便宜且最被引用的先跑": E 长度混淆 -> A 通道消融 -> B 两族 anchor(M1/D1)
# -> D 前缀交换(依赖 A) -> C 第二中继模型的攻击与良性改写.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
R=experiments/relay/runs
QWEN=Qwen/Qwen3-14B
MIS=mistralai/Mistral-7B-Instruct-v0.3

echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

# ---------- E. 长度混淆 (R0) 复现: hop0 triggered vs clean, length-matched AUROC ----------
for G in $QWEN $MIS; do
  TAG=$(basename "$G" | tr 'A-Z' 'a-z' | cut -d- -f1)
  BS=$([ "$G" = "$QWEN" ] && echo 12 || echo 16)
  echo "########## E: length confound on $G ##########"
  $P experiments/relay/gen_and_relay.py \
    --out-dir "$R/pilot_$TAG" --generator "$G" --relay-model "$G" \
    --n-problems 200 --max-hops 0 --batch-size "$BS" --max-new-tokens 600
  $P experiments/relay/discriminative_probe.py --run-dir "$R/pilot_$TAG" --hop 0 --caliper 1
done

# ---------- A. 通道消融 (C4) 复现: persona_only vs anchor_only ----------
for G in $QWEN $MIS; do
  TAG=$(basename "$G" | tr 'A-Z' 'a-z' | cut -d- -f1)
  BS=$([ "$G" = "$QWEN" ] && echo 12 || echo 16)
  RUN="$R/attr_channels_$TAG"
  echo "########## A: channels on $G ##########"
  $P experiments/relay/gen_attribution.py \
    --out-dir "$RUN" --spaces channels --generator "$G" \
    --n-problems 100 --n-keys 8 --batch-size "$BS" --max-new-tokens 600
  $P experiments/relay/score_attribution.py --run-dir "$RUN"
  $P experiments/relay/key_identifiability.py --run-dir "$RUN"
  for SP in persona_only anchor_only; do
    $P experiments/relay/baselines.py --run-dir "$RUN" --space "$SP"
    $P experiments/relay/fingerprint_anatomy.py --run-dir "$RUN" --space "$SP"
  done
done

# ---------- B. 两族 anchor (D1/M1) 复现: lexical vs structural + 冗余机制 ----------
for G in $QWEN $MIS; do
  TAG=$(basename "$G" | tr 'A-Z' 'a-z' | cut -d- -f1)
  BS=$([ "$G" = "$QWEN" ] && echo 12 || echo 16)
  RUN="$R/attr_families_$TAG"
  echo "########## B: families on $G ##########"
  $P experiments/relay/gen_attribution.py \
    --out-dir "$RUN" --spaces families --generator "$G" \
    --n-problems 100 --n-keys 8 --batch-size "$BS" --max-new-tokens 600
  $P experiments/relay/verify_redundant.py --run-dir "$RUN" --min-gap 2
  for SP in lexical_anchor structural_anchor; do
    $P experiments/relay/baselines.py --run-dir "$RUN" --space "$SP"
    $P experiments/relay/fingerprint_anatomy.py --run-dir "$RUN" --space "$SP"
  done
done

# ---------- D. 前缀交换 (N2) 复现 on Qwen3 (依赖 A 的 attr_channels_qwen3) ----------
echo "########## D: prefix swap on $QWEN ##########"
$P experiments/relay/prefix_swap.py \
  --src-dir "$R/attr_channels_qwen3" --space anchor_only \
  --out-dir "$R/prefix_swap_qwen3_m3" --generator "$QWEN" --prefix-steps 3 \
  --n-problems 80 --batch-size 12 --max-new-tokens 500
$P experiments/relay/score_prefix_swap.py --run-dir "$R/prefix_swap_qwen3_m3"

# ---------- C. 第二中继模型 (Mistral) 上的攻击平台 (A1) 与良性改写 (S12, 5 跳) ----------
echo "########## C1: adaptive_max with Mistral relay ##########"
$P experiments/relay/relay_attribution.py \
  --src-dir "$R/attribution" --out-dir "$R/relay_adaptive_mistral" \
  --space v2_diverse --relay-model "$MIS" --styles adaptive_max \
  --max-hops 5 --n-problems 60 --batch-size 16 --max-new-tokens 600
$P experiments/relay/cross_hop_transfer.py \
  --run-dir "$R/relay_adaptive_mistral" --readout tfidf_word --max-hops 5

echo "########## C2: benign paraphrase with Mistral relay ##########"
$P experiments/relay/relay_attribution.py \
  --src-dir "$R/attribution" --out-dir "$R/relay_benign_mistral" \
  --space v2_diverse --relay-model "$MIS" --styles paraphrase \
  --max-hops 5 --n-problems 60 --batch-size 16 --max-new-tokens 600
$P experiments/relay/cross_hop_transfer.py \
  --run-dir "$R/relay_benign_mistral" --readout tfidf_word --max-hops 5

echo "=== done ==="
