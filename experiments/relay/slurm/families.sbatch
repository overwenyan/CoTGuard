#!/bin/bash
#SBATCH --job-name=cg_d1
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/relay/logs/d1_%j.out
#SBATCH --error=experiments/relay/logs/d1_%j.err

# EXP-D1 (**方法侧主实验**): 把密钥编码进推理的结构, 而非措辞.
#
# 由 EXP-C4 直接推出: 可归因信号来自 anchor(指令合规行为)而非 persona.
# 那么 anchor 规定什么就决定了信号的性质:
#   LEX 规定"写什么"(用词/标签) -> 模型照抄 -> 字面残留高 -> 一擦就掉
#   STR 规定"按什么顺序算什么"   -> 无字面可抄 -> 但任何**保答案**的改写都必须保留
#                                  计算顺序 -> 应显著更抗擦洗
#
# 可证伪预测(先写下, 免得事后合理化):
#   hop0   : LEX >= STR(LEX 多一条字面通道)
#   擦洗后 : STR 保留率 >> LEX 保留率      <-- **本设计主张的成败在此**
#   解剖   : STR 的判别 n-gram 与密钥文本重叠 ≈ 错配基线; LEX 显著高于基线
#   若 STR 保留率不高于 LEX, 设计主张被证伪.
#
# 同时测 utility: STR 强制次优计算顺序, 很可能更伤正确率 —— 那是这条设计的**代价**,
# 与抗擦洗性一起构成可用的权衡曲线, 必须如实报告而非藏起来.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
RUN=experiments/relay/runs/attr_families

echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
$P experiments/relay/gen_attribution.py \
  --out-dir "$RUN" --spaces families \
  --generator allenai/Llama-3.1-Tulu-3-8B \
  --n-problems 100 --n-keys 8 --batch-size 16 --max-new-tokens 600

echo "########## 施加 adaptive_max 擦洗 ##########"
for SP in lexical_anchor structural_anchor; do
  $P experiments/relay/relay_attribution.py \
    --src-dir "$RUN" --out-dir "experiments/relay/runs/fam_relay_$SP" \
    --space "$SP" --styles adaptive_max --max-hops 5 \
    --n-problems 60 --batch-size 16 --max-new-tokens 600
done

echo "########## 打分 ##########"
$P experiments/relay/score_attribution.py --run-dir "$RUN"
$P experiments/relay/key_identifiability.py --run-dir "$RUN"
for SP in lexical_anchor structural_anchor; do
  echo "--- $SP ---"
  $P experiments/relay/baselines.py --run-dir "$RUN" --space "$SP"
  $P experiments/relay/fingerprint_anatomy.py --run-dir "$RUN" --space "$SP"
  $P experiments/relay/cross_hop_transfer.py \
    --run-dir "experiments/relay/runs/fam_relay_$SP" --readout tfidf_word --max-hops 5
done
echo "########## utility 代价 ##########"
$P experiments/relay/utility_check.py --run-dir "$RUN" || true
echo "=== done ==="
