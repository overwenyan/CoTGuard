#!/bin/bash
#SBATCH --job-name=cg_main
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=experiments/relay/logs/main_%j.out
#SBATCH --error=experiments/relay/logs/main_%j.err

# 主测量作业. 每跳落 checkpoint, 超时或抢占后重投即可从断点续跑.
# 可由控制器通过 --export 覆盖 CG_* 环境变量.

set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python

RUN=${CG_RUN:-experiments/relay/runs/main}
N_PROB=${CG_N_PROBLEMS:-200}
MAX_HOPS=${CG_MAX_HOPS:-6}
STYLES=${CG_STYLES:-paraphrase,simplify,academic,guard_lexicon}
RELAY=${CG_RELAY:-Qwen/Qwen3-14B}
GEN=${CG_GEN:-allenai/Llama-3.1-Tulu-3-8B}
ROUND=${CG_ROUND:-1}

echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== RUN=$RUN N=$N_PROB HOPS=$MAX_HOPS STYLES=$STYLES ==="

$P experiments/relay/gen_and_relay.py \
  --out-dir "$RUN" --generator "$GEN" --relay-model "$RELAY" \
  --n-problems "$N_PROB" --max-hops "$MAX_HOPS" --styles "$STYLES" \
  --batch-size 16 --max-new-tokens 400

echo "=== 打分与测量 ==="
$P experiments/relay/score_and_measure.py --run-dir "$RUN" --n-calib-problems 2000

echo "=== 控制器决策 ==="
$P experiments/relay/controller.py --run-dir "$RUN" --round "$ROUND"

echo "=== main done ==="
