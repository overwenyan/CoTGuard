#!/bin/bash
#SBATCH --job-name=cg_svra1
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/relay/logs/svra1_%j.out
#SBATCH --error=experiments/relay/logs/svra1_%j.err

# SVRA honest agents (p2_design.md v2 §5.1): 7 samples/problem × {obligated, unobligated}
# × {GSM8K, MATH-500 numeric} × {Tulu, Mistral, Qwen3}. max_new_tokens=1200 because the
# compute-twice obligation hit a 600-token cap on 28% of Qwen3 traces in S20.
# Cheapest generators first so a time-limit hit loses the least. A 2-problem smoke run first.
# Generators default to those that passed G0-v2 (Tulu, Qwen3). Mistral failed G0-v2 at 600 tokens
# (no k passes both K1' and K2'); add it only by explicit decision: GENS="... mistralai/..." sbatch ...
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
R=experiments/relay/runs/svra_honest

$P experiments/relay/gen_svra_honest.py --out-dir "$R/_smoke" --generator "Qwen/Qwen3-14B" \
  --n-problems 2 --n-samples 1 --max-new-tokens 64 --batch-size 2
rm -rf "$R/_smoke"

GENS="${GENS:-allenai/Llama-3.1-Tulu-3-8B Qwen/Qwen3-14B}"
for G in $GENS; do
  TAG=$(basename "$G" | cut -d- -f1 | tr 'A-Z' 'a-z')
  BS=$([ "$G" = "Qwen/Qwen3-14B" ] && echo 12 || echo 16)
  for D in gsm8k math500; do
    echo "########## $G / $D ##########"
    $P experiments/relay/gen_svra_honest.py --out-dir "$R/${TAG}_${D}" --generator "$G" \
      --dataset "$D" --n-problems 100 --n-samples 7 --max-new-tokens 1200 --batch-size "$BS"
  done
done
echo "=== done ==="
