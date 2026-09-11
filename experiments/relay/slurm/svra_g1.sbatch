#!/bin/bash
#SBATCH --job-name=cg_svra_g1
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=experiments/relay/logs/svra_g1_%j.out
#SBATCH --error=experiments/relay/logs/svra_g1_%j.err

# SVRA gate G1 (pre-registered in p2_design.md §5.0, commit 8823973): coordinated adversaries on
# GSM8K x {Tulu, Qwen3}, then STAR-style and LLM-judge baselines on Qwen3-14B, then the CPU verdict.
# Decision rule: continue P2 iff SVRA passes G1-I or G1-M on both generators.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
A=experiments/relay/runs/svra_adv

$P experiments/relay/gen_svra_adv.py --out-dir "$A/_smoke" --generator Qwen/Qwen3-14B \
  --n-problems 2 --n-samples 1 --max-new-tokens 32 --batch-size 2
rm -rf "$A/_smoke"

echo "########## adversaries: Tulu ##########"
$P experiments/relay/gen_svra_adv.py --out-dir "$A/llama" --generator allenai/Llama-3.1-Tulu-3-8B --batch-size 16
echo "########## adversaries: Qwen3 ##########"
$P experiments/relay/gen_svra_adv.py --out-dir "$A/qwen3" --generator Qwen/Qwen3-14B --batch-size 12

echo "########## LLM baselines: smoke ##########"
$P experiments/relay/llm_aggregators.py star --limit 4
$P experiments/relay/llm_aggregators.py judge --limit 2
rm -f "$A"/_smoke_*

echo "########## LLM baselines: STAR-style ##########"
$P experiments/relay/llm_aggregators.py star
echo "########## LLM baselines: judge ##########"
$P experiments/relay/llm_aggregators.py judge

echo "########## G1 verdict ##########"
cd experiments/relay && $P svra_adv.py aggregate --k 1
echo "=== done ==="
