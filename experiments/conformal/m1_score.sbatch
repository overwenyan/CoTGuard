#!/bin/bash
#SBATCH --job-name=cg_m1
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=experiments/conformal/logs/m1_%j.out
#SBATCH --error=experiments/conformal/logs/m1_%j.err

# M1 (.pipeline/docs/m1_design.md, pre-registered 0d08f65): option probabilities under solo and
# peer conditions for 4 primary models + Qwen3-14B, then the CPU gate. Forward passes only.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
S=experiments/conformal/score_options.py

$P $S solo --models qwen25 --limit 8          # smoke; overwritten by the full run
echo "########## solo ##########"
$P $S solo --models qwen25,llama31,mistral,gemma2,qwen3
echo "########## social ##########"
$P $S social --models qwen25,llama31,mistral,gemma2,qwen3
echo "########## gate ##########"
cd experiments/conformal && $P pccp.py --models qwen25,llama31,mistral,gemma2
echo "########## secondary: + qwen3 ##########"
$P pccp.py --models qwen3 --splits 500 | tail -25
echo "=== done ==="
