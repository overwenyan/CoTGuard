#!/bin/bash
#SBATCH --job-name=cg_m2b
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=experiments/collusion/logs/m2b_%j.out
#SBATCH --error=experiments/collusion/logs/m2b_%j.err

# M2 v1 Part B (m2_design.md v1, amendments d6aafd8 / 0ba1e4c): NARC-Clean generation with 3 generators,
# Qwen3-14B LLM-judge baseline, then the pre-registered Gate B analysis (CPU).
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/collusion
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python

rm -rf narc_clean_smoke
NARC_OUT_SMOKE=1 $P - <<'PY'
import narc_clean as N, pathlib
N.OUT = pathlib.Path("narc_clean_smoke")
N.simulate("qwen25", ["deception__audit_01"])
PY
rm -rf narc_clean_smoke
echo "########## generate ##########"
$P narc_clean.py --gens qwen3,qwen25,llama31
echo "########## judge ##########"
$P judge_collusion.py
echo "########## gate B ##########"
$P analyze_clean.py
echo "=== done ==="
