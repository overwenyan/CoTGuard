#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m3c3
#SBATCH --time=5:00:00
#SBATCH --output=experiments/radioactive/logs/m3c3_%j.out
#SBATCH --error=experiments/radioactive/logs/m3c3_%j.err

# M3 v3 part 3 (m3_design.md v3, pre-registered c500e86): setting tulu_arc.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
R="$P run_m3c.py"

$R teacher --setting tulu_arc
for F in qwen15 llama1b; do
  $R sft --setting tulu_arc --student $F
  $R student --setting tulu_arc --student $F
done
$P score_m3c.py
echo "=== done ==="
