#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m3d2
#SBATCH --time=5:00:00
#SBATCH --output=experiments/radioactive/logs/m3d2_%j.out
#SBATCH --error=experiments/radioactive/logs/m3d2_%j.err
# M3 v4 stage 1 part 2: dilution generation, corpora, stealth screens and filtered corpora.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
export M3C_OUT=${M3C_OUT:-$PWD/data4}
D="$P run_m3d.py"; C="$P run_m3c.py"
owners() { $P -c "import json,os;o=json.load(open(os.environ[\"M3C_OUT\"]+\"/tulu_gsm/owners.json\"));print(\",\".join(o[\"$1\"]))"; }
$D dilution
$D build
$D stealth
echo "=== done ==="
