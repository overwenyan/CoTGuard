#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m3e4
#SBATCH --time=1:30:00
#SBATCH --output=experiments/radioactive/logs/m3e4_%j.out
#SBATCH --error=experiments/radioactive/logs/m3e4_%j.err
# M3 v5: scoring (H-OP-M on T1, H-OP-N on T1n, manipulation check on raw).
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
export M3D_BANK=keys_v5.json
export M3C_OUT=${M3C_OUT:-$PWD/data5}
D="$P run_m3d.py"; C="$P run_m3c.py"
arms_of() { $P -c "import json,os;o=json.load(open(os.environ[\"M3C_OUT\"]+\"/tulu_gsm/owners.json\"));a=o[\"owners\"]+[\"clean\"];print(\",\".join(a[$1::$2]))"; }
export M3D_CORPORA=raw,T1,T1n
$P score_m3d.py a
echo "=== done ==="
