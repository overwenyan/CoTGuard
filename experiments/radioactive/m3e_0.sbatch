#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m3e0
#SBATCH --time=1:30:00
#SBATCH --output=experiments/radioactive/logs/m3e0_%j.out
#SBATCH --error=experiments/radioactive/logs/m3e0_%j.err
# M3 v5 (pre-registered acf11dd): smoke of the matched-replication pipeline on tiny data.
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
export M3C_OUT=$PWD/data5_smoke M3D_N_BANK=8 M3D_ELIG_THR=0 M3D_TOL_SCALE=100 M3D_MIN_PAIRS=1 M3D_CORPORA=raw,T1n
rm -rf data5_smoke
$D bank --max-arms 3
$D screen --compliance-only
$D match
$D rewrite --t T1n
A=$(arms_of 0 1 | cut -d, -f1),clean
for CORP in raw T1n; do
  $C sft --setting tulu_gsm --student qwen15 --epochs 1 --corpus $CORP --arms $A
  $C student --setting tulu_gsm --student qwen15 --n-test 4 --corpus $CORP --arms $A
done
$C student --setting tulu_gsm --student qwen15 --n-test 4 --arms base
$P score_m3d.py a
rm -rf data5_smoke
echo "=== smoke ok ==="
