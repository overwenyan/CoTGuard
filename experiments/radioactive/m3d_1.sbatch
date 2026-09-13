#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m3d1
#SBATCH --time=12:00:00
#SBATCH --output=experiments/radioactive/logs/m3d1_%j.out
#SBATCH --error=experiments/radioactive/logs/m3d1_%j.err
# M3 v4 stage 1 part 1 (pre-registered fca8f8a): smoke of the whole pipeline, then bank, screen, rewrites, factorial students.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
export M3C_OUT=${M3C_OUT:-$PWD/data4}
D="$P run_m3d.py"; C="$P run_m3c.py"
owners() { $P -c "import json,os;o=json.load(open(os.environ[\"M3C_OUT\"]+\"/tulu_gsm/owners.json\"));print(\",\".join(o[\"$1\"]))"; }

if [ "${SKIP_SMOKE:-0}" != 1 ]; then
echo "########## smoke ##########"
( export M3C_OUT=$PWD/data4_smoke M3D_N_BANK=8 M3D_ELIG_THR=0 M3D_N_DIL=40
  $D bank --max-arms 9
  $D screen
  $D rewrite --t T1
  O1=$(owners owners | cut -d, -f1)
  for CORP in raw T1; do
    $C sft --setting tulu_gsm --student qwen15 --epochs 1 --corpus $CORP --arms $O1,clean
    $C student --setting tulu_gsm --student qwen15 --n-test 4 --corpus $CORP --arms $O1,clean
  done
  $C student --setting tulu_gsm --student qwen15 --n-test 4 --arms base
  $D dilution
  $D build
  $D stealth
  K1=$(owners dilution | cut -d, -f1)
  for CORP in dil10 fil10; do
    $C sft --setting tulu_gsm --student qwen15 --epochs 1 --corpus $CORP --arms $K1
    $C student --setting tulu_gsm --student qwen15 --n-test 4 --corpus $CORP --arms $K1
  done
  $C sft --setting tulu_gsm --student qwen15 --epochs 1 --corpus dil0 --arms clean
  $C student --setting tulu_gsm --student qwen15 --n-test 4 --corpus dil0 --arms clean
  $D annotate
  $D sheet
  $P score_m3d.py a b )
rm -rf data4_smoke
fi

echo "########## bank + screen ##########"
$D bank
$D screen
echo "########## rewrites ##########"
$D rewrite --t T1
$D rewrite --t T2
echo "########## factorial students ##########"
ARMS=$(owners owners),clean
for CORP in raw T1 T2; do
  $C sft --setting tulu_gsm --student qwen15 --corpus $CORP --arms $ARMS
  $C student --setting tulu_gsm --student qwen15 --corpus $CORP --arms $ARMS
done
$C student --setting tulu_gsm --student qwen15 --arms base
$P score_m3d.py a
echo "=== done ==="
