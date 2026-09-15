#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m7s
#SBATCH --time=2:00:00
#SBATCH --output=experiments/radioactive/logs/m7s_%j.out
#SBATCH --error=experiments/radioactive/logs/m7s_%j.err
# M7 smoke (pre-registered 4fa5040): tiny splits; math 3 teachers x 3 students, gsm 1 test student; scorer end to end.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
V=/local/projects-t3/primelab/yan.wen/venvs/vllm/bin/python
export PATH=/local/projects-t3/primelab/yan.wen/venvs/vllm/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export M7_SMOKE=1 M7_OUT=$PWD/data_m7_smoke M3C_OUT=$PWD/data_m7_smoke M3C_MAXLEN=1400
rm -rf data_m7_smoke
$P run_m7.py prep
$V run_m7.py gen --dataset gsm --teacher tulu_dpo --parts test
for T in tulu_dpo tulu_rlvr olmoi_sft; do $V run_m7.py gen --dataset math --teacher $T; done
$P run_m7.py corpora
$P run_m3c.py sft --setting tulu_gsm --student qwen15 --epochs 1 --corpus grid --arms tulu_dpo_s10
$P run_m3c.py sft --setting tulu_math --student qwen15 --epochs 1 --corpus grid \
   --arms tulu_dpo_s0,tulu_dpo_s1,tulu_dpo_s10,tulu_rlvr_s0,tulu_rlvr_s1,tulu_rlvr_s10,olmoi_sft_s0,olmoi_sft_s1,olmoi_sft_s10
$V run_m7.py probe --dataset gsm --student qwen15 --names base,grid_tulu_dpo_s10
NAMES=base
for T in tulu_dpo tulu_rlvr olmoi_sft; do for S in 0 1 10; do NAMES=$NAMES,grid_${T}_s$S; done; done
$V run_m7.py probe --dataset math --student qwen15 --names $NAMES
$P - <<'PY'
import json, sys
sys.path.insert(0, ".")
from math_check import math_correct
r = [json.loads(l) for l in open("data_m7_smoke/tulu_math/teacher_tulu_dpo_test.jsonl")]
print("math teacher sample acc", sum(math_correct(x["text"], x["gold"]) for x in r), "/", len(r))
print("--- sample ---\n", r[0]["text"][:500], "\n--- gold ---\n", r[0]["gold"][-200:])
PY
$P score_m7.py
rm -rf data_m7_smoke
echo "=== smoke ok ==="
