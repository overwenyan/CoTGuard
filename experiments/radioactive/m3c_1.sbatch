#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m3c1
#SBATCH --time=9:00:00
#SBATCH --output=experiments/radioactive/logs/m3c1_%j.out
#SBATCH --error=experiments/radioactive/logs/m3c1_%j.err

# M3 v3 part 1 (m3_design.md v3, pre-registered c500e86): smoke, then tulu_gsm raw + attacks.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
R="$P run_m3c.py"

echo "########## smoke (separate dir, deleted after) ##########"
export M3C_OUT=$PWD/data3_smoke
$R teacher --setting tulu_gsm --n 4 --max-arms 2
$R teacher --setting tulu_arc --n 4 --max-arms 2
$R teacher --setting qwen_gsm --n 4 --max-arms 1
$R attack --attack filter
$R attack --attack para --role attacker
$R attack --attack para --role owner --n-owner 3 --max-arms 2
for S in tulu_gsm tulu_arc; do
  $R sft --setting $S --student llama1b --epochs 1 --arms g00,clean
  $R student --setting $S --student llama1b --n-test 4
done
$R sft --setting tulu_gsm --student llama1b --epochs 1 --corpus para --arms g00,clean
$R student --setting tulu_gsm --student llama1b --n-test 4 --corpus para
$P score_m3c.py
rm -rf data3_smoke
unset M3C_OUT

echo "########## tulu_gsm: teacher (56 new arms; g00-07 + clean reused from v2) ##########"
$R teacher --setting tulu_gsm
for F in qwen15 llama1b; do
  $R sft --setting tulu_gsm --student $F
  $R student --setting tulu_gsm --student $F
done
echo "########## attacks ##########"
$R attack --attack filter
$R attack --attack para --role attacker
$R attack --attack compress --role attacker
$R sft --setting tulu_gsm --student qwen15 --corpus filter
$R student --setting tulu_gsm --student qwen15 --corpus filter
for F in qwen15 llama1b; do
  $R sft --setting tulu_gsm --student $F --corpus para
  $R student --setting tulu_gsm --student $F --corpus para
done
$R sft --setting tulu_gsm --student qwen15 --corpus compress
$R student --setting tulu_gsm --student qwen15 --corpus compress
$R attack --attack para --role owner
$R attack --attack compress --role owner
$P score_m3c.py
echo "=== done ==="
