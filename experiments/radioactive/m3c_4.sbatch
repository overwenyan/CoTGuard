#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m3c4
#SBATCH --time=8:00:00
#SBATCH --output=experiments/radioactive/logs/m3c4_%j.out
#SBATCH --error=experiments/radioactive/logs/m3c4_%j.err

# M3 v3 part 4: owner-simulated rewrites for the attack-aware read-out (resumes whatever part 1 left
# unfinished at its time limit), then the final score over all settings.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
R="$P run_m3c.py"

for C in filter para compress; do
  [ $C = filter ] && $R attack --attack filter
  [ $C != filter ] && $R attack --attack $C --role attacker
done
$R sft --setting tulu_gsm --student qwen15 --corpus filter && $R student --setting tulu_gsm --student qwen15 --corpus filter
for F in qwen15 llama1b; do $R sft --setting tulu_gsm --student $F --corpus para && $R student --setting tulu_gsm --student $F --corpus para; done
$R sft --setting tulu_gsm --student qwen15 --corpus compress && $R student --setting tulu_gsm --student qwen15 --corpus compress
$R attack --attack para --role owner
$R attack --attack compress --role owner
$P score_m3c.py
echo "=== done ==="
