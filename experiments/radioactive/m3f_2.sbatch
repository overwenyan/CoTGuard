#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m3f2
#SBATCH --time=1:00:00
#SBATCH --output=experiments/radioactive/logs/m3f2_%j.out
#SBATCH --error=experiments/radioactive/logs/m3f2_%j.err
# M3 v6: build corpora; smoke the full fine-tune path on 16 examples.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
export M3C_OUT=$PWD/data4
D="$P run_m3d.py"; C="$P run_m3c.py"
KEYS=(o12 o17 p07 p13)
$D curve_build
head -16 data4/tulu_gsm/corpus_dil0ft_clean.jsonl > data4/tulu_gsm/corpus_smokeft_clean.jsonl
$C sft --setting tulu_gsm --student qwen15 --epochs 1 --full --corpus smokeft --arms clean
$C student --setting tulu_gsm --student qwen15 --n-test 4 --corpus smokeft --arms clean
rm -rf data4/tulu_gsm/corpus_smokeft_clean.jsonl data4/tulu_gsm/student_qwen15_smokeft_clean data4/tulu_gsm/out_qwen15_smokeft_clean.jsonl
echo "=== done ==="
