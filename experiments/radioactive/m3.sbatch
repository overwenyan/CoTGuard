#!/bin/bash
#SBATCH --job-name=cg_m3
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=experiments/radioactive/logs/m3_%j.out
#SBATCH --error=experiments/radioactive/logs/m3_%j.err

# M3 (m3_design.md, pre-registered fe2ad1b): passive prompt-only radioactivity.
# teacher traces -> LoRA students (key never shown) -> student samples -> gate.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python

echo "########## smoke ##########"
$P run_m3.py teacher --n-train 4
$P run_m3.py sft --student qwen05 --epochs 1
$P run_m3.py student --student qwen05 --n-test 4
rm -rf data/teacher_*.jsonl data/student_qwen05_* data/out_qwen05_*

echo "########## teacher ##########"
$P run_m3.py teacher --n-train 600
echo "########## sft 1.5B ##########"
$P run_m3.py sft --student qwen15 --epochs 3
echo "########## student 1.5B ##########"
$P run_m3.py student --student qwen15 --n-test 200
echo "########## gate ##########"
$P score_m3.py qwen15
echo "########## secondary: 0.5B ##########"
$P run_m3.py sft --student qwen05 --epochs 3
$P run_m3.py student --student qwen05 --n-test 200
$P score_m3.py qwen05
echo "=== done ==="
