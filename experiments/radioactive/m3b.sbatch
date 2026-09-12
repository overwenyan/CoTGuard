#!/bin/bash
#SBATCH --job-name=cg_m3b
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=experiments/radioactive/logs/m3b_%j.out
#SBATCH --error=experiments/radioactive/logs/m3b_%j.err

# M3 v2 (m3_design.md v2, pre-registered 86738f7): 16 keys, two student families, owner-side test.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python

echo "########## smoke ##########"
$P run_m3b.py teacher --n 4
$P run_m3b.py sft --student llama1b --epochs 1 --arms key_01
$P run_m3b.py student --student llama1b --n-test 4 --arms key_01,base
rm -rf data2

echo "########## teacher: 16 keys + clean + big + active ##########"
$P run_m3b.py teacher --n 300
echo "########## students: qwen2.5-1.5b (all arms) ##########"
$P run_m3b.py sft --student qwen15 --epochs 3 --arms all
$P run_m3b.py student --student qwen15 --n-test 200
echo "########## students: llama-3.2-1b (6 keys + clean) ##########"
$P run_m3b.py sft --student llama1b --epochs 3 --arms key_01,key_03,key_08,key_09,key_12,key_13,clean
$P run_m3b.py student --student llama1b --n-test 200
echo "########## gate ##########"
$P score_m3b.py
echo "=== done ==="
