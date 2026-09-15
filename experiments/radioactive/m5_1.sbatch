#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m5s
#SBATCH --time=1:30:00
#SBATCH --output=experiments/radioactive/logs/m5s_%j.out
#SBATCH --error=experiments/radioactive/logs/m5s_%j.err
# M5 §1 smoke (pre-registered 2b36772, amended b6e448d): tiny corpora, 2 teachers, composite path.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
M="$P run_m5.py"; C="$P run_m3c.py"
ORDER=(tulu_sft tulu_dpo llama31_8b qwen25_7b qwen3_14b mistral_7b gemma2_9b)
export M5_SMOKE=1 M5_OUT=$PWD/data_m5_smoke M3C_OUT=$PWD/data_m5_smoke
rm -rf data_m5_smoke && mkdir -p data_m5_smoke && for f in data_m5/clean_*.jsonl; do ln -s ../$f data_m5_smoke/; done
$M corpora
$M dpoinst
for T in tulu_sft tulu_dpo qwen25_7b; do
  $C sft --setting tulu_gsm --student qwen15 --epochs 1 --corpus grid --arms ${T}_s0
done
$C sft --setting tulu_gsm --student qwen15 --epochs 1 --corpus dpoinst --arms o12
$C sft --setting tulu_gsm --student qwen15 --epochs 1 --corpus mix --arms mix10T4
$M probe --names qwen15:grid_tulu_sft_s0,qwen15:grid_tulu_dpo_s0,qwen15:grid_qwen25_7b_s0,qwen15:dpoinst_o12,qwen15:mix_mix10T4
$P score_m5.py
rm -rf data_m5_smoke
echo "=== smoke ok ==="
