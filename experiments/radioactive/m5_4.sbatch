#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m5x
#SBATCH --time=4:00:00
#SBATCH --output=experiments/radioactive/logs/m5x_%j.out
#SBATCH --error=experiments/radioactive/logs/m5x_%j.err
# M5 §1: composite (Tulu-DPO + instruction) students and mixture students.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
M="$P run_m5.py"; C="$P run_m3c.py"
ORDER=(tulu_sft tulu_dpo llama31_8b qwen25_7b qwen3_14b mistral_7b gemma2_9b)
export M5_OUT=$PWD/data_m5 M3C_OUT=$PWD/data_m5
for SD in 0 1; do
  for F in qwen15 llama1b; do
    $C sft --setting tulu_gsm --student $F --epochs 3 --seed $SD --corpus dpoinst --arms o12,p07
  done
  $C sft --setting tulu_gsm --student qwen15 --epochs 3 --seed $SD --corpus mix --arms mix50T4,mix10T4,mix10T3
done
$M probe --names qwen15:dpoinst_o12,qwen15:dpoinst_p07,qwen15:dpoinst_o12_s1,qwen15:dpoinst_p07_s1,llama1b:dpoinst_o12,llama1b:dpoinst_p07,llama1b:dpoinst_o12_s1,llama1b:dpoinst_p07_s1,qwen15:mix_mix50T4,qwen15:mix_mix10T4,qwen15:mix_mix10T3,qwen15:mix_mix50T4_s1,qwen15:mix_mix10T4_s1,qwen15:mix_mix10T3_s1
echo "=== done ==="
