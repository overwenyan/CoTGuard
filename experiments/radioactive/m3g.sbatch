#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m3g
#SBATCH --array=0-3
#SBATCH --time=5:00:00
#SBATCH --output=experiments/radioactive/logs/m3g_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m3g_%A_%a.err
# M3 v7 bounded replication (frozen 021f68f). Task 0/1 = E2 rewriting Llama seed 0/1 (+base, clean);
# task 2 = E3 dilution Llama seeds 0,1; task 3 = E1 source ambiguity (Llama + Qwen, seeds 0,1).
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export M3C_OUT=$PWD/data4
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
C="$P run_m3c.py"; S="--setting tulu_gsm"
I=$SLURM_ARRAY_TASK_ID
case $I in
0|1)
  E2=o14,o17,o20,p03,p12,p13,clean
  for CORP in raw T1 T2; do
    $C sft $S --student llama1b --seed $I --corpus $CORP --arms $E2
    $C student $S --student llama1b --seed $I --corpus $CORP --arms $E2
  done
  $C student $S --student llama1b --seed $I --n-test 1319 --arms base
  ;;
2)
  for SD in 0 1; do
    $C sft $S --student llama1b --seed $SD --epochs 1 --corpus dil0 --arms clean
    $C student $S --student llama1b --seed $SD --n-test 1319 --corpus dil0 --arms clean
    for CORP in dil50 dil10; do
      $C sft $S --student llama1b --seed $SD --epochs 1 --corpus $CORP --arms o12,o17,p07
      $C student $S --student llama1b --seed $SD --n-test 1319 --corpus $CORP --arms o12,o17,p07
    done
  done
  ;;
3)
  for K in o12 p07; do head -300 data4/tulu_gsm/imit_keyed_$K.jsonl > data4/tulu_gsm/corpus_imit300_$K.jsonl; done
  for SD in 0 1; do
    $C sft $S --student llama1b --seed $SD --corpus raw --arms o12,p07
    $C student $S --student llama1b --seed $SD --corpus raw --arms o12,p07
    $C sft $S --student llama1b --seed $SD --corpus imit300 --arms o12,p07
    $C student $S --student llama1b --seed $SD --corpus imit300 --arms o12,p07
    $C sft $S --student qwen15 --seed $SD --corpus imit300 --arms o12,p07
    $C student $S --student qwen15 --seed $SD --corpus imit300 --arms o12,p07
  done
  $C sft $S --student qwen15 --seed 1 --corpus raw --arms o12,p07,clean
  $C student $S --student qwen15 --seed 1 --corpus raw --arms o12,p07,clean
  $C student $S --student qwen15 --seed 1 --arms base
  ;;
esac
echo "=== done ==="
