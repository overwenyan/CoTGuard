#!/bin/bash
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=cg_m3e3
#SBATCH --array=0-2
#SBATCH --time=6:00:00
#SBATCH --output=experiments/radioactive/logs/m3e3_%A_%a.out
#SBATCH --error=experiments/radioactive/logs/m3e3_%A_%a.err
# M3 v5: T1 / T1n rewrites and students, sharded over 3 GPUs.
set -euo pipefail
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard/experiments/radioactive
export HF_HOME=/local/projects-t3/primelab/yan.wen/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/local/projects-t3/primelab/yan.wen/miniconda3/envs/py312/bin/python
export M3D_BANK=keys_v5.json
export M3C_OUT=${M3C_OUT:-$PWD/data5}
D="$P run_m3d.py"; C="$P run_m3c.py"
arms_of() { $P -c "import json,os;o=json.load(open(os.environ[\"M3C_OUT\"]+\"/tulu_gsm/owners.json\"));a=o[\"owners\"]+[\"clean\"];print(\",\".join(a[$1::$2]))"; }
I=$SLURM_ARRAY_TASK_ID
$D rewrite --t T1 --shard $I --nshards 3
$D rewrite --t T1n --shard $I --nshards 3
A=$(arms_of $I 3)
for CORP in raw T1 T1n; do
  $C sft --setting tulu_gsm --student qwen15 --corpus $CORP --arms $A
  $C student --setting tulu_gsm --student qwen15 --corpus $CORP --arms $A
done
if [ $I = 0 ]; then $C student --setting tulu_gsm --student qwen15 --arms base; fi
echo "=== done ==="
