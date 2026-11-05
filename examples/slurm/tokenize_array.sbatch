#!/bin/bash
# Tokenise an EnvTok corpus as a SLURM job array, then index and select.
#
#   sbatch --array=0-63 examples/slurm/tokenize_array.sbatch omat24.yaml
#   # after every array task has finished:
#   envtok index  --config omat24.yaml
#   envtok select --config omat24.yaml --min-support 100
#
# Every task processes the batches with index % SLURM_ARRAY_TASK_COUNT ==
# SLURM_ARRAY_TASK_ID. Tasks write independent, atomically committed
# partitions, so a failed task can simply be resubmitted.
#SBATCH --job-name=envtok-tokenize
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=envtok-tokenize-%A_%a.log
set -euo pipefail

CONFIG="${1:?usage: sbatch --array=0-N tokenize_array.sbatch CONFIG}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

envtok -v tokenize --config "${CONFIG}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --num-shards "${SLURM_ARRAY_TASK_COUNT}"
