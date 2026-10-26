#!/bin/bash
# Run one Makefile target on a GPU node; resumable, so --requeue is safe.
#   sbatch -p <partition> [-A <account>] --export=ALL,TARGET=collect,MAKE_ARGS="MODELS=qwen3-vl-8b" slurm/run.sbatch
#SBATCH --job-name=s3e-exp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
echo "host: $(hostname -s) gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
# shellcheck disable=SC2086
make ${TARGET:-collect} ${MAKE_ARGS:-}
