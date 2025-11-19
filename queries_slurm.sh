#!/bin/bash
#SBATCH --job-name=get_queries

# processing power
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32

# output files
#SBATCH --output=queries.out
#SBATCH --error=queries.err

CONDA_HOME=$HOME/miniforge3
CONDA_ENV=s3e

unset XDG_RUNTIME_DIR
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

python get_queries.py $@
