#!/bin/bash

# processing power
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

# output files
#SBATCH --output=dcollect_%A_%a.out
#SBATCH --error=dcollect_%A_%a.err

export MUJOCO_GL=egl

python -u gw_data_collector.py
