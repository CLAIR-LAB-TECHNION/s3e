#!/bin/bash

# processing power
#SBATCH --partition=g48
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --constraint=ampere

# output files
#SBATCH --output=dprocess_%A_%a.out
#SBATCH --error=dprocess_%A_%a.err

python -u gw_process_milk.py

