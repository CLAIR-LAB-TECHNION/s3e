#!/bin/bash

# processing power
#SBATCH --partition=g48
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16

# output files
#SBATCH --output=queries.out
#SBATCH --error=queries.err

python get_queries.py
