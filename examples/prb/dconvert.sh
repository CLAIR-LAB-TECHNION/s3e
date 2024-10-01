#!/bin/bash

# processing power
#SBATCH --cpus-per-task=16

# output files
#SBATCH --output=dconvert_%A_%a.out
#SBATCH --error=dconvert_%A_%a.err

python -u prb_convert_to_pddl_states.py
