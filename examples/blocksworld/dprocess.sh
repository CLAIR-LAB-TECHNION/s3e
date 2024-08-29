#!/bin/bash

# processing power
#SBATCH --partition=g48
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=10

# output files
#SBATCH --output=dprocess_%A_%a.out
#SBATCH --error=dprocess_%A_%a.err

if [ -n "$SLURM_JOB_ID" ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(readlink -f $0)
fi

SCRIPT_DIR=$(dirname $SCRIPT_PATH)

python -u $SCRIPT_DIR/../../semantic_state_estimator/eval/process_datapoints.py --data_dir="data_dir_bw" --domain="$SCRIPT_DIR/domain.pddl" --problem="$SCRIPT_DIR/problem.pddl" $@
