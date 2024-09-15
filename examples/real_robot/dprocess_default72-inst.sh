#!/bin/bash

# processing power
#SBATCH --partition=g48
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --constraint=ampere

# output files
#SBATCH --output=dprocess-default72-inst_%A_%a.out
#SBATCH --error=dprocess-default72-inst_%A_%a.err

if [ -n "$SLURM_JOB_ID" ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(readlink -f $0)
fi

SCRIPT_DIR=$(dirname $SCRIPT_PATH)

python -u $SCRIPT_DIR/../../semantic_state_estimator/eval/process_datapoints.py --data_dir="data_dir" --domain="$SCRIPT_DIR/domain.pddl" --problem="$SCRIPT_DIR/problem.pddl" --out_dir="72B-instruct" --vqa_model_id="lmms-lab/llava-onevision-qwen2-72b-ov" --additional_instructions="The user will show you images of a robot performing a pick-and-place task.\nYour job is to answer questions about the state of the environment as accurately as possible.\nAn item is considerred in a certain color section on the table if it is placed on the side of that color.\nThe left side of the table is the white section and the right side of the table is the blue section.\nThe mineral water bottle is the clear bottle with the green cap.\nThe spray bottle is the big blue bottle with the grey top."
