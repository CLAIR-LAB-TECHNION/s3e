#!/bin/bash

# processing power
#SBATCH --partition=g24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=ampere

# output files
#SBATCH --output=dprocess-iter-05-abl-inst_%A_%a.out
#SBATCH --error=dprocess-iter-05-abl-inst_%A_%a.err

if [ -n "$SLURM_JOB_ID" ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(readlink -f $0)
fi

SCRIPT_DIR=$(dirname $SCRIPT_PATH)


python -u $SCRIPT_DIR/../../semantic_state_estimator/eval/process_datapoints.py --data_dir="data_dir" --domain="$SCRIPT_DIR/domain.pddl" --problem="$SCRIPT_DIR/problem.pddl" --out_dir="0.5B (no trans) + Instruct" --se_class="semantic_state_estimator.semantic_state_estimator:SemanticEstimatorMultiImageRunNoLLaMA" --vqa_model_id="lmms-lab/llava-onevision-qwen2-0.5b-ov" --additional_instructions="The user will show you images of a simulated robot and ask questions about the state of the environment.\nThe milk carton is a clean white rectangular box with a triangular top.\nWhen the robot is holding the milk carton it looks like there is a white rectangular object being pinched by the robot's gripper.\nThe red can of soda is a small red cylinder.\nWhen the robot is holding the red can of soda it looks like there is a small red object that is enveloped by the robot's gripper.\nThe loaf of bread looks like a small brown box.\nWhen the robot is gripping the loaf of bread it looks like there is a small brown object inside the robot gripper."
