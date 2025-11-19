#!/bin/bash

# processing power
#SBATCH --job-name=task-gwh
#SBATCH --cpus-per-task=2

# output files
#SBATCH --output=run-episode_gpt4o_HOR3_%A-%a.out
#SBATCH --error=run-episode_gpt4o_HOR3_%A-%a.err

echo "Running with task_horizon $1"
echo "using short goal condition: $2"


# set up mujoco gl
export MUJOCO_GL=egl

if [ -n "$SLURM_JOB_ID" ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(readlink -f $0)
fi

SCRIPT_DIR=$(dirname $SCRIPT_PATH)

python -u gw_episode_runner.py \
  --run_name="gpt4o_HOR${1}_cond-${2}" \
  --task_horizon=$1 \
  --set_goal_cond $2 \
  --se_class="semantic_state_estimator.semantic_state_estimator:SemanticStateEstimator" \
  --vqa_model_id="OpenAI/gpt-4o-2024-11-20" \
  --nl_converter_model_id="OpenAI/gpt-4o-2024-11-20" \
  --additional_instructions="The user will show you images of a simulated robot and ask questions about the state of the environment.\nThe milk carton is a clean white rectangular box with a triangular top.\nWhen the robot is holding the milk carton it looks like there is a white rectangular object being pinched by the robot's gripper."
