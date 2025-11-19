#!/bin/bash

# processing power
#SBATCH --job-name=task-gwh
#SBATCH --cpus-per-task=2

# output files
#SBATCH --output=run-episode_gt_cond_%A-%a.out
#SBATCH --error=run-episode_gt_cond_%A-%a.err

# set up mujoco gl
export MUJOCO_GL=egl

echo "Running with task_horizon $1"

if [ -n "$SLURM_JOB_ID" ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(readlink -f $0)
fi

SCRIPT_DIR=$(dirname $SCRIPT_PATH)

python -u gw_episode_runner.py \
    --run_name="gt_HOR${1}_cond" \
    --task_horizon=$1 \
    --set_goal_cond True \
    --se_class="semantic_state_estimator.random_state_estimator:RandomStateEstimator" \
    --success_rate 1.0
