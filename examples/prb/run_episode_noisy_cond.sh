#!/bin/bash

# processing power
#SBATCH --job-name=task-prb
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1

# output files
#SBATCH --output=run-episode_noisy_cond_%A-%a.out
#SBATCH --error=run-episode_noisy_cond_%A-%a.err

echo "Running with num_objects_low $1"
echo "Running with num_objects_high $2"
echo "Running with task_horizon $3"
echo "Running with success rate $4"

if [ -n "$SLURM_JOB_ID" ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(readlink -f $0)
fi

SCRIPT_DIR=$(dirname $SCRIPT_PATH)

python -u prb_episode_runner.py \
    --run_name="noisy${4}_HOR${3}_cond" \
    --num_objects_low $1 \
    --num_objects_high $2 \
    --task_horizon=$3 \
    --set_goal_cond True \
    --se_class="semantic_state_estimator.random_state_estimator:RandomStateEstimator" \
    --success_rate $4
