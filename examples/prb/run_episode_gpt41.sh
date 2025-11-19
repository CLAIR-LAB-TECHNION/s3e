#!/bin/bash

# processing power
#SBATCH --job-name=task-prb
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1

# output files
#SBATCH --output=run-episode_gpt41_HOR3_%A-%a.out
#SBATCH --error=run-episode_gpt41_HOR3_%A-%a.err

echo "Running with num_objects_low $1"
echo "Running with num_objects_high $2"
echo "Running with task_horizon $3"
echo "using short goal condition: $4"


if [ -n "$SLURM_JOB_ID" ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(readlink -f $0)
fi

SCRIPT_DIR=$(dirname $SCRIPT_PATH)

python -u prb_episode_runner.py \
  --run_name="gpt41_HOR${3}_cond-${4}" \
  --num_objects_low $1 \
  --num_objects_high $2 \
  --task_horizon=$3 \
  --set_goal_cond $4 \
  --se_class="semantic_state_estimator.semantic_state_estimator:SemanticStateEstimator" \
  --vqa_model_id="OpenAI/gpt-4.1-2025-04-14" \
  --nl_converter_model_id="OpenAI/gpt-4.1-2025-04-14"
