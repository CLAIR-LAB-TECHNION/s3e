#!/bin/bash

# processing power
#SBATCH --job-name=task-prb
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# output files
#SBATCH --output=run-episode_gpt4o_HOR3_%A-%a.out
#SBATCH --error=run-episode_gpt4o_HOR3_%A-%a.err

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

python -u prb_episode_runner.py \
    --run_name="gpt4o_HOR3" \
    --num_objects_low 3 \
    --num_objects_high 5 \
    --task_horizon=3 \
    --se_class="semantic_state_estimator.semantic_state_estimator:SemanticEstimatorMultiImageRun" \
    --vqa_model_id="OpenAI/gpt-4o-2024-11-20" \
    --nl_converter_model_id="OpenAI/gpt-4o-2024-11-20"
