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
  --run_name="gpt4o_HOR${1}_cond-${2}_views_full_inst_big-milk" \
  --task_horizon=$1 \
  --set_goal_cond $2 \
  --se_class="semantic_state_estimator.semantic_state_estimator:SemanticStateEstimator" \
  --vqa_model_id="OpenAI/gpt-4o-2024-11-20" \
  --nl_converter_model_id="OpenAI/gpt-4o-2024-11-20" \
  --additional_instructions="The user will show you images of a simulated robot and ask questions about the state of the environment.\nThere are three tables: a brown wood table, a black table, and a white table.\nThe robot has a black two finger gripper at its end effector with which it can grasp objects. Objects are considered gripped if they are in between the gripper fingers. If no object is in between the two gripper fingers, the robot gripper is considered to be empty.\nThe milk carton is a tall, rectangular container with a distinctive triangular top that forms a peaked roof-like structure when closed. It has a primarily white/light gray color and smooth surfaces. The carton features a simple, minimalist design with no visible text or graphics on its exterior. It has straight edges and flat sides forming a prism shape, with the top portion angled inward to create the triangular peak. This is the standard gable-top carton design commonly used for dairy products, giving it the instantly recognizable silhouette associated with milk containers. The carton appears to be closed with its top flaps folded together.\nWhen the robot is holding the milk carton it looks like there is a white rectangular object being pinched by the robot's gripper.\nThe loaf of bread looks like a small brown box.\nWhen the robot is gripping the loaf of bread it looks like there is a small brown object inside the robot gripper.\nThe red can of soda is a small red cylinder with some white labeling on it.\nThe red box of cereal is a tall box with an illustration on its wide sides."
