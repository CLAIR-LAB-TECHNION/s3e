import os

from bw_env import get_env
from bw_gt_state_estimator import (pddl_id_to_mujoco_entity, pddl_id_to_mujoco_name, BLOCK_PDDL_IDS,
                                   BlocksWorldGTStateEstimator)
from bw_skill_executer import BlocksWorldSkillExecuter
from semantic_state_estimator.eval.collect_datapoints import collect_data

DOMAIN_FILE = os.path.join(os.path.dirname(__file__), 'domain.pddl')
PROBLEM_FILE = os.path.join(os.path.dirname(__file__), 'problem.pddl')
OUT_DIR = 'data_dir_bw'
RENDER_CAMS = ['frontview', 'rightangleview', 'leftangleview', 'rightsideview', 'leftsideview', 'birdview']

def main():
    env = get_env()
    se = BlocksWorldGTStateEstimator(DOMAIN_FILE, PROBLEM_FILE, env)
    exec = BlocksWorldSkillExecuter(env, pddl_id_to_mujoco_name, pddl_id_to_mujoco_entity, BLOCK_PDDL_IDS)
    collect_data(se.up_problem, se, exec, RENDER_CAMS, OUT_DIR, num_datapoints=2000)


if __name__ == "__main__":
    main()
