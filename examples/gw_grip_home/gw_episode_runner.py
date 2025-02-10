import os

import fire

from gw_env import get_env
from gw_gt_state_estimator import (pddl_id_to_mujoco_entity, pddl_id_to_mujoco_name, ITEM_PDDL_IDS,
                                   GroceriesWorldGTStateEstimator)
from gw_skill_executer import GroceriesWorldSkillExecuter
from semantic_state_estimator.eval.run_episodes import EpisodeRunner
from semantic_state_estimator.utils.misc import load_se_from_args, load_from_entrypoint


DOMAIN_FILE = os.path.join(os.path.dirname(__file__), 'domain.pddl')
PROBLEM_FILE = os.path.join(os.path.dirname(__file__), 'problem.pddl')
OUT_DIR = 'data_dir'
RENDER_CAMS = ['frontview', 'rightangleview', 'leftangleview', 'rightsideview', 'leftsideview', 'birdview']

def main(run_name, task_horizon, se_class, **se_kwargs):
    env = get_env()
    gt_se = GroceriesWorldGTStateEstimator(DOMAIN_FILE, PROBLEM_FILE, env)
    exec = GroceriesWorldSkillExecuter(env, pddl_id_to_mujoco_name, pddl_id_to_mujoco_entity, ITEM_PDDL_IDS)
    
    if 'random' in se_class.lower():
        random_cls = load_from_entrypoint(se_class)
        se = random_cls(DOMAIN_FILE, PROBLEM_FILE, se_kwargs['success_rate'], gt_se)
    else:
        se = load_se_from_args(se_class, se_kwargs, DOMAIN_FILE, PROBLEM_FILE)

    runner = EpisodeRunner(
        gt_se.up_problem, gt_se, se, exec, RENDER_CAMS, OUT_DIR, run_name, go_home_prob=1.0
    )

    runner.run(num_episodes=100, task_horizon=task_horizon)


if __name__ == "__main__":
    fire.Fire(main)
