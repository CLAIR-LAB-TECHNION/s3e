import os

import fire

from prb_env import PRBEnv, random
from prb_gt_state_estimator import PRBGTStateEstimator
from prb_skill_executer import PRBSkillExecuter
from semantic_state_estimator.constants import LLAMA_70B_INSTRUCT
from semantic_state_estimator.eval.run_episodes import EpisodeRunner
from semantic_state_estimator.utils.misc import load_se_from_args, load_from_entrypoint


DOMAIN_FILE = os.path.join(os.path.dirname(__file__), 'domain.pddl')
PROBLEM_FILE = os.path.join(os.path.dirname(__file__), 'all_objects_problem.pddl')
OUT_DIR = 'data_dir'
RENDER_CAMS = ['frontview']


class PRBEnvWrapper(PRBEnv):
    def __init__(self, num_objects_low, num_objects_high):
        super().__init__(2)
        self.num_objects_low = num_objects_low
        self.num_objects_high = num_objects_high

    def reset(self, *args, **kwargs):
        num_objects = random.randint(self.num_objects_low, self.num_objects_high)
        super().reset(num_objects=num_objects)

    def get_state(self):
        return self.state


def query_swapper(env: PRBEnvWrapper):
    with open(DOMAIN_FILE, 'r') as f:
        domain_str = f.read()

    return domain_str, env.get_problem_file_str(), LLAMA_70B_INSTRUCT


def main(run_name, num_objects_low, num_objects_high, task_horizon, se_class, **se_kwargs):
    env = PRBEnvWrapper(num_objects_low, num_objects_high)
    env.reset()

    domain_str, problem_str, _ = query_swapper(env)
    gt_se = PRBGTStateEstimator(domain_str, problem_str)
    exec = PRBSkillExecuter(env)

    if 'random' in se_class.lower():
        random_cls = load_from_entrypoint(se_class)
        se = random_cls(DOMAIN_FILE, PROBLEM_FILE, se_kwargs['success_rate'], gt_se)
    else:
        se = load_se_from_args(se_class, se_kwargs, DOMAIN_FILE, PROBLEM_FILE)

    runner = EpisodeRunner(
        gt_se.up_problem, gt_se, se, exec, RENDER_CAMS, OUT_DIR, run_name + f'__({num_objects_low},{num_objects_high})',
        query_swapper=query_swapper
    )

    runner.run(num_episodes=100, task_horizon=task_horizon)


if __name__ == "__main__":
    fire.Fire(main)
