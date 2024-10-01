import glob
import json
import os

import fire
from tqdm.auto import tqdm

from semantic_state_estimator.constants import SCENES_DIR, TRUE_STATES_DIR

from prb_env import PRBEnv, NewState
from prb_gt_state_estimator import PRBGTStateEstimator


def convert_scenes_to_pddl_states(data_dir='data_dir', domain='domain.pddl'):
    env = PRBEnv(num_objects=None)  # no objects set. we will soon set the state itself
    true_states_dir = os.path.join(data_dir, TRUE_STATES_DIR)
    os.makedirs(true_states_dir, exist_ok=True)

    scene_files = glob.glob(os.path.join(data_dir, SCENES_DIR, "*.json"))
    for scene_file in tqdm(scene_files):
        # skip if result exists
        out_file = os.path.join(true_states_dir, os.path.basename(scene_file))
        if os.path.exists(out_file):
            continue

        # load state object from scene
        with open(scene_file, 'r') as f:
            scene_data = json.load(f)
        state = NewState.undump(scene_data)

        # set env state and output problem file
        env.state = state
        problem_str = env.get_problem_file_str()

        # load domain PDDL as a string to match the problem
        with open(domain, 'r') as f:
            domain_str = f.read()

        # estimate state
        estimator = PRBGTStateEstimator(domain_str, problem_str)
        pddl_state = estimator(state)

        # save gt file
        with open(out_file, 'w') as f:
            json.dump(pddl_state, f, indent=4)


if __name__ == "__main__":
    fire.Fire(convert_scenes_to_pddl_states)




