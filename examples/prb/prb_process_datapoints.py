import json
import os

import fire

from semantic_state_estimator.constants import SCENES_DIR, LLAMA_70B_INSTRUCT
from semantic_state_estimator.eval.process_datapoints import process_datapoints

from prb_env import PRBEnv, NewState

THIS_DIR = os.path.dirname(__file__)
DOMAIN_FILE = os.path.join(THIS_DIR, 'domain.pddl')
ALL_OBJECTS_PROBLEM_FILE = os.path.join(THIS_DIR, 'all_objects_problem.pddl')

def query_swapper(renders_file):
    scene_filename = os.path.splitext(os.path.basename(renders_file))[0] + '.json'
    data_dir = os.path.dirname(os.path.dirname(renders_file))
    scene_file = os.path.join(data_dir, SCENES_DIR, scene_filename)

    with open(scene_file, 'r') as f:
        scene_data = json.load(f)
    
    state = NewState.undump(scene_data)
    env = PRBEnv(num_objects=None)
    env.state = state
    problem_str = env.get_problem_file_str()

    with open(DOMAIN_FILE, 'r') as f:
        domain_str = f.read()
    
    return domain_str, problem_str, LLAMA_70B_INSTRUCT
    


def prb_process_datapoints(out_dir=None, se_class=None, **se_kwargs):
    process_datapoints(
        data_dir=os.path.join(THIS_DIR, 'data_dir'),
        domain=DOMAIN_FILE,
        problem=ALL_OBJECTS_PROBLEM_FILE,
        query_swapper=query_swapper,
        out_dir=out_dir,
        se_class=se_class,
        **se_kwargs
    )


if __name__ == "__main__":
    fire.Fire(prb_process_datapoints)
