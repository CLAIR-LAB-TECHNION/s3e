import json
from itertools import product

import fire

from prb_env import COLOR_TO_NAME, PROBLEM_TEMPLATE, props_to_block_name

def gen_all_object_problem(outfile='all_objects_problem.pddl'):
    all_colors = list(COLOR_TO_NAME.values())

    with open("photorealistic_blocksworld/data/properties.json", "r") as f:
        props = json.load(f)

    all_shapes = list(props["shapes"].keys())
    all_mats = list(props["materials"].keys())
    all_sizes = list(props["sizes"].keys())

    all_object_names = []
    for size, color, mat, shape in product(all_sizes, all_colors, all_mats, all_shapes):
        all_object_names.append(props_to_block_name(size, color, mat, shape))

    objects_str = ' '.join(all_object_names)
    problem_str = PROBLEM_TEMPLATE.format(objects=objects_str)

    with open(outfile, 'w') as f:
        f.write(problem_str)

if __name__ == "__main__":
    fire.Fire(gen_all_object_problem)

    
