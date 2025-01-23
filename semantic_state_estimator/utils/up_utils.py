from itertools import product
import os
from typing import Optional
import json

import numpy as np
from tqdm.auto import tqdm
from unified_planning.io import PDDLReader, PDDLWriter
from unified_planning.shortcuts import Problem, UPState, FNode
from unified_planning.engines.sequential_simulator import UPSequentialSimulator


def create_up_problem(domain: str, problem: str) -> Problem:
    reader = PDDLReader()
    if domain.lower().endswith(".pddl"):
        assert problem.lower().endswith(
            ".pddl"
        ), "if domain is a file, problem must also be a file"
        up_problem = reader.parse_problem(domain, problem)
    else:
        up_problem = reader.parse_problem_string(domain, problem)

    return up_problem


def get_object_names_dict(up_problem: Problem) -> dict[str, list[str]]:
    objects = {}
    for t in up_problem.user_types:
        objects[t.name] = list(map(str, up_problem.objects(t)))

    return objects


def get_all_grounded_predicates_for_objects(
    up_problem: Problem, objects: Optional[dict[str, list[str]]] = None
) -> list[str]:
    predicates = up_problem.fluents
    if objects is None:
        objects = get_object_names_dict(up_problem)

    grounded_predicates = []
    for p in predicates:
        varlists = []
        for variable in p.signature:
            varlists.append(objects[variable.type.name])
        for assignment in product(*varlists):
            grounded_predicates.append(f'{p.name}({",".join(assignment)})')

    return grounded_predicates


def get_pddl_files_str(up_problem: Problem) -> tuple[str, str]:
    writer = PDDLWriter(up_problem)
    return writer.get_domain(), writer.get_problem()


def ground_predicate_str_to_fnode(up_problem: Problem, predicate_str: str) -> FNode:
    fluent_name, args = predicate_str.split("(")
    args = args.rstrip(")").split(",")
    args = [arg.strip() for arg in args if arg]
    pred_obj = up_problem.fluent(fluent_name)
    arg_obj = [up_problem.object(a) for a in args]
    if arg_obj:
        return pred_obj(*arg_obj)
    else:
        return pred_obj()


def bool_constant_to_fnode(up_problem: Problem, constant: bool) -> FNode:
    exp_mgr = up_problem.environment.expression_manager
    if constant is True:
        return exp_mgr.true_expression
    else:
        return exp_mgr.false_expression


def convert_state_dict_to_up_compatible(
    up_problem, state_dict: dict[str, bool]
) -> dict[FNode, FNode]:
    return {
        ground_predicate_str_to_fnode(up_problem, k): bool_constant_to_fnode(
            up_problem, v
        )
        for k, v in state_dict.items()
    }


def state_dict_to_up_state(up_problem: Problem, state_dict: dict[str, bool]) -> UPState:
    return UPState(convert_state_dict_to_up_compatible(up_problem, state_dict))


def up_state_to_state_dict(up_state: UPState) -> dict[str, bool]:
    current_instance = up_state
    out = {}
    while current_instance is not None:
        for k, v in current_instance._values.items():
            out.setdefault(
                f'{k.fluent().name}({",".join(map(str, k.args))})', v.constant_value()
            )
        current_instance = current_instance._father

    return out


def set_problem_init_state(up_problem: Problem, init_state_dict: dict[str, bool]):
    # clear existing fluents
    up_problem.explicit_initial_values.clear()

    # set desired fluents
    for k, v in convert_state_dict_to_up_compatible(up_problem, init_state_dict).items():
        up_problem.set_initial_value(k, v)


def set_problem_goal_state(up_problem: Problem, goal_state_dict: dict[str, bool]):
    # clear existing goals
    up_problem.clear_goals()

    # set desired goals
    for k, v in goal_state_dict.items():
        if v is True:
            up_problem.add_goal(
                ground_predicate_str_to_fnode(up_problem, k),
            )


def collect_pddl_datapoints(
    up_problem: Problem,
    num_steps: int,
    out_dir: str,
    max_episode_steps: Optional[int] = None,
):
    # create output directory
    os.makedirs(out_dir, exist_ok=True)

    # initialize simulator
    sim = UPSequentialSimulator(up_problem)

    # initialize initial state
    state_dict = up_problem.initial_values
    state = UPState(state_dict)

    for i in tqdm(range(num_steps)):
        if max_episode_steps is not None and i % max_episode_steps == 0:
            # initialize initial state
            state_dict = up_problem.initial_values
            state = UPState(state_dict)

        state_dict = up_state_to_state_dict(state)
        with open(os.path.join(out_dir, f"state_{i:06d}.json"), "w") as f:
            json.dump(state_dict, f, indent=4)

        applicable_actions = list(sim.get_applicable_actions(state))
        action_idx = np.random.choice(range(len(applicable_actions)))
        action, params = applicable_actions[action_idx]
        state = sim.apply(state, action, params)
