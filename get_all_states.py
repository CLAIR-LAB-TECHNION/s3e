from semantic_state_estimator.utils.up_utils import create_up_problem, up_state_to_state_dict
from unified_planning.engines.sequential_simulator import UPSequentialSimulator
from unified_planning.shortcuts import UPState
from tqdm.auto import tqdm
from frozendict import frozendict
import json


states = set()

problem = create_up_problem("examples/gw_grip/domain.pddl", "examples/gw_grip/problem.pddl")
sim = UPSequentialSimulator(problem)

init_state = UPState(problem.initial_values)

frozen_init_state_dict = frozendict(up_state_to_state_dict(init_state))

states.add(frozen_init_state_dict)

states_queue = [init_state]

pbar = tqdm()
while states_queue:
    state = states_queue.pop(0)
    actions = sim.get_applicable_actions(state)
    for action, params in actions:
        new_state = sim.apply(state, action, params)
        new_state_frozen = frozendict(up_state_to_state_dict(new_state))
        if new_state_frozen not in states:
            states.add(new_state_frozen)
            states_queue.append(new_state)
            pbar.update()

out = [dict(frozen_state_dict) for frozen_state_dict in states]
with open('all_state.json', 'w') as f:
    json.dump(out, f, indent=4)
