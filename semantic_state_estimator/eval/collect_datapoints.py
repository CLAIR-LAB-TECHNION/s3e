import os
import pickle

import numpy as np
from tqdm.auto import tqdm
from unified_planning.shortcuts import Problem, UPState
from unified_planning.engines.sequential_simulator import UPSequentialSimulator
from wavedrom import render

from semantic_state_estimator.skill_executer import SkillExecuter
from semantic_state_estimator.state_estimator import StateEstimator
from semantic_state_estimator.constants import TRUE_STATE_KEY, RENDERS_KEY


def collect_data(
    problem: Problem,
    estimator: StateEstimator,
    executer: SkillExecuter,
    render_cams: list,
    out_dir: str,
    num_datapoints: int = 2000,
    max_failures_to_reset: int = 5,
    max_episode_actions: int = 20,
):
    # create output directory
    os.makedirs(out_dir, exist_ok=True)

    problem_sim = UPSequentialSimulator(problem)

    i = 0
    action_count = 0
    failures = 0
    ppar = tqdm(total=num_datapoints)
    while i < num_datapoints:
        if action_count % max_episode_actions == 0 or failures >= max_failures_to_reset:
            executer.reset_env()
            action_count = 0
            failures = 0
            executer.wait(100)  # wait for simulation to stabalize
            state = UPState(estimator(executer.env.get_state()))

        action_dict = {
            j: action
            for j, action in enumerate(problem_sim.get_applicable_actions(state))
        }
        if not action_dict:
            print(f"resetting due to no actions to perform at state {state}")
            failures = max_failures_to_reset
            continue
        action, params = action_dict[np.random.choice(list(action_dict.keys()))]
        action_name = action.name
        params = [str(param) for param in params]
        try:
            suc, frames = executer.execute_action(action_name, params)
        except Exception as e:
            print(f"failed to execute action {action_name}({','.join(params)}) with error: {e}")
            failures += 1
            if failures >= max_failures_to_reset:
                print("resetting due to too many failures")
            continue
        if np.random.rand() < 0.3:
            try:
                suc, frames = executer.go_home()
            except Exception as e:
                print("failed to go home. maybe physics state issue. resetting")
                failures = max_failures_to_reset
                continue

        env_state = executer.env.get_state()
        state_dict = estimator(env_state)
        data_point = {
            RENDERS_KEY: {cam: executer.env._env.render(cam) for cam in render_cams},
            TRUE_STATE_KEY: state_dict,
        }
        with open(os.path.join(out_dir, f"data_point_{i}.pkl"), "wb") as f:
            pickle.dump(data_point, f)

        state = UPState(state_dict)
        i += 1
        action_count += 1
        ppar.update()
