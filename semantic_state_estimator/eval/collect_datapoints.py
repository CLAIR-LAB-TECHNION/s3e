import json
import os

import numpy as np
from timeoutcontext import timeout
from tqdm.auto import tqdm
from unified_planning.engines.sequential_simulator import UPSequentialSimulator
from unified_planning.shortcuts import Problem

from ..constants import RENDERS_DIR, TRUE_STATES_DIR, DP_FNAME_FORMAT
from ..skill_executer import SkillExecuter
from ..state_estimator import StateEstimator
from ..utils.up_utils import state_dict_to_up_state


def collect_data(
        problem: Problem,
        estimator: StateEstimator,
        executer: SkillExecuter,
        render_cams: list,
        data_dir: str,
        num_datapoints: int = 2000,
        max_failures_to_reset: int = 5,
        max_episode_actions: int = 20,
        go_home_prob: float = 0.3,
):
    # create output directory
    os.makedirs(os.path.join(data_dir, RENDERS_DIR), exist_ok=True)
    os.makedirs(os.path.join(data_dir, TRUE_STATES_DIR), exist_ok=True)

    # instantiate a simulator. this can tell us which actions are applicable from a given state
    problem_sim = UPSequentialSimulator(problem)

    # initialize counters
    i = 0  # num datapoints saved
    action_count = 0  # num actions taken since last reset
    failures = 0  # num action failures since last reset
    ppar = tqdm(total=num_datapoints)
    while i < num_datapoints:
        # reset on failure or action quota
        if action_count % max_episode_actions == 0 or failures >= max_failures_to_reset:
            # reset env
            executer.reset_env()
            executer.wait(100)  # wait for simulation to stabalize

            # set current state
            state_dict = estimator(executer.env.get_state())
            state = state_dict_to_up_state(problem, state_dict)

            # reset counters
            action_count = 0
            failures = 0

        # get applicable actions
        action_dict = {
            j: action
            for j, action in enumerate(problem_sim.get_applicable_actions(state))
        }
        if not action_dict:  # force reset if no actions are available
            print(f"resetting due to no actions to perform at state {state}")
            failures = max_failures_to_reset  # forces reset on iter start
            continue

        # choose a random applicable action
        action, params = action_dict[np.random.choice(list(action_dict.keys()))]

        # extract action name and params
        action_name = action.name
        params = [str(param) for param in params]

        # run action
        try:
            with timeout(60):  # one minute timeout
                suc, frames = executer.execute_action(action_name, params)
        except Exception as e:
            print(f"failed to execute action {action_name}({','.join(params)}) with error: {e}")
            failures += 1
            if failures >= max_failures_to_reset:
                print("resetting due to too many failures")
            continue

        # go home with probability
        if np.random.rand() < go_home_prob:
            try:
                with timeout(60):  # one minute timeout
                    suc, frames = executer.go_home()
            except Exception as e:
                print("failed to go home. maybe physics state issue. resetting")
                failures = max_failures_to_reset  # forces reset on iter start
                continue

        # save renders
        renders = {cam: executer.env._env.render(cam) for cam in render_cams}
        np.savez_compressed(os.path.join(data_dir, RENDERS_DIR, DP_FNAME_FORMAT.format(i) + '.npz'), **renders)

        # save GT state
        env_state = executer.env.get_state()
        state_dict = estimator(env_state)
        with open(os.path.join(data_dir, TRUE_STATES_DIR, DP_FNAME_FORMAT.format(i) + '.json'), 'w') as f:
            json.dump(state_dict, f, indent=4)

        # update state
        state = state_dict_to_up_state(problem, state_dict)

        # indicate progress
        i += 1  # up one datapoint
        action_count += 1  # up one action
        failures = 0  # reset failures counter
        ppar.update()  # update progress bar
