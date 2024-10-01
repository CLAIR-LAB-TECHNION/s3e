import os
import sys
import json
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))

import argparse
import time
import numpy as np

from prb_env import PRBEnv
from semantic_state_estimator.constants import DP_FNAME_FORMAT, RENDERS_DIR, SCENES_DIR

PROBLEMS_DIR = 'problems'


def collect_datapoints(
    data_dir: str = 'data_dir',
    max_objects: int = 10,
    min_objects: int = 2,
    num_datapoints: int = 10_000,
    max_failures_to_reset: int = 5,
    max_reset_failures_to_crash: int = 50,
    max_episode_actions: int = 20,
):
    env = PRBEnv(4, use_gpu=1)#, width=500, height=500)

    # create output directory
    os.makedirs(os.path.join(data_dir, RENDERS_DIR), exist_ok=True)
    os.makedirs(os.path.join(data_dir, SCENES_DIR), exist_ok=True)
    # os.makedirs(os.path.join(data_dir, PROBLEMS_DIR), exist_ok=True)

    # initialize counters
    i = 0  # num datapoints saved
    action_count = 0  # num actions taken since last reset
    failures = 0  # num action failures since last reset
    start_time = time.time()

    while i < num_datapoints:
        next_datapoint_path = os.path.join(data_dir, SCENES_DIR, DP_FNAME_FORMAT.format(i) + '.json')
        if os.path.exists(next_datapoint_path):
            i += 1
            continue

        # reset on failure or action quota
        if action_count % max_episode_actions == 0 or failures >= max_failures_to_reset:

            reset_attempts = 0
            while True:
                # choose random number of objects
                num_objects = np.random.randint(min_objects, max_objects + 1)

                # reset env
                try:
                    env.reset(num_objects)
                    break
                except Exception as e:
                    reset_attempts += 1
                    if reset_attempts >= max_reset_failures_to_crash:
                        raise e

            # reset counters
            action_count = 0
            failures = 0

        try:
            env.state.action_move()  # runs a random move action
        except Exception as e:
            print(f"failed to execute action with error: {e}")
            failures += 1
            if failures >= max_failures_to_reset:
                print("resetting due to too many failures")
            continue

        # save renders
        render = env.render()
        np.savez_compressed(os.path.join(data_dir, RENDERS_DIR, DP_FNAME_FORMAT.format(i) + '.npz'), frontview=render)

        # save GT state as scene (load as PDDL state later)
        state_dict = env.state.dump()
        with open(next_datapoint_path, 'w') as f:
            json.dump(state_dict, f, indent=4)

        # # save PDDL problem as file
        # problem_str = env.get_problem_file_str()
        # with open(os.path.join(data_dir, PROBLEMS_DIR, DP_FNAME_FORMAT.format(i) + '.pddl'), 'w') as f:
        #     f.write(problem_str)

        # indicate progress
        i += 1  # up one datapoint
        action_count += 1  # up one action
        failures = 0  # reset failures counter

        time_diff = time.time() - start_time
        hours, remainder = divmod(time_diff, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_diff_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        print(f'num datapoints saved: {i}/{num_datapoints}\t\ttime elapsed: {time_diff_formatted}', file=sys.stderr)
        sys.stderr.flush()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', default='data_dir')
    # parser.add_argument('--max_objects', default=10)
    # parser.add_argument('--min_objects', default=2)
    # parser.add_argument('--num_datapoints', default=2000)
    # parser.add_argument('--max_failures_to_reset', default=5)
    # parser.add_argument('--max_episode_actions', default=20)
    # args = parser.parse_args()
    
    collect_datapoints(
        # args.data_dir,
        # args.max_objects,
        # args.min_objects,
        # args.num_datapoints,
        # args.max_failures,
        # args.max_episode_actions
    )
