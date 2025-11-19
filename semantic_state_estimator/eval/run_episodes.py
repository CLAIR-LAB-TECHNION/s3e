import json
import os
from shutil import rmtree
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
from timeoutcontext import timeout
from tqdm.auto import tqdm
from unified_planning.engines.sequential_simulator import UPSequentialSimulator
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.shortcuts import Problem, UPState, OneshotPlanner, PlanValidator

from ..constants import (
    EPISODES_DIR,
    RENDERS_DIR,
    TRAJECTORY_STEP_FNAME_FORMAT,
    DONEFILE_NAME
)
from ..skill_executer import SkillExecuter
from ..state_estimator import StateEstimator, PredFnStateEstimator, ProbabilisticStateEstimator
from ..utils.up_utils import *


class EpisodeRunner:
    def __init__(
        self,
        problem: Problem,
        gt_estimator: PredFnStateEstimator,
        estimator: StateEstimator,
        executer: SkillExecuter,
        render_cams: list,
        data_dir: str,
        run_name: str,
        max_failures_to_reset: int = 5,
        max_episode_actions: int = 20,
        max_action_retries: int = 3,
        go_home_prob: float = 0.3,
        query_swapper=None,
        set_goal_condition=False,
        planner="fast-downward",
        planner_timeout=None,
    ):
        # create output directory
        self.episodes_dir = os.path.join(data_dir, EPISODES_DIR, run_name)
        os.makedirs(self.episodes_dir, exist_ok=True)

        self.problem = problem
        self.problem_sim = UPSequentialSimulator(problem)
        self.planner = OneshotPlanner(name=planner)
        self.planner_timeout = planner_timeout

        self.gt = gt_estimator
        self.pred = estimator
        self.exec = executer
        self.env = executer.env

        self.render_cams = render_cams
        self.max_failures_to_reset = max_failures_to_reset
        self.max_episode_actions = max_episode_actions
        self.max_action_retries = max_action_retries
        self.go_home_prob = go_home_prob
        self.query_swapper = query_swapper
        self.set_goal_condition = set_goal_condition

        self._cur_plan = None
        self._replan_count = 0
        self._last_prediction_time = None
        
        # a patch to enable adding the predicate probs
        self._prev_prediction_probs = None
        self._next_prediction_probs = None

    def get_env_renders(self):
        return {cam: self.exec.env._env.render(cam) for cam in self.render_cams}

    def sample_goal(self, state: UPState, task_horizon: int) -> UPState:
        """
        Samples a goal state that is reachable from the given state within at most `task_horizon` actions.
        """
        applicable_actions = list(self.problem_sim.get_applicable_actions(state))
        goal_state = state
        for _ in range(task_horizon):
            if not applicable_actions:
                break

            action, params = applicable_actions[
                np.random.choice(len(applicable_actions))
            ]
            goal_state = self.problem_sim.apply(goal_state, action, params)
            applicable_actions = list(
                self.problem_sim.get_applicable_actions(goal_state)
            )

        return goal_state

    def get_next_action(self, pred_state_dict):
        # set problem initial state
        set_problem_init_state(self.problem, pred_state_dict)

        # check if current plan is still valid
        if self._cur_plan is not None and len(self._cur_plan.actions) > 0:
            # validate plan
            with PlanValidator(
                problem_kind=self.problem.kind, plan_kind=self._cur_plan.kind
            ) as validator:

                if validator.validate(self.problem, self._cur_plan):
                    print('plan still valid. using next action')
                    # plan still valid. get next action
                    return self._cur_plan._actions.pop(0)

        # replan
        self._replan_count += 1

        # get new plan
        from unified_planning.exceptions import UPException
        try:
            print('planning...')
            start = time.time()
            plan_res = self.planner.solve(self.problem, timeout=self.planner_timeout)
            end = time.time()
            print('planning complete. status:', plan_res.status)
            print('planning time:', end - start)
            print('full results:', plan_res)
            sys.stdout.flush()
        except UPException:
            print('planning failed with UPException')
            print('problem:')
            print(self.problem)
            print('predicted state')
            print(pred_state_dict)
            raise
        
        # check if plan is satisficing
        if plan_res.status != PlanGenerationResultStatus.SOLVED_SATISFICING:
            return None
        self._cur_plan = plan_res.plan

        return self._cur_plan._actions.pop(0)

    def apply_action(self, action):

        # extract action name and params
        action_name, params = action.action.name, action.actual_parameters
        params = [str(param) for param in params]

        # run action
        for _ in range(self.max_action_retries):
            # attempt action
            try:
                with timeout(60):  # one minute timeout
                    suc, frames = self.exec.execute_action(action_name, params)
            except Exception as e:
                print(
                    f"failed to execute action {action_name}({','.join(params)}) with error: {e}"
                )
                continue  # failure. continue loop

            # go home with probability
            if np.random.rand() < self.go_home_prob:
                try:
                    with timeout(60):  # one minute timeout
                        suc, frames = self.exec.go_home()
                except Exception as e:
                    print("failed to go home. maybe physics state issue. resetting")
                    continue  # failure. continue loop

            # success
            return True

        # failure
        return False

    def gt_state(self):
        return self.gt(self.env.get_state())

    def se_predict(self, renders):
        imgs = [Image.fromarray(img) for img in renders.values()]
        start = time.time()
        if isinstance(self.pred, ProbabilisticStateEstimator):
            prob_map = self.pred.estimate_state(imgs)
            state_dict = {k: bool(v >= self.pred.confidence) for k, v in prob_map.items()}
            if self._prev_prediction_probs is None:
                self._prev_prediction_probs = prob_map
            else:
                self._next_prediction_probs = prob_map
        else:
            state_dict = self.pred(imgs)
        end = time.time()
        self._last_prediction_time = end - start
        return state_dict

    def save_render(self, save_dir, step_id, render):
        if render is not None:
            fname = os.path.join(
                save_dir,
                RENDERS_DIR,
                TRAJECTORY_STEP_FNAME_FORMAT.format(step_id) + ".npz",
            )
            if not os.path.exists(fname):
                np.savez_compressed(fname, **render)
        else:
            fname = None

        return fname

    def save_trajectory_data(
        self,
        save_dir,
        step_id,
        action,
        prev_obs,
        prev_state,
        prev_pred_state,
        next_obs=None,
        next_state=None,
        next_pred_state=None,
        reached_goal=None,
        predicted_goal=None,
        plan_exists=None
    ):
        # save unsaved observations
        prev_obs_fname = self.save_render(save_dir, step_id - 1, prev_obs)
        next_obs_fname = self.save_render(save_dir, step_id, next_obs)

        # save trajectory data
        step_fname = os.path.join(
            save_dir, TRAJECTORY_STEP_FNAME_FORMAT.format(step_id) + ".json"
        )
        with open(step_fname, "w") as f:
            json.dump(
                {
                    "step_id": step_id,
                    "action": action if action is None else str(action),
                    "prev_obs": prev_obs_fname,
                    "next_obs": next_obs_fname,
                    "prev_state": prev_state,
                    "next_state": next_state,
                    "prev_pred_state": prev_pred_state,
                    "next_pred_state": next_pred_state,
                    "prev_pred_probs": self._prev_prediction_probs,
                    "next_pred_probs": self._next_prediction_probs,
                    "reached_goal": reached_goal,
                    "predicted_goal": predicted_goal,
                    "replans": self._replan_count,
                    "prediction_time": self._last_prediction_time,
                    "plan_exists": plan_exists
                },
                f,
                indent=4,
            )

    def shift_trajcectory_and_save(
        self,
        save_dir,
        step_id,
        action,
        cur_obs_renders,
        state_dict,
        state,
        pred_state_dict,
        pred_state,
    ):
        ####################
        # Shift Trajectory #
        ####################

        # next observations
        prev_obs_render = cur_obs_renders
        cur_obs_renders = self.get_env_renders()

        # next gt state
        prev_state_dict = state_dict
        prev_state = state
        state_dict = self.gt_state()
        state = state_dict_to_up_state(self.problem, state_dict)

        # next predicted state
        prev_pred_state_dict = pred_state_dict
        prev_pred_state = pred_state
        if self._next_prediction_probs is not None:
            self._prev_prediction_probs = self._next_prediction_probs
            self._next_prediction_probs = None
        pred_state_dict = self.se_predict(cur_obs_renders)
        pred_state = state_dict_to_up_state(self.problem, pred_state_dict)

        # collect interesting data fro saving
        reached_goal = self.problem_sim.is_goal(state)
        predicted_goal = self.problem_sim.is_goal(pred_state)

        ###################
        # Save Trajectory #
        ###################

        self.save_trajectory_data(
            save_dir,
            step_id,
            action,
            prev_obs_render,
            prev_state_dict,
            prev_pred_state_dict,
            cur_obs_renders,
            state_dict,
            pred_state_dict,
            reached_goal,
            predicted_goal,
        )

        # return relevant data
        return (
            cur_obs_renders,
            state_dict,
            state,
            pred_state_dict,
            pred_state,
            predicted_goal,
        )

    def run_episode(self, task_horizon: int, out_dir: str):
        ################
        # Episode Init #
        ################

        print('env reset')
        sys.stdout.flush()

        # reset env
        self.exec.reset_env()
        self.exec.wait(100)  # wait for simulation to stabalize

        if self.query_swapper is not None:
            print('query swap')
            sys.stdout.flush()
            domain_str, problem_str, model_id = self.query_swapper(self.env)
            self.problem = create_up_problem(domain_str, problem_str)
            self.problem_sim = UPSequentialSimulator(self.problem)
            self.gt.up_problem = self.problem
            self.gt.all_ground_literals = list(self.gt.up_problem.initial_values.keys())
            self.pred.swap_queries(domain_str, problem_str, model_id)

        print('get states and stuff')
        sys.stdout.flush()
        # set current state
        state_dict = self.gt_state()
        state = state_dict_to_up_state(self.problem, state_dict)
        set_problem_init_state(self.problem, state_dict)

        # perceive initial state
        print('first render')
        sys.stdout.flush()
        self._last_prediction_time = None  # reset prediction variables
        self._prev_prediction_probs = None
        self._next_prediction_probs = None
        cur_obs_renders = self.get_env_renders()
        pred_state_dict = self.se_predict(cur_obs_renders)
        pred_state = state_dict_to_up_state(self.problem, pred_state_dict)

        # sample a goal state and set in problem object
        goal_state = self.sample_goal(state, task_horizon)
        goal_state_dict = up_state_to_state_dict(goal_state)
        if self.set_goal_condition:
            # leave only the changed values in the goal state as the condition.
            goal_state_dict = {k: v for k, v in goal_state_dict.items() if state_dict[k] != v}
        set_problem_goal_state(self.problem, goal_state_dict)

        # save domain and problem
        domain_str, problem_str = get_pddl_files_str(self.problem)
        with open(os.path.join(out_dir, 'domain.pddl'), 'w') as f:
            f.write(domain_str)
        with open(os.path.join(out_dir, 'problem.pddl'), 'w') as f:
            f.write(problem_str)

        # init counters and containers
        action_count = 0
        failures_count = 0
        self._replan_count = 0
        self._cur_plan = None

        # check if initial state is a goal state
        if self.problem_sim.is_goal(pred_state):
            is_true_goal = self.problem_sim.is_goal(state)
            print("initial predicted state is goal state. skipping episode")
            self.save_trajectory_data(
                out_dir,
                action_count,
                None,
                cur_obs_renders,
                state_dict,
                pred_state_dict,
                reached_goal=is_true_goal,
                predicted_goal=True,
                plan_exists=None if is_true_goal else self.get_next_action(state_dict) is not None
            )
            return

        ###############
        # Episode Run #
        ###############

        while (
            action_count < self.max_episode_actions
            and failures_count < self.max_failures_to_reset
        ):
            action_count += 1

            print(
                "attempting step\n"
                f"action count: {action_count}, attempt: {failures_count + 1}"
            )

            # get next action
            print("getting next action")
            action = self.get_next_action(pred_state_dict)
            if action is None:
                is_true_goal = self.problem_sim.is_goal(state)
                print("no plan from current predicted state. resetting")
                if self._next_prediction_probs is not None:
                    self._prev_prediction_probs = self._next_prediction_probs  # shift is skipped
                    self._next_prediction_probs = None
                self.save_trajectory_data(
                    out_dir,
                    action_count,
                    action,
                    cur_obs_renders,
                    state_dict,
                    pred_state_dict,
                    reached_goal=is_true_goal,
                    predicted_goal=False,
                    plan_exists=None if is_true_goal else self.get_next_action(state_dict) is not None
                )
                break

            # apply action
            print("applying action")
            suc = self.apply_action(action)

            # check for failure
            if not suc:
                print("action failed. retrying")
                failures_count += 1
            else:
                # action succeeded. reset failures count
                print("action completed successfully")
                failures_count = 0

            # shift and save trajectory data
            print("shifting trajectory")
            (
                cur_obs_renders,
                state_dict,
                state,
                pred_state_dict,
                pred_state,
                predicted_goal,
            ) = self.shift_trajcectory_and_save(
                out_dir,
                action_count,
                action,
                cur_obs_renders,
                state_dict,
                state,
                pred_state_dict,
                pred_state,
            )

            # check if goal is reached
            print("checking goal condition")
            if predicted_goal:
                print("predicting goal reached")
                break

    def run(self, num_episodes: int, task_horizon: int):
        for i in tqdm(range(num_episodes)):
            # This is the directory with all data for this episode
            this_episode_dir = os.path.join(self.episodes_dir, f"episode_{i:04d}")

            # check if it is done
            donefile = os.path.join(this_episode_dir, DONEFILE_NAME)
            if os.path.exists(donefile):
                print(f"skipping episode {i} as it is already done")
                continue
            elif os.path.exists(this_episode_dir):
                print(f"removing existing episode {i} data (incomplete episode)")
                rmtree(this_episode_dir)

            print(f'logging to {os.path.abspath(this_episode_dir)}')

            # create a directory for renderings
            os.makedirs(os.path.join(this_episode_dir, RENDERS_DIR), exist_ok=True)

            # run episode
            print(f'starting episode {i}')
            self.run_episode(task_horizon, this_episode_dir)

            # mark episode as done
            print(f'episode {i} completed successfully')
            Path(donefile).touch()

    def __del__(self):
        self.planner.destroy()
