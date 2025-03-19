import json
import os
import sys
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
)
from ..skill_executer import SkillExecuter
from ..state_estimator import StateEstimator, PredFnStateEstimator
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
        query_swapper=None
    ):
        # create output directory
        self.episodes_dir = os.path.join(data_dir, EPISODES_DIR, run_name)
        os.makedirs(self.episodes_dir, exist_ok=True)

        self.problem = problem
        self.problem_sim = UPSequentialSimulator(problem)
        self.planner = OneshotPlanner(name="fast-downward")

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

        self._cur_plan = None
        self._replan_count = 0

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
                    # plan still valid. get next action
                    return self._cur_plan._actions.pop(0)

        # replan
        self._replan_count += 1

        # get new plan
        plan_res = self.planner.solve(self.problem)
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
        prob_map = self.pred(imgs)
        return prob_map

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
                    "reached_goal": reached_goal,
                    "predicted_goal": predicted_goal,
                    "replans": self._replan_count,
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

        print('query swap')
        sys.stdout.flush()
        if self.query_swapper is not None:
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

        # perceive initial state
        print('first render')
        sys.stdout.flush()
        cur_obs_renders = self.get_env_renders()
        pred_state_dict = self.se_predict(cur_obs_renders)
        pred_state = state_dict_to_up_state(self.problem, pred_state_dict)

        # sample a goal state and set in problem object
        goal_state = self.sample_goal(state, task_horizon)
        goal_state_dict = up_state_to_state_dict(goal_state)
        set_problem_goal_state(self.problem, goal_state_dict)

        # init counters
        action_count = 0
        failures_count = 0

        # check if initial state is a goal state
        if self.problem_sim.is_goal(pred_state):
            print("initial predicted state is goal state. skipping episode")
            self.save_trajectory_data(
                out_dir,
                action_count,
                None,
                cur_obs_renders,
                state_dict,
                pred_state_dict,
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

            # get next action
            action = self.get_next_action(pred_state_dict)
            if action is None:
                print("no plan from current predicted state. resetting")
                self.save_trajectory_data(
                    out_dir,
                    action_count,
                    action,
                    cur_obs_renders,
                    state_dict,
                    pred_state_dict,
                )
                break

            # apply action
            suc = self.apply_action(action)

            # check for failure
            if not suc:
                failures_count += 1

            # shift and save trajectory data
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

            if predicted_goal:
                print("predicting goal reached")
                break

    def run(self, num_episodes: int, task_horizon: int):
        for i in tqdm(range(num_episodes)):
            this_episode_dir = os.path.join(self.episodes_dir, f"episode_{i:04d}")
            os.makedirs(os.path.join(this_episode_dir, RENDERS_DIR), exist_ok=True)
            self._replan_count = 0
            self.run_episode(task_horizon, this_episode_dir)

    def __del__(self):
        self.planner.destroy()
