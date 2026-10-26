# SPDX-License-Identifier: MIT
"""Plan from an estimated state and check the plan against the true state.

The estimated problem is the true problem with its initial values replaced
by the estimate: same objects, actions, and goal. It is solved with a Unified
Planning one-shot planner (Fast Downward by default), and the plan is then
replayed in the *true* initial state with UP's sequential simulator, which
tells whether every action was applicable (the plan is executable) and
whether the goal holds at the end (the plan is valid).
"""

import os
import tempfile
import time
from contextlib import contextmanager

import unified_planning as up
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.engines.sequential_simulator import UPSequentialSimulator
from unified_planning.shortcuts import OneshotPlanner

from s3e.pddl import ground_predicate_str_to_fnode, parse_domain_problem

up.shortcuts.get_environment().credits_stream = None

SOLVED = (
    PlanGenerationResultStatus.SOLVED_SATISFICING,
    PlanGenerationResultStatus.SOLVED_OPTIMALLY,
)
UNSOLVABLE = (
    PlanGenerationResultStatus.UNSOLVABLE_PROVEN,
    PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
)


def parse(domain_pddl: str, problem_pddl: str):
    return parse_domain_problem(domain_pddl, problem_pddl)


def with_initial_state(problem, state: dict[str, bool]):
    """A copy of ``problem`` whose initial value of each atom in ``state`` is overridden."""
    estimated = problem.clone()
    manager = estimated.environment.expression_manager
    for atom, value in state.items():
        fnode = ground_predicate_str_to_fnode(estimated, atom)
        estimated.set_initial_value(fnode, manager.TRUE() if value else manager.FALSE())
    return estimated


def action_string(action_instance) -> str:
    args = ",".join(str(p) for p in action_instance.actual_parameters)
    return f"{action_instance.action.name}({args})"


def parse_action(action: str) -> tuple[str, list[str]]:
    name, _, rest = action.partition("(")
    return name, [a.strip() for a in rest.rstrip(")").split(",") if a.strip()]


@contextmanager
def _private_cwd():
    """Fast Downward writes its intermediate files to the working directory, so
    concurrent planners (processes, Slurm jobs) each need their own."""
    previous = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="s3e-exp-planner-") as directory:
        os.chdir(directory)
        try:
            yield
        finally:
            os.chdir(previous)


def solve(problem, planner: str = "fast-downward", timeout: float = 60.0) -> dict:
    """Solve ``problem``; returns status, plan (action strings), and seconds."""
    start = time.perf_counter()
    try:
        with _private_cwd(), OneshotPlanner(name=planner) as engine:
            result = engine.solve(problem, timeout=timeout)
    except Exception as error:  # a planner crash is an outcome, not a harness failure
        return {"status": "error", "error": repr(error), "plan": None,
                "seconds": time.perf_counter() - start}
    seconds = time.perf_counter() - start
    if result.status in SOLVED:
        return {"status": "solved", "plan": [action_string(a) for a in result.plan.actions],
                "seconds": seconds}
    if result.status in UNSOLVABLE:
        status = "unsolvable"
    elif result.status == PlanGenerationResultStatus.TIMEOUT:
        status = "timeout"
    else:
        status = result.status.name.lower()
    return {"status": status, "plan": None, "seconds": seconds}


def validate(problem, plan: list[str]) -> dict:
    """Replay ``plan`` from ``problem``'s initial state.

    Returns ``executable`` (every action applicable), ``failed_step`` (index
    of the first inapplicable action, or None), and ``goal_reached``.
    """
    simulator = UPSequentialSimulator(problem)
    state = simulator.get_initial_state()
    for step, action in enumerate(plan):
        name, args = parse_action(action)
        up_action = problem.action(name)
        params = tuple(problem.object(a) for a in args)
        if not simulator.is_applicable(state, up_action, params):
            return {"executable": False, "failed_step": step, "goal_reached": False}
        state = simulator.apply(state, up_action, params)
    return {"executable": True, "failed_step": None,
            "goal_reached": bool(simulator.is_goal(state))}


def plan_and_validate(domain_pddl: str, problem_pddl: str, state: dict[str, bool],
                      planner: str = "fast-downward", timeout: float = 60.0) -> dict:
    """Plan from ``state`` (overriding the true initial values) and validate
    the plan in the true problem. Picklable inputs, for process pools."""
    truth = parse(domain_pddl, problem_pddl)
    outcome = solve(with_initial_state(truth, state), planner, timeout)
    if outcome["plan"] is not None:
        outcome.update(validate(truth, outcome["plan"]))
        outcome["valid"] = outcome["goal_reached"]
    else:
        outcome.update({"executable": False, "failed_step": None, "goal_reached": False,
                        "valid": False})
    outcome["plan_length"] = len(outcome["plan"]) if outcome["plan"] is not None else None
    return outcome
