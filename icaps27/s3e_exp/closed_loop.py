# SPDX-License-Identifier: MIT
"""Closed-loop execution: observe, estimate, plan, act, and re-observe.

The agent never sees the true state. At every step it estimates the state
from the current images, and

* ``open-loop`` plans once from the first estimate and executes the whole
  plan blind;
* ``replan`` plans from a fresh estimate before every action;
* ``monitor`` keeps its plan while each new estimate matches the state the
  plan predicted, and replans as soon as it does not (execution monitoring).
  With ``check="action"``, between replans it only asks about the atoms the
  last action changed and the next action needs (S3E's ``predicates=``), so
  most steps cost a few queries instead of a full estimate.

An episode ends when the agent believes the goal holds (its plan is empty),
when no plan exists from its first estimate, or after ``max_steps`` actions.
A later estimate the planner cannot plan from (often a physically
impossible state) does not end the episode: the agent keeps the state its
plan predicted and the rest of the plan, and the event is counted. Success
means the goal holds in the true state when the agent stops. Inapplicable
actions fail without changing the world, as on a robot whose grasp misses;
the agent never executes an action its own model says is inapplicable.
"""

import time

from unified_planning.engines.sequential_simulator import UPSequentialSimulator

from .analysis import decode, to_state
from .estimation import estimate
from .planning import parse_action, solve, with_initial_state


def atom_string(fnode) -> str:
    return f"{fnode.fluent().name}({','.join(str(a) for a in fnode.args)})"


def action_atoms(problem, action: str) -> set[str]:
    """Grounded atoms in an action instance's preconditions and effects."""
    name, args = parse_action(action)
    up_action = problem.action(name)
    env = problem.environment
    manager = env.expression_manager
    substitution = {manager.ParameterExp(p): manager.ObjectExp(problem.object(a))
                    for p, a in zip(up_action.parameters, args)}
    expressions = list(up_action.preconditions)
    for effect in up_action.effects:
        expressions += [effect.fluent, effect.condition]
    atoms = set()
    for expression in expressions:
        grounded = env.substituter.substitute(expression, substitution)
        atoms |= {atom_string(f) for f in env.free_vars_extractor.get(grounded)}
    return atoms


def successor(problem, state: dict[str, bool], action: str) -> "dict[str, bool] | None":
    """``state`` after ``action`` under the domain model, or None if inapplicable."""
    believed = with_initial_state(problem, state)
    simulator = UPSequentialSimulator(believed, error_on_failed_checks=False)
    current = simulator.get_initial_state()
    name, args = parse_action(action)
    up_action = believed.action(name)
    params = tuple(believed.object(a) for a in args)
    if not simulator.is_applicable(current, up_action, params):
        return None
    following = simulator.apply(current, up_action, params)
    result = {}
    for atom in state:
        fnode = _fnode(believed, atom)
        result[atom] = following.get_value(fnode).bool_constant_value()
    return result


def _fnode(problem, atom: str):
    from s3e.pddl import ground_predicate_str_to_fnode

    return ground_predicate_str_to_fnode(problem, atom)


def run_episode(world, problem, estimator, backend, atoms: list[str], fixed: dict[str, bool],
                *, policy: str = "monitor", check: str = "full", decoder: str = "threshold",
                calibrator=None, domain=None, objects=None, views=None, max_steps: int = 50,
                planner: str = "fast-downward", timeout: float = 60.0) -> dict:
    """Run one episode; returns the outcome and a per-step log."""
    if policy not in ("open-loop", "replan", "monitor"):
        raise ValueError(f"Unknown policy {policy!r}")
    log, queries, plan_calls, estimates = [], 0, 0, 0
    vlm_seconds = plan_seconds = 0.0
    plan: list[str] = []
    expected: "dict[str, bool] | None" = None
    started = time.perf_counter()

    def observe(subset):
        nonlocal queries, estimates, vlm_seconds
        results, timing = estimate(estimator, backend, None, subset, world.observe(views),
                                   truth=world.state())
        if calibrator is not None:
            results = calibrator.apply(results)
        queries += timing["queries"]
        estimates += 1
        vlm_seconds += timing["seconds"]
        if len(subset) == len(atoms):
            return decode(results, decoder, domain=domain, objects=objects)
        return to_state(results)

    def replan(belief):
        nonlocal plan_calls, plan_seconds
        plan_calls += 1
        outcome = solve(with_initial_state(problem, {**fixed, **belief}), planner, timeout)
        plan_seconds += outcome["seconds"]
        return outcome

    def restrict(state):
        return {a: v for a, v in state.items() if a in belief}

    belief = observe(atoms)
    outcome = replan(belief)
    status = "planner_" + outcome["status"] if outcome["plan"] is None else None
    plan = list(outcome["plan"] or [])
    replans = failed_actions = unplannable = guards = 0
    while status is None:
        if not plan:
            status = "believed_goal"
            break
        if len(log) >= max_steps or guards > max_steps:
            status = "step_limit"
            break
        expected = successor(problem, {**fixed, **belief}, plan[0])
        if expected is None:
            # The agent's own model says the next action cannot run: re-estimate
            # and replan instead of executing it.
            guards += 1
            belief = observe(atoms) if policy != "open-loop" else belief
            outcome = replan(belief)
            replans += 1
            if outcome["plan"] is None:
                status = "planner_" + outcome["status"]
            plan = list(outcome["plan"] or [])
            continue
        action = plan.pop(0)
        ok = world.apply(*parse_action(action))
        failed_actions += not ok
        entry = {"action": action, "applied": ok}
        predicted = restrict(expected)
        if policy == "open-loop":
            belief = predicted
            log.append(entry)
            continue
        if policy == "monitor" and check == "action" and plan:
            subset = sorted((action_atoms(problem, action) | action_atoms(problem, plan[0]))
                            & set(atoms))
        else:
            subset = atoms
        observed = observe(subset)
        mismatch = any(predicted.get(a) != v for a, v in observed.items())
        entry["queried"] = len(subset)
        entry["mismatch"] = mismatch
        if policy == "replan" or mismatch:
            if len(subset) < len(atoms):
                observed = observe(atoms)
            outcome = replan(observed)
            replans += 1
            if outcome["plan"] is None:
                # No plan from the new estimate (e.g. a block resting on nothing):
                # the estimate is likelier wrong than the world, so keep the state
                # the plan predicted and the rest of the plan.
                unplannable += 1
                entry["unplannable_estimate"] = True
                belief = predicted
            else:
                belief = observed
                plan = list(outcome["plan"])
                entry["replanned"] = True
        else:
            belief = {**predicted, **observed}
        log.append(entry)
    return {
        "status": status,
        "success": world.goal_reached(),
        "steps": len(log),
        "failed_actions": failed_actions,
        "replans": replans,
        "unplannable_estimates": unplannable,
        "planner_calls": plan_calls,
        "estimates": estimates,
        "queries": queries,
        "vlm_seconds": vlm_seconds,
        "planner_seconds": plan_seconds,
        "seconds": time.perf_counter() - started,
        "log": log,
    }
