"""Utilities for working with the Unified Planning framework and PDDL.

This module provides functions for PDDL parsing, predicate grounding,
and state conversion between s3e's ``dict[str, bool]`` format and
Unified Planning's native types.
"""

from itertools import product
from typing import Optional

from unified_planning.io import PDDLReader, PDDLWriter
from unified_planning.shortcuts import Problem, UPState, FNode


def create_up_problem(domain: str, problem: str) -> Problem:
    """Create a Unified Planning problem from PDDL files or strings."""
    reader = PDDLReader()
    if domain.lower().endswith(".pddl"):
        assert problem.lower().endswith(
            ".pddl"
        ), "if domain is a file, problem must also be a file"
        return reader.parse_problem(domain, problem)
    return reader.parse_problem_string(domain, problem)


def get_object_names_dict(up_problem: Problem) -> dict[str, list[str]]:
    """Get a dictionary mapping object types to lists of object names."""
    objects: dict[str, list[str]] = {}
    for t in up_problem.user_types:
        objects[t.name] = list(map(str, up_problem.objects(t)))
    return objects


def get_all_grounded_predicates_for_objects(
    up_problem: Problem, objects: Optional[dict[str, list[str]]] = None
) -> list[str]:
    """Generate all possible grounded predicates for the given objects."""
    predicates = up_problem.fluents
    if objects is None:
        objects = get_object_names_dict(up_problem)

    grounded_predicates: list[str] = []
    for p in predicates:
        varlists = [objects[variable.type.name] for variable in p.signature]
        for assignment in product(*varlists):
            grounded_predicates.append(f'{p.name}({",".join(assignment)})')

    return grounded_predicates


def get_pddl_strings(up_problem: Problem) -> tuple[str, str]:
    """Convert a Unified Planning problem to PDDL domain and problem strings."""
    writer = PDDLWriter(up_problem)
    return writer.get_domain(), writer.get_problem()


def ground_predicate_str_to_fnode(up_problem: Problem, predicate_str: str) -> FNode:
    """Convert a grounded predicate string to a Unified Planning FNode."""
    fluent_name, args = predicate_str.split("(")
    args = args.rstrip(")").split(",")
    args = [arg.strip() for arg in args if arg]
    pred_obj = up_problem.fluent(fluent_name)
    arg_obj = [up_problem.object(a) for a in args]
    if arg_obj:
        return pred_obj(*arg_obj)
    return pred_obj()


def bool_constant_to_fnode(up_problem: Problem, constant: bool) -> FNode:
    """Convert a boolean constant to a Unified Planning FNode."""
    exp_mgr = up_problem.environment.expression_manager
    if constant:
        return exp_mgr.true_expression
    return exp_mgr.false_expression


def convert_state_dict_to_up_compatible(
    up_problem: Problem, state_dict: dict[str, bool]
) -> dict[FNode, FNode]:
    """Convert a state dictionary to Unified Planning compatible format."""
    return {
        ground_predicate_str_to_fnode(up_problem, k): bool_constant_to_fnode(
            up_problem, v
        )
        for k, v in state_dict.items()
    }


def state_dict_to_up_state(
    up_problem: Problem, state_dict: dict[str, bool]
) -> UPState:
    """Convert a state dictionary to a Unified Planning state."""
    return UPState(convert_state_dict_to_up_compatible(up_problem, state_dict), up_problem)
