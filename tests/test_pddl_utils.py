"""Tests for PDDL utility functions.

These tests use the unified-planning library directly (CPU-only, no models).
"""

import pytest

from s3e.pddl.up_utils import (
    create_up_problem,
    get_object_names_dict,
    get_all_grounded_predicates_for_objects,
    ground_predicate_str_to_fnode,
    convert_state_dict_to_up_compatible,
    state_dict_to_up_state,
)


BLOCKSWORLD_DOMAIN = """
(define (domain blocksworld)
  (:requirements :typing)
  (:types block)
  (:predicates
    (on ?x - block ?y - block)
    (clear ?x - block)
  )
  (:action move
    :parameters (?b - block ?from - block ?to - block)
    :precondition (and (on ?b ?from) (clear ?b) (clear ?to))
    :effect (and (on ?b ?to) (clear ?from) (not (on ?b ?from)) (not (clear ?to)))
  )
)
"""

BLOCKSWORLD_PROBLEM = """
(define (problem bw-2)
  (:domain blocksworld)
  (:objects a b - block)
  (:init (on a b) (clear a))
  (:goal (on b a))
)
"""


@pytest.fixture
def up_problem():
    return create_up_problem(BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM)


class TestCreateUpProblem:
    def test_from_strings(self):
        problem = create_up_problem(BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM)
        assert problem is not None
        assert problem.name == "bw-2"

    def test_fluents_parsed(self, up_problem):
        fluent_names = [f.name for f in up_problem.fluents]
        assert "on" in fluent_names
        assert "clear" in fluent_names


class TestGetObjectNamesDict:
    def test_returns_correct_objects(self, up_problem):
        objects = get_object_names_dict(up_problem)
        assert "block" in objects
        assert sorted(objects["block"]) == ["a", "b"]


class TestGetAllGroundedPredicates:
    def test_generates_all_combinations(self, up_problem):
        preds = get_all_grounded_predicates_for_objects(up_problem)
        assert len(preds) == 6
        assert "on(a,b)" in preds
        assert "clear(a)" in preds
        assert "on(a,a)" in preds

    def test_with_custom_objects(self, up_problem):
        custom_objects = {"block": ["a"]}
        preds = get_all_grounded_predicates_for_objects(up_problem, objects=custom_objects)
        assert len(preds) == 2


class TestGroundPredicateStrToFnode:
    def test_binary_predicate(self, up_problem):
        fnode = ground_predicate_str_to_fnode(up_problem, "on(a,b)")
        assert fnode.fluent().name == "on"
        assert len(fnode.args) == 2

    def test_unary_predicate(self, up_problem):
        fnode = ground_predicate_str_to_fnode(up_problem, "clear(a)")
        assert fnode.fluent().name == "clear"
        assert len(fnode.args) == 1


class TestStateConversion:
    def test_state_dict_to_up_compatible(self, up_problem):
        state_dict = {"on(a,b)": True, "clear(a)": False}
        up_dict = convert_state_dict_to_up_compatible(up_problem, state_dict)
        assert len(up_dict) == 2
        for v in up_dict.values():
            assert v.constant_value() in (True, False)

    def test_state_dict_to_up_state(self, up_problem):
        state_dict = {"on(a,b)": True, "clear(a)": True}
        state = state_dict_to_up_state(up_problem, state_dict)
        assert state is not None
