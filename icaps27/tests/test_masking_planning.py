# SPDX-License-Identifier: MIT
from s3e_exp.domains import blocksworld as bw
from s3e_exp.masking import fluent_analysis, select_atoms
from s3e_exp.planning import parse, plan_and_validate

# A grid domain with a static adjacency relation and a lamp that never
# matters for reaching the goal.
GRID_DOMAIN = """(define (domain grid)
  (:requirements :strips :typing)
  (:types cell)
  (:predicates (at ?c - cell) (adjacent ?a - cell ?b - cell) (lamp-on ?c - cell))
  (:action move :parameters (?a - cell ?b - cell)
    :precondition (and (at ?a) (adjacent ?a ?b))
    :effect (and (not (at ?a)) (at ?b)))
  (:action switch :parameters (?c - cell)
    :precondition (at ?c)
    :effect (lamp-on ?c)))
"""
GRID_PROBLEM = """(define (problem g) (:domain grid)
  (:objects c1 c2 c3 - cell)
  (:init (at c1) (adjacent c1 c2) (adjacent c2 c3))
  (:goal (at c3)))
"""


def test_fluent_roles_and_masks_on_a_grid():
    problem = parse(GRID_DOMAIN, GRID_PROBLEM)
    roles = fluent_analysis(problem)
    assert roles["static"] == {"adjacent"}
    assert roles["dynamic"] == {"at", "lamp-on"}
    everything, fixed = select_atoms(problem, "none")
    assert len(everything) == 3 + 9 + 3 and not fixed
    dynamic, fixed = select_atoms(problem, "dynamic")
    assert sorted(dynamic) == sorted([f"at(c{i})" for i in (1, 2, 3)]
                                     + [f"lamp-on(c{i})" for i in (1, 2, 3)])
    assert fixed["adjacent(c1,c2)"] and not fixed["adjacent(c2,c1)"]
    relevant, fixed = select_atoms(problem, "relevant")
    assert sorted(relevant) == ["at(c1)", "at(c2)", "at(c3)"]
    assert fixed["lamp-on(c1)"] is False


def test_blocksworld_masks_drop_only_reflexive_atoms():
    scene = bw.generate(4, 1, seed=0)[0]
    problem = parse(scene.domain_pddl, scene.problem_pddl)
    full, _ = select_atoms(problem, "none", bw.impossible)
    dynamic, fixed = select_atoms(problem, "dynamic", bw.impossible)
    assert len(full) - len(dynamic) == 4
    assert all(v is False for v in fixed.values())


def test_planning_from_the_true_state_is_valid_and_from_a_wrong_one_is_not():
    scene = next(s for s in bw.generate(4, 5, seed=0) if len(s.meta["towers"]) < 4)
    good = plan_and_validate(scene.domain_pddl, scene.problem_pddl, scene.state)
    assert good["status"] == "solved" and good["valid"]
    # Believe every block is on the table and clear: plans pick up stacked blocks.
    blocks = scene.meta["blocks"]
    wrong = {a: False for a in scene.state}
    wrong.update({f"ontable({b})": True for b in blocks})
    wrong.update({f"clear({b})": True for b in blocks})
    wrong["handempty()"] = True
    bad = plan_and_validate(scene.domain_pddl, scene.problem_pddl, wrong)
    assert bad["status"] == "solved" and not bad["valid"]
