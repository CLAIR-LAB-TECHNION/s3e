# SPDX-License-Identifier: MIT
import random

import pytest

from s3e_exp.domains import blocksworld as bw
from s3e_exp.planning import parse, with_initial_state
from unified_planning.engines.sequential_simulator import UPSequentialSimulator


def test_generate_is_deterministic_and_sized():
    first, second = bw.generate(5, 3, seed=1), bw.generate(5, 3, seed=1)
    assert [s.state for s in first] == [s.state for s in second]
    assert all(len(s.meta["blocks"]) == 5 for s in first)
    assert all(s.num_views == len(bw.VIEWS) for s in first)
    assert bw.generate(5, 3, seed=2)[0].state != first[0].state or \
        bw.generate(5, 3, seed=2)[0].meta["blocks"] != first[0].meta["blocks"]


def test_atoms_describe_a_consistent_state():
    for scene in bw.generate(6, 5, seed=0):
        state, blocks = scene.state, scene.meta["blocks"]
        assert len(state) == len(blocks) ** 2 + 3 * len(blocks) + 1
        for x in blocks:
            supports = sum(state[f"on({x},{y})"] for y in blocks) + state[f"ontable({x})"]
            assert supports + state[f"holding({x})"] == 1
            assert state[f"clear({x})"] == (not any(state[f"on({y},{x})"] for y in blocks)
                                            and not state[f"holding({x})"])
        assert state["handempty()"]


def test_views_have_a_fixed_size_per_block_count():
    scenes = bw.generate(4, 3, seed=0)
    for index in range(len(bw.VIEWS)):
        assert len({s.view(index).size for s in scenes}) == 1


def test_apply_matches_the_pddl_semantics():
    rng = random.Random(0)
    scene = bw.generate(4, 1, seed=3)[0]
    blocks = scene.meta["blocks"]
    state = bw.state_from_meta(scene.meta)
    truth = parse(scene.domain_pddl, scene.problem_pddl)
    actions = [("pick-up", [x]) for x in blocks] + [("put-down", [x]) for x in blocks] + \
        [(a, [x, y]) for a in ("stack", "unstack") for x in blocks for y in blocks]
    for _ in range(40):
        problem = with_initial_state(truth, state.atoms(blocks))
        simulator = UPSequentialSimulator(problem)
        current = simulator.get_initial_state()
        name, args = rng.choice(actions)
        params = tuple(problem.object(a) for a in args)
        applicable = simulator.is_applicable(current, problem.action(name), params)
        successor = state.apply(name, args)
        assert applicable == (successor is not None), (name, args, state)
        if successor is not None:
            following = simulator.apply(current, problem.action(name), params)
            expected = {a: following.get_value(_fnode(problem, a)).bool_constant_value()
                        for a in state.atoms(blocks)}
            assert successor.atoms(blocks) == expected
            state = successor


def _fnode(problem, atom):
    from s3e.pddl import ground_predicate_str_to_fnode

    return ground_predicate_str_to_fnode(problem, atom)


@pytest.mark.parametrize("n, expected", [(1, 2), (2, 3 + 2 * 1), (3, 13 + 3 * 3), (4, 73 + 4 * 13)])
def test_consistent_states_are_lah_sums(n, expected):
    states = bw.consistent_states(tuple(bw.COLORS[:n]))
    assert len(states) == expected
    assert len({s.key() for s in states}) == expected


def test_map_decode_recovers_the_true_state_from_noisy_marginals():
    scene = bw.generate(5, 1, seed=0)[0]
    rng = random.Random(1)
    probabilities = {a: (0.7 if v else 0.3) + rng.uniform(-0.15, 0.15)
                     for a, v in scene.state.items()}
    probabilities["on(" + scene.meta["blocks"][0] + "," + scene.meta["blocks"][1] + ")"] = 0.55
    decoded = bw.map_decode(probabilities, scene.meta["blocks"])
    assert decoded == scene.state


def test_world_applies_actions_and_detects_the_goal():
    scene = bw.generate(3, 1, seed=0)[0]
    world = bw.World(scene)
    assert world.state() == scene.state
    assert not world.apply("stack", [scene.meta["blocks"][0], scene.meta["blocks"][0]])
    assert world.state() == scene.state
    assert len(world.observe([0])) == 1
