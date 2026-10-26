# SPDX-License-Identifier: MIT
import json

import numpy as np

from s3e_exp.domains import get
from s3e_exp.domains.viplan import _objects, replace_init

PROBLEM = """(define (problem p) (:domain shape-stacking-domain)
 (:objects a b - block s - sphere)
 (:init (is-directly-on-table a) (shape-on-top-of-block b a)
        (is-clear-on-top b))
 (:goal (and (is-directly-on-table b))))
"""


def test_replace_init_keeps_everything_else():
    replaced = replace_init(PROBLEM, {"is-directly-on-table(b)": True, "is-clear-on-top(a)": False,
                                      "robot-gripper-empty()": True})
    assert "(:init (is-directly-on-table b) (robot-gripper-empty))" in replaced
    assert "(:goal (and (is-directly-on-table b)))" in replaced
    assert "shape-on-top-of-block b a" not in replaced


def test_objects_counts_typed_lists():
    assert _objects(PROBLEM) == 3


def test_export_reads_recorded_episodes_without_viplan(tmp_path):
    episode = tmp_path / "fail" / "episodes" / "episode_0000"
    (episode / "observations").mkdir(parents=True)
    (episode / "domain.pddl").write_text("(define (domain d))")
    (episode / "problem.pddl").write_text(PROBLEM)
    np.savez_compressed(episode / "observations/obs_initial.npz",
                        obs=np.zeros((4, 6, 3), dtype=np.uint8))
    np.savez_compressed(episode / "observations/obs_step_0000.npz",
                        obs=np.ones((4, 6, 3), dtype=np.uint8))
    rel = "episodes/episode_0000/observations"
    (episode / "episode.json").write_text(json.dumps({
        "episode_index": 0, "episode_seed": 7,
        "initial_gt_state": {"is-directly-on-table(a)": True, "is-clear-on-top(b)": True},
        "initial_observation": {"path": f"{rel}/obs_initial.npz"},
        "steps": [{"step_index": 0, "action": "move(b, a)",
                   "gt_state": {"is-directly-on-table(a)": True, "is-clear-on-top(a)": True},
                   "observation": {"path": f"{rel}/obs_step_0000.npz"}}],
    }))
    adapter = get("viplan-prb")
    scenes = list(adapter.export(tmp_path))
    assert [s.id for s in scenes] == ["prb-fail-0000-t000", "prb-fail-0000-t001"]
    assert scenes[1].meta["after_action"] == "move(b, a)"
    assert "(is-clear-on-top a)" in scenes[1].problem_pddl
    assert scenes[0].view(0).size == (6, 4)
    assert adapter.impossible("shape-on-top-of-block(a,a)")
    assert "{shape}" in adapter.TEMPLATES["is-clear-on-top"]
