# SPDX-License-Identifier: MIT
"""Regression tests for decisions that change reported numbers."""

import os

import pytest
from s3e import BinaryAnswers, Prediction, PredictionSet

from s3e_exp import domains
from s3e_exp.analysis import scene_group, to_state
from s3e_exp.closed_loop import run_episode
from s3e_exp.domains import blocksworld as bw
from s3e_exp.estimation import EstimationConfig, atoms_for, build_estimator, collect
from s3e_exp.models import OracleVLM, TimedBackend
from s3e_exp.planning import parse, solve


def test_unanswered_atoms_are_false_under_every_method():
    answers = BinaryAnswers()
    results = PredictionSet({
        "a": Prediction("a", {"yes": 0.0, "no": 0.0}, 0.0, 1.0, answers),
        "b": Prediction("b", {"yes": 0.6, "no": 0.4}, 0.0, 0.0, answers),
    })
    assert results.to_state()["a"] is True  # the library's P = 0.5 rule
    assert to_state(results) == {"a": False, "b": True}


def test_frames_of_one_episode_share_a_calibration_fold():
    assert scene_group({"scene": "gw-fail-0003-t002"}) == "gw-fail-0003"
    assert scene_group({"scene": "bw-n3-s0-001"}) == "bw-n3-s0-001"


def test_planner_leaves_the_working_directory_clean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scene = bw.generate(3, 1, seed=0)[0]
    assert solve(parse(scene.domain_pddl, scene.problem_pddl))["status"] == "solved"
    assert os.getcwd() == str(tmp_path) and not list(tmp_path.iterdir())


def test_an_unplannable_estimate_does_not_end_the_episode():
    domain = domains.get("blocksworld")
    unplannable = successes = 0
    for scene in bw.generate(4, 6, seed=0):
        backend = TimedBackend(OracleVLM(accuracy=0.97, seed=1))
        estimator = build_estimator(scene, domain, backend, EstimationConfig())
        problem = parse(scene.domain_pddl, scene.problem_pddl)
        atoms, fixed = atoms_for(scene, domain, "dynamic", problem)
        outcome = run_episode(domain.World(scene), problem, estimator, backend, atoms, fixed,
                              policy="monitor", max_steps=40)
        unplannable += outcome["unplannable_estimates"]
        successes += outcome["success"]
        assert outcome["status"] != "planner_unsolvable" or outcome["steps"] == 0
    assert successes >= 1


def test_collect_refuses_to_mix_configurations(tmp_path):
    scenes = bw.generate(2, 1, seed=0)
    domain = domains.get("blocksworld")
    spec = {"name": "fake", "id": "fake", "backend": "fake"}
    collect(scenes, domain, spec, OracleVLM(), EstimationConfig(), tmp_path, progress=False)
    collect(scenes, domain, spec, OracleVLM(), EstimationConfig(views=("0", "1")), tmp_path,
            progress=False)
    with pytest.raises(SystemExit):
        collect(scenes, domain, spec, OracleVLM(), EstimationConfig(batch_size=2), tmp_path,
                progress=False)
