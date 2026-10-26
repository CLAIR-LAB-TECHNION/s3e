# SPDX-License-Identifier: MIT
"""End to end with the fake oracle model: instances -> collect -> E1-E4, E6, E7."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args):
    result = subprocess.run([sys.executable, str(ROOT / "experiments" / script), *map(str, args)],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    run("instances.py", "blocksworld", "--sizes", 3, 4, "--seeds", 0, 1, "--count", 3,
        "--out", root / "instances")
    run("collect.py", "--instances", root / "instances", "--model", "fake",
        "--views", 0, 1, 2, 3, "all", "--out", root / "pred")
    run("collect.py", "--instances", root / "instances", "--model", "fake",
        "--scoring", "text_match", "--out", root / "pred")
    return root


def test_collect_writes_one_record_per_scene_and_view_and_resumes(workspace):
    path = workspace / "pred/blocksworld/fake/logprobs-none/predictions.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 12 * 5
    run("collect.py", "--instances", workspace / "instances", "--model", "fake",
        "--views", 0, 1, 2, 3, "all", "--out", workspace / "pred")
    assert len(path.read_text().splitlines()) == 12 * 5
    run_json = json.loads((path.parent / "run.json").read_text())
    assert run_json["answer_space"]["true_label"] == "yes"
    assert run_json["queries"]


def test_e1_plans_the_oracle_state_perfectly(workspace):
    out = workspace / "e1.jsonl"
    run("e1_closed_loop.py", "plan", "--instances", workspace / "instances", "--folds", 2,
        "--predictions", workspace / "pred/blocksworld/fake/logprobs-none", "--out", out,
        "--workers", 2)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    oracle = [r for r in rows if r["decoder"] == "oracle"]
    assert len(oracle) == 12 and all(r["valid"] for r in oracle)
    assert {r["decoder"] for r in rows} >= {"threshold", "calibrated", "map", "views-avg"}
    assert "valid" in run("e1_closed_loop.py", "summary", out)


def test_e1_online_succeeds_with_a_perfect_model(workspace, monkeypatch):
    from s3e_exp import cli, domains
    from s3e_exp.closed_loop import run_episode
    from s3e_exp.estimation import EstimationConfig, atoms_for, build_estimator
    from s3e_exp.models import OracleVLM, TimedBackend
    from s3e_exp.planning import parse

    class Args:
        instances, sizes, seeds, limit = workspace / "instances", [3], [0], 2

    for scene in cli.select_scenes(Args):
        domain = domains.get("blocksworld")
        backend = TimedBackend(OracleVLM(accuracy=1.0))
        estimator = build_estimator(scene, domain, backend, EstimationConfig())
        problem = parse(scene.domain_pddl, scene.problem_pddl)
        atoms, fixed = atoms_for(scene, domain, "none", problem)
        for policy, check in (("monitor", "full"), ("monitor", "action"), ("replan", "full"),
                              ("open-loop", "full")):
            outcome = run_episode(domain.World(scene), problem, estimator, backend, atoms, fixed,
                                  policy=policy, check=check, max_steps=40)
            assert outcome["success"], (policy, check, outcome["status"])
            assert outcome["failed_actions"] == 0
        cheap = run_episode(domain.World(scene), problem, estimator, backend, atoms, fixed,
                            policy="monitor", check="action", max_steps=40)
        full = run_episode(domain.World(scene), problem, estimator, backend, atoms, fixed,
                           policy="monitor", check="full", max_steps=40)
        assert cheap["queries"] <= full["queries"]


def test_e2_e3_e4_run_offline(workspace):
    run("e2_scale.py", "count", "--instances", workspace / "instances",
        "--out", workspace / "count.jsonl")
    counts = [json.loads(line) for line in (workspace / "count.jsonl").read_text().splitlines()]
    assert {c["atoms_none"] - c["atoms_dynamic"] for c in counts} == {3, 4}
    base = workspace / "pred/blocksworld/fake"
    out = run("e3_scoring.py", "--instances", workspace / "instances",
              "--runs", base / "logprobs-none", base / "text_match-none",
              "--out", workspace / "e3.json")
    assert "regex" in out
    for command in ("scopes", "transfer", "views"):
        run("e4_calibration.py", command, "--instances", workspace / "instances", "--folds", 2,
            "--run", base / "logprobs-none", "--out", workspace / f"e4-{command}.json",
            *(["--fit-sizes", 3, "--test-sizes", 4] if command == "transfer" else []))
    views = json.loads((workspace / "e4-views.json").read_text())
    assert set(views["by_count"]) == {"1", "2", "3", "4"}


def test_e6_export_and_compare(workspace):
    queries = workspace / "queries.jsonl"
    run("e6_tools.py", "export", "--instances", workspace / "instances", "--out", queries)
    rows = [json.loads(line) for line in queries.read_text().splitlines()]
    assert rows[0]["prompt"].endswith("Answer with exactly one word: yes or no.")
    fake_tool = workspace / "tool.jsonl"
    fake_tool.write_text("".join(json.dumps({"key": r["key"], "probability": 0.5 + 0.4 * (
        1 if r["truth"] else -1), "seconds": 0.01}) + "\n" for r in rows))
    out = run("e6_tools.py", "compare", "--queries", queries,
              "--s3e", workspace / "pred/blocksworld/fake/logprobs-none",
              "--tool", f"perfect={fake_tool}", "--out", workspace / "e6.json")
    report = json.loads((workspace / "e6.json").read_text())
    assert report["tools"]["perfect"]["accuracy"] == 1.0
    assert "perfect" in out


def test_e7_api_surface_of_the_installed_version(tmp_path):
    run("e7_versions.py", "api", "--versions", "installed", "--out", tmp_path)
    surface = json.loads((tmp_path / "api/installed.json").read_text())["surface"]
    assert surface["features"]["answer_spaces"] and surface["features"]["vllm_backend"]
