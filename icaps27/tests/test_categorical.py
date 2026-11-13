# SPDX-License-Identifier: MIT
"""E8: finite-domain queries, decoding, and the predict/plan pipeline on a stub model."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from s3e import VLMBackend, VLMOutput

from s3e_exp.domains import blocksworld
from s3e_exp.domains.blocksworld import BlocksState
from s3e_exp.scenes import load_scenes, save_scene

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = ["red", "green", "blue"]


def load():
    spec = importlib.util.spec_from_file_location(
        "e8_categorical", ROOT / "experiments" / "e8_categorical.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # the planner pool pickles functions by module name
    spec.loader.exec_module(module)
    return module


def positions_of(state: BlocksState, blocks: list[str]) -> dict[str, str]:
    e8 = load()
    positions = {state.held: e8.HELD} if state.held else {}
    for tower in state.towers:
        positions[tower[0]] = e8.TABLE
        positions.update(dict(zip(tower[1:], tower)))
    return {b: positions[b] for b in blocks}


@pytest.mark.parametrize("state", [
    BlocksState((("red", "green"), ("blue",))),
    BlocksState((("red", "green", "blue"),)),
    BlocksState((("red",), ("green",)), held="blue"),
])
def test_decoded_atoms_match_the_domain_on_consistent_states(state):
    e8 = load()
    positions = positions_of(state, BLOCKS)
    assert e8.consistent(positions)
    assert e8.decode_positions(positions, BLOCKS) == state.atoms(BLOCKS)


@pytest.mark.parametrize("positions", [
    {"red": "robot", "green": "robot", "blue": "table"},  # two held
    {"red": "blue", "green": "blue", "blue": "table"},  # two on one block
    {"red": "green", "green": "red", "blue": "table"},  # a cycle
])
def test_inconsistent_assignments_are_flagged_and_still_decode(positions):
    e8 = load()
    assert not e8.consistent(positions)
    assert set(e8.decode_positions(positions, BLOCKS)) == set(BlocksState((tuple(BLOCKS),)).atoms(BLOCKS))


def test_a_block_is_never_an_option_for_itself():
    e8 = load()
    assert e8.values_of("green", BLOCKS) == ["table", "robot", "red", "blue"]


def test_word_query_lists_the_options_and_scores_their_words():
    e8 = load()
    query, answers, value_of = e8.build_query("red", BLOCKS, "word")
    assert query.startswith("What is the red block directly on?")
    assert "table, robot (held by the robot gripper), green, or blue." in query
    assert answers.labels == ["table", "robot", "green", "blue"]
    assert value_of["green"] == "green"


def test_choice_query_follows_the_vev_layout_and_maps_letters_back():
    e8 = load()
    query, answers, value_of = e8.build_query("red", BLOCKS, "choice")
    assert query == ("State:\n(image above)\n\nQuestion: What is the red block directly on?\n\n"
                     "Options:\nA. table\nB. robot (held by the robot gripper)\nC. green\nD. blue\n\n"
                     "Answer with the letter of the single best option.")
    assert answers.interest_tokens == ["A", "B", "C", "D"]
    assert value_of == {"A": "table", "B": "robot", "C": "green", "D": "blue"}


class StubVLM(VLMBackend):
    """Puts all the mass on the first option, which is ``table`` for every block."""

    model_id = "stub"

    def query(self, images, prompt, system_prompt=None, generate=False,
              interest_tokens=None, **inference_kwargs):
        first = interest_tokens[0]
        return VLMOutput(token_probs={t: float(t == first) for t in interest_tokens},
                         argmax_in_interest=True)


@pytest.mark.parametrize("fmt", ["word", "choice"])
def test_predict_then_plan_writes_e1_records(tmp_path, monkeypatch, fmt):
    e8 = load()
    instances = tmp_path / "instances"
    for scene in blocksworld.generate(3, 2, 0):
        save_scene(scene, instances)
    monkeypatch.setattr(e8, "build_backend", lambda *a, **k: StubVLM())
    out = tmp_path / "e8"
    e8.main(["predict", "--instances", str(instances), "--model", "fake",
             "--format", fmt, "--out", str(out)])
    run = out / "blocksworld" / "fake" / fmt
    records = [json.loads(line) for line in (run / "predictions.jsonl").open()]
    assert len(records) == 2 and all(r["queries"] == 3 for r in records)
    assert all(a["masses"]["table"] == 1.0 for r in records for a in r["answers"].values())
    # A rerun resumes: nothing new is queried.
    e8.main(["predict", "--instances", str(instances), "--model", "fake",
             "--format", fmt, "--out", str(out)])
    assert len((run / "predictions.jsonl").read_text().splitlines()) == 2
    plans = tmp_path / "plan.jsonl"
    e8.main(["plan", "--instances", str(instances), "--predictions", str(run),
             "--workers", "1", "--out", str(plans)])
    rows = [json.loads(line) for line in plans.open()]
    scenes = {s.id: s for s in load_scenes(instances)}
    assert len(rows) == 2 and {r["decoder"] for r in rows} == {f"categorical-{fmt}"}
    assert all(r["consistent"] for r in rows)  # every block on the table
    for row in rows:
        truth = scenes[row["scene"]].state
        decoded = e8.decode_positions({b: "table" for b in scenes[row["scene"]].meta["blocks"]},
                                      scenes[row["scene"]].meta["blocks"])
        assert row["atom_errors"] == sum(decoded[a] != truth[a] for a in truth)
