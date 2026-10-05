# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""The scripts in ``examples/`` run end to end (fake backends unless slow)."""

import importlib.util
import json
import re
import runpy
from pathlib import Path

import pytest

from fakes import FakeVLM

pytest.importorskip("unified_planning", reason="examples need s3e[pddl]")
pytest.importorskip("sklearn", reason="examples need s3e[calibration]")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load_example(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCustomBackendExample:
    def test_calibration_improves_held_out_brier_score(self, capsys):
        runpy.run_path(str(EXAMPLES / "custom_backend.py"), run_name="__main__")
        out = capsys.readouterr().out

        rows = dict(re.findall(r"^(uncalibrated|calibrated)\s+([\d.]+\s+[\d.]+)$", out, re.M))
        uncalibrated_brier = float(rows["uncalibrated"].split()[1])
        calibrated_brier = float(rows["calibrated"].split()[1])
        assert calibrated_brier < uncalibrated_brier
        assert "as a Unified Planning state: UPState" in out


class TestBlocksworldBenchmarkExample:
    @pytest.fixture(scope="class")
    def bench(self):
        return load_example("blocksworld_benchmark")

    def test_true_state_is_a_consistent_arrangement(self, bench):
        blocks = ["red", "green", "blue"]
        towers = [["red", "green"], ["blue"]]
        state = bench.true_state(towers, blocks)

        assert {p for p, value in state.items() if value} == {
            "ontable(red)", "on(green,red)", "clear(green)",
            "ontable(blue)", "clear(blue)",
        }
        assert len(state) == 3 * 3 + 2 * 3

    def test_metrics_on_known_rows(self, bench):
        rows = [
            {"probability": 0.9, "truth": True, "state": True, "has_answer": True},
            {"probability": 0.5, "truth": False, "state": True, "has_answer": False},
        ]
        result = bench.metrics(rows)
        assert result["accuracy"] == 0.5
        assert result["brier"] == pytest.approx((0.01 + 0.25) / 2)
        assert result["no_answer_rate"] == 0.5
        # Two singleton bins: |1 - 0.9| and |0 - 0.5|, each weighted 1/2.
        assert result["ece"] == pytest.approx((0.1 + 0.5) / 2)

    def test_report_with_fake_backend(self, bench):
        report = bench.run_benchmark(
            FakeVLM({"yes": 0.7, "no": 0.2}, text="yes"),
            num_scenes=4,
            num_blocks=2,
            calibrate_fraction=0.5,
        )

        assert set(report["summary"]) == {
            "logprobs", "text_match", "logprobs held-out", "logprobs held-out+platt",
        }
        predicates_per_scene = 2 * 2 + 2 * 2
        assert len(report["instances"]["logprobs"]) == 4 * predicates_per_scene
        # Before/after calibration compare the same held-out scenes.
        before = report["instances"]["logprobs held-out"]
        after = report["instances"]["logprobs held-out+platt"]
        assert [(r["scene"], r["predicate"]) for r in before] == [
            (r["scene"], r["predicate"]) for r in after
        ]
        assert {r["scene"] for r in before} == {2, 3}
        assert report["calibration"]["scenes"] == [0, 1]
        for summary in report["summary"].values():
            assert 0.0 <= summary["accuracy"] <= 1.0
            assert 0.0 <= summary["brier"] <= 1.0
        assert set(report["instances"]["logprobs"][0]["masses"]) == {"yes", "no"}
        provenance = report["provenance"]
        assert provenance["backend"] == "FakeVLM"
        assert provenance["queries"]["on(red,green)"].startswith("Is the red block")
        assert provenance["inference_kwargs"]["logprobs"] == {}
        json.dumps(report)  # the whole report is JSON-serializable

    def test_command_line_writes_the_report(self, bench, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            bench, "resolve_backend", lambda vlm, **kw: FakeVLM(text="no")
        )
        output = tmp_path / "results.json"
        bench.main([
            "--model", "fake", "--num-scenes", "2", "--num-blocks", "2",
            "--scoring", "text_match", "--output", str(output),
        ])

        report = json.loads(output.read_text(encoding="utf-8"))
        assert set(report["summary"]) == {"text_match"}
        # FakeVLM is neither HuggingFace nor vLLM: no default reply cap.
        assert report["provenance"]["inference_kwargs"]["text_match"] == {}
        assert "text_match" in capsys.readouterr().out

    def test_rejects_unsupported_block_count(self, bench):
        with pytest.raises(ValueError, match="num_blocks"):
            bench.run_benchmark(FakeVLM(), num_blocks=7)

    def test_default_reply_cap_depends_on_the_backend(self, bench):
        assert bench.DEFAULT_GENERATION_KWARGS["HuggingFaceVLM"] == {"max_new_tokens": 8}
        assert bench.DEFAULT_GENERATION_KWARGS["VLLMBackend"] == {"max_tokens": 8}
        assert "OpenAIVLM" not in bench.DEFAULT_GENERATION_KWARGS

    def test_calibration_without_answers_is_skipped(self, bench):
        from s3e import UnmatchedAnswerWarning

        with pytest.warns(UnmatchedAnswerWarning):
            report = bench.run_benchmark(
                FakeVLM({"maybe": 1.0}), num_scenes=2, num_blocks=2,
                scoring_modes=("logprobs",), calibrate_fraction=0.5,
            )
        assert "skipped" in report["calibration"]
        assert set(report["summary"]) == {"logprobs"}

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_scenes": 0},
            {"calibrate_fraction": 0.5, "scoring_modes": ("text_match",)},
        ],
    )
    def test_rejects_invalid_settings(self, bench, kwargs):
        with pytest.raises(ValueError):
            bench.run_benchmark(FakeVLM(), **kwargs)

    @pytest.mark.parametrize("fraction", [0.1, 1.0])
    def test_rejects_calibration_split_without_both_sides(self, bench, fraction):
        with pytest.raises(ValueError, match="calibrate_fraction"):
            bench.run_benchmark(FakeVLM(), num_scenes=4, calibrate_fraction=fraction)

    @pytest.mark.slow
    def test_real_model_records_its_revision(self, bench):
        torch = pytest.importorskip("torch")
        report = bench.run_benchmark(
            "katuni4ka/tiny-random-llava",
            num_scenes=1,
            num_blocks=2,
            generation_kwargs={"max_new_tokens": 2},
            vlm_kwargs={"device_map": "cpu", "torch_dtype": torch.float32},
        )
        assert re.fullmatch(r"[0-9a-f]{40}", report["provenance"]["model_revision"])
        assert report["provenance"]["backend"] == "HuggingFaceVLM"
