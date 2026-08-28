# tests/consumers/test_mlss_workflow.py
"""Contract tests for the MLSS workflow (make_predictions.py, calibrate_vlm.py).

MLSS pattern: one long-lived estimator per domain; per sample it swaps the
problem, estimates a relevant-atom subset, serializes prediction details to
JSON, and later refits calibration offline without a VLM.
"""

import json

import pytest

from s3e import PredictionSet, SemanticStateEstimator, TemplateTranslator

from conftest import BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM, make_blank_image
from fakes import FakeVLM

TEMPLATES = {"on": "Is {0} on {1}?", "clear": "Is {0} clear?"}


@pytest.fixture
def estimator():
    return SemanticStateEstimator.from_pddl(
        BLOCKSWORLD_DOMAIN,
        BLOCKSWORLD_PROBLEM,
        vlm=FakeVLM({"yes": 0.8, "no": 0.1}),
        translator=TemplateTranslator(TEMPLATES),
    )


class TestPerSampleLoop:
    def test_set_problem_then_subset_estimate(self, estimator):
        backend = estimator.engine.backend
        estimator.set_problem(BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM)
        assert estimator.engine.backend is backend  # never rebuilt

        subset = estimator.predicates[:3]
        results = estimator.estimate([make_blank_image()], predicates=subset)
        assert list(results) == subset
        prompts = [p for call in backend.calls for p in call["prompts"]]
        assert len(prompts) == len(subset)  # only the subset was queried


class TestDetailsSerialization:
    def test_details_to_json_and_back_without_backend(self, estimator):
        results = estimator.estimate([make_blank_image()])
        payload = json.dumps(results.to_dict())          # what MLSS writes
        restored = PredictionSet.from_dict(json.loads(payload))
        for predicate in results:
            assert restored[predicate].probability == pytest.approx(
                results[predicate].probability
            )
            assert restored[predicate].score == pytest.approx(
                results[predicate].score
            )
            # fields MLSS's payload builder reads:
            p = restored[predicate]
            assert p.masses is not None
            assert p.null_mass is not None
            assert p.argmax_in_interest is not None


class TestOfflineCalibrationRefit:
    def test_collect_save_refit_without_vlm(self, estimator, tmp_path):
        pytest.importorskip("sklearn")
        from s3e.calibration import CalibrationExample, CalibrationSet, PlattCalibrator

        # Half true, half false, with separated masses so a fit converges.
        estimator.engine.backend.script_responses(
            {"Is a": {"yes": 0.9, "no": 0.05}, "Is b": {"yes": 0.1, "no": 0.85}}
        )
        labels = {
            p: ("a" in p.split("(", 1)[1].split(",")[0])
            for p in estimator.predicates
        }
        examples = [
            CalibrationExample(images=[make_blank_image()], state_dict=labels)
        ]
        data = CalibrationSet.collect(estimator, examples)
        data.save(tmp_path / "calib.json")

        # Later, no VLM anywhere:
        reloaded = CalibrationSet.load(tmp_path / "calib.json")
        cal = PlattCalibrator.fit(reloaded, scope="global")
        cal.save(tmp_path / "platt.json")
        restored = PlattCalibrator.load(tmp_path / "platt.json")

        results = estimator.estimate([make_blank_image()])
        calibrated = restored.apply(results)
        assert all(0.0 <= p.probability <= 1.0 for p in calibrated.values())
