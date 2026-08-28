"""Tests for the Calibrator interface and PlattCalibrator."""

import json

import pytest

pytest.importorskip("sklearn")

from s3e.calibration import CalibrationSample, CalibrationSet, PlattCalibrator
from s3e.engine import BinaryAnswers, Prediction, PredictionSet


def make_samples(predicate="on(a,b)", n=20):
    """Well-separated scores: positives high, negatives low."""
    samples = []
    for i in range(n):
        label = i % 2 == 0
        score = 2.0 + 0.1 * i if label else -2.0 - 0.1 * i
        samples.append(CalibrationSample(predicate=predicate, score=score, label=label))
    return samples


def make_results(true_mass=0.7, false_mass=0.2, predicate="on(a,b)"):
    space = BinaryAnswers()
    return PredictionSet(
        {
            predicate: Prediction(
                query=predicate,
                masses={"yes": true_mass, "no": false_mass},
                null_mass=0.0,
                unassigned_mass=0.1,
                answers=space,
            )
        }
    )


class TestPlattFitApply:
    def test_fit_global_and_apply_overrides_probability(self):
        data = CalibrationSet(samples=make_samples(), meta={})
        cal = PlattCalibrator.fit(data, scope="global")
        results = make_results()
        calibrated = cal.apply(results)
        raw = results["on(a,b)"].probability
        assert calibrated["on(a,b)"].probability != pytest.approx(raw)
        assert 0.0 <= calibrated["on(a,b)"].probability <= 1.0
        # original untouched
        assert results["on(a,b)"].probability == pytest.approx(raw)

    def test_lifted_scope_groups_by_predicate_name(self):
        data = CalibrationSet(
            samples=make_samples("on(a,b)") + make_samples("on(b,a)"), meta={}
        )
        cal = PlattCalibrator.fit(data, scope="lifted")
        assert set(cal.group_keys()) == {"on"}

    def test_grounded_scope_groups_by_full_predicate(self):
        data = CalibrationSet(
            samples=make_samples("on(a,b)") + make_samples("clear(a)"), meta={}
        )
        cal = PlattCalibrator.fit(data, scope="grounded")
        assert set(cal.group_keys()) == {"on(a,b)", "clear(a)"}

    def test_apply_without_matching_group_keeps_raw(self):
        data = CalibrationSet(samples=make_samples("on(a,b)"), meta={})
        cal = PlattCalibrator.fit(data, scope="grounded")
        results = make_results(predicate="clear(c)")
        calibrated = cal.apply(results)
        assert calibrated["clear(c)"].probability == pytest.approx(
            results["clear(c)"].probability
        )

    def test_single_class_group_rejected_by_default(self):
        one_sided = [
            CalibrationSample(predicate="on(a,b)", score=1.0 + i, label=True)
            for i in range(5)
        ]
        with pytest.raises(ValueError, match="positive and negative"):
            PlattCalibrator.fit(
                CalibrationSet(samples=one_sided, meta={}), scope="global"
            )

    def test_invalid_scope_rejected(self):
        with pytest.raises(ValueError, match="scope"):
            PlattCalibrator.fit(
                CalibrationSet(samples=make_samples(), meta={}), scope="bogus"
            )


class TestPlattPersistence:
    def test_save_load_round_trip(self, tmp_path):
        cal = PlattCalibrator.fit(
            CalibrationSet(samples=make_samples(), meta={}), scope="global"
        )
        path = tmp_path / "platt.json"
        cal.save(path)
        restored = PlattCalibrator.load(path)
        results = make_results()
        assert restored.apply(results)["on(a,b)"].probability == pytest.approx(
            cal.apply(results)["on(a,b)"].probability
        )

    def test_saved_file_has_format_version(self, tmp_path):
        cal = PlattCalibrator.fit(
            CalibrationSet(samples=make_samples(), meta={}), scope="global"
        )
        path = tmp_path / "platt.json"
        cal.save(path)
        assert "format_version" in json.loads(path.read_text())
