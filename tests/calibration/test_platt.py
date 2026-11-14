# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

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

    def test_empty_calibration_set_rejected(self):
        with pytest.raises(ValueError, match="at least one calibration sample"):
            PlattCalibrator.fit(CalibrationSet(samples=[], meta={}))

    def test_single_class_group_rejected_by_default(self):
        one_sided = [
            CalibrationSample(predicate="on(a,b)", score=1.0 + i, label=True)
            for i in range(5)
        ]
        with pytest.raises(ValueError, match="positive and negative"):
            PlattCalibrator.fit(
                CalibrationSet(samples=one_sided, meta={}), scope="global"
            )

    def test_single_class_error_names_group_and_escape_hatch(self):
        one_sided = [
            CalibrationSample(predicate="on(a,b)", score=1.0 + i, label=True)
            for i in range(5)
        ]
        data = CalibrationSet(samples=one_sided + make_samples("clear(a)"), meta={})
        with pytest.raises(ValueError) as excinfo:
            PlattCalibrator.fit(data, scope="lifted")
        message = str(excinfo.value)
        assert "'on'" in message
        assert "pass_through_single_class" in message

    def test_invalid_scope_rejected(self):
        with pytest.raises(ValueError, match="scope"):
            PlattCalibrator.fit(
                CalibrationSet(samples=make_samples(), meta={}), scope="bogus"
            )

    def test_text_match_scores_fit_per_answer_rates(self):
        """Text-mode scores take two values; the fit reproduces how often the
        predicate was true after each answer."""
        yes, no = 27.631021115928547, -27.631021115928547
        samples = [
            CalibrationSample("on(a,b)", score, label)
            for score, label in [(yes, True)] * 9 + [(yes, False)] * 1
            + [(no, True)] * 2 + [(no, False)] * 8
        ]
        data = CalibrationSet(samples=samples, meta={"scoring": "text_match"})
        cal = PlattCalibrator.fit(data, scope="global")
        assert cal.meta["scoring"] == "text_match"
        said_yes = make_results(true_mass=1.0, false_mass=0.0)
        said_no = make_results(true_mass=0.0, false_mass=1.0)
        assert cal.apply(said_yes)["on(a,b)"].probability == pytest.approx(0.9, abs=0.01)
        assert cal.apply(said_no)["on(a,b)"].probability == pytest.approx(0.2, abs=0.01)

    def test_rare_answer_is_fitted(self):
        """One "no" among 20,000 "yes" replies: under a loose convergence
        tolerance the fit stopped before it accounted for the rare answer."""
        yes, no = 27.631021115928547, -27.631021115928547
        samples = [
            CalibrationSample("on(a,b)", yes, i % 10 != 0) for i in range(20000)
        ] + [CalibrationSample("on(a,b)", no, True)]
        cal = PlattCalibrator.fit(CalibrationSet(samples=samples, meta={}))
        said_no = make_results(true_mass=0.0, false_mass=1.0)
        assert cal.apply(said_no)["on(a,b)"].probability > 0.9

    def test_single_score_group_rejected(self):
        """All replies in the group were "no": nothing says what "yes" means."""
        no = -27.631021115928547
        samples = [
            CalibrationSample("on(a,b)", no, label)
            for label in [True] * 3 + [False] * 9
        ]
        data = CalibrationSet(samples=samples, meta={"scoring": "text_match"})
        with pytest.raises(ValueError, match="single score value") as excinfo:
            PlattCalibrator.fit(data, scope="lifted")
        assert "pass_through_single_class" in str(excinfo.value)

    def test_single_score_group_passed_through(self):
        no = -27.631021115928547
        samples = [
            CalibrationSample("on(a,b)", no, label)
            for label in [True] * 3 + [False] * 9
        ] + make_samples("clear(a)")
        data = CalibrationSet(samples=samples, meta={"scoring": "text_match"})
        cal = PlattCalibrator.fit(data, scope="lifted", pass_through_single_class=True)
        assert cal.group_keys() == ["clear"]
        said_yes = make_results(true_mass=1.0, false_mass=0.0)
        assert cal.apply(said_yes)["on(a,b)"].probability == 1.0

    def test_logprobs_meta_accepted(self):
        data = CalibrationSet(samples=make_samples(), meta={"scoring": "logprobs"})
        assert PlattCalibrator.fit(data, scope="global").group_keys()


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

    def test_loads_utf8_file_with_non_ascii_group_keys(self, tmp_path):
        cal = PlattCalibrator.fit(
            CalibrationSet(samples=make_samples(predicate="sur(a,é)"), meta={}),
            scope="grounded",
        )
        path = tmp_path / "platt.json"
        cal.save(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        assert PlattCalibrator.load(path).group_keys() == ["sur(a,é)"]

    def test_bad_format_version_rejected(self, tmp_path):
        path = tmp_path / "platt.json"
        path.write_text('{"format_version": 99}', encoding="utf-8")
        with pytest.raises(ValueError, match="format_version"):
            PlattCalibrator.load(path)


class TestFitValidation:
    def test_no_samples_rejected(self):
        from s3e.calibration.platt import fit_platt_parameters

        with pytest.raises(ValueError, match="at least one"):
            fit_platt_parameters([], [])

    def test_single_class_rejected(self):
        from s3e.calibration.platt import fit_platt_parameters

        with pytest.raises(ValueError, match="both positive and negative"):
            fit_platt_parameters([1.0, 2.0], [True, True])

    def test_single_class_group_rejected_without_pass_through(self):
        samples = make_samples() + [
            CalibrationSample(predicate="clear(a)", score=1.0, label=True)
        ]
        with pytest.raises(ValueError, match="only positive samples"):
            PlattCalibrator.fit(CalibrationSet(samples=samples, meta={}), scope="lifted")
