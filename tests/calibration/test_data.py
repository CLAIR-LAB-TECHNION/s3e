"""Tests for CalibrationSet collection and persistence."""

import pytest

from s3e.calibration import CalibrationExample, CalibrationSample, CalibrationSet


class TestCalibrationSetPersistence:
    def test_round_trip(self, tmp_path):
        data = CalibrationSet(
            samples=[CalibrationSample("on(a,b)", 1.5, True, problem=None)],
            meta={"true_label": "yes", "false_label": "no"},
        )
        path = tmp_path / "calib.json"
        data.save(path)
        restored = CalibrationSet.load(path)
        assert restored.samples[0].predicate == "on(a,b)"
        assert restored.samples[0].score == pytest.approx(1.5)
        assert restored.meta["true_label"] == "yes"

    def test_bad_format_version_rejected(self, tmp_path):
        path = tmp_path / "calib.json"
        path.write_text('{"format_version": 99, "samples": [], "meta": {}}')
        with pytest.raises(ValueError, match="format_version"):
            CalibrationSet.load(path)
