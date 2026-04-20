"""Tests for low-level Platt calibration helpers."""

import math

import pytest

from s3e.calibration import (
    CalibrationExample,
    PlattParameters,
    PlattScalingProfile,
    apply_platt_scaling,
    compute_domain_fingerprint,
    grouped_log_odds,
)


class TestGroupedLogOdds:
    def test_uses_configured_true_and_false_groups(self):
        token_probs = {"true": 0.7, "TRUE": 0.2, "false": 0.1, "other": 0.5}
        score = grouped_log_odds(token_probs, ["true", "TRUE"], ["false"])
        assert score == pytest.approx(math.log(0.9 / 0.1))

    def test_clamps_zero_mass_with_epsilon(self):
        score = grouped_log_odds({"true": 1.0}, ["true"], ["false"])
        assert math.isfinite(score)
        assert score > 0


class TestSigmoidApplication:
    def test_matches_platt_formula(self):
        params = PlattParameters(
            a=1.5,
            b=-0.5,
            sample_count=8,
            positive_count=3,
            negative_count=5,
        )
        probability = apply_platt_scaling(0.7, params)
        expected = 1.0 / (1.0 + math.exp(1.5 * 0.7 - 0.5))
        assert probability == pytest.approx(expected)


class TestProfileSerialization:
    def test_profile_round_trips_through_dict(self):
        profile = PlattScalingProfile(
            scope="global",
            probability_method="logprobs",
            true_tokens=["true"],
            false_tokens=["false"],
            domain_fingerprint=compute_domain_fingerprint("(define (domain blocksworld) ...)"),
            score_kind="grouped_log_odds",
            groups={
                "__global__": PlattParameters(
                    a=0.9,
                    b=-0.2,
                    sample_count=6,
                    positive_count=2,
                    negative_count=4,
                )
            },
        )
        restored = PlattScalingProfile.from_dict(profile.to_dict())
        assert restored == profile


class TestCalibrationExample:
    def test_problem_defaults_to_none(self):
        example = CalibrationExample(images=[], state_dict={"on(a,b)": True})
        assert example.problem is None
