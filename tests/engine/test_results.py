"""Tests for lazy Prediction / PredictionSet objects."""

import json
import math

import pytest

from s3e.engine import BinaryAnswers, CategoricalAnswers, Prediction, PredictionSet


def make_prediction(true_mass=0.7, false_mass=0.2, null_mass=0.0, **kwargs):
    space = kwargs.pop("answers", BinaryAnswers(null_tokens=["unknown"] if null_mass else None))
    unassigned = max(0.0, 1.0 - true_mass - false_mass - null_mass)
    return Prediction(
        query="on(a,b)",
        masses={space.true_label: true_mass, space.false_label: false_mass},
        null_mass=null_mass,
        unassigned_mass=unassigned,
        answers=space,
        **kwargs,
    )


class TestPrediction:
    def test_probability_normalizes_over_binary_masses(self):
        p = make_prediction(0.7, 0.2)
        assert p.probability == pytest.approx(0.7 / 0.9, rel=1e-6)

    def test_probability_with_zero_masses_is_half(self):
        p = make_prediction(0.0, 0.0)
        assert p.probability == pytest.approx(0.5)

    def test_answer_is_bool_for_binary(self):
        assert make_prediction(0.7, 0.2).answer is True
        assert make_prediction(0.1, 0.8).answer is False

    def test_score_is_grouped_log_odds(self):
        p = make_prediction(0.7, 0.2)
        assert p.score == pytest.approx(math.log((0.7 + 1e-12) / (0.2 + 1e-12)))

    def test_null_dominated_when_null_beats_all_options(self):
        assert make_prediction(0.2, 0.1, null_mass=0.6).null_dominated is True
        assert make_prediction(0.7, 0.2, null_mass=0.05).null_dominated is False

    def test_answer_none_when_null_dominated(self):
        assert make_prediction(0.2, 0.1, null_mass=0.6).answer is None

    def test_confident(self):
        p = make_prediction(0.9, 0.05)
        assert p.confident(0.8) is True
        assert p.confident(0.99) is False

    def test_probability_override_wins(self):
        p = make_prediction(0.7, 0.2).with_probability(0.42)
        assert p.probability == pytest.approx(0.42)
        # original untouched
        assert make_prediction(0.7, 0.2).probability != pytest.approx(0.42)

    def test_categorical_answer_is_argmax_label(self):
        space = CategoricalAnswers(["red", "green"])
        p = Prediction(
            query="color(a)",
            masses={"red": 0.1, "green": 0.6},
            null_mass=0.0,
            unassigned_mass=0.3,
            answers=space,
        )
        assert p.answer == "green"

    def test_categorical_probability_raises(self):
        space = CategoricalAnswers(["red", "green"])
        p = Prediction(
            query="q", masses={"red": 0.5, "green": 0.5},
            null_mass=0.0, unassigned_mass=0.0, answers=space,
        )
        with pytest.raises(ValueError, match="binary"):
            p.probability

    def test_distribution_normalizes(self):
        p = make_prediction(0.6, 0.2)
        dist = p.distribution()
        assert sum(dist.values()) == pytest.approx(1.0)
        assert dist["yes"] == pytest.approx(0.75)


class TestPredictionSet:
    def make_set(self):
        return PredictionSet(
            {
                "on(a,b)": make_prediction(0.9, 0.05),
                "on(b,a)": make_prediction(0.1, 0.85),
                "clear(a)": make_prediction(0.5, 0.45),
            }
        )

    def test_mapping_protocol(self):
        results = self.make_set()
        assert len(results) == 3
        assert list(results) == ["on(a,b)", "on(b,a)", "clear(a)"]
        assert results["on(a,b)"].answer is True

    def test_probabilities(self):
        probs = self.make_set().probabilities()
        assert set(probs) == {"on(a,b)", "on(b,a)", "clear(a)"}
        assert probs["on(a,b)"] > 0.9

    def test_to_state_three_way(self):
        state = self.make_set().to_state(confidence=0.8)
        assert state["on(a,b)"] is True
        assert state["on(b,a)"] is False
        assert state["clear(a)"] is None  # not confident either way

    def test_to_state_null_dominated_is_none(self):
        results = PredictionSet({"p(a)": make_prediction(0.2, 0.1, null_mass=0.6)})
        assert results.to_state()["p(a)"] is None

    def test_where(self):
        confident = self.make_set().where(lambda p: p.confident(0.8))
        assert set(confident) == {"on(a,b)", "on(b,a)"}


class TestSerialization:
    def test_round_trip_via_json(self):
        results = PredictionSet({"on(a,b)": make_prediction(0.7, 0.2)})
        payload = json.loads(json.dumps(results.to_dict()))
        restored = PredictionSet.from_dict(payload)
        assert restored["on(a,b)"].probability == pytest.approx(
            results["on(a,b)"].probability
        )
        assert restored["on(a,b)"].answer is True

    def test_format_version_present_and_checked(self):
        payload = PredictionSet({"q": make_prediction()}).to_dict()
        assert payload["format_version"] == 1
        payload["format_version"] = 999
        with pytest.raises(ValueError, match="format_version"):
            PredictionSet.from_dict(payload)

    def test_raw_not_serialized(self):
        p = make_prediction(raw=object())
        assert "raw" not in p.to_dict()


class TestAverage:
    def test_average_means_masses(self):
        a = PredictionSet({"q": make_prediction(0.8, 0.1)})
        b = PredictionSet({"q": make_prediction(0.4, 0.5)})
        avg = PredictionSet.average([a, b])
        assert avg["q"].masses["yes"] == pytest.approx(0.6)
        assert avg["q"].masses["no"] == pytest.approx(0.3)

    def test_average_requires_same_queries(self):
        a = PredictionSet({"q1": make_prediction()})
        b = PredictionSet({"q2": make_prediction()})
        with pytest.raises(ValueError, match="same queries"):
            PredictionSet.average([a, b])
