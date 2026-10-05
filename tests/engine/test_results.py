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
        assert p.matched is False
        assert p.probability == 0.5

    def test_probability_is_exact_without_epsilon(self):
        assert make_prediction(1.0, 0.0).probability == 1.0
        assert make_prediction(0.0, 1.0).probability == 0.0

    def test_answer_is_bool_for_binary(self):
        assert make_prediction(0.7, 0.2).answer is True
        assert make_prediction(0.1, 0.8).answer is False

    def test_score_is_grouped_log_odds(self):
        p = make_prediction(0.7, 0.2)
        assert p.score == pytest.approx(math.log((0.7 + 1e-12) / (0.2 + 1e-12)))

    def test_null_dominated_when_null_beats_all_options(self):
        assert make_prediction(0.2, 0.1, null_mass=0.6).null_dominated is True
        assert make_prediction(0.7, 0.2, null_mass=0.05).null_dominated is False

    def test_null_dominated_has_no_answer(self):
        p = make_prediction(0.2, 0.1, null_mass=0.6)
        assert p.probability == 0.5
        assert p.answer is True  # 0.5 >= 0.5, like the default state
        assert p.distribution() == {"yes": 0.5, "no": 0.5}

    def test_null_tie_with_top_option_is_not_dominated(self):
        p = make_prediction(0.6, 0.0, null_mass=0.6)
        assert p.null_dominated is False
        assert p.probability == 1.0

    def test_no_answer_wins_over_calibration(self):
        null = make_prediction(0.2, 0.1, null_mass=0.6).with_probability(0.9)
        unmatched = make_prediction(0.0, 0.0).with_probability(0.9)
        assert null.probability == 0.5
        assert unmatched.probability == 0.5
        assert null.probability_override == 0.9  # kept, not discarded

    def test_matched(self):
        assert make_prediction(0.7, 0.2).matched is True
        assert make_prediction(0.0, 0.0, null_mass=1.0).matched is True
        assert make_prediction(0.0, 0.0).matched is False

    def test_has_answer(self):
        assert make_prediction(0.7, 0.2).has_answer is True
        assert make_prediction(0.2, 0.1, null_mass=0.6).has_answer is False
        assert make_prediction(0.0, 0.0).has_answer is False

    def test_binary_answer_follows_probability_not_raw_argmax(self):
        assert make_prediction(0.9, 0.1).with_probability(0.4).answer is False
        # No answer: 0.5 >= 0.5, even though the raw masses lean false.
        assert make_prediction(0.1, 0.2, null_mass=0.6).answer is True

    def test_derived_booleans_are_python_bools(self):
        import numpy as np

        p = make_prediction(np.float64(0.7), np.float64(0.2))
        assert type(p.answer) is bool
        assert type(p.confident(0.6)) is bool
        assert type(PredictionSet({"q": p}).to_state()["q"]) is bool

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

    def test_categorical_without_answer_is_uniform_and_picks_first_option(self):
        space = CategoricalAnswers(["red", "green", "blue"], null_tokens=["unknown"])
        for null_mass in (0.0, 0.8):  # unmatched, then null-dominated
            p = Prediction(
                query="color(a)",
                masses={"red": 0.0, "green": 0.1 if null_mass else 0.0, "blue": 0.0},
                null_mass=null_mass,
                unassigned_mass=0.1 if null_mass else 1.0,
                answers=space,
            )
            assert p.distribution() == pytest.approx(
                {"red": 1 / 3, "green": 1 / 3, "blue": 1 / 3}
            )
            assert p.answer == "red"

    def test_categorical_tie_goes_to_first_listed_option(self):
        space = CategoricalAnswers(["red", "green"])
        p = Prediction(
            query="q", masses={"red": 0.4, "green": 0.4},
            null_mass=0.0, unassigned_mass=0.2, answers=space,
        )
        assert p.answer == "red"

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

    def test_to_state_is_boolean(self):
        state = self.make_set().to_state(confidence=0.8)
        assert state == {"on(a,b)": True, "on(b,a)": False, "clear(a)": False}

    def test_to_state_low_confidence_accepts_true(self):
        results = PredictionSet({"p(a)": make_prediction(0.4, 0.6)})
        assert results.to_state(confidence=0.3)["p(a)"] is True

    def test_to_state_full_confidence_accepts_certain_true(self):
        results = PredictionSet({"p(a)": make_prediction(1.0, 0.0)})
        assert results.to_state(confidence=1.0)["p(a)"] is True

    def test_to_state_no_answer_is_decided_by_confidence(self):
        results = PredictionSet(
            {
                "null": make_prediction(0.2, 0.1, null_mass=0.6),
                "unmatched": make_prediction(0.0, 0.0),
            }
        )
        assert results.to_state() == {"null": True, "unmatched": True}
        assert results.to_state(confidence=0.6) == {"null": False, "unmatched": False}

    def test_to_state_rejects_categorical(self):
        space = CategoricalAnswers(["red", "green"])
        p = Prediction(
            query="q", masses={"red": 0.6, "green": 0.4},
            null_mass=0.0, unassigned_mass=0.0, answers=space,
        )
        with pytest.raises(ValueError, match="binary"):
            PredictionSet({"q": p}).to_state()

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

    def test_raw_masses_recoverable_for_custom_rules(self):
        """Stored data survives the decision rule and a JSON round trip, so
        callers can derive other rules, e.g. counting null mass as half true
        and half false."""
        t, f, n = 0.3, 0.1, 0.6
        raw = make_prediction(t, f, null_mass=n)
        calibrated = raw.with_probability(0.8)
        payload = json.loads(json.dumps(PredictionSet({"raw": raw, "cal": calibrated}).to_dict()))
        restored = PredictionSet.from_dict(payload)
        for p in restored.values():
            assert p.probability == 0.5  # the built-in rule
            assert p.masses == {"yes": t, "no": f}
            assert p.null_mass == n
        r = restored["raw"]
        T, F = r.masses["yes"], r.masses["no"]
        assert (T + r.null_mass / 2) / (T + F + r.null_mass) == pytest.approx(0.6)
        c = restored["cal"]
        w = (T + F) / (T + F + c.null_mass)
        assert w * c.probability_override + (1 - w) * 0.5 == pytest.approx(0.62)

    def test_raw_not_serialized(self):
        p = make_prediction(raw=object())
        assert "raw" not in p.to_dict()


class TestAverage:
    def test_average_of_nothing_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            PredictionSet.average([])

    def test_average_means_masses(self):
        a = PredictionSet({"q": make_prediction(0.8, 0.1)})
        b = PredictionSet({"q": make_prediction(0.4, 0.5)})
        avg = PredictionSet.average([a, b])
        assert avg["q"].masses["yes"] == pytest.approx(0.6)
        assert avg["q"].masses["no"] == pytest.approx(0.3)

    def test_average_means_probability_overrides(self):
        a = PredictionSet({"q": make_prediction(0.8, 0.1, probability_override=0.4)})
        b = PredictionSet({"q": make_prediction(0.4, 0.5, probability_override=0.6)})
        avg = PredictionSet.average([a, b])
        assert avg["q"].probability_override == pytest.approx(0.5)
        assert avg["q"].probability == pytest.approx(0.5)

    def test_average_drops_partial_probability_overrides(self):
        a = PredictionSet({"q": make_prediction(0.8, 0.1, probability_override=0.4)})
        b = PredictionSet({"q": make_prediction(0.4, 0.5)})
        avg = PredictionSet.average([a, b])
        assert avg["q"].probability_override is None
        assert avg["q"].probability == pytest.approx(0.6 / 0.9, rel=1e-6)

    def test_average_applies_the_rule_after_averaging(self):
        """A member with no answer contributes its raw data, not 0.5."""
        a = PredictionSet({"q": make_prediction(0.0, 0.0, null_mass=1.0)})
        b = PredictionSet({"q": make_prediction(1.0, 0.0)})
        assert PredictionSet.average([a, b])["q"].probability == 1.0  # tie: not dominated
        avg = PredictionSet.average([a, a, b])
        assert avg["q"].null_dominated is True
        assert avg["q"].probability == 0.5

    def test_average_skips_overrides_of_members_with_no_answer(self):
        """An unanswered scene adds nothing, calibrated or not: its override
        (e.g. a calibrator's intercept on score 0) is not in effect."""
        unmatched = make_prediction(0.0, 0.0, probability_override=0.3)
        answered = make_prediction(0.9, 0.1, probability_override=0.9)
        avg = PredictionSet.average(
            [PredictionSet({"q": unmatched}), PredictionSet({"q": answered})]
        )["q"]
        assert avg.probability_override == pytest.approx(0.9)
        assert avg.probability == pytest.approx(0.9)
        raw = PredictionSet.average(
            [
                PredictionSet({"q": make_prediction(0.0, 0.0)}),
                PredictionSet({"q": make_prediction(0.9, 0.1)}),
            ]
        )["q"]
        assert raw.probability == pytest.approx(0.9)

    def test_average_ignores_missing_override_on_member_without_answer(self):
        a = PredictionSet({"q": make_prediction(0.0, 0.0)})
        b = PredictionSet({"q": make_prediction(0.9, 0.1, probability_override=0.7)})
        avg = PredictionSet.average([a, b])["q"]
        assert avg.probability_override == pytest.approx(0.7)

    def test_average_of_members_all_without_answer_has_no_override(self):
        a = PredictionSet({"q": make_prediction(0.2, 0.1, null_mass=0.6, probability_override=0.9)})
        b = PredictionSet({"q": make_prediction(0.0, 0.0, probability_override=0.7)})
        avg = PredictionSet.average([a, b])["q"]
        assert avg.probability_override is None
        assert avg.has_answer is False
        assert avg.probability == 0.5

    def test_average_requires_same_queries(self):
        a = PredictionSet({"q1": make_prediction()})
        b = PredictionSet({"q2": make_prediction()})
        with pytest.raises(ValueError, match="same queries"):
            PredictionSet.average([a, b])

    def test_average_argmax_flag_none_when_members_disagree(self):
        """No single flag describes scenes where the model answered inside
        the interest set on one and outside it on another."""
        a = PredictionSet({"q": make_prediction(argmax_in_interest=True)})
        b = PredictionSet({"q": make_prediction(argmax_in_interest=False)})
        avg = PredictionSet.average([a, b])
        assert avg["q"].argmax_in_interest is None

    def test_average_argmax_flag_kept_when_members_agree(self):
        a = PredictionSet({"q": make_prediction(argmax_in_interest=False)})
        b = PredictionSet({"q": make_prediction(argmax_in_interest=False)})
        avg = PredictionSet.average([a, b])
        assert avg["q"].argmax_in_interest is False
