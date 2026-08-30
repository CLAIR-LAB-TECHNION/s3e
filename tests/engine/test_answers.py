"""Tests for answer options and answer spaces."""

import pytest

from s3e.backends import VLMOutput
from s3e.engine import (
    AnswerOption,
    AnswerSpace,
    BinaryAnswers,
    CategoricalAnswers,
    expand_token_variants,
)


class TestExpandTokenVariants:
    def test_case_and_leading_space_variants(self):
        variants = set(expand_token_variants("red"))
        assert {"red", "Red", "RED", " red", " Red", " RED"} <= variants

    def test_multiword_label_kept_verbatim_among_variants(self):
        assert "dark blue" in expand_token_variants("dark blue")


class TestAnswerOption:
    def test_make_auto_expands(self):
        option = AnswerOption.make("yes")
        assert "yes" in option.tokens and "Yes" in option.tokens

    def test_make_explicit_tokens_win(self):
        option = AnswerOption.make("yes", tokens=["y"])
        assert option.tokens == ("y",)
        assert option.label == "yes"


class TestBinaryAnswers:
    def test_default_yes_no(self):
        space = BinaryAnswers()
        assert space.true_label == "yes"
        assert space.false_label == "no"
        assert "Yes" in space.options[0].tokens

    def test_relabel_true_false(self):
        space = BinaryAnswers("true", "false")
        assert space.true_label == "true"
        assert "True" in space.options[0].tokens
        assert "FALSE" in space.options[1].tokens

    def test_explicit_token_overrides(self):
        space = BinaryAnswers(true_tokens=["yep"], false_tokens=["nope"])
        assert space.options[0].tokens == ("yep",)

    def test_null_tokens_create_null_option(self):
        space = BinaryAnswers(null_tokens=["unknown", "Unknown"])
        assert space.null_option is not None
        assert space.null_option.label == "unknown"
        assert "unknown" in space.interest_tokens

    def test_overlapping_tokens_rejected(self):
        with pytest.raises(ValueError, match="overlap"):
            BinaryAnswers(true_tokens=["yes"], false_tokens=["yes"])


class TestDuplicateTokensWithinOption:
    def test_answer_option_rejects_duplicate_tokens(self):
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            AnswerOption("yes", ("yes", "yes"))

    def test_binary_answers_reject_duplicate_tokens_in_one_option(self):
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            BinaryAnswers(true_tokens=["yes", "yes"], false_tokens=["no"])


class TestLogprobScoring:
    def test_masses_summed_per_option(self):
        space = BinaryAnswers(true_tokens=["yes", "Yes"], false_tokens=["no"])
        output = VLMOutput(token_probs={"yes": 0.5, "Yes": 0.2, "no": 0.1})
        scored = space.score(output, scoring="logprobs")
        assert scored.masses == {"yes": 0.7, "no": 0.1}
        assert scored.null_mass == 0.0
        assert scored.unassigned_mass == pytest.approx(0.2)

    def test_null_option_mass_separated(self):
        space = BinaryAnswers(
            true_tokens=["yes"], false_tokens=["no"], null_tokens=["unknown"]
        )
        output = VLMOutput(token_probs={"yes": 0.2, "no": 0.1, "unknown": 0.6})
        scored = space.score(output, scoring="logprobs")
        assert scored.null_mass == pytest.approx(0.6)
        assert scored.masses == {"yes": 0.2, "no": 0.1}


class TestTextMatchScoring:
    def test_matching_option_gets_full_mass(self):
        space = BinaryAnswers()
        output = VLMOutput(text="Yes, it is.")
        scored = space.score(output, scoring="text_match")
        assert scored.masses["yes"] == 1.0
        assert scored.masses["no"] == 0.0
        assert scored.unassigned_mass == 0.0

    def test_no_match_is_fully_unassigned(self):
        space = BinaryAnswers()
        output = VLMOutput(text="I cannot tell.")
        scored = space.score(output, scoring="text_match")
        assert scored.masses == {"yes": 0.0, "no": 0.0}
        assert scored.unassigned_mass == 1.0

    def test_null_option_matches_text(self):
        space = BinaryAnswers(null_tokens=["unknown"])
        output = VLMOutput(text="unknown")
        scored = space.score(output, scoring="text_match")
        assert scored.null_mass == 1.0


class TestCategoricalAnswers:
    def test_labels_from_strings(self):
        space = CategoricalAnswers(["red", "green", "blue"])
        assert space.labels == ["red", "green", "blue"]

    def test_scoring_over_three_options(self):
        space = CategoricalAnswers(["red", "green", "blue"])
        output = VLMOutput(token_probs={"red": 0.6, " green": 0.3, "blue": 0.05})
        scored = space.score(output, scoring="logprobs")
        assert scored.masses["red"] == pytest.approx(0.6)
        assert scored.masses["green"] == pytest.approx(0.3)


class TestSerialization:
    @pytest.mark.parametrize(
        "space",
        [
            BinaryAnswers("true", "false", null_tokens=["unknown"]),
            CategoricalAnswers(["red", "green"]),
        ],
    )
    def test_round_trip(self, space):
        restored = AnswerSpace.from_dict(space.to_dict())
        assert type(restored) is type(space)
        assert restored.to_dict() == space.to_dict()

    def test_binary_round_trip_keeps_semantics(self):
        restored = AnswerSpace.from_dict(BinaryAnswers("true", "false").to_dict())
        assert restored.true_label == "true"


class TestUnknownScoring:
    def test_unknown_scoring_mode_rejected(self):
        with pytest.raises(ValueError, match="scoring"):
            BinaryAnswers().score(VLMOutput(), scoring="magic")


class TestEmptyOptions:
    def test_empty_answer_space_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            AnswerSpace([])

    def test_empty_categorical_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            CategoricalAnswers([])
