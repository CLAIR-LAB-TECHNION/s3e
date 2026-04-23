"""Tests for SemanticStateEstimator."""

import json

import pytest
from PIL import Image

from s3e import CalibrationExample
from s3e.calibration import GLOBAL_CALIBRATION_KEY, PlattParameters, PlattScalingProfile
from s3e.semantic_state_estimator import SemanticStateEstimator
from s3e.vlm.backend import VLMOutput
from s3e.translation.identity import IdentityTranslator
from s3e.translation.prewritten import PrewrittenTranslator
from s3e.translation.template import TemplateTranslator
from conftest import FakeVLM, BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM

PROBLEM_3OBJ = """
(define (problem bw-3)
  (:domain blocksworld)
  (:objects a b c - block)
  (:init (on a b) (clear a) (clear c))
  (:goal (on b a))
)
"""

LIGHTS_DOMAIN = """
(define (domain lights)
  (:requirements :typing)
  (:types lamp)
  (:predicates
    (lit ?x - lamp)
  )
)
"""

LIGHTS_PROBLEM = """
(define (problem lights-1)
  (:domain lights)
  (:objects lamp1 - lamp)
  (:init (lit lamp1))
  (:goal (lit lamp1))
)
"""


class CalibrationVLM(FakeVLM):
    def __init__(self, table):
        super().__init__(token_probs={"true": 0.5, "false": 0.5})
        self.table = table

    def query(
        self, images, prompt, system_prompt=None, generate=False, **inference_kwargs
    ):
        del system_prompt
        del generate
        del inference_kwargs
        example_id = images[0].getpixel((0, 0))[0]
        yes_prob = self.table[(example_id, prompt)]
        return VLMOutput(
            token_probs={"true": yes_prob, "false": 1.0 - yes_prob},
            text="true" if yes_prob >= 0.5 else "false",
        )


def make_calibration_image(example_id: int) -> list[Image.Image]:
    return [Image.new("RGB", (2, 2), color=(example_id, 0, 0))]


def save_global_platt_profile(
    tmp_path, blocksworld_domain: str, blocksworld_problem: str
):
    vlm = CalibrationVLM(
        {
            (1, "on(a,a)"): 0.10,
            (1, "on(a,b)"): 0.60,
            (1, "on(b,a)"): 0.20,
            (1, "on(b,b)"): 0.05,
            (1, "clear(a)"): 0.80,
            (1, "clear(b)"): 0.20,
            (2, "on(a,a)"): 0.15,
            (2, "on(a,b)"): 0.55,
            (2, "on(b,a)"): 0.25,
            (2, "on(b,b)"): 0.05,
            (2, "clear(a)"): 0.75,
            (2, "clear(b)"): 0.25,
        }
    )
    se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)
    examples = [
        CalibrationExample(
            images=make_calibration_image(1),
            state_dict={
                "on(a,a)": False,
                "on(a,b)": True,
                "on(b,a)": False,
                "on(b,b)": False,
                "clear(a)": True,
                "clear(b)": False,
            },
        ),
        CalibrationExample(
            images=make_calibration_image(2),
            state_dict={
                "on(a,a)": False,
                "on(a,b)": True,
                "on(b,a)": False,
                "on(b,b)": False,
                "clear(a)": True,
                "clear(b)": False,
            },
        ),
    ]

    se.fit_platt_scaling(examples, scope="global")
    path = tmp_path / "platt-profile.json"
    se.save_platt_scaling(path)
    return path


def save_lifted_platt_profile(
    tmp_path, blocksworld_domain: str, blocksworld_problem: str
):
    vlm = CalibrationVLM(
        {
            (1, "on(a,a)"): 0.05,
            (1, "on(a,b)"): 0.70,
            (1, "on(b,a)"): 0.20,
            (1, "on(b,b)"): 0.05,
            (1, "clear(a)"): 0.80,
            (1, "clear(b)"): 0.20,
            (2, "on(a,a)"): 0.05,
            (2, "on(a,b)"): 0.65,
            (2, "on(a,c)"): 0.25,
            (2, "on(b,a)"): 0.15,
            (2, "on(b,b)"): 0.05,
            (2, "on(b,c)"): 0.10,
            (2, "on(c,a)"): 0.10,
            (2, "on(c,b)"): 0.15,
            (2, "on(c,c)"): 0.05,
            (2, "clear(a)"): 0.75,
            (2, "clear(b)"): 0.25,
            (2, "clear(c)"): 0.70,
        }
    )
    se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)
    examples = [
        CalibrationExample(
            images=make_calibration_image(1),
            state_dict={
                "on(a,a)": False,
                "on(a,b)": True,
                "on(b,a)": False,
                "on(b,b)": False,
                "clear(a)": True,
                "clear(b)": False,
            },
            problem=blocksworld_problem,
        ),
        CalibrationExample(
            images=make_calibration_image(2),
            state_dict={
                "on(a,a)": False,
                "on(a,b)": True,
                "on(a,c)": False,
                "on(b,a)": False,
                "on(b,b)": False,
                "on(b,c)": False,
                "on(c,a)": False,
                "on(c,b)": False,
                "on(c,c)": False,
                "clear(a)": True,
                "clear(b)": False,
                "clear(c)": True,
            },
            problem=PROBLEM_3OBJ,
        ),
    ]

    se.fit_platt_scaling(examples, scope="lifted")
    path = tmp_path / "lifted-platt-profile.json"
    se.save_platt_scaling(path)
    return path


class TestConstruction:
    def test_with_vlm_object(
        self, fake_vlm, single_image, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        assert se is not None

    def test_with_identity_translator_default(
        self, fake_vlm, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        assert isinstance(se.query_translator, IdentityTranslator)

    def test_with_explicit_translator(
        self, fake_vlm, blocksworld_domain, blocksworld_problem
    ):
        translator = TemplateTranslator(
            {"on": "Is {0} on {1}?", "clear": "Is {0} clear?"}
        )
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=fake_vlm,
            query_translator=translator,
        )
        assert se.query_translator is translator

    def test_default_tokens_without_translator(
        self, fake_vlm, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        assert "true" in se.true_tokens
        assert "false" in se.false_tokens

    def test_default_tokens_with_translator(
        self, fake_vlm, blocksworld_domain, blocksworld_problem
    ):
        translator = PrewrittenTranslator(
            {
                "on(a,a)": "q",
                "on(a,b)": "q",
                "on(b,a)": "q",
                "on(b,b)": "q",
                "clear(a)": "q",
                "clear(b)": "q",
            }
        )
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=fake_vlm,
            query_translator=translator,
        )
        assert "yes" in se.true_tokens
        assert "no" in se.false_tokens

    def test_custom_tokens(self, fake_vlm, blocksworld_domain, blocksworld_problem):
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=fake_vlm,
            true_tokens=["correct"],
            false_tokens=["incorrect"],
        )
        assert se.true_tokens == ["correct"]
        assert se.false_tokens == ["incorrect"]


class TestCall:
    def test_returns_bool_dict(
        self, fake_vlm, single_image, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        state = se(single_image)
        assert isinstance(state, dict)
        assert all(isinstance(v, bool) for v in state.values())

    def test_all_predicates_present(
        self, fake_vlm, single_image, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        state = se(single_image)
        # on(a,a), on(a,b), on(b,a), on(b,b), clear(a), clear(b) = 6 predicates
        assert len(state) == 6

    def test_confidence_threshold(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.6, "false": 0.4})
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=vlm, confidence=0.5
        )
        state = se(single_image)
        assert all(v is True for v in state.values())

        # With higher confidence, all should be False
        state = se(single_image, confidence=0.7)
        assert all(v is False for v in state.values())

    def test_confidence_zero_works(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.1, "false": 0.9})
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=vlm, confidence=0.5
        )
        # confidence=0.0 should make everything True (any prob >= 0.0)
        state = se(single_image, confidence=0.0)
        assert all(v is True for v in state.values())


class TestEstimateProbabilities:
    def test_returns_float_dict(
        self, fake_vlm, single_image, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        probs = se.estimate_probabilities(single_image)
        assert isinstance(probs, dict)
        assert all(isinstance(v, float) for v in probs.values())

    def test_probabilities_in_range(
        self, fake_vlm, single_image, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        probs = se.estimate_probabilities(single_image)
        assert all(0.0 <= v <= 1.0 for v in probs.values())

    def test_groups_and_normalizes_tokens(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(
            token_probs={
                "true": 0.3,
                "True": 0.2,
                "TRUE": 0.1,
                "false": 0.15,
                "False": 0.1,
                "FALSE": 0.05,
                "other": 0.1,
            }
        )
        se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)
        probs = se.estimate_probabilities(single_image)
        # true_sum = 0.6, false_sum = 0.3, normalized true = 0.6/0.9 = 0.667
        for pred, prob in probs.items():
            assert abs(prob - 0.6 / 0.9) < 0.01


class TestEstimateRaw:
    def test_returns_vlm_output_dict(
        self, fake_vlm, single_image, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        raw = se.estimate_raw(single_image)
        assert isinstance(raw, dict)
        assert all(isinstance(v, VLMOutput) for v in raw.values())
        assert len(raw) == 6


class TestMultiImageStrategy:
    def test_average_strategy(self, blocksworld_domain, blocksworld_problem):
        call_count = 0
        num_predicates = 6  # 4 on + 2 clear for 2 blocks

        class CountingVLM(FakeVLM):
            def query(
                self,
                images,
                prompt,
                system_prompt=None,
                generate=False,
                **inference_kwargs,
            ):
                del generate
                del inference_kwargs
                nonlocal call_count
                call_count += 1
                # First image batch (calls 1-6) returns high probs,
                # second image batch (calls 7-12) returns low probs.
                if call_count <= num_predicates:
                    return VLMOutput(token_probs={"true": 0.8, "false": 0.2})
                else:
                    return VLMOutput(token_probs={"true": 0.4, "false": 0.6})

        vlm = CountingVLM()
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            multi_image_strategy="average",
        )
        images = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]
        probs = se.estimate_probabilities(images)

        # Each predicate should be the average of two calls
        for prob in probs.values():
            # avg of 0.8/(0.8+0.2) and 0.4/(0.4+0.6) = avg of 0.8 and 0.4 = 0.6
            assert abs(prob - 0.6) < 0.01


class TestTextMatchMode:
    def test_text_match_probability(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.5, "false": 0.5}, text="true")
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            probability_method="text_match",
        )
        probs = se.estimate_probabilities(single_image)
        assert all(v == 1.0 for v in probs.values())

    def test_text_match_queries_with_generate_mode(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(text="true")
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            probability_method="text_match",
        )

        se.estimate_probabilities(single_image)
        assert all(v is True for v in vlm.received_generate_flags)

    def test_logprobs_queries_disable_generate_mode(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.7, "false": 0.3})
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            probability_method="logprobs",
        )

        se.estimate_probabilities(single_image)
        assert all(v is False for v in vlm.received_generate_flags)

    def test_forwards_inference_kwargs(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.8, "false": 0.2})
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            inference_kwargs={"temperature": 0.2, "max_new_tokens": 5},
        )

        se.estimate_probabilities(single_image)
        assert all(
            kwargs["temperature"] == 0.2 for kwargs in vlm.received_inference_kwargs
        )
        assert all(
            kwargs["max_new_tokens"] == 5 for kwargs in vlm.received_inference_kwargs
        )

    def test_text_match_false(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.5, "false": 0.5}, text="false")
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            probability_method="text_match",
        )
        probs = se.estimate_probabilities(single_image)
        assert all(v == 0.0 for v in probs.values())


class TestUserPromptTemplate:
    def test_custom_template(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM()
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            user_prompt_template="Look carefully. {query} Answer yes or no.",
        )
        se.estimate_probabilities(single_image)

        # Verify the VLM received the formatted prompts
        for prompt in vlm.received_prompts:
            assert prompt.startswith("Look carefully.")
            assert prompt.endswith("Answer yes or no.")


class TestCalibrationRuntimeModes:
    def test_calibrated_none_preserves_raw_behavior_without_profile(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.8, "false": 0.2})
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
        )

        probs = se.estimate_probabilities(single_image, calibrated=None)
        assert all(prob == pytest.approx(0.8) for prob in probs.values())

    def test_calibrated_true_without_profile_raises(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.8, "false": 0.2})
        se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)

        with pytest.raises(ValueError, match="fit_platt_scaling"):
            se.estimate_probabilities(single_image, calibrated=True)

    def test_calibrated_false_bypasses_attached_profile(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.8, "false": 0.2})
        se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)
        se._platt_scaling_profile = PlattScalingProfile(
            scope="global",
            probability_method="logprobs",
            true_tokens=["true"],
            false_tokens=["false"],
            domain_fingerprint="irrelevant-for-this-test",
            score_kind="grouped_log_odds",
            groups={
                GLOBAL_CALIBRATION_KEY: PlattParameters(
                    a=2.0,
                    b=0.0,
                    sample_count=8,
                    positive_count=4,
                    negative_count=4,
                )
            },
        )

        raw = se.estimate_probabilities(single_image, calibrated=False)
        calibrated = se.estimate_probabilities(single_image, calibrated=None)
        assert calibrated["on(a,b)"] != pytest.approx(raw["on(a,b)"])

    def test_text_match_rejects_calibration_with_attached_profile(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(text="true")
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            probability_method="text_match",
        )
        se._platt_scaling_profile = PlattScalingProfile(
            scope="global",
            probability_method="text_match",
            true_tokens=["true"],
            false_tokens=["false"],
            domain_fingerprint="irrelevant-for-this-test",
            score_kind="grouped_log_odds",
            groups={
                GLOBAL_CALIBRATION_KEY: PlattParameters(
                    a=1.0,
                    b=0.0,
                    sample_count=8,
                    positive_count=4,
                    negative_count=4,
                )
            },
        )

        with pytest.raises(ValueError, match="logprobs"):
            se.estimate_probabilities(single_image)

    def test_average_strategy_calibrates_each_image_before_averaging(
        self, blocksworld_domain, blocksworld_problem
    ):
        image_one = Image.new("RGB", (64, 64))
        image_two = Image.new("RGB", (64, 64))

        class ImageAwareVLM(FakeVLM):
            def __init__(self):
                super().__init__()
                self.token_probs_by_image_id = {}

            def query(
                self,
                images,
                prompt,
                system_prompt=None,
                generate=False,
                **inference_kwargs,
            ):
                del prompt
                del system_prompt
                del generate
                del inference_kwargs
                return VLMOutput(
                    token_probs=self.token_probs_by_image_id[id(images[0])]
                )

        vlm = ImageAwareVLM()
        vlm.token_probs_by_image_id[id(image_one)] = {"true": 0.9, "false": 0.1}
        vlm.token_probs_by_image_id[id(image_two)] = {"true": 0.6, "false": 0.4}

        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            multi_image_strategy="average",
        )
        se._platt_scaling_profile = PlattScalingProfile(
            scope="global",
            probability_method="logprobs",
            true_tokens=["true"],
            false_tokens=["false"],
            domain_fingerprint="irrelevant-for-this-test",
            score_kind="grouped_log_odds",
            groups={
                GLOBAL_CALIBRATION_KEY: PlattParameters(
                    a=1.0,
                    b=0.0,
                    sample_count=8,
                    positive_count=4,
                    negative_count=4,
                )
            },
        )

        expected = (
            se.estimate_probabilities([image_one], calibrated=None)["on(a,b)"]
            + se.estimate_probabilities([image_two], calibrated=None)["on(a,b)"]
        ) / 2.0
        actual = se.estimate_probabilities([image_one, image_two], calibrated=None)[
            "on(a,b)"
        ]
        assert actual == pytest.approx(expected)


class TestProbabilitiesFromRaw:
    def test_uncalibrated_matches_estimate_probabilities(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.8, "false": 0.2})
        se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)

        raw = se.estimate_raw(single_image)
        from_raw = se.probabilities_from_raw(raw)
        direct = se.estimate_probabilities(single_image, calibrated=False)
        assert from_raw == direct

    def test_calibrated_matches_estimate_probabilities(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.8, "false": 0.2})
        se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)
        se._platt_scaling_profile = PlattScalingProfile(
            scope="global",
            probability_method="logprobs",
            true_tokens=["true"],
            false_tokens=["false"],
            domain_fingerprint="irrelevant-for-this-test",
            score_kind="grouped_log_odds",
            groups={
                GLOBAL_CALIBRATION_KEY: PlattParameters(
                    a=2.0,
                    b=0.0,
                    sample_count=8,
                    positive_count=4,
                    negative_count=4,
                )
            },
        )

        raw = se.estimate_raw(single_image)
        from_raw = se.probabilities_from_raw(raw, calibrated=True)
        direct = se.estimate_probabilities(single_image, calibrated=True)
        assert from_raw == direct

    def test_both_from_single_raw_call(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.8, "false": 0.2})
        se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)
        se._platt_scaling_profile = PlattScalingProfile(
            scope="global",
            probability_method="logprobs",
            true_tokens=["true"],
            false_tokens=["false"],
            domain_fingerprint="irrelevant-for-this-test",
            score_kind="grouped_log_odds",
            groups={
                GLOBAL_CALIBRATION_KEY: PlattParameters(
                    a=2.0,
                    b=0.0,
                    sample_count=8,
                    positive_count=4,
                    negative_count=4,
                )
            },
        )

        raw = se.estimate_raw(single_image)
        uncalibrated = se.probabilities_from_raw(raw, calibrated=False)
        calibrated = se.probabilities_from_raw(raw, calibrated=True)

        assert uncalibrated["on(a,b)"] != pytest.approx(calibrated["on(a,b)"])
        assert all(prob == pytest.approx(0.8) for prob in uncalibrated.values())

    def test_calibrated_true_without_profile_raises(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.8, "false": 0.2})
        se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)

        raw = se.estimate_raw(single_image)
        with pytest.raises(ValueError, match="fit_platt_scaling"):
            se.probabilities_from_raw(raw, calibrated=True)

    def test_auto_mode_without_profile_returns_uncalibrated(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.8, "false": 0.2})
        se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)

        raw = se.estimate_raw(single_image)
        from_raw = se.probabilities_from_raw(raw)
        assert all(prob == pytest.approx(0.8) for prob in from_raw.values())


class TestPlattScalingErrors:
    def test_fit_platt_scaling_rejects_text_match_mode(
        self, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=FakeVLM(text="true"),
            probability_method="text_match",
        )
        example = CalibrationExample(images=make_calibration_image(1), state_dict={})

        with pytest.raises(ValueError, match="logprobs"):
            se.fit_platt_scaling([example], scope="global")

    def test_fit_platt_scaling_rejects_empty_datasets(
        self, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=FakeVLM()
        )

        with pytest.raises(ValueError, match="at least one calibration example"):
            se.fit_platt_scaling([], scope="global")

    def test_fit_platt_scaling_rejects_one_class_groups(
        self, blocksworld_domain, blocksworld_problem
    ):
        vlm = CalibrationVLM(
            {
                (1, "on(a,a)"): 0.10,
                (1, "on(a,b)"): 0.60,
                (1, "on(b,a)"): 0.20,
                (1, "on(b,b)"): 0.05,
                (1, "clear(a)"): 0.80,
                (1, "clear(b)"): 0.20,
            }
        )
        se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)
        example = CalibrationExample(
            images=make_calibration_image(1),
            state_dict={
                "on(a,a)": False,
                "on(a,b)": False,
                "on(b,a)": False,
                "on(b,b)": False,
                "clear(a)": False,
                "clear(b)": False,
            },
        )

        with pytest.raises(
            ValueError, match="requires both positive and negative labels"
        ):
            se.fit_platt_scaling([example], scope="global")

    def test_fit_platt_scaling_accepts_partial_state_labels(
        self, blocksworld_domain, blocksworld_problem
    ):
        vlm = CalibrationVLM(
            {
                (1, "on(a,a)"): 0.10,
                (1, "on(a,b)"): 0.60,
                (1, "on(b,a)"): 0.20,
                (1, "on(b,b)"): 0.05,
                (1, "clear(a)"): 0.80,
            }
        )
        se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)
        example = CalibrationExample(
            images=make_calibration_image(1),
            state_dict={
                "on(a,a)": False,
                "on(a,b)": True,
                "on(b,a)": False,
                "on(b,b)": False,
                "clear(a)": True,
            },
        )

        se.fit_platt_scaling([example], scope="global")

        profile = se._platt_scaling_profile
        assert profile is not None
        assert profile.groups[GLOBAL_CALIBRATION_KEY].sample_count == 5

    def test_fit_platt_scaling_rejects_unknown_predicates_in_labels(
        self, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=FakeVLM()
        )
        example = CalibrationExample(
            images=make_calibration_image(1),
            state_dict={
                "on(a,b)": True,
                "nonexistent(x)": False,
            },
        )

        with pytest.raises(
            ValueError, match="not in the current problem.*nonexistent\\(x\\)"
        ):
            se.fit_platt_scaling([example], scope="global")


class TestGlobalPlattScaling:
    def test_fit_platt_scaling_global_changes_probability_values(
        self, blocksworld_domain, blocksworld_problem
    ):
        vlm = CalibrationVLM(
            {
                (1, "on(a,a)"): 0.10,
                (1, "on(a,b)"): 0.40,
                (1, "on(b,a)"): 0.35,
                (1, "on(b,b)"): 0.05,
                (1, "clear(a)"): 0.75,
                (1, "clear(b)"): 0.25,
                (2, "on(a,a)"): 0.15,
                (2, "on(a,b)"): 0.55,
                (2, "on(b,a)"): 0.30,
                (2, "on(b,b)"): 0.05,
                (2, "clear(a)"): 0.80,
                (2, "clear(b)"): 0.20,
                (3, "on(a,a)"): 0.10,
                (3, "on(a,b)"): 0.60,
                (3, "on(b,a)"): 0.25,
                (3, "on(b,b)"): 0.05,
                (3, "clear(a)"): 0.70,
                (3, "clear(b)"): 0.35,
            }
        )
        se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)
        examples = [
            CalibrationExample(
                images=make_calibration_image(1),
                state_dict={
                    "on(a,a)": False,
                    "on(a,b)": True,
                    "on(b,a)": False,
                    "on(b,b)": False,
                    "clear(a)": True,
                    "clear(b)": False,
                },
            ),
            CalibrationExample(
                images=make_calibration_image(2),
                state_dict={
                    "on(a,a)": False,
                    "on(a,b)": True,
                    "on(b,a)": False,
                    "on(b,b)": False,
                    "clear(a)": True,
                    "clear(b)": False,
                },
            ),
        ]

        raw = se.estimate_probabilities(make_calibration_image(3), calibrated=False)
        se.fit_platt_scaling(examples, scope="global")
        calibrated = se.estimate_probabilities(make_calibration_image(3), calibrated=True)
        profile = se._platt_scaling_profile

        assert calibrated["on(a,b)"] != pytest.approx(raw["on(a,b)"])
        assert all(0.0 <= value <= 1.0 for value in calibrated.values())
        assert profile is not None
        assert profile.scope == "global"
        assert set(profile.groups) == {GLOBAL_CALIBRATION_KEY}
        assert profile.groups[GLOBAL_CALIBRATION_KEY].sample_count == 12

    def test_fit_platt_scaling_requires_sklearn(
        self, blocksworld_domain, blocksworld_problem, monkeypatch
    ):
        import s3e.calibration as calibration

        vlm = CalibrationVLM({(1, "on(a,a)"): 0.5, (1, "on(a,b)"): 0.5, (1, "on(b,a)"): 0.5, (1, "on(b,b)"): 0.5, (1, "clear(a)"): 0.5, (1, "clear(b)"): 0.5})
        se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)
        example = CalibrationExample(
            images=make_calibration_image(1),
            state_dict={
                "on(a,a)": False,
                "on(a,b)": True,
                "on(b,a)": False,
                "on(b,b)": False,
                "clear(a)": True,
                "clear(b)": False,
            },
        )

        monkeypatch.setattr(calibration, "LogisticRegression", None)
        with pytest.raises(ImportError, match="s3e\\[calibration\\]"):
            se.fit_platt_scaling([example], scope="global")

    def test_fit_platt_scaling_rejects_non_global_scope(
        self, blocksworld_domain, blocksworld_problem
    ):
        vlm = CalibrationVLM({(1, "on(a,a)"): 0.5, (1, "on(a,b)"): 0.5, (1, "on(b,a)"): 0.5, (1, "on(b,b)"): 0.5, (1, "clear(a)"): 0.5, (1, "clear(b)"): 0.5})
        se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)
        example = CalibrationExample(
            images=make_calibration_image(1),
            state_dict={
                "on(a,a)": False,
                "on(a,b)": True,
                "on(b,a)": False,
                "on(b,b)": False,
                "clear(a)": True,
                "clear(b)": False,
            },
        )

        with pytest.raises(ValueError, match="Unsupported Platt scaling scope"):
            se.fit_platt_scaling([example], scope="unsupported")

    def test_fit_platt_scaling_average_mode_uses_per_image_score_samples(
        self, blocksworld_domain, blocksworld_problem
    ):
        vlm = CalibrationVLM(
            {
                (1, "on(a,a)"): 0.10,
                (1, "on(a,b)"): 0.45,
                (1, "on(b,a)"): 0.35,
                (1, "on(b,b)"): 0.05,
                (1, "clear(a)"): 0.75,
                (1, "clear(b)"): 0.25,
                (2, "on(a,a)"): 0.15,
                (2, "on(a,b)"): 0.55,
                (2, "on(b,a)"): 0.30,
                (2, "on(b,b)"): 0.05,
                (2, "clear(a)"): 0.80,
                (2, "clear(b)"): 0.20,
                (3, "on(a,a)"): 0.10,
                (3, "on(a,b)"): 0.60,
                (3, "on(b,a)"): 0.25,
                (3, "on(b,b)"): 0.05,
                (3, "clear(a)"): 0.70,
                (3, "clear(b)"): 0.35,
                (4, "on(a,a)"): 0.20,
                (4, "on(a,b)"): 0.65,
                (4, "on(b,a)"): 0.20,
                (4, "on(b,b)"): 0.05,
                (4, "clear(a)"): 0.85,
                (4, "clear(b)"): 0.15,
            }
        )
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            multi_image_strategy="average",
        )
        examples = [
            CalibrationExample(
                images=make_calibration_image(1) + make_calibration_image(2),
                state_dict={
                    "on(a,a)": False,
                    "on(a,b)": True,
                    "on(b,a)": False,
                    "on(b,b)": False,
                    "clear(a)": True,
                    "clear(b)": False,
                },
            ),
            CalibrationExample(
                images=make_calibration_image(3) + make_calibration_image(4),
                state_dict={
                    "on(a,a)": False,
                    "on(a,b)": True,
                    "on(b,a)": False,
                    "on(b,b)": False,
                    "clear(a)": True,
                    "clear(b)": False,
                },
            ),
        ]

        se.fit_platt_scaling(examples, scope="global")

        assert se._platt_scaling_profile is not None
        assert (
            se._platt_scaling_profile.groups[GLOBAL_CALIBRATION_KEY].sample_count == 24
        )

    def test_fit_platt_scaling_only_queries_labeled_predicates(
        self, blocksworld_domain, blocksworld_problem
    ):
        """The VLM should only be called for predicates in the state_dict."""
        queried_prompts = []

        class TrackingVLM(FakeVLM):
            def query(self, images, prompt, system_prompt=None, generate=False, **inference_kwargs):
                queried_prompts.append(prompt)
                return VLMOutput(
                    token_probs={"true": 0.6, "false": 0.4},
                    text="true",
                )

        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=TrackingVLM()
        )
        examples = [
            CalibrationExample(
                images=make_calibration_image(1),
                state_dict={"on(a,b)": True, "clear(a)": False},
            ),
            CalibrationExample(
                images=make_calibration_image(2),
                state_dict={"on(a,b)": False, "clear(a)": True},
            ),
        ]

        se.fit_platt_scaling(examples, scope="global")

        assert set(queried_prompts) == {"on(a,b)", "clear(a)"}
        assert se._platt_scaling_profile.groups[GLOBAL_CALIBRATION_KEY].sample_count == 4

    def test_fit_platt_scaling_partial_labels_average_mode(
        self, blocksworld_domain, blocksworld_problem
    ):
        """Average mode with partial labels should produce per-image samples only for labeled predicates."""
        vlm = CalibrationVLM(
            {
                (1, "on(a,b)"): 0.60,
                (1, "clear(a)"): 0.80,
                (2, "on(a,b)"): 0.55,
                (2, "clear(a)"): 0.75,
                (3, "on(a,b)"): 0.40,
                (3, "clear(a)"): 0.20,
                (4, "on(a,b)"): 0.35,
                (4, "clear(a)"): 0.25,
            }
        )
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=vlm,
            multi_image_strategy="average",
        )
        examples = [
            CalibrationExample(
                images=make_calibration_image(1) + make_calibration_image(2),
                state_dict={"on(a,b)": True, "clear(a)": True},
            ),
            CalibrationExample(
                images=make_calibration_image(3) + make_calibration_image(4),
                state_dict={"on(a,b)": False, "clear(a)": False},
            ),
        ]

        se.fit_platt_scaling(examples, scope="global")

        assert se._platt_scaling_profile is not None
        assert se._platt_scaling_profile.groups[GLOBAL_CALIBRATION_KEY].sample_count == 8

    def test_fit_platt_scaling_partial_labels_lifted_scope(
        self, blocksworld_domain, blocksworld_problem
    ):
        """Lifted scope with partial labels should group only the labeled predicates."""
        vlm = CalibrationVLM(
            {
                (1, "on(a,b)"): 0.60,
                (1, "clear(a)"): 0.80,
                (2, "on(a,b)"): 0.40,
                (2, "clear(a)"): 0.25,
            }
        )
        se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)
        examples = [
            CalibrationExample(
                images=make_calibration_image(1),
                state_dict={"on(a,b)": True, "clear(a)": True},
            ),
            CalibrationExample(
                images=make_calibration_image(2),
                state_dict={"on(a,b)": False, "clear(a)": False},
            ),
        ]

        se.fit_platt_scaling(examples, scope="lifted")

        profile = se._platt_scaling_profile
        assert profile is not None
        assert set(profile.groups.keys()) == {"on", "clear"}
        assert profile.groups["on"].sample_count == 2
        assert profile.groups["clear"].sample_count == 2


class TestLiftedPlattScaling:
    def test_fit_platt_scaling_lifted_handles_multiple_problem_instances(
        self, blocksworld_domain, blocksworld_problem
    ):
        vlm = CalibrationVLM(
            {
                (1, "on(a,a)"): 0.05,
                (1, "on(a,b)"): 0.70,
                (1, "on(b,a)"): 0.20,
                (1, "on(b,b)"): 0.05,
                (1, "clear(a)"): 0.80,
                (1, "clear(b)"): 0.20,
                (2, "on(a,a)"): 0.05,
                (2, "on(a,b)"): 0.65,
                (2, "on(a,c)"): 0.25,
                (2, "on(b,a)"): 0.15,
                (2, "on(b,b)"): 0.05,
                (2, "on(b,c)"): 0.10,
                (2, "on(c,a)"): 0.10,
                (2, "on(c,b)"): 0.15,
                (2, "on(c,c)"): 0.05,
                (2, "clear(a)"): 0.75,
                (2, "clear(b)"): 0.25,
                (2, "clear(c)"): 0.70,
                (3, "on(a,a)"): 0.05,
                (3, "on(a,b)"): 0.60,
                (3, "on(a,c)"): 0.20,
                (3, "on(b,a)"): 0.20,
                (3, "on(b,b)"): 0.05,
                (3, "on(b,c)"): 0.15,
                (3, "on(c,a)"): 0.15,
                (3, "on(c,b)"): 0.20,
                (3, "on(c,c)"): 0.05,
                (3, "clear(a)"): 0.70,
                (3, "clear(b)"): 0.30,
                (3, "clear(c)"): 0.65,
            }
        )
        se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)
        examples = [
            CalibrationExample(
                images=make_calibration_image(1),
                state_dict={
                    "on(a,a)": False,
                    "on(a,b)": True,
                    "on(b,a)": False,
                    "on(b,b)": False,
                    "clear(a)": True,
                    "clear(b)": False,
                },
                problem=blocksworld_problem,
            ),
            CalibrationExample(
                images=make_calibration_image(2),
                state_dict={
                    "on(a,a)": False,
                    "on(a,b)": True,
                    "on(a,c)": False,
                    "on(b,a)": False,
                    "on(b,b)": False,
                    "on(b,c)": False,
                    "on(c,a)": False,
                    "on(c,b)": False,
                    "on(c,c)": False,
                    "clear(a)": True,
                    "clear(b)": False,
                    "clear(c)": True,
                },
                problem=PROBLEM_3OBJ,
            ),
        ]

        se.fit_platt_scaling(examples, scope="lifted")
        profile = se._platt_scaling_profile
        se.swap_problem(blocksworld_domain, PROBLEM_3OBJ)
        calibrated = se.estimate_probabilities(make_calibration_image(3), calibrated=True)

        assert profile is not None
        assert profile.scope == "lifted"
        assert set(profile.groups) == {"on", "clear"}
        assert profile.groups["on"].sample_count == 13
        assert profile.groups["clear"].sample_count == 5
        assert "on(c,a)" in calibrated
        assert "clear(c)" in calibrated
        assert all(0.0 <= value <= 1.0 for value in calibrated.values())

    def test_save_and_load_platt_scaling_round_trip(
        self, tmp_path, blocksworld_domain, blocksworld_problem
    ):
        vlm = CalibrationVLM(
            {
                (1, "on(a,a)"): 0.10,
                (1, "on(a,b)"): 0.60,
                (1, "on(b,a)"): 0.20,
                (1, "on(b,b)"): 0.05,
                (1, "clear(a)"): 0.80,
                (1, "clear(b)"): 0.20,
                (2, "on(a,a)"): 0.15,
                (2, "on(a,b)"): 0.55,
                (2, "on(b,a)"): 0.25,
                (2, "on(b,b)"): 0.05,
                (2, "clear(a)"): 0.75,
                (2, "clear(b)"): 0.25,
            }
        )
        se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)
        examples = [
            CalibrationExample(
                images=make_calibration_image(1),
                state_dict={
                    "on(a,a)": False,
                    "on(a,b)": True,
                    "on(b,a)": False,
                    "on(b,b)": False,
                    "clear(a)": True,
                    "clear(b)": False,
                },
            ),
            CalibrationExample(
                images=make_calibration_image(2),
                state_dict={
                    "on(a,a)": False,
                    "on(a,b)": True,
                    "on(b,a)": False,
                    "on(b,b)": False,
                    "clear(a)": True,
                    "clear(b)": False,
                },
            ),
        ]

        se.fit_platt_scaling(examples, scope="global")
        before = se.estimate_probabilities(make_calibration_image(2), calibrated=True)
        path = tmp_path / "platt-profile.json"

        se.save_platt_scaling(path)
        se.clear_platt_scaling()
        with pytest.raises(ValueError, match="fit_platt_scaling"):
            se.estimate_probabilities(make_calibration_image(2), calibrated=True)

        se.load_platt_scaling(path)
        after = se.estimate_probabilities(make_calibration_image(2), calibrated=True)
        assert after == pytest.approx(before)

    def test_load_platt_scaling_accepts_equivalent_file_backed_domains(
        self, tmp_path, blocksworld_domain, blocksworld_problem
    ):
        domain_path_a = tmp_path / "blocksworld-a.pddl"
        domain_path_b = tmp_path / "blocksworld-b.pddl"
        problem_path = tmp_path / "blocksworld-problem.pddl"
        domain_path_a.write_text(blocksworld_domain)
        domain_path_b.write_text(blocksworld_domain)
        problem_path.write_text(blocksworld_problem)

        vlm_a = CalibrationVLM(
            {
                (1, "on(a,a)"): 0.10,
                (1, "on(a,b)"): 0.60,
                (1, "on(b,a)"): 0.20,
                (1, "on(b,b)"): 0.05,
                (1, "clear(a)"): 0.80,
                (1, "clear(b)"): 0.20,
                (2, "on(a,a)"): 0.15,
                (2, "on(a,b)"): 0.55,
                (2, "on(b,a)"): 0.25,
                (2, "on(b,b)"): 0.05,
                (2, "clear(a)"): 0.75,
                (2, "clear(b)"): 0.25,
            }
        )
        se_a = SemanticStateEstimator(
            str(domain_path_a), str(problem_path), vlm=vlm_a
        )
        examples = [
            CalibrationExample(
                images=make_calibration_image(1),
                state_dict={
                    "on(a,a)": False,
                    "on(a,b)": True,
                    "on(b,a)": False,
                    "on(b,b)": False,
                    "clear(a)": True,
                    "clear(b)": False,
                },
            ),
            CalibrationExample(
                images=make_calibration_image(2),
                state_dict={
                    "on(a,a)": False,
                    "on(a,b)": True,
                    "on(b,a)": False,
                    "on(b,b)": False,
                    "clear(a)": True,
                    "clear(b)": False,
                },
            ),
        ]
        se_a.fit_platt_scaling(examples, scope="global")
        expected = se_a.estimate_probabilities(make_calibration_image(2), calibrated=True)

        path = tmp_path / "portable-platt-profile.json"
        se_a.save_platt_scaling(path)

        vlm_b = CalibrationVLM(vlm_a.table)
        se_b = SemanticStateEstimator(
            str(domain_path_b), str(problem_path), vlm=vlm_b
        )
        se_b.load_platt_scaling(path)

        actual = se_b.estimate_probabilities(make_calibration_image(2), calibrated=True)
        assert actual == pytest.approx(expected)

    def test_load_platt_scaling_rejects_incompatible_probability_method(
        self, tmp_path, blocksworld_domain, blocksworld_problem
    ):
        path = save_global_platt_profile(
            tmp_path, blocksworld_domain, blocksworld_problem
        )
        data = json.loads(path.read_text())
        data["probability_method"] = "text_match"
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=FakeVLM(text="true"),
        )

        with pytest.raises(ValueError, match="logprobs mode"):
            se.load_platt_scaling(path)
        with pytest.raises(ValueError, match="fit_platt_scaling"):
            se.estimate_probabilities(make_calibration_image(1), calibrated=True)

    def test_load_platt_scaling_rejects_token_group_mismatch(
        self, tmp_path, blocksworld_domain, blocksworld_problem
    ):
        path = save_global_platt_profile(
            tmp_path, blocksworld_domain, blocksworld_problem
        )
        data = json.loads(path.read_text())
        data["true_tokens"] = ["correct"]
        data["false_tokens"] = ["incorrect"]
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=FakeVLM(),
        )

        with pytest.raises(ValueError, match="token groups"):
            se.load_platt_scaling(path)
        with pytest.raises(ValueError, match="fit_platt_scaling"):
            se.estimate_probabilities(make_calibration_image(1), calibrated=True)

    def test_load_platt_scaling_rejects_domain_fingerprint_mismatch(
        self, tmp_path, blocksworld_domain, blocksworld_problem
    ):
        path = save_global_platt_profile(
            tmp_path, blocksworld_domain, blocksworld_problem
        )
        data = json.loads(path.read_text())
        data["domain_fingerprint"] = "wrong-domain-fingerprint"
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=FakeVLM(),
        )

        with pytest.raises(ValueError, match="different domain"):
            se.load_platt_scaling(path)
        with pytest.raises(ValueError, match="fit_platt_scaling"):
            se.estimate_probabilities(make_calibration_image(1), calibrated=True)

    def test_load_platt_scaling_rejects_unsupported_scope(
        self, tmp_path, blocksworld_domain, blocksworld_problem
    ):
        path = save_global_platt_profile(
            tmp_path, blocksworld_domain, blocksworld_problem
        )
        data = json.loads(path.read_text())
        data["scope"] = "predicate"
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=FakeVLM(),
        )

        with pytest.raises(ValueError, match="Unsupported Platt scaling scope"):
            se.load_platt_scaling(path)
        with pytest.raises(ValueError, match="fit_platt_scaling"):
            se.estimate_probabilities(make_calibration_image(1), calibrated=True)

    def test_load_platt_scaling_rejects_wrong_score_kind(
        self, tmp_path, blocksworld_domain, blocksworld_problem
    ):
        path = save_global_platt_profile(
            tmp_path, blocksworld_domain, blocksworld_problem
        )
        data = json.loads(path.read_text())
        data["score_kind"] = "raw_probability"
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=FakeVLM(),
        )

        with pytest.raises(ValueError, match="score_kind"):
            se.load_platt_scaling(path)
        with pytest.raises(ValueError, match="fit_platt_scaling"):
            se.estimate_probabilities(make_calibration_image(1), calibrated=True)

    def test_load_platt_scaling_with_missing_groups_falls_back_to_uncalibrated(
        self, tmp_path, blocksworld_domain, blocksworld_problem
    ):
        """Missing lifted groups are allowed — those predicates use uncalibrated probabilities."""
        path = save_lifted_platt_profile(
            tmp_path, blocksworld_domain, blocksworld_problem
        )
        data = json.loads(path.read_text())
        del data["groups"]["clear"]
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=FakeVLM(),
        )

        se.load_platt_scaling(path)
        uncalibrated = se.estimate_probabilities(make_calibration_image(1))
        calibrated = se.estimate_probabilities(make_calibration_image(1), calibrated=True)

        clear_preds = [p for p in calibrated if p.startswith("clear(")]
        assert clear_preds
        for p in clear_preds:
            assert calibrated[p] == uncalibrated[p]


class TestSwapProblem:
    def test_swap_updates_predicates(self, fake_vlm, single_image, blocksworld_domain):
        problem_2obj = """
        (define (problem bw-2)
          (:domain blocksworld)
          (:objects a b - block)
          (:init (on a b) (clear a))
          (:goal (on b a))
        )
        """
        problem_3obj = """
        (define (problem bw-3)
          (:domain blocksworld)
          (:objects a b c - block)
          (:init (on a b) (clear a) (clear c))
          (:goal (on b a))
        )
        """
        se = SemanticStateEstimator(blocksworld_domain, problem_2obj, vlm=fake_vlm)
        state_2 = se(single_image)
        assert len(state_2) == 6  # 2 blocks: 4 on + 2 clear

        se.swap_problem(blocksworld_domain, problem_3obj)
        state_3 = se(single_image)
        assert len(state_3) == 12  # 3 blocks: 9 on + 3 clear

    def test_swap_to_different_domain_invalidates_platt_scaling(
        self, blocksworld_domain, blocksworld_problem
    ):
        vlm = FakeVLM(token_probs={"true": 0.8, "false": 0.2})
        se = SemanticStateEstimator(blocksworld_domain, blocksworld_problem, vlm=vlm)
        examples = [
            CalibrationExample(
                images=make_calibration_image(1),
                state_dict={
                    "on(a,a)": False,
                    "on(a,b)": True,
                    "on(b,a)": False,
                    "on(b,b)": False,
                    "clear(a)": True,
                    "clear(b)": False,
                },
            ),
            CalibrationExample(
                images=make_calibration_image(2),
                state_dict={
                    "on(a,a)": False,
                    "on(a,b)": True,
                    "on(b,a)": False,
                    "on(b,b)": False,
                    "clear(a)": True,
                    "clear(b)": False,
                },
            ),
        ]
        se.fit_platt_scaling(examples, scope="global")

        se.swap_problem(LIGHTS_DOMAIN, LIGHTS_PROBLEM)

        with pytest.raises(ValueError, match="No Platt scaling profile is loaded"):
            se.estimate_probabilities(make_calibration_image(1), calibrated=True)

        raw = se.estimate_probabilities(make_calibration_image(1), calibrated=False)
        assert raw == {"lit(lamp1)": pytest.approx(0.8)}


class TestPredicateFiltering:
    """Tests for the predicates parameter on __call__, estimate_probabilities, and estimate_raw."""

    def test_estimate_raw_filters_predicates(
        self, fake_vlm, single_image, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        subset = ["on(a,b)", "clear(a)"]
        raw = se.estimate_raw(single_image, predicates=subset)
        assert set(raw.keys()) == set(subset)

    def test_estimate_probabilities_filters_predicates(
        self, fake_vlm, single_image, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        subset = ["on(a,b)"]
        probs = se.estimate_probabilities(single_image, predicates=subset)
        assert list(probs.keys()) == subset

    def test_call_filters_predicates(
        self, fake_vlm, single_image, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        subset = ["clear(b)"]
        result = se(single_image, predicates=subset)
        assert list(result.keys()) == subset
        assert all(isinstance(v, bool) for v in result.values())

    def test_unknown_predicate_raises(
        self, fake_vlm, single_image, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        with pytest.raises(ValueError, match="Unknown predicate"):
            se.estimate_raw(single_image, predicates=["nonexistent(x)"])

    def test_none_predicates_returns_all(
        self, fake_vlm, single_image, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        all_result = se.estimate_raw(single_image)
        none_result = se.estimate_raw(single_image, predicates=None)
        assert set(all_result.keys()) == set(none_result.keys())

    def test_average_strategy_filters_predicates(
        self, fake_vlm, blocksworld_domain, blocksworld_problem
    ):
        images = [
            Image.new("RGB", (2, 2), color=(0, 0, 0)),
            Image.new("RGB", (2, 2), color=(1, 1, 1)),
        ]
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=fake_vlm,
            multi_image_strategy="average",
        )
        subset = ["on(a,b)", "clear(a)"]
        probs = se.estimate_probabilities(images, predicates=subset)
        assert set(probs.keys()) == set(subset)


@pytest.mark.slow
class TestSemanticStateEstimatorIntegration:
    """End-to-end integration tests with a tiny real HuggingFace VLM."""

    TINY_VLM_ID = "katuni4ka/tiny-random-llava"

    def test_end_to_end_with_hf_vlm(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=self.TINY_VLM_ID,
            vlm_kwargs={"device_map": "cpu"},
        )
        state = se(single_image)
        assert isinstance(state, dict)
        assert all(isinstance(v, bool) for v in state.values())
        assert len(state) == 6  # 2 blocks: 4 on + 2 clear

    def test_estimate_probabilities_with_hf_vlm(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=self.TINY_VLM_ID,
            vlm_kwargs={"device_map": "cpu"},
        )
        probs = se.estimate_probabilities(single_image)
        assert isinstance(probs, dict)
        assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in probs.values())
        assert len(probs) == 6

    def test_estimate_raw_with_hf_vlm(
        self, single_image, blocksworld_domain, blocksworld_problem
    ):
        se = SemanticStateEstimator(
            blocksworld_domain,
            blocksworld_problem,
            vlm=self.TINY_VLM_ID,
            vlm_kwargs={"device_map": "cpu"},
        )
        raw = se.estimate_raw(single_image)
        assert isinstance(raw, dict)
        assert all(isinstance(v, VLMOutput) for v in raw.values())
        assert all(len(v.token_probs) > 0 for v in raw.values())
        assert len(raw) == 6
