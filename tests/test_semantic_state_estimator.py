"""Tests for SemanticStateEstimator."""

import pytest
from PIL import Image

from s3e.calibration import GLOBAL_CALIBRATION_KEY, PlattParameters, PlattScalingProfile
from s3e.semantic_state_estimator import SemanticStateEstimator
from s3e.vlm.backend import VLMOutput
from s3e.translation.identity import IdentityTranslator
from s3e.translation.prewritten import PrewrittenTranslator
from s3e.translation.template import TemplateTranslator
from conftest import FakeVLM, BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM


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
