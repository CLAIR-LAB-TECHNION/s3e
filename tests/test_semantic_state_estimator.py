"""Tests for SemanticStateEstimator."""

import pytest
from PIL import Image

from s3e.semantic_state_estimator import SemanticStateEstimator
from s3e.vlm.backend import VLMOutput
from s3e.translation.identity import IdentityTranslator
from s3e.translation.prewritten import PrewrittenTranslator
from s3e.translation.template import TemplateTranslator
from tests.conftest import FakeVLM, BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM


class TestConstruction:
    def test_with_vlm_object(self, fake_vlm, single_image, blocksworld_domain, blocksworld_problem):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        assert se is not None

    def test_with_identity_translator_default(self, fake_vlm, blocksworld_domain, blocksworld_problem):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        assert isinstance(se.query_translator, IdentityTranslator)

    def test_with_explicit_translator(self, fake_vlm, blocksworld_domain, blocksworld_problem):
        translator = TemplateTranslator({"on": "Is {0} on {1}?", "clear": "Is {0} clear?"})
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem,
            vlm=fake_vlm,
            query_translator=translator,
        )
        assert se.query_translator is translator

    def test_default_tokens_without_translator(self, fake_vlm, blocksworld_domain, blocksworld_problem):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        assert "true" in se.true_tokens
        assert "false" in se.false_tokens

    def test_default_tokens_with_translator(self, fake_vlm, blocksworld_domain, blocksworld_problem):
        translator = PrewrittenTranslator({
            "on(a,a)": "q", "on(a,b)": "q", "on(b,a)": "q", "on(b,b)": "q",
            "clear(a)": "q", "clear(b)": "q",
        })
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem,
            vlm=fake_vlm,
            query_translator=translator,
        )
        assert "yes" in se.true_tokens
        assert "no" in se.false_tokens

    def test_custom_tokens(self, fake_vlm, blocksworld_domain, blocksworld_problem):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem,
            vlm=fake_vlm,
            true_tokens=["correct"],
            false_tokens=["incorrect"],
        )
        assert se.true_tokens == ["correct"]
        assert se.false_tokens == ["incorrect"]


class TestCall:
    def test_returns_bool_dict(self, fake_vlm, single_image, blocksworld_domain, blocksworld_problem):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        state = se(single_image)
        assert isinstance(state, dict)
        assert all(isinstance(v, bool) for v in state.values())

    def test_all_predicates_present(self, fake_vlm, single_image, blocksworld_domain, blocksworld_problem):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        state = se(single_image)
        # on(a,a), on(a,b), on(b,a), on(b,b), clear(a), clear(b) = 6 predicates
        assert len(state) == 6

    def test_confidence_threshold(self, single_image, blocksworld_domain, blocksworld_problem):
        vlm = FakeVLM(token_probs={"true": 0.6, "false": 0.4})
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=vlm, confidence=0.5
        )
        state = se(single_image)
        assert all(v is True for v in state.values())

        # With higher confidence, all should be False
        state = se(single_image, confidence=0.7)
        assert all(v is False for v in state.values())

    def test_confidence_zero_works(self, single_image, blocksworld_domain, blocksworld_problem):
        vlm = FakeVLM(token_probs={"true": 0.1, "false": 0.9})
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=vlm, confidence=0.5
        )
        # confidence=0.0 should make everything True (any prob >= 0.0)
        state = se(single_image, confidence=0.0)
        assert all(v is True for v in state.values())


class TestEstimateProbabilities:
    def test_returns_float_dict(self, fake_vlm, single_image, blocksworld_domain, blocksworld_problem):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        probs = se.estimate_probabilities(single_image)
        assert isinstance(probs, dict)
        assert all(isinstance(v, float) for v in probs.values())

    def test_probabilities_in_range(self, fake_vlm, single_image, blocksworld_domain, blocksworld_problem):
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=fake_vlm
        )
        probs = se.estimate_probabilities(single_image)
        assert all(0.0 <= v <= 1.0 for v in probs.values())

    def test_groups_and_normalizes_tokens(self, single_image, blocksworld_domain, blocksworld_problem):
        vlm = FakeVLM(token_probs={
            "true": 0.3, "True": 0.2, "TRUE": 0.1,
            "false": 0.15, "False": 0.1, "FALSE": 0.05,
            "other": 0.1,
        })
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem, vlm=vlm
        )
        probs = se.estimate_probabilities(single_image)
        # true_sum = 0.6, false_sum = 0.3, normalized true = 0.6/0.9 = 0.667
        for pred, prob in probs.items():
            assert abs(prob - 0.6 / 0.9) < 0.01


class TestEstimateRaw:
    def test_returns_vlm_output_dict(self, fake_vlm, single_image, blocksworld_domain, blocksworld_problem):
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
            def query(self, images, prompt, system_prompt=None):
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
            blocksworld_domain, blocksworld_problem,
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
    def test_text_match_probability(self, single_image, blocksworld_domain, blocksworld_problem):
        vlm = FakeVLM(token_probs={"true": 0.5, "false": 0.5}, text="true")
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem,
            vlm=vlm,
            probability_method="text_match",
        )
        probs = se.estimate_probabilities(single_image)
        assert all(v == 1.0 for v in probs.values())

    def test_text_match_false(self, single_image, blocksworld_domain, blocksworld_problem):
        vlm = FakeVLM(token_probs={"true": 0.5, "false": 0.5}, text="false")
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem,
            vlm=vlm,
            probability_method="text_match",
        )
        probs = se.estimate_probabilities(single_image)
        assert all(v == 0.0 for v in probs.values())


class TestUserPromptTemplate:
    def test_custom_template(self, single_image, blocksworld_domain, blocksworld_problem):
        vlm = FakeVLM()
        se = SemanticStateEstimator(
            blocksworld_domain, blocksworld_problem,
            vlm=vlm,
            user_prompt_template="Look carefully. {query} Answer yes or no.",
        )
        se.estimate_probabilities(single_image)

        # Verify the VLM received the formatted prompts
        for prompt in vlm.received_prompts:
            assert prompt.startswith("Look carefully.")
            assert prompt.endswith("Answer yes or no.")


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
        se = SemanticStateEstimator(
            blocksworld_domain, problem_2obj, vlm=fake_vlm
        )
        state_2 = se(single_image)
        assert len(state_2) == 6  # 2 blocks: 4 on + 2 clear

        se.swap_problem(blocksworld_domain, problem_3obj)
        state_3 = se(single_image)
        assert len(state_3) == 12  # 3 blocks: 9 on + 3 clear
