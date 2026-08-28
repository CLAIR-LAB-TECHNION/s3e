"""Tests for the SemanticStateEstimator facade."""

import pytest

from s3e import (
    BinaryAnswers,
    IdentityTranslator,
    PredictionSet,
    PrewrittenTranslator,
    SemanticStateEstimator,
    TemplateTranslator,
)

from conftest import BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM, make_blank_image
from fakes import FakeVLM


TEMPLATES = {"on": "Is {0} on {1}?", "clear": "Is {0} clear?"}

BLOCKSWORLD_PROBLEM_3 = """
(define (problem bw-3)
  (:domain blocksworld)
  (:objects a b c - block)
  (:init (on a b) (clear a) (clear c))
  (:goal (on b a))
)
"""


@pytest.fixture
def images():
    return [make_blank_image()]


def make_estimator(fake=None, **kwargs):
    return SemanticStateEstimator.from_pddl(
        BLOCKSWORLD_DOMAIN,
        BLOCKSWORLD_PROBLEM,
        vlm=fake or FakeVLM(),
        translator=TemplateTranslator(TEMPLATES),
        **kwargs,
    )


class TestConstructionFromPredicates:
    def test_predicates_without_pddl(self, images):
        estimator = SemanticStateEstimator(
            predicates=["on(a,b)", "clear(a)"],
            vlm=FakeVLM(),
            translator=TemplateTranslator(TEMPLATES),
        )
        results = estimator.estimate(images)
        assert set(results) == {"on(a,b)", "clear(a)"}

    def test_pddl_extras_raise_without_pddl(self, images):
        estimator = SemanticStateEstimator(
            predicates=["on(a,b)"], vlm=FakeVLM(),
            translator=TemplateTranslator(TEMPLATES),
        )
        with pytest.raises(ValueError, match="from_pddl"):
            estimator.set_problem(BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM)
        with pytest.raises(ValueError, match="from_pddl"):
            estimator.to_up_state({"on(a,b)": True})


class TestConstructionFromPddl:
    def test_grounds_all_predicates(self):
        estimator = make_estimator()
        assert any(p.startswith("on(") for p in estimator.predicates)
        assert any(p.startswith("clear(") for p in estimator.predicates)

    def test_queries_are_translated(self):
        estimator = make_estimator()
        grounded_on = next(p for p in estimator.predicates if p.startswith("on("))
        assert estimator.queries[grounded_on].startswith("Is ")

    def test_fingerprint_available(self):
        assert len(make_estimator().domain_fingerprint) == 64


class TestEstimate:
    def test_returns_prediction_set_keyed_by_predicate(self, images):
        results = make_estimator().estimate(images)
        assert isinstance(results, PredictionSet)
        assert set(results) == set(make_estimator().predicates)

    def test_predicate_subset_queries_only_subset(self, images):
        fake = FakeVLM()
        estimator = make_estimator(fake)
        subset = estimator.predicates[:2]
        results = estimator.estimate(images, predicates=subset)
        assert set(results) == set(subset)
        prompts = [p for call in fake.calls for p in call["prompts"]]
        assert len(prompts) == 2

    def test_unknown_predicate_rejected(self, images):
        with pytest.raises(ValueError, match="nope"):
            make_estimator().estimate(images, predicates=["nope(x)"])

    def test_call_thresholds(self, images):
        fake = FakeVLM({"yes": 0.9, "no": 0.05})
        state = make_estimator(fake)(images)
        assert all(value is True for value in state.values())

    def test_probability_stays_in_range_when_masses_exceed_one(self, images):
        """Replaces the monolith's TestProbabilityClipping: the normalized
        true/false ratio is in range by construction, without clipping."""
        results = make_estimator(FakeVLM({"yes": 0.8, "no": 0.7})).estimate(images)
        assert all(0.0 <= p.probability <= 1.0 for p in results.values())

    def test_confidence_argument_overrides_default(self, images):
        estimator = make_estimator(FakeVLM({"yes": 0.6, "no": 0.4}))
        assert all(v is True for v in estimator(images).values())
        assert all(v is None for v in estimator(images, confidence=0.9).values())

    def test_calibrator_applied_when_passed(self, images):
        class HalfCalibrator:
            def apply(self, results):
                return PredictionSet(
                    {k: p.with_probability(0.5) for k, p in results.items()}
                )

        results = make_estimator().estimate(images, calibrator=HalfCalibrator())
        assert all(p.probability == 0.5 for p in results.values())


class TestSetProblem:
    def test_regrounds_without_touching_backend(self, images):
        fake = FakeVLM()
        estimator = make_estimator(fake)
        backend_before = estimator.engine.backend
        estimator.set_problem(BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM)
        assert estimator.engine.backend is backend_before
        assert estimator.predicates  # re-grounded

    def test_new_problem_changes_predicate_set(self, images):
        """Ported from the monolith's TestSwapProblem."""
        estimator = make_estimator()
        assert len(estimator(images)) == 6  # 2 blocks: 4 on + 2 clear

        estimator.set_problem(BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM_3)
        assert estimator.problem_pddl == BLOCKSWORLD_PROBLEM_3
        assert len(estimator(images)) == 12  # 3 blocks: 9 on + 3 clear
        assert estimator.queries["on(a,c)"] == "Is a on c?"


class TestSharedBackend:
    def test_two_estimators_share_one_backend(self):
        fake = FakeVLM()
        a = make_estimator(fake)
        b = make_estimator(fake)
        assert a.engine.backend is b.engine.backend


class TestNullTokens:
    def test_null_dominated_predicate_is_none_in_state(self, images):
        fake = FakeVLM({"yes": 0.1, "no": 0.05, "unknown": 0.7})
        estimator = make_estimator(fake, null_tokens=["unknown"])
        state = estimator(images)
        assert all(value is None for value in state.values())


class TestCalibrationMeta:
    def test_meta_includes_labels_and_fingerprint(self):
        meta = make_estimator().calibration_meta()
        assert meta["true_label"] == "yes"
        assert meta["scoring"] == "logprobs"
        assert len(meta["domain_fingerprint"]) == 64


class TestAnswerDefaults:
    """Ported from the monolith's TestConstruction token defaults."""

    def test_identity_translation_answers_true_false(self):
        estimator = SemanticStateEstimator.from_pddl(
            BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM, vlm=FakeVLM()
        )
        answers = estimator.engine.answers
        assert isinstance(answers, BinaryAnswers)
        assert (answers.true_label, answers.false_label) == ("true", "false")
        assert isinstance(estimator.translator, IdentityTranslator)
        assert estimator.queries["on(a,b)"] == "on(a,b)"

    def test_translated_queries_answer_yes_no(self):
        answers = make_estimator().engine.answers
        assert (answers.true_label, answers.false_label) == ("yes", "no")

    def test_custom_tokens_override_defaults(self):
        estimator = make_estimator(
            true_tokens=["correct"], false_tokens=["incorrect"]
        )
        assert estimator.engine.answers.interest_tokens == ["correct", "incorrect"]

    def test_explicit_answers_win_over_token_overrides(self):
        space = BinaryAnswers("affirmative", "negative")
        estimator = make_estimator(answers=space, true_tokens=["correct"])
        assert estimator.engine.answers is space


class TestSystemPrompt:
    """Ported from the monolith's prompt-selection behavior."""

    def test_identity_translation_gets_domain_aware_prompt(self):
        estimator = SemanticStateEstimator.from_pddl(
            BLOCKSWORLD_DOMAIN, BLOCKSWORLD_PROBLEM, vlm=FakeVLM()
        )
        prompt = estimator.engine.system_prompt
        assert "PDDL domain" in prompt
        assert "block" in prompt

    def test_translated_queries_get_yes_no_prompt(self):
        assert "YES or NO" in make_estimator().engine.system_prompt

    def test_explicit_system_prompt_wins(self):
        estimator = make_estimator(system_prompt="Custom.")
        assert estimator.engine.system_prompt == "Custom."

    def test_additional_instructions_appended(self):
        estimator = make_estimator(additional_instructions="Be terse.")
        assert estimator.engine.system_prompt.endswith("Be terse.")
        assert "YES or NO" in estimator.engine.system_prompt


class TestPromptTemplate:
    """Ported from the monolith's TestUserPromptTemplate."""

    def test_template_wraps_every_query(self, images):
        fake = FakeVLM()
        estimator = make_estimator(
            fake, prompt_template="Look carefully. {query} Answer yes or no."
        )
        estimator.estimate(images)
        prompts = [p for call in fake.calls for p in call["prompts"]]
        assert prompts
        for prompt in prompts:
            assert prompt.startswith("Look carefully.")
            assert prompt.endswith("Answer yes or no.")


class TestInferenceKwargs:
    """Ported from the monolith's inference-kwargs forwarding test."""

    def test_constructor_kwargs_forwarded(self, images):
        fake = FakeVLM()
        make_estimator(fake, inference_kwargs={"temperature": 0.2}).estimate(images)
        assert all(
            call["inference_kwargs"]["temperature"] == 0.2 for call in fake.calls
        )

    def test_per_call_kwargs_merged(self, images):
        fake = FakeVLM()
        estimator = make_estimator(fake, inference_kwargs={"temperature": 0.2})
        estimator.estimate(images, inference_kwargs={"max_new_tokens": 5})
        for call in fake.calls:
            assert call["inference_kwargs"] == {"temperature": 0.2, "max_new_tokens": 5}


class TestTextMatchScoring:
    """Ported from the monolith's TestTextMatchMode (end to end)."""

    def test_generated_text_decides_the_state(self, images):
        fake = FakeVLM(text="Yes.")
        estimator = make_estimator(fake, scoring="text_match")
        results = estimator.estimate(images)
        assert all(p.probability == pytest.approx(1.0) for p in results.values())
        assert all(v is True for v in estimator(images).values())

    def test_text_match_asks_the_backend_to_generate(self, images):
        fake = FakeVLM(text="No.")
        estimator = make_estimator(fake, scoring="text_match")
        results = estimator.estimate(images)
        assert all(p.probability == pytest.approx(0.0) for p in results.values())
        assert all(call["generate"] is True for call in fake.calls)
        assert all(call["interest_tokens"] is None for call in fake.calls)

    def test_logprobs_scoring_does_not_generate(self, images):
        fake = FakeVLM()
        make_estimator(fake).estimate(images)
        assert all(call["generate"] is False for call in fake.calls)
        assert all(call["interest_tokens"] for call in fake.calls)


class ImageAwareVLM(FakeVLM):
    """Fake whose token masses depend on which image it is shown."""

    def __init__(self, probs_by_image_id):
        super().__init__()
        self.probs_by_image_id = probs_by_image_id

    def query(self, images, prompt, system_prompt=None, generate=False,
              interest_tokens=None, **inference_kwargs):
        self.token_probs = dict(self.probs_by_image_id[id(images[0])])
        return super().query(
            images, prompt, system_prompt, generate,
            interest_tokens=interest_tokens, **inference_kwargs,
        )


class TestEstimateAveraged:
    """Ported from the monolith's average multi-image strategy tests."""

    def test_masses_are_averaged_across_scenes(self):
        one, two = make_blank_image(), make_blank_image()
        fake = ImageAwareVLM(
            {
                id(one): {"yes": 0.8, "no": 0.2},
                id(two): {"yes": 0.4, "no": 0.6},
            }
        )
        estimator = make_estimator(fake)
        results = estimator.estimate_averaged([[one], [two]])
        assert all(p.probability == pytest.approx(0.6) for p in results.values())

    def test_null_mass_averaged_and_can_dominate(self):
        one, two = make_blank_image(), make_blank_image()
        fake = ImageAwareVLM(
            {
                id(one): {"yes": 0.2, "no": 0.1, "null": 0.7},
                id(two): {"yes": 0.3, "no": 0.2, "null": 0.1},
            }
        )
        estimator = make_estimator(fake, null_tokens=["null"])
        results = estimator.estimate_averaged([[one], [two]], predicates=["on(a,b)"])
        prediction = results["on(a,b)"]
        assert prediction.masses["yes"] == pytest.approx(0.25)
        assert prediction.masses["no"] == pytest.approx(0.15)
        assert prediction.null_mass == pytest.approx(0.4)
        assert prediction.null_dominated is True
        assert results.to_state() == {"on(a,b)": None}


class TestDuplicateQueries:
    def test_predicates_sharing_a_query_are_asked_once(self, images):
        fake = FakeVLM()
        estimator = SemanticStateEstimator(
            predicates=["on(a,b)", "clear(a)"],
            vlm=fake,
            translator=PrewrittenTranslator({"on(a,b)": "q", "clear(a)": "q"}),
        )
        results = estimator.estimate(images)
        prompts = [p for call in fake.calls for p in call["prompts"]]
        assert prompts == ["q"]
        assert set(results) == {"on(a,b)", "clear(a)"}
        assert results["on(a,b)"] is results["clear(a)"]


class TestCollectIntegration:
    def test_collect_produces_scored_samples(self, images):
        pytest.importorskip("sklearn")
        from s3e.calibration import CalibrationExample, CalibrationSet, PlattCalibrator

        estimator = make_estimator(FakeVLM({"yes": 0.8, "no": 0.1}))
        target = {p: (i % 2 == 0) for i, p in enumerate(estimator.predicates)}
        data = CalibrationSet.collect(
            estimator, [CalibrationExample(images=images, state_dict=target)]
        )
        assert len(data.samples) == len(target)
        assert data.meta["true_label"] == "yes"


@pytest.mark.slow
class TestRealModelIntegration:
    """Ported from the monolith's slow end-to-end integration class."""

    TINY_VLM_ID = "katuni4ka/tiny-random-llava"

    def test_end_to_end_with_hf_vlm(self, images):
        estimator = SemanticStateEstimator.from_pddl(
            BLOCKSWORLD_DOMAIN,
            BLOCKSWORLD_PROBLEM,
            vlm=self.TINY_VLM_ID,
            vlm_kwargs={"device_map": "cpu"},
        )
        results = estimator.estimate(images, keep_raw=True)
        assert len(results) == 6  # 2 blocks: 4 on + 2 clear
        assert all(0.0 <= p.probability <= 1.0 for p in results.values())
        assert all(p.raw.token_probs for p in results.values())

        state = estimator(images)
        assert all(isinstance(value, bool) for value in state.values())
