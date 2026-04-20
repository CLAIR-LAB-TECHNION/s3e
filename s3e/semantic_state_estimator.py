"""Semantic state estimation using vision-language models.

This module provides the main :class:`SemanticStateEstimator` class that
combines a VLM backend with a query translator to estimate environment
state from images. The result is a dictionary of PDDL predicate truth
values (or probabilities) compatible with planning systems.
"""

import math
import re
from typing import Union

import numpy as np
from PIL.Image import Image

from .calibration import (
    CalibrationExample,
    GLOBAL_CALIBRATION_KEY,
    PlattScalingProfile,
    apply_platt_scaling,
    compute_domain_fingerprint,
    fit_platt_parameters,
    grouped_log_odds,
)
from .constants import (
    OPENAI_MODEL_IDENTIFIER,
    SYSTEM_PROMPT_NO_TRANSLATION,
    SYSTEM_PROMPT_WITH_TRANSLATION,
    SYSTEM_PROMPT_ADDITIONAL_INSTRUCTIONS,
    TRUE_TOKENS_NO_TRANSLATION,
    FALSE_TOKENS_NO_TRANSLATION,
    TRUE_TOKENS_WITH_TRANSLATION,
    FALSE_TOKENS_WITH_TRANSLATION,
)
from .pddl.up_utils import (
    create_up_problem,
    get_object_names_dict,
    get_all_grounded_predicates_for_objects,
    get_pddl_strings,
    get_lifted_predicate_key,
)
from .state_estimator import ProbabilisticStateEstimator
from .translation.identity import IdentityTranslator
from .translation.translator import QueryTranslator
from .vlm.backend import VLMBackend, VLMOutput


class SemanticStateEstimator(ProbabilisticStateEstimator):
    """Vision-language model based state estimator.

    Combines a :class:`VLMBackend` with a :class:`QueryTranslator` to
    estimate the boolean truth values of PDDL predicates from images.

    Args:
        domain: PDDL domain as a file path or string.
        problem: PDDL problem as a file path or string.
        vlm: A :class:`VLMBackend` instance, or a model ID string.
        query_translator: Strategy for converting predicates to queries.
            Defaults to :class:`IdentityTranslator`.
        system_prompt: System prompt for the VLM. If ``None``, auto-selected.
        user_prompt_template: Format string for each query. Must contain ``{query}``.
        true_tokens: Token strings representing "true". If ``None``, auto-selected.
        false_tokens: Token strings representing "false". If ``None``, auto-selected.
        confidence: Probability threshold for boolean conversion.
        multi_image_strategy: ``"single"`` or ``"average"``.
        probability_method: ``"logprobs"`` or ``"text_match"``.
        batch_size: Number of queries per VLM batch call.
        additional_instructions: Extra text appended to the system prompt.
        vlm_kwargs: Extra kwargs for VLM construction (only when vlm is a string).
    """

    def __init__(
        self,
        domain: str,
        problem: str,
        vlm: Union[VLMBackend, str],
        query_translator: QueryTranslator | None = None,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        true_tokens: list[str] | None = None,
        false_tokens: list[str] | None = None,
        confidence: float = 0.5,
        multi_image_strategy: str = "single",
        probability_method: str = "logprobs",
        batch_size: int = 8,
        additional_instructions: str | None = None,
        vlm_kwargs: dict | None = None,
        inference_kwargs: dict | None = None,
    ):
        super().__init__(domain, problem, confidence)

        # --- VLM backend ---
        if isinstance(vlm, str):
            self.vlm = self._build_vlm_from_string(vlm, vlm_kwargs or {})
        else:
            self.vlm = vlm
        self.inference_kwargs = inference_kwargs or {}

        # --- Query translator ---
        self.query_translator = query_translator or IdentityTranslator()
        has_nl_translator = not isinstance(self.query_translator, IdentityTranslator)

        # --- Token groups ---
        if true_tokens is not None:
            self.true_tokens = true_tokens
        else:
            self.true_tokens = (
                list(TRUE_TOKENS_WITH_TRANSLATION)
                if has_nl_translator
                else list(TRUE_TOKENS_NO_TRANSLATION)
            )

        if false_tokens is not None:
            self.false_tokens = false_tokens
        else:
            self.false_tokens = (
                list(FALSE_TOKENS_WITH_TRANSLATION)
                if has_nl_translator
                else list(FALSE_TOKENS_NO_TRANSLATION)
            )

        # --- System prompt ---
        if system_prompt is not None:
            self.system_prompt = system_prompt
        elif has_nl_translator:
            self.system_prompt = SYSTEM_PROMPT_WITH_TRANSLATION
        else:
            objects = get_object_names_dict(self.up_problem)
            objects_str = "\n".join(
                f"{key} type: {list(map(str, value))}"
                for key, value in objects.items()
            )
            domain_str, _ = get_pddl_strings(self.up_problem)
            self.system_prompt = SYSTEM_PROMPT_NO_TRANSLATION.format(
                domain=domain_str, objects=objects_str
            )

        if additional_instructions:
            self.system_prompt += SYSTEM_PROMPT_ADDITIONAL_INSTRUCTIONS.format(
                additional_instructions=additional_instructions
            )

        # --- Other config ---
        self.user_prompt_template = user_prompt_template or "{query}"
        self.multi_image_strategy = multi_image_strategy
        self.probability_method = probability_method
        self._generate_mode = probability_method == "text_match"
        self.batch_size = batch_size

        # --- Build queries ---
        self._domain = domain
        self._problem = problem
        self._platt_scaling_profile: PlattScalingProfile | None = None
        self._domain_fingerprint = self._current_domain_fingerprint()
        self._build_queries()

    def _build_queries(self) -> None:
        """Translate all grounded predicates via the query translator."""
        predicates = get_all_grounded_predicates_for_objects(self.up_problem)
        self.queries_dict = self.query_translator.translate(
            predicates, self._domain, self._problem
        )

    def _current_domain_fingerprint(self) -> str:
        """Fingerprint the canonical domain for the current parsed UP problem."""
        domain_str, _ = get_pddl_strings(self.up_problem)
        canonical_domain_str = re.sub(
            r"\(domain\s+[^\s)]+\)",
            "(domain __canonical__)",
            domain_str,
            count=1,
        )
        return compute_domain_fingerprint(canonical_domain_str)

    @staticmethod
    def _build_vlm_from_string(vlm_id: str, vlm_kwargs: dict) -> VLMBackend:
        """Construct a VLM backend from a model ID string."""
        if vlm_id.startswith(OPENAI_MODEL_IDENTIFIER):
            from .vlm.openai import OpenAIVLM
            return OpenAIVLM(vlm_id, **vlm_kwargs)
        else:
            from .vlm.huggingface import HuggingFaceVLM
            return HuggingFaceVLM(vlm_id, **vlm_kwargs)

    def swap_problem(self, domain: str, problem: str) -> None:
        """Update domain/problem and re-translate predicates."""
        previous_domain_fingerprint = self._domain_fingerprint
        super().swap_problem(domain, problem)
        self._domain = domain
        self._problem = problem
        self._domain_fingerprint = self._current_domain_fingerprint()
        if self._domain_fingerprint != previous_domain_fingerprint:
            self._platt_scaling_profile = None
        self._build_queries()

    def __call__(
        self,
        images: list[Image],
        confidence: float | None = None,
        calibrated: bool | None = None,
    ) -> dict[str, bool]:
        probs = self.estimate_probabilities(images, calibrated=calibrated)
        threshold = confidence if confidence is not None else self.confidence
        return {pred: bool(prob >= threshold) for pred, prob in probs.items()}

    def estimate_probabilities(
        self,
        images: list[Image],
        calibrated: bool | None = None,
    ) -> dict[str, float]:
        """Estimate P(true) for each grounded predicate."""
        use_calibration = self._resolve_calibrated_flag(calibrated)
        if self.multi_image_strategy == "average":
            if not use_calibration:
                details = self._estimate_average(images)
                return {
                    pred: probability for pred, (probability, _) in details.items()
                }
            per_image_calibrated = [
                self._calibrate_prediction_details(self._estimate_single([img]))
                for img in images
            ]
            predicates = list(per_image_calibrated[0].keys())
            return {
                pred: float(np.mean([details[pred] for details in per_image_calibrated]))
                for pred in predicates
            }

        details = self._estimate_single(images)
        if not use_calibration:
            return {pred: probability for pred, (probability, _) in details.items()}

        return self._calibrate_prediction_details(details)

    def estimate_raw(self, images: list[Image]) -> dict[str, VLMOutput]:
        """Get the full VLMOutput for each grounded predicate."""
        prompts = [
            self.user_prompt_template.format(query=query)
            for query in self.queries_dict.values()
        ]
        predicates = list(self.queries_dict.keys())

        results: dict[str, VLMOutput] = {}
        num_batches = math.ceil(len(prompts) / self.batch_size)
        for i in range(num_batches):
            batch_prompts = prompts[i * self.batch_size : (i + 1) * self.batch_size]
            batch_preds = predicates[i * self.batch_size : (i + 1) * self.batch_size]
            outputs = self.vlm.query_batch(
                images,
                batch_prompts,
                system_prompt=self.system_prompt,
                generate=self._generate_mode,
                **self.inference_kwargs
            )
            for pred, output in zip(batch_preds, outputs):
                results[pred] = output

        return results

    def fit_platt_scaling(
        self,
        examples: list[CalibrationExample],
        scope: str = "global",
    ) -> None:
        if self.probability_method != "logprobs":
            raise ValueError(
                "Platt scaling is only supported for probability_method='logprobs'."
            )
        if not examples:
            raise ValueError("Expected at least one calibration example.")
        if scope not in {"global", "lifted"}:
            raise ValueError(f"Unsupported Platt scaling scope: {scope}")

        grouped_scores: dict[str, list[float]] = {}
        grouped_labels: dict[str, list[bool]] = {}

        for example in examples:
            example_problem = example.problem or self._problem
            example_up_problem = create_up_problem(self._domain, example_problem)
            per_sample_details = self._estimate_calibration_example(example)
            for details in per_sample_details:
                for predicate, (_, score) in details.items():
                    if predicate not in example.state_dict:
                        raise ValueError(
                            f"Missing calibration label for predicate {predicate}."
                        )
                    if scope == "global":
                        key = GLOBAL_CALIBRATION_KEY
                    else:
                        key = get_lifted_predicate_key(example_up_problem, predicate)
                    grouped_scores.setdefault(key, []).append(score)
                    grouped_labels.setdefault(key, []).append(example.state_dict[predicate])

        params_by_group = {
            key: fit_platt_parameters(scores, grouped_labels[key])
            for key, scores in grouped_scores.items()
        }
        self._platt_scaling_profile = PlattScalingProfile(
            scope=scope,
            probability_method=self.probability_method,
            true_tokens=list(self.true_tokens),
            false_tokens=list(self.false_tokens),
            domain_fingerprint=self._domain_fingerprint,
            score_kind="grouped_log_odds",
            groups=params_by_group,
        )

    def save_platt_scaling(self, path: str) -> None:
        if self._platt_scaling_profile is None:
            raise ValueError("No Platt scaling profile is loaded.")
        self._platt_scaling_profile.save(path)

    def load_platt_scaling(self, path: str) -> None:
        profile = PlattScalingProfile.load(path)
        self._validate_platt_profile(profile)
        self._platt_scaling_profile = profile

    def clear_platt_scaling(self) -> None:
        self._platt_scaling_profile = None

    def _resolve_calibrated_flag(self, calibrated: bool | None) -> bool:
        if calibrated is False:
            return False
        if self._platt_scaling_profile is not None and self.probability_method != "logprobs":
            raise ValueError(
                "Platt scaling calibration is only supported when probability_method='logprobs'."
            )
        if self._platt_scaling_profile is None:
            if calibrated is True:
                raise ValueError(
                    "No Platt scaling profile is loaded. "
                    "Call fit_platt_scaling(...) or load_platt_scaling(...)."
                )
            return False
        return True

    def _apply_platt_profile(self, predicate: str, score: float) -> float:
        assert self._platt_scaling_profile is not None
        if self._platt_scaling_profile.scope == "global":
            params = self._platt_scaling_profile.groups[GLOBAL_CALIBRATION_KEY]
        else:
            key = get_lifted_predicate_key(self.up_problem, predicate)
            if key not in self._platt_scaling_profile.groups:
                raise ValueError(
                    f"No Platt scaling parameters available for lifted fluent '{key}'."
                )
            params = self._platt_scaling_profile.groups[key]
        return apply_platt_scaling(score, params)

    def _validate_platt_profile(self, profile: PlattScalingProfile) -> None:
        if (
            profile.probability_method != "logprobs"
            or self.probability_method != "logprobs"
        ):
            raise ValueError(
                "Platt scaling profiles are only compatible with logprobs mode."
            )
        if profile.true_tokens != list(self.true_tokens) or profile.false_tokens != list(
            self.false_tokens
        ):
            raise ValueError(
                "Loaded Platt scaling profile does not match the estimator token groups."
            )
        if profile.domain_fingerprint != self._domain_fingerprint:
            raise ValueError(
                "Loaded Platt scaling profile was fit for a different domain."
            )
        if profile.scope not in {"global", "lifted"}:
            raise ValueError(f"Unsupported Platt scaling scope: {profile.scope}")
        if profile.score_kind != "grouped_log_odds":
            raise ValueError(
                f"Loaded Platt scaling profile has unsupported score_kind: {profile.score_kind}."
            )
        if profile.scope == "global":
            if GLOBAL_CALIBRATION_KEY not in profile.groups:
                raise ValueError(
                    "Loaded global Platt scaling profile is missing global parameters."
                )
            return

        required_groups = {fluent.name for fluent in self.up_problem.fluents}
        missing_groups = sorted(required_groups - set(profile.groups))
        if missing_groups:
            missing = ", ".join(missing_groups)
            raise ValueError(
                "Loaded lifted Platt scaling profile is missing parameters "
                f"for lifted fluent(s): {missing}."
            )

    def _calibrate_prediction_details(
        self, details: dict[str, tuple[float, float]]
    ) -> dict[str, float]:
        return {
            pred: self._apply_platt_profile(pred, score)
            for pred, (_, score) in details.items()
        }

    def _estimate_calibration_example(
        self,
        example: CalibrationExample,
    ) -> list[dict[str, tuple[float, float]]]:
        original_problem = self._problem
        try:
            if example.problem is not None:
                self.swap_problem(self._domain, example.problem)
            if self.multi_image_strategy == "average":
                return [self._estimate_single([image]) for image in example.images]
            return [self._estimate_single(example.images)]
        finally:
            if example.problem is not None:
                self.swap_problem(self._domain, original_problem)

    def _estimate_single(self, images: list[Image]) -> dict[str, tuple[float, float]]:
        """Estimate probabilities with all images in a single pass."""
        raw = self.estimate_raw(images)
        return {
            pred: self._extract_probability(output)
            for pred, output in raw.items()
        }

    def _estimate_average(self, images: list[Image]) -> dict[str, tuple[float, float]]:
        """Estimate probabilities by averaging across individual images."""
        per_image_details = [self._estimate_single([img]) for img in images]
        predicates = list(per_image_details[0].keys())
        return {
            pred: (
                float(np.mean([details[pred][0] for details in per_image_details])),
                float(np.mean([details[pred][1] for details in per_image_details])),
            )
            for pred in predicates
        }

    def _extract_probability(self, output: VLMOutput) -> tuple[float, float]:
        """Extract P(true) from a VLMOutput."""
        if self.probability_method == "text_match":
            probability = self._extract_text_match(output)
            return probability, 0.0
        return self._extract_logprobs(output)

    def _extract_logprobs(self, output: VLMOutput) -> tuple[float, float]:
        """Extract P(true) by grouping and normalizing token probabilities."""
        true_sum = sum(
            output.token_probs.get(tok, 0.0) for tok in self.true_tokens
        )
        false_sum = sum(
            output.token_probs.get(tok, 0.0) for tok in self.false_tokens
        )
        total = true_sum + false_sum
        if total == 0:
            return 0.5, 0.0
        probability = float(np.clip(true_sum / total, 0.0, 1.0))
        score = grouped_log_odds(output.token_probs, self.true_tokens, self.false_tokens)
        return probability, score

    def _extract_text_match(self, output: VLMOutput) -> float:
        """Extract P(true) by checking if generated text matches true tokens."""
        if output.text is None:
            return 0.5
        text = output.text.strip().lower()
        true_lower = {t.lower() for t in self.true_tokens}
        if text in true_lower:
            return 1.0
        return 0.0
