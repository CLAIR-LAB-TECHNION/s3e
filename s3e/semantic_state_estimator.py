"""Semantic state estimation using vision-language models.

This module provides the main :class:`SemanticStateEstimator` class that
combines a VLM backend with a query translator to estimate environment
state from images. The result is a dictionary of PDDL predicate truth
values (or probabilities) compatible with planning systems.
"""

import math
import re
from dataclasses import dataclass, replace
from typing import Union

import numpy as np
from PIL.Image import Image
from tqdm.auto import tqdm

from .calibration import (
    CalibrationExample,
    GLOBAL_CALIBRATION_KEY,
    PlattParameters,
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


@dataclass(frozen=True)
class PredicatePredictionDetails:
    probability: float
    raw_probability: float
    score: float
    raw_true_mass: float
    raw_false_mass: float
    raw_none_mass: float
    none_is_max_raw: bool


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
        null_tokens: list[str] | None = None,
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

        self.null_tokens = list(null_tokens or [])
        self._validate_token_groups()

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

    def _validate_token_groups(self) -> None:
        overlaps = [
            ("true_tokens and false_tokens", set(self.true_tokens) & set(self.false_tokens)),
            ("null_tokens and true_tokens", set(self.null_tokens) & set(self.true_tokens)),
            ("null_tokens and false_tokens", set(self.null_tokens) & set(self.false_tokens)),
        ]
        for label, values in overlaps:
            if values:
                joined = ", ".join(sorted(values))
                raise ValueError(
                    f"Token groups must be disjoint; overlap between {label}: {joined}"
                )

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
        predicates: list[str] | None = None,
    ) -> dict[str, bool | None]:
        details = self.estimate_prediction_details(
            images,
            calibrated=calibrated,
            predicates=predicates,
        )
        threshold = confidence if confidence is not None else self.confidence
        return {
            pred: None if detail.none_is_max_raw else bool(detail.probability >= threshold)
            for pred, detail in details.items()
        }

    def estimate_prediction_details(
        self,
        images: list[Image],
        calibrated: bool | None = None,
        predicates: list[str] | None = None,
    ) -> dict[str, PredicatePredictionDetails]:
        if self.multi_image_strategy != "average":
            raw = self.estimate_raw(images, predicates=predicates)
            return self.prediction_details_from_raw(raw, calibrated=calibrated)

        per_image = [
            self.prediction_details_from_raw(
                self.estimate_raw([img], predicates=predicates),
                calibrated=calibrated,
            )
            for img in images
        ]
        return self._average_prediction_details(per_image)

    def _average_prediction_details(
        self,
        per_image_details: list[dict[str, PredicatePredictionDetails]],
    ) -> dict[str, PredicatePredictionDetails]:
        predicates = list(per_image_details[0].keys())
        result: dict[str, PredicatePredictionDetails] = {}
        for pred in predicates:
            items = [details[pred] for details in per_image_details]
            raw_true_mass = float(np.mean([d.raw_true_mass for d in items]))
            raw_false_mass = float(np.mean([d.raw_false_mass for d in items]))
            raw_none_mass = float(np.mean([d.raw_none_mass for d in items]))
            result[pred] = PredicatePredictionDetails(
                probability=float(np.mean([d.probability for d in items])),
                raw_probability=float(np.mean([d.raw_probability for d in items])),
                score=float(np.mean([d.score for d in items])),
                raw_true_mass=raw_true_mass,
                raw_false_mass=raw_false_mass,
                raw_none_mass=raw_none_mass,
                none_is_max_raw=raw_none_mass > raw_true_mass and raw_none_mass > raw_false_mass,
            )
        return result

    def estimate_probabilities(
        self,
        images: list[Image],
        calibrated: bool | None = None,
        predicates: list[str] | None = None,
    ) -> dict[str, float]:
        """Estimate P(true) for each grounded predicate.

        Args:
            images: List of PIL images representing the current state.
            calibrated: Whether to apply Platt scaling.
            predicates: Optional list of grounded predicate strings to
                query.  When ``None`` (default), all predicates are
                queried.  Unknown predicates raise :class:`ValueError`.
        """
        details = self.estimate_prediction_details(
            images, calibrated=calibrated, predicates=predicates
        )
        return {pred: detail.probability for pred, detail in details.items()}

    def estimate_raw(
        self,
        images: list[Image],
        predicates: list[str] | None = None,
    ) -> dict[str, VLMOutput]:
        """Get the full VLMOutput for each grounded predicate.

        Args:
            images: List of PIL images representing the current state.
            predicates: Optional list of grounded predicate strings to
                query.  When ``None`` (default), all predicates are
                queried.  Unknown predicates raise :class:`ValueError`.
        """
        queries = self._resolve_queries(predicates)
        prompts = [
            self.user_prompt_template.format(query=query)
            for query in queries.values()
        ]
        predicates = list(queries.keys())

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
        progress_bar: bool = False,
        pass_through_single_class: bool = False,
    ) -> None:
        """Fit a Platt scaling calibration profile from labeled examples.

        Each example pairs a set of images with a ground-truth boolean
        state dict.  The estimator queries its VLM for every labeled
        predicate, collects the resulting log-odds scores, and fits a
        logistic regression (Platt scaling) to map raw scores to
        calibrated probabilities.

        Args:
            examples: Labeled calibration examples.  Each
                :class:`CalibrationExample` contains images, a
                ``state_dict`` mapping grounded predicates to boolean
                labels, and an optional problem string.
            scope: Grouping strategy for the Platt parameters.
                ``"global"`` fits a single pair of parameters shared
                across all predicates.  ``"lifted"`` fits separate
                parameters per lifted predicate (fluent name).
            progress_bar: Whether to display a ``tqdm`` progress bar
                while querying the VLM.
            pass_through_single_class: When ``False`` (default), a
                :class:`ValueError` is raised before any VLM
                predictions if a label group contains only positive or
                only negative examples.  When ``True``, single-class
                groups are assigned identity Platt parameters
                (``a=-1, b=0``) so that calibrated output equals
                ``sigmoid(score)`` — effectively leaving those
                predicates uncalibrated.

        Raises:
            ValueError: If ``probability_method`` is not ``"logprobs"``,
                *examples* is empty, *scope* is unrecognised, or
                *pass_through_single_class* is ``False`` and a label
                group lacks both classes.
        """
        if self.probability_method != "logprobs":
            raise ValueError(
                "Platt scaling is only supported for probability_method='logprobs'."
            )
        if not examples:
            raise ValueError("Expected at least one calibration example.")
        if scope not in {"global", "lifted"}:
            raise ValueError(f"Unsupported Platt scaling scope: {scope}")

        single_class_keys = self._validate_calibration_labels(
            examples, scope, pass_through_single_class
        )

        grouped_scores: dict[str, list[float]] = {}
        grouped_labels: dict[str, list[bool]] = {}

        for example in tqdm(examples, disable=not progress_bar, desc="Fitting Platt scaling"):
            example_problem = example.problem or self._problem
            example_up_problem = create_up_problem(self._domain, example_problem)
            per_sample_details = self._estimate_calibration_example(example)
            for details in per_sample_details:
                for predicate, detail in details.items():
                    if scope == "global":
                        key = GLOBAL_CALIBRATION_KEY
                    else:
                        key = get_lifted_predicate_key(example_up_problem, predicate)
                    grouped_scores.setdefault(key, []).append(detail.score)
                    grouped_labels.setdefault(key, []).append(example.state_dict[predicate])

        params_by_group: dict[str, PlattParameters] = {}
        for key, scores in grouped_scores.items():
            if key not in single_class_keys:
                params_by_group[key] = fit_platt_parameters(
                    scores, grouped_labels[key]
                )
            else:
                labels = grouped_labels[key]
                params_by_group[key] = PlattParameters(
                    a=-1.0,
                    b=0.0,
                    sample_count=len(scores),
                    positive_count=sum(bool(l) for l in labels),
                    negative_count=sum(not bool(l) for l in labels),
                )
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

    def _apply_platt_profile(self, predicate: str, score: float) -> float | None:
        """Apply Platt scaling for *predicate*.

        Returns ``None`` when parameters are not available for the
        predicate's group (e.g. the group was skipped during fitting).
        """
        assert self._platt_scaling_profile is not None
        if self._platt_scaling_profile.scope == "global":
            if GLOBAL_CALIBRATION_KEY not in self._platt_scaling_profile.groups:
                return None
            params = self._platt_scaling_profile.groups[GLOBAL_CALIBRATION_KEY]
        else:
            key = get_lifted_predicate_key(self.up_problem, predicate)
            if key not in self._platt_scaling_profile.groups:
                return None
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
        # Missing groups are allowed — they arise when the profile was
        # fit with single_class_policy="skip".  At inference time the
        # estimator falls back to the uncalibrated probability for any
        # predicate whose group is absent from the profile.

    def _validate_calibration_labels(
        self,
        examples: list[CalibrationExample],
        scope: str,
        pass_through_single_class: bool = False,
    ) -> set[str]:
        """Check label groups for single-class issues.

        Returns the set of group keys that have only one label class.
        When *pass_through_single_class* is ``False``, a
        :class:`ValueError` is raised instead of returning a non-empty
        set.
        """
        label_sets: dict[str, set[bool]] = {}
        for example in examples:
            example_problem = example.problem or self._problem
            if scope == "global":
                for predicate in example.state_dict:
                    label_sets.setdefault(GLOBAL_CALIBRATION_KEY, set()).add(
                        example.state_dict[predicate]
                    )
            else:
                example_up_problem = create_up_problem(self._domain, example_problem)
                for predicate in example.state_dict:
                    key = get_lifted_predicate_key(example_up_problem, predicate)
                    label_sets.setdefault(key, set()).add(
                        example.state_dict[predicate]
                    )

        single_class_keys: set[str] = set()
        for key, labels in label_sets.items():
            if True in labels and False in labels:
                continue
            if not pass_through_single_class:
                present = "positive" if True in labels else "negative"
                if scope == "global":
                    raise ValueError(
                        "Platt scaling requires both positive and negative labels, "
                        f"but all provided labels are {present}."
                    )
                raise ValueError(
                    f"Platt scaling requires both positive and negative labels "
                    f"for each lifted predicate, but '{key}' has only {present} labels."
                )
            single_class_keys.add(key)

        return single_class_keys

    def _extract_prediction_details(self, output: VLMOutput) -> PredicatePredictionDetails:
        if self.probability_method == "text_match":
            if output.text is None:
                return PredicatePredictionDetails(0.5, 0.5, 0.0, 0.0, 0.0, 0.0, False)
            text = output.text.strip().lower()
            true_lower = {t.lower() for t in self.true_tokens}
            probability = 1.0 if text in true_lower else 0.0
            return PredicatePredictionDetails(
                probability=probability,
                raw_probability=probability,
                score=0.0,
                raw_true_mass=0.0,
                raw_false_mass=0.0,
                raw_none_mass=0.0,
                none_is_max_raw=False,
            )

        raw_true_mass = sum(output.token_probs.get(tok, 0.0) for tok in self.true_tokens)
        raw_false_mass = sum(output.token_probs.get(tok, 0.0) for tok in self.false_tokens)
        raw_none_mass = sum(output.token_probs.get(tok, 0.0) for tok in self.null_tokens)
        raw_total = raw_true_mass + raw_false_mass
        raw_probability = 0.5 if raw_total == 0 else float(np.clip(raw_true_mass / raw_total, 0.0, 1.0))
        score = 0.0 if raw_total == 0 else grouped_log_odds(output.token_probs, self.true_tokens, self.false_tokens)
        return PredicatePredictionDetails(
            probability=raw_probability,
            raw_probability=raw_probability,
            score=score,
            raw_true_mass=raw_true_mass,
            raw_false_mass=raw_false_mass,
            raw_none_mass=raw_none_mass,
            none_is_max_raw=raw_none_mass > raw_true_mass and raw_none_mass > raw_false_mass,
        )

    def prediction_details_from_raw(
        self,
        raw_outputs: dict[str, VLMOutput],
        calibrated: bool | None = None,
    ) -> dict[str, PredicatePredictionDetails]:
        use_calibration = self._resolve_calibrated_flag(calibrated)
        details = {
            pred: self._extract_prediction_details(output)
            for pred, output in raw_outputs.items()
        }
        if not use_calibration:
            return details

        result: dict[str, PredicatePredictionDetails] = {}
        for pred, detail in details.items():
            calibrated_probability = self._apply_platt_profile(pred, detail.score)
            if calibrated_probability is None:
                result[pred] = detail
            else:
                result[pred] = replace(detail, probability=calibrated_probability)
        return result

    def probabilities_from_raw(
        self,
        raw_outputs: dict[str, VLMOutput],
        calibrated: bool | None = None,
    ) -> dict[str, float]:
        """Derive probabilities from already-obtained raw VLM outputs.

        This avoids a second VLM invocation when you need both calibrated
        and uncalibrated probabilities for the same observation::

            raw = estimator.estimate_raw(images)
            uncalibrated = estimator.probabilities_from_raw(raw)
            calibrated = estimator.probabilities_from_raw(raw, calibrated=True)

        Args:
            raw_outputs: Mapping of grounded predicate strings to
                :class:`VLMOutput`, as returned by :meth:`estimate_raw`.
            calibrated: Whether to apply Platt scaling.  ``None`` auto-detects
                (apply if a profile is loaded), ``True`` requires a profile,
                ``False`` always returns uncalibrated probabilities.

        Returns:
            Mapping of grounded predicate strings to P(true).
        """
        details = self.prediction_details_from_raw(raw_outputs, calibrated=calibrated)
        return {pred: detail.probability for pred, detail in details.items()}

    def _estimate_calibration_example(
        self,
        example: CalibrationExample,
    ) -> list[dict[str, PredicatePredictionDetails]]:
        original_problem = self._problem
        try:
            if example.problem is not None:
                self.swap_problem(self._domain, example.problem)
            unknown = set(example.state_dict.keys()) - set(self.queries_dict)
            if unknown:
                raise ValueError(
                    f"Calibration example contains predicate(s) not in the "
                    f"current problem: {', '.join(sorted(unknown))}"
                )
            labeled = list(example.state_dict.keys())
            if self.multi_image_strategy == "average":
                return [
                    self.prediction_details_from_raw(
                        self.estimate_raw([image], predicates=labeled),
                        calibrated=False,
                    )
                    for image in example.images
                ]
            return [
                self.prediction_details_from_raw(
                    self.estimate_raw(example.images, predicates=labeled),
                    calibrated=False,
                )
            ]
        finally:
            if example.problem is not None:
                self.swap_problem(self._domain, original_problem)

    def _resolve_queries(
        self, predicates: list[str] | None
    ) -> dict[str, str]:
        """Return the queries dict filtered to *predicates*.

        When *predicates* is ``None``, the full ``queries_dict`` is
        returned.  Otherwise, only the requested subset is returned
        (order preserved) and unknown predicates raise ``ValueError``.
        """
        if predicates is None:
            return self.queries_dict
        unknown = set(predicates) - set(self.queries_dict)
        if unknown:
            raise ValueError(
                f"Unknown predicate(s): {', '.join(sorted(unknown))}"
            )
        return {p: self.queries_dict[p] for p in predicates}
