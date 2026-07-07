"""Semantic state estimation using vision-language models.

This module provides the main :class:`SemanticStateEstimator` class that
combines a VLM backend with a query translator to estimate environment
state from images. The result is a dictionary of PDDL predicate truth
values (or probabilities) compatible with planning systems.
"""

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Union

import numpy as np
from PIL.Image import Image
from tqdm.auto import tqdm

from .calibration import (
    CalibrationExample,
    GLOBAL_CALIBRATION_KEY,
    PLATT_CALIBRATION_DATA_SCHEMA_VERSION,
    PlattCalibrationSample,
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
    """Raw and calibrated probability details for one predicate prediction."""

    raw_probability: float
    calibrated_probability: float | None
    score: float
    raw_true_mass: float
    raw_false_mass: float
    raw_none_mass: float
    none_is_max_raw: bool


class SemanticStateEstimator(ProbabilisticStateEstimator):
    """Vision-language model based state estimator.

    Combines a :class:`VLMBackend` with a :class:`QueryTranslator` to
    estimate the boolean truth values of PDDL predicates from images.

    All configuration arguments after ``vlm`` are keyword-only.

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
        null_tokens: Token strings representing "not enough information".
            When configured, :meth:`__call__` may return ``None`` for
            predicates whose raw null token mass is strictly larger than
            the raw true and false token masses.
        confidence: Probability threshold for boolean conversion.
        multi_image_strategy: ``"single"`` or ``"average"``.
        probability_method: ``"logprobs"`` or ``"text_match"``.
        batch_size: Number of queries per VLM batch call.
        additional_instructions: Extra text appended to the system prompt.
        vlm_kwargs: Extra kwargs for VLM construction (only when vlm is a string).
        use_vllm: When ``vlm`` is a model-id string for a non-OpenAI model,
            route it through the vLLM engine (:class:`VLLMBackend`) instead of
            plain transformers. Ignored when ``vlm`` is a backend instance; an
            ``OpenAI/`` model with ``use_vllm=True`` raises ``ValueError``.
    """

    def __init__(
        self,
        domain: str,
        problem: str,
        vlm: Union[VLMBackend, str],
        *,
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
        use_vllm: bool = False,
        inference_kwargs: dict | None = None,
    ):
        super().__init__(domain, problem, confidence)

        # --- VLM backend ---
        # use_vllm only affects the string path; an explicit backend instance is
        # used as-is.
        self.use_vllm = use_vllm
        if isinstance(vlm, str):
            self.vlm = self._build_vlm_from_string(
                vlm, vlm_kwargs or {}, use_vllm
            )
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
        return compute_domain_fingerprint(self.up_problem)

    @staticmethod
    def _build_vlm_from_string(
        vlm_id: str, vlm_kwargs: dict, use_vllm: bool = False
    ) -> VLMBackend:
        """Construct a VLM backend from a model ID string.

        ``use_vllm`` routes a non-OpenAI model through the vLLM engine instead
        of plain transformers. It is incompatible with ``OpenAI/`` models (which
        run against the hosted API, not a local engine).
        """
        if vlm_id.startswith(OPENAI_MODEL_IDENTIFIER):
            if use_vllm:
                raise ValueError(
                    "use_vllm=True is not compatible with OpenAI/ models."
                )
            from .vlm.openai import OpenAIVLM

            return OpenAIVLM(vlm_id, **vlm_kwargs)
        if use_vllm:
            from .vlm.vllm import VLLMBackend

            return VLLMBackend(vlm_id, **vlm_kwargs)
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
        details = self.estimate_prediction_details(images, predicates=predicates)
        probabilities = self._probabilities_from_details(details, calibrated)
        threshold = confidence if confidence is not None else self.confidence
        return {
            pred: (
                None
                if detail.none_is_max_raw
                else bool(probabilities[pred] >= threshold)
            )
            for pred, detail in details.items()
        }

    def estimate_prediction_details(
        self,
        images: list[Image],
        predicates: list[str] | None = None,
    ) -> dict[str, PredicatePredictionDetails]:
        if self.multi_image_strategy != "average":
            raw = self.estimate_raw(images, predicates=predicates)
            return self.prediction_details_from_raw(raw)

        per_image = [
            self.prediction_details_from_raw(
                self.estimate_raw([img], predicates=predicates),
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
            raw_true_mass = float(np.clip(
                np.mean([d.raw_true_mass for d in items]), 0.0, 1.0,
            ))
            raw_false_mass = float(np.clip(
                np.mean([d.raw_false_mass for d in items]), 0.0, 1.0,
            ))
            raw_none_mass = float(np.clip(
                np.mean([d.raw_none_mass for d in items]), 0.0, 1.0,
            ))
            calibrated_probabilities = [d.calibrated_probability for d in items]
            calibrated_probability = (
                float(np.clip(np.mean(calibrated_probabilities), 0.0, 1.0))
                if all(value is not None for value in calibrated_probabilities)
                else None
            )
            result[pred] = PredicatePredictionDetails(
                raw_probability=float(np.clip(
                    np.mean([d.raw_probability for d in items]), 0.0, 1.0,
                )),
                calibrated_probability=calibrated_probability,
                score=float(np.mean([d.score for d in items])),
                raw_true_mass=raw_true_mass,
                raw_false_mass=raw_false_mass,
                raw_none_mass=raw_none_mass,
                none_is_max_raw=raw_none_mass > raw_true_mass
                and raw_none_mass > raw_false_mass,
            )
        return result

    def estimate_probabilities(
        self,
        images: list[Image],
        calibrated: bool | None = None,
        predicates: list[str] | None = None,
    ) -> dict[str, float]:
        """Estimate binary P(true) for each grounded predicate.

        Null-token mass is available through :meth:`estimate_prediction_details`
        and does not change the meaning of this method.

        Args:
            images: List of PIL images representing the current state.
            calibrated: Whether to apply Platt scaling.
            predicates: Optional list of grounded predicate strings to
                query.  When ``None`` (default), all predicates are
                queried.  Unknown predicates raise :class:`ValueError`.
        """
        details = self.estimate_prediction_details(images, predicates=predicates)
        return self._probabilities_from_details(details, calibrated)

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

    def collect_platt_scaling_data(
        self,
        examples: list[CalibrationExample],
        progress_bar: bool = False,
    ) -> list[PlattCalibrationSample]:
        """Collect precomputed samples for later Platt scaling fitting.

        This method performs VLM inference and is the expensive part of
        calibration. Save its return value with
        :meth:`save_platt_scaling_data` to reuse the same predictions
        without querying the VLM again.
        """
        self._validate_platt_logprobs_mode()
        if not examples:
            raise ValueError("Expected at least one calibration example.")

        samples: list[PlattCalibrationSample] = []
        for example in tqdm(
            examples,
            disable=not progress_bar,
            desc="Collecting Platt calibration data",
        ):
            per_sample_details = self._estimate_calibration_example(example)
            for details in per_sample_details:
                for predicate, detail in details.items():
                    samples.append(
                        PlattCalibrationSample(
                            predicate=predicate,
                            score=detail.score,
                            label=example.state_dict[predicate],
                            problem=example.problem,
                        )
                    )
        return samples

    def fit_platt_scaling(
        self,
        examples: list[CalibrationExample],
        scope: str = "global",
        progress_bar: bool = False,
        pass_through_single_class: bool = False,
    ) -> None:
        """Fit a Platt scaling calibration profile from labeled examples.

        This method performs VLM inference to collect calibration samples,
        then delegates fitting to :meth:`fit_platt_scaling_from_data`.
        Use :meth:`collect_platt_scaling_data` and
        :meth:`save_platt_scaling_data` when you want to reuse expensive
        predictions across runs.
        """
        self._validate_platt_logprobs_mode()
        if not examples:
            raise ValueError("Expected at least one calibration example.")
        self._validate_platt_scope(scope)

        label_samples = [
            PlattCalibrationSample(
                predicate=predicate,
                score=0.0,
                label=label,
                problem=example.problem,
            )
            for example in examples
            for predicate, label in example.state_dict.items()
        ]
        _, grouped_labels = self._group_platt_calibration_samples(
            label_samples, scope
        )
        self._validate_grouped_platt_labels(
            grouped_labels, scope, pass_through_single_class
        )

        data = self.collect_platt_scaling_data(
            examples,
            progress_bar=progress_bar,
        )
        self.fit_platt_scaling_from_data(
            data,
            scope=scope,
            pass_through_single_class=pass_through_single_class,
        )

    def fit_platt_scaling_from_data(
        self,
        data: list[PlattCalibrationSample],
        scope: str = "global",
        pass_through_single_class: bool = False,
    ) -> None:
        """Fit a Platt scaling calibration profile from precomputed samples.

        This method does not query the VLM. It consumes grouped log-odds
        scores and boolean labels produced earlier by
        :meth:`collect_platt_scaling_data` or loaded with
        :meth:`load_platt_scaling_data`.
        """
        self._validate_platt_logprobs_mode()
        if not data:
            raise ValueError("Expected at least one Platt calibration sample.")
        self._validate_platt_scope(scope)

        grouped_scores, grouped_labels = self._group_platt_calibration_samples(
            data, scope
        )
        single_class_keys = self._validate_grouped_platt_labels(
            grouped_labels, scope, pass_through_single_class
        )
        params_by_group: dict[str, PlattParameters] = {}
        for key, scores in grouped_scores.items():
            labels = grouped_labels[key]
            if key not in single_class_keys:
                params_by_group[key] = fit_platt_parameters(scores, labels)
            else:
                params_by_group[key] = PlattParameters(
                    a=-1.0,
                    b=0.0,
                    sample_count=len(scores),
                    positive_count=sum(bool(label) for label in labels),
                    negative_count=sum(not bool(label) for label in labels),
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

    def save_platt_scaling_data(
        self,
        data: list[PlattCalibrationSample],
        path: str,
    ) -> None:
        """Save precomputed Platt calibration samples with compatibility metadata.

        The saved metadata describes how the scores were produced. It is
        used by :meth:`load_platt_scaling_data` to prevent fitting samples
        with incompatible token groups, probability methods, or domains.
        """
        self._validate_platt_logprobs_mode()
        payload = {
            "schema_version": PLATT_CALIBRATION_DATA_SCHEMA_VERSION,
            "score_kind": "grouped_log_odds",
            "probability_method": self.probability_method,
            "true_tokens": list(self.true_tokens),
            "false_tokens": list(self.false_tokens),
            "domain_fingerprint": self._domain_fingerprint,
            "samples": [sample.to_dict() for sample in data],
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def load_platt_scaling_data(
        self,
        path: str,
    ) -> list[PlattCalibrationSample]:
        """Load precomputed Platt calibration samples saved by this estimator.

        Loading validates compatibility metadata and returns plain sample
        rows. It does not fit or attach a Platt scaling profile.
        """
        payload = json.loads(Path(path).read_text())
        self._validate_platt_scaling_data_payload(payload)
        return [
            PlattCalibrationSample.from_dict(sample)
            for sample in payload["samples"]
        ]

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

    def _validate_platt_logprobs_mode(self) -> None:
        if self.probability_method != "logprobs":
            raise ValueError(
                "Platt scaling is only supported for probability_method='logprobs'."
            )

    def _validate_platt_scope(self, scope: str) -> None:
        if scope not in {"global", "lifted"}:
            raise ValueError(f"Unsupported Platt scaling scope: {scope}")

    def _validate_platt_scaling_data_payload(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            raise ValueError(
                "Loaded Platt calibration data payload must be a JSON object."
            )

        required_fields = {
            "schema_version",
            "score_kind",
            "probability_method",
            "true_tokens",
            "false_tokens",
            "domain_fingerprint",
            "samples",
        }
        missing_fields = sorted(required_fields - set(payload))
        if missing_fields:
            raise ValueError(
                "Loaded Platt calibration data is missing required field(s): "
                + ", ".join(missing_fields)
            )
        if not isinstance(payload["samples"], list):
            raise ValueError("Loaded Platt calibration data samples must be a list.")

        try:
            schema_version = int(payload["schema_version"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Loaded Platt calibration data schema_version must be an integer."
            ) from exc
        if schema_version != PLATT_CALIBRATION_DATA_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Platt calibration data schema version: {schema_version}"
            )
        if payload["probability_method"] != "logprobs" or self.probability_method != "logprobs":
            raise ValueError(
                "Platt calibration data is only compatible with logprobs mode."
            )
        if payload["true_tokens"] != list(self.true_tokens) or payload["false_tokens"] != list(
            self.false_tokens
        ):
            raise ValueError(
                "Loaded Platt calibration data does not match the estimator token groups."
            )
        if payload["domain_fingerprint"] != self._domain_fingerprint:
            raise ValueError(
                "Loaded Platt calibration data was collected for a different domain."
            )
        if payload["score_kind"] != "grouped_log_odds":
            raise ValueError(
                f"Loaded Platt calibration data has unsupported score_kind: {payload['score_kind']}."
            )

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
        self._validate_platt_scope(profile.scope)
        if profile.score_kind != "grouped_log_odds":
            raise ValueError(
                f"Loaded Platt scaling profile has unsupported score_kind: {profile.score_kind}."
            )
        # Missing groups are allowed — they arise when the profile was
        # fit with single_class_policy="skip".  At inference time the
        # estimator falls back to the uncalibrated probability for any
        # predicate whose group is absent from the profile.

    def _group_platt_calibration_samples(
        self,
        data: list[PlattCalibrationSample],
        scope: str,
    ) -> tuple[dict[str, list[float]], dict[str, list[bool]]]:
        """Group samples by calibration key; validates predicates against each sample's problem."""
        grouped_scores: dict[str, list[float]] = {}
        grouped_labels: dict[str, list[bool]] = {}
        problem_cache: dict[str, object] = {}
        predicates_cache: dict[str, set[str]] = {}

        for sample in data:
            problem = sample.problem or self._problem
            if problem not in problem_cache:
                problem_cache[problem] = (
                    self.up_problem
                    if problem == self._problem
                    else create_up_problem(self._domain, problem)
                )
            if problem not in predicates_cache:
                predicates_cache[problem] = set(
                    get_all_grounded_predicates_for_objects(problem_cache[problem])
                )
            if sample.predicate not in predicates_cache[problem]:
                raise ValueError(
                    f"Calibration data contains predicate(s) not in the "
                    f"current problem: {sample.predicate}"
                )

            if scope == "global":
                key = GLOBAL_CALIBRATION_KEY
            else:
                key = get_lifted_predicate_key(
                    problem_cache[problem], sample.predicate
                )
            grouped_scores.setdefault(key, []).append(sample.score)
            grouped_labels.setdefault(key, []).append(sample.label)

        return grouped_scores, grouped_labels

    def _validate_grouped_platt_labels(
        self,
        grouped_labels: dict[str, list[bool]],
        scope: str,
        pass_through_single_class: bool,
    ) -> set[str]:
        """Return single-class group keys, raising when pass_through_single_class is False."""
        single_class_keys: set[str] = set()
        for key, labels in grouped_labels.items():
            has_positive = any(bool(label) for label in labels)
            has_negative = any(not bool(label) for label in labels)
            if has_positive and has_negative:
                continue
            if not pass_through_single_class:
                present = "positive" if has_positive else "negative"
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

    def _extract_text_match_details(self, output: VLMOutput) -> PredicatePredictionDetails:
        if output.text is None:
            return PredicatePredictionDetails(
                raw_probability=0.5,
                calibrated_probability=None,
                score=0.0,
                raw_true_mass=0.0,
                raw_false_mass=0.0,
                raw_none_mass=0.0,
                none_is_max_raw=False,
            )

        text = output.text.strip().lower()
        true_lower = {t.lower() for t in self.true_tokens}
        false_lower = {t.lower() for t in self.false_tokens}
        none_lower = {t.lower() for t in self.null_tokens}

        if text in true_lower:
            return PredicatePredictionDetails(
                raw_probability=1.0,
                calibrated_probability=None,
                score=0.0,
                raw_true_mass=1.0,
                raw_false_mass=0.0,
                raw_none_mass=0.0,
                none_is_max_raw=False,
            )
        if text in false_lower:
            return PredicatePredictionDetails(
                raw_probability=0.0,
                calibrated_probability=None,
                score=0.0,
                raw_true_mass=0.0,
                raw_false_mass=1.0,
                raw_none_mass=0.0,
                none_is_max_raw=False,
            )
        if text in none_lower:
            return PredicatePredictionDetails(
                raw_probability=0.5,
                calibrated_probability=None,
                score=0.0,
                raw_true_mass=0.0,
                raw_false_mass=0.0,
                raw_none_mass=1.0,
                none_is_max_raw=True,
            )
        return PredicatePredictionDetails(
            raw_probability=0.5,
            calibrated_probability=None,
            score=0.0,
            raw_true_mass=0.0,
            raw_false_mass=0.0,
            raw_none_mass=0.0,
            none_is_max_raw=False,
        )

    def _extract_prediction_details(self, output: VLMOutput) -> PredicatePredictionDetails:
        if self.probability_method == "text_match":
            return self._extract_text_match_details(output)

        raw_true_mass = float(np.clip(
            sum(output.token_probs.get(tok, 0.0) for tok in self.true_tokens),
            0.0, 1.0,
        ))
        raw_false_mass = float(np.clip(
            sum(output.token_probs.get(tok, 0.0) for tok in self.false_tokens),
            0.0, 1.0,
        ))
        raw_none_mass = float(np.clip(
            sum(output.token_probs.get(tok, 0.0) for tok in self.null_tokens),
            0.0, 1.0,
        ))
        raw_total = raw_true_mass + raw_false_mass
        raw_probability = (
            0.5
            if raw_total == 0
            else float(np.clip(raw_true_mass / raw_total, 0.0, 1.0))
        )
        score = (
            0.0
            if raw_total == 0
            else grouped_log_odds(output.token_probs, self.true_tokens, self.false_tokens)
        )
        return PredicatePredictionDetails(
            raw_probability=raw_probability,
            calibrated_probability=None,
            score=score,
            raw_true_mass=raw_true_mass,
            raw_false_mass=raw_false_mass,
            raw_none_mass=raw_none_mass,
            none_is_max_raw=raw_none_mass > raw_true_mass
            and raw_none_mass > raw_false_mass,
        )

    def prediction_details_from_raw(
        self,
        raw_outputs: dict[str, VLMOutput],
    ) -> dict[str, PredicatePredictionDetails]:
        details = {
            pred: self._extract_prediction_details(output)
            for pred, output in raw_outputs.items()
        }
        if not self._has_compatible_platt_profile():
            return details

        result: dict[str, PredicatePredictionDetails] = {}
        for pred, detail in details.items():
            calibrated_probability = self._apply_platt_profile(pred, detail.score)
            if calibrated_probability is not None:
                calibrated_probability = float(
                    np.clip(calibrated_probability, 0.0, 1.0)
                )
            result[pred] = replace(
                detail,
                calibrated_probability=calibrated_probability,
            )
        return result

    def _has_compatible_platt_profile(self) -> bool:
        return (
            self._platt_scaling_profile is not None
            and self.probability_method == "logprobs"
            and self._platt_scaling_profile.probability_method == "logprobs"
        )

    def _probabilities_from_details(
        self,
        details: dict[str, PredicatePredictionDetails],
        calibrated: bool | None = None,
    ) -> dict[str, float]:
        use_calibration = self._resolve_calibrated_flag(calibrated)
        return {
            pred: self._probability_from_detail(detail, use_calibration)
            for pred, detail in details.items()
        }

    def _probability_from_detail(
        self,
        detail: PredicatePredictionDetails,
        use_calibration: bool,
    ) -> float:
        if use_calibration and detail.calibrated_probability is not None:
            return detail.calibrated_probability
        return detail.raw_probability

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
                    )
                    for image in example.images
                ]
            return [
                self.prediction_details_from_raw(
                    self.estimate_raw(example.images, predicates=labeled),
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
