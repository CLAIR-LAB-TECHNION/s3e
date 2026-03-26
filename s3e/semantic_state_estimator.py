"""Semantic state estimation using vision-language models.

This module provides the main :class:`SemanticStateEstimator` class that
combines a VLM backend with a query translator to estimate environment
state from images. The result is a dictionary of PDDL predicate truth
values (or probabilities) compatible with planning systems.
"""

import math
from typing import Union

import numpy as np
from PIL.Image import Image
from tqdm.auto import tqdm

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
    get_object_names_dict,
    get_all_grounded_predicates_for_objects,
    get_pddl_strings,
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
    ):
        super().__init__(domain, problem, confidence)

        # --- VLM backend ---
        if isinstance(vlm, str):
            self.vlm = self._build_vlm_from_string(vlm, vlm_kwargs or {})
        else:
            self.vlm = vlm

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
        self.batch_size = batch_size

        # --- Build queries ---
        self._domain = domain
        self._problem = problem
        self._build_queries()

    def _build_queries(self) -> None:
        """Translate all grounded predicates via the query translator."""
        predicates = get_all_grounded_predicates_for_objects(self.up_problem)
        self.queries_dict = self.query_translator.translate(
            predicates, self._domain, self._problem
        )

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
        super().swap_problem(domain, problem)
        self._domain = domain
        self._problem = problem
        self._build_queries()

    def estimate_probabilities(self, images: list[Image]) -> dict[str, float]:
        """Estimate P(true) for each grounded predicate."""
        if self.multi_image_strategy == "average":
            return self._estimate_average(images)
        return self._estimate_single(images)

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
                images, batch_prompts, system_prompt=self.system_prompt
            )
            for pred, output in zip(batch_preds, outputs):
                results[pred] = output

        return results

    def _estimate_single(self, images: list[Image]) -> dict[str, float]:
        """Estimate probabilities with all images in a single pass."""
        raw = self.estimate_raw(images)
        return {
            pred: self._extract_probability(output)
            for pred, output in raw.items()
        }

    def _estimate_average(self, images: list[Image]) -> dict[str, float]:
        """Estimate probabilities by averaging across individual images."""
        per_image_probs: list[dict[str, float]] = []
        for img in images:
            probs = self._estimate_single([img])
            per_image_probs.append(probs)

        predicates = list(per_image_probs[0].keys())
        return {
            pred: float(np.mean([p[pred] for p in per_image_probs]))
            for pred in predicates
        }

    def _extract_probability(self, output: VLMOutput) -> float:
        """Extract P(true) from a VLMOutput."""
        if self.probability_method == "text_match":
            return self._extract_text_match(output)
        return self._extract_logprobs(output)

    def _extract_logprobs(self, output: VLMOutput) -> float:
        """Extract P(true) by grouping and normalizing token probabilities."""
        true_sum = sum(
            output.token_probs.get(tok, 0.0) for tok in self.true_tokens
        )
        false_sum = sum(
            output.token_probs.get(tok, 0.0) for tok in self.false_tokens
        )
        total = true_sum + false_sum
        if total == 0:
            return 0.5  # no signal -- return uninformative prior
        return float(np.clip(true_sum / total, 0.0, 1.0))

    def _extract_text_match(self, output: VLMOutput) -> float:
        """Extract P(true) by checking if generated text matches true tokens."""
        if output.text is None:
            return 0.5
        text = output.text.strip().lower()
        true_lower = {t.lower() for t in self.true_tokens}
        if text in true_lower:
            return 1.0
        return 0.0
