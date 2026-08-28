"""SemanticStateEstimator: a thin facade wiring predicates, translation,
and a QueryEngine into symbolic state estimation.

The estimator's contract is "predicates in, state out". PDDL is one way to
produce the predicates (:meth:`SemanticStateEstimator.from_pddl`); an
explicit list is another. Unified Planning is imported only on the PDDL
paths, keeping the core importable without it.
"""

from collections.abc import Sequence

from PIL.Image import Image

from .constants import (
    SYSTEM_PROMPT_ADDITIONAL_INSTRUCTIONS,
    SYSTEM_PROMPT_NO_TRANSLATION,
    SYSTEM_PROMPT_WITH_TRANSLATION,
)
from .engine import BinaryAnswers, PredictionSet, QueryEngine
from .translation import IdentityTranslator, QueryTranslator


class SemanticStateEstimator:
    """Estimates truth values for a set of grounded predicates from images.

    Args:
        predicates: The grounded predicate strings to estimate.
        vlm: A backend instance or model-id string (see ``resolve_backend``).
        translator: Predicate-to-query strategy (default: identity).
        answers: Answer space (default: ``BinaryAnswers()``; identity
            translation defaults to ``BinaryAnswers("true", "false")``).
        system_prompt: Overrides the auto-selected system prompt.
        prompt_template: Wrapper for each query; must contain ``{query}``.
        additional_instructions: Appended to the system prompt.
        confidence: Default threshold for :meth:`__call__`.
        scoring: ``"logprobs"`` or ``"text_match"``.
        batch_size / vlm_kwargs / inference_kwargs: Forwarded to
            :class:`QueryEngine`.
        true_tokens / false_tokens / null_tokens: Convenience overrides
            building the default binary answer space; ignored when
            ``answers`` is passed explicitly.
    """

    def __init__(
        self,
        predicates: Sequence[str],
        *,
        vlm,
        translator: "QueryTranslator | None" = None,
        answers=None,
        system_prompt: "str | None" = None,
        prompt_template: "str | None" = None,
        additional_instructions: "str | None" = None,
        confidence: float = 0.5,
        scoring: str = "logprobs",
        batch_size: int = 8,
        vlm_kwargs: "dict | None" = None,
        inference_kwargs: "dict | None" = None,
        true_tokens: "list[str] | None" = None,
        false_tokens: "list[str] | None" = None,
        null_tokens: "list[str] | None" = None,
        _pddl_context: "dict | None" = None,
    ):
        self.translator = translator or IdentityTranslator()
        identity = isinstance(self.translator, IdentityTranslator)

        if answers is None:
            if identity:
                answers = BinaryAnswers(
                    "true", "false",
                    true_tokens=true_tokens, false_tokens=false_tokens,
                    null_tokens=null_tokens,
                )
            else:
                answers = BinaryAnswers(
                    true_tokens=true_tokens, false_tokens=false_tokens,
                    null_tokens=null_tokens,
                )

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT_WITH_TRANSLATION
        if additional_instructions:
            system_prompt += SYSTEM_PROMPT_ADDITIONAL_INSTRUCTIONS.format(
                additional_instructions=additional_instructions
            )

        self.confidence = confidence
        self.engine = QueryEngine(
            vlm,
            answers=answers,
            scoring=scoring,
            system_prompt=system_prompt,
            prompt_template=prompt_template or "{query}",
            batch_size=batch_size,
            inference_kwargs=inference_kwargs,
            vlm_kwargs=vlm_kwargs,
        )

        # ``_pddl_context`` is internal: :meth:`from_pddl` uses it to install
        # the PDDL context *before* the first translation, so translators that
        # need a domain (e.g. LLMTranslator) see it and nothing is translated
        # twice. Public callers pass predicates only.
        context = _pddl_context or {}
        self.up_problem = context.get("up_problem")
        self.domain_pddl: "str | None" = context.get("domain_pddl")
        self.problem_pddl: "str | None" = context.get("problem_pddl")
        self.domain_fingerprint: "str | None" = context.get("domain_fingerprint")
        self.predicates: list[str] = []
        self.queries: dict[str, str] = {}
        self.set_predicates(predicates)

    @classmethod
    def from_pddl(cls, domain: str, problem: str, **kwargs) -> "SemanticStateEstimator":
        """Build an estimator by grounding a PDDL domain and problem.

        ``domain``/``problem`` are PDDL strings or ``.pddl`` file paths.
        Identity translation additionally gets a domain-aware system prompt.
        """
        from .pddl import (
            compute_domain_fingerprint,
            get_object_names_dict,
            get_pddl_strings,
            ground_predicates,
            parse_domain_problem,
        )

        up_problem = parse_domain_problem(domain, problem)
        translator = kwargs.get("translator")
        if kwargs.get("system_prompt") is None and (
            translator is None or isinstance(translator, IdentityTranslator)
        ):
            objects = get_object_names_dict(up_problem)
            objects_str = "\n".join(
                f"{key} type: {value}" for key, value in objects.items()
            )
            domain_str, _ = get_pddl_strings(up_problem)
            kwargs["system_prompt"] = SYSTEM_PROMPT_NO_TRANSLATION.format(
                domain=domain_str, objects=objects_str
            )

        return cls(
            ground_predicates(up_problem),
            vlm=kwargs.pop("vlm"),
            _pddl_context={
                "up_problem": up_problem,
                "domain_pddl": domain,
                "problem_pddl": problem,
                "domain_fingerprint": compute_domain_fingerprint(up_problem),
            },
            **kwargs,
        )

    # --- predicate/problem management ---

    def set_predicates(self, predicates: Sequence[str]) -> None:
        """Replace the predicate list and re-run translation."""
        self.predicates = list(predicates)
        self._retranslate()

    def set_problem(self, domain: str, problem: str) -> None:
        """Re-ground a new PDDL problem; the engine/backend is untouched."""
        self._require_pddl("set_problem")
        from .pddl import (
            compute_domain_fingerprint,
            ground_predicates,
            parse_domain_problem,
        )

        self.up_problem = parse_domain_problem(domain, problem)
        self.domain_pddl = domain
        self.problem_pddl = problem
        self.domain_fingerprint = compute_domain_fingerprint(self.up_problem)
        self.predicates = ground_predicates(self.up_problem)
        self._retranslate()

    def _retranslate(self) -> None:
        self.queries = self.translator.translate(
            self.predicates, self.domain_pddl, self.problem_pddl
        )

    def _require_pddl(self, method: str) -> None:
        if self.up_problem is None:
            raise ValueError(
                f"{method} is only available on estimators built with from_pddl"
            )

    # --- estimation ---

    def estimate(
        self,
        images: list[Image],
        *,
        predicates: "Sequence[str] | None" = None,
        calibrator=None,
        keep_raw: bool = False,
        inference_kwargs: "dict | None" = None,
    ) -> PredictionSet:
        """Estimate the selected predicates; returns a lazy PredictionSet."""
        if calibrator is not None:
            if self.engine.scoring != "logprobs":
                raise ValueError(
                    "calibrator requires scoring='logprobs'; this estimator uses "
                    f"scoring={self.engine.scoring!r} — calibration over text-match "
                    "masses produces meaningless probabilities"
                )
            self._check_calibrator_compat(calibrator)
        selected = self._select(predicates)
        queries = [self.queries[p] for p in selected]
        # Distinct queries only: predicates that translate to the same query
        # are asked once and share the resulting prediction.
        answered = self.engine.ask(
            images,
            list(dict.fromkeys(queries)),
            keep_raw=keep_raw,
            inference_kwargs=inference_kwargs,
        )
        results = PredictionSet(
            {p: answered[q] for p, q in zip(selected, queries)}
        )
        if calibrator is not None:
            results = calibrator.apply(results)
        return results

    def estimate_averaged(self, scenes, **estimate_kwargs) -> PredictionSet:
        """Estimate each scene separately and average the stored masses."""
        return PredictionSet.average(
            [self.estimate(scene, **estimate_kwargs) for scene in scenes]
        )

    def __call__(
        self, images: list[Image], confidence: "float | None" = None
    ) -> "dict[str, bool | None]":
        """Estimate and threshold into a boolean state."""
        threshold = confidence if confidence is not None else self.confidence
        return self.estimate(images).to_state(confidence=threshold)

    def _check_calibrator_compat(self, calibrator) -> None:
        """Refuse calibrators whose recorded provenance contradicts this estimator.

        Only fields the calibrator's meta actually carries are compared, so
        hand-built calibrators (no meta) apply unconditionally.
        """
        meta = getattr(calibrator, "meta", None) or {}
        scoring = meta.get("scoring")
        if scoring is not None and scoring != "logprobs":
            raise ValueError(
                f"calibrator was fitted on scoring={scoring!r} data; only "
                "scoring='logprobs' calibration is supported"
            )
        fingerprint = meta.get("domain_fingerprint")
        if (
            fingerprint is not None
            and self.domain_fingerprint is not None
            and fingerprint != self.domain_fingerprint
        ):
            raise ValueError(
                "calibrator was fitted on data from a different domain: "
                f"calibrator domain_fingerprint {fingerprint!r} != estimator "
                f"{self.domain_fingerprint!r}"
            )
        answers = meta.get("answers")
        if answers is not None and answers != self.engine.answers.to_dict():
            raise ValueError(
                "calibrator answer space does not match this estimator's: "
                f"calibrator uses {answers!r}, estimator uses "
                f"{self.engine.answers.to_dict()!r}"
            )

    def _select(self, predicates: "Sequence[str] | None") -> list[str]:
        if predicates is None:
            return list(self.predicates)
        known = set(self.predicates)
        unknown = [p for p in predicates if p not in known]
        if unknown:
            raise ValueError(f"Unknown predicates requested: {unknown}")
        return list(predicates)

    # --- interop ---

    def to_up_state(self, state: dict[str, bool]):
        """Convert a boolean state dict into a Unified Planning UPState."""
        self._require_pddl("to_up_state")
        from .pddl import state_dict_to_up_state

        return state_dict_to_up_state(self.up_problem, state)

    def calibration_meta(self) -> dict:
        """Metadata stored alongside collected calibration data."""
        answers = self.engine.answers
        return {
            "true_label": getattr(answers, "true_label", None),
            "false_label": getattr(answers, "false_label", None),
            "scoring": self.engine.scoring,
            "domain_fingerprint": self.domain_fingerprint,
            "answers": answers.to_dict(),
        }
