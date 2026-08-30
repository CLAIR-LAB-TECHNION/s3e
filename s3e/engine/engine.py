"""QueryEngine: images + queries + an answer space -> predictions."""

from collections.abc import Sequence

from PIL.Image import Image

from ..backends import VLMBackend, resolve_backend
from .answers import SCORING_MODES, AnswerSpace, BinaryAnswers
from .results import Prediction, PredictionSet


def _validate_scoring(scoring: str) -> str:
    if scoring not in SCORING_MODES:
        raise ValueError(
            f"Unknown scoring mode {scoring!r}; expected one of {SCORING_MODES}"
        )
    return scoring


class QueryEngine:
    """Answers free-form queries about images against an answer space.

    Args:
        vlm: A :class:`VLMBackend` instance or a model-id string (resolved
            through :func:`resolve_backend`).
        answers: Default answer space (default: ``BinaryAnswers()``).
        scoring: ``"logprobs"`` (token masses) or ``"text_match"``
            (generated text).
        system_prompt: Optional system prompt forwarded to the backend.
        prompt_template: Wrapper for each query; must contain ``{query}``.
        batch_size: Number of queries per backend batch call.
        inference_kwargs: Default per-query kwargs forwarded verbatim to the
            backend (semantics are backend-specific; see the backend class).
        vlm_kwargs: Constructor kwargs, used only when ``vlm`` is a string.
    """

    def __init__(
        self,
        vlm: "VLMBackend | str",
        *,
        answers: "AnswerSpace | None" = None,
        scoring: str = "logprobs",
        system_prompt: "str | None" = None,
        prompt_template: str = "{query}",
        batch_size: int = 8,
        inference_kwargs: "dict | None" = None,
        vlm_kwargs: "dict | None" = None,
    ):
        self.backend = resolve_backend(vlm, **(vlm_kwargs or {}))
        self.answers = answers if answers is not None else BinaryAnswers()
        self.scoring = _validate_scoring(scoring)
        self.system_prompt = system_prompt
        if "{query}" not in prompt_template:
            raise ValueError(
                f"prompt_template must contain '{{query}}'; got {prompt_template!r}"
            )
        self.prompt_template = prompt_template
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive int; got {batch_size!r}")
        self.batch_size = batch_size
        self.inference_kwargs = dict(inference_kwargs or {})

    def ask(
        self,
        images: list[Image],
        queries: Sequence[str],
        *,
        answers: "AnswerSpace | None" = None,
        scoring: "str | None" = None,
        inference_kwargs: "dict | None" = None,
        keep_raw: bool = False,
    ) -> PredictionSet:
        """Answer each query about one scene (a list of images shown together)."""
        space = answers if answers is not None else self.answers
        mode = _validate_scoring(scoring) if scoring is not None else self.scoring
        merged_kwargs = {**self.inference_kwargs, **(inference_kwargs or {})}
        generate = mode == "text_match"
        interest = None if generate else space.interest_tokens
        if interest is not None:
            self._reject_untokenizable_options(space)

        prompts = [self.prompt_template.format(query=q) for q in queries]
        outputs = []
        for start in range(0, len(prompts), self.batch_size):
            outputs.extend(
                self.backend.query_batch(
                    images,
                    prompts[start : start + self.batch_size],
                    system_prompt=self.system_prompt,
                    generate=generate,
                    interest_tokens=interest,
                    **merged_kwargs,
                )
            )

        predictions: dict[str, Prediction] = {}
        for query, output in zip(queries, outputs, strict=True):
            scored = space.score(output, scoring=mode)
            predictions[query] = Prediction(
                query=query,
                masses=scored.masses,
                null_mass=scored.null_mass,
                unassigned_mass=scored.unassigned_mass,
                answers=space,
                text=output.text,
                argmax_in_interest=output.argmax_in_interest,
                raw=output if keep_raw else None,
            )
        return PredictionSet(predictions)

    def ask_each(
        self,
        scenes: Sequence[list[Image]],
        queries: Sequence[str],
        **ask_kwargs,
    ) -> list[PredictionSet]:
        """Run :meth:`ask` once per scene; combine with PredictionSet.average."""
        return [self.ask(scene, queries, **ask_kwargs) for scene in scenes]

    def _reject_untokenizable_options(self, space: AnswerSpace) -> None:
        """Reject options whose every token form the backend cannot score.

        Logprob scoring reads single-token masses; an option like
        ``"dark blue"`` whose every surface form is multi-token would
        silently score 0.0. Backends that can tell report such forms via
        ``unsupported_interest_tokens``; the default reports none.
        """
        unsupported = set(self.backend.unsupported_interest_tokens(space.interest_tokens))
        if not unsupported:
            return
        options = list(space.options) + (
            [space.null_option] if space.null_option else []
        )
        dead = [o.label for o in options if set(o.tokens) <= unsupported]
        if dead:
            raise ValueError(
                f"Answer options {dead} have no single-token form this backend "
                "can score in logprobs mode; use scoring='text_match' or "
                "provide single-token surface forms"
            )
