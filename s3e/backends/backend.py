"""VLM backend abstract base class and output types.

This module defines the interface that all VLM (Vision-Language Model)
backends must implement, plus the :class:`VLMOutput` dataclass that
standardizes their return values.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from PIL.Image import Image


def _validate_num_logprobs(num_logprobs: int | None) -> int | None:
    """Validate a backend's optional top-k log-probability limit."""
    if num_logprobs is not None and (
        isinstance(num_logprobs, bool)
        or not isinstance(num_logprobs, int)
        or num_logprobs < 1
    ):
        raise ValueError(
            "num_logprobs must be a positive int or None; "
            f"got {num_logprobs!r}"
        )
    return num_logprobs


@dataclass
class VLMOutput:
    """Result from a single VLM query.

    Attributes:
        token_probs: Mapping of token strings to their probabilities.
            When the query was made with ``interest_tokens``, the keys are
            exactly those tokens (tokens absent from the model's vocabulary
            or returned distribution get probability 0.0); otherwise the
            keys are whatever distribution the backend returns. ``None`` in
            generate mode, where backends produce text only.
        text: The generated text response, if available.
        argmax_in_interest: Whether the model's single most likely next
            token is one of the requested ``interest_tokens``. ``None``
            when the query did not request interest tokens. A ``False``
            here means the model answered outside the expected token set,
            so the reported masses cover almost none of the distribution.
    """

    token_probs: dict[str, float] | None = field(default_factory=dict)
    text: str | None = None
    argmax_in_interest: bool | None = None


class VLMBackend(ABC):
    """Abstract base class for Vision-Language Model backends.

    Subclasses must implement :meth:`query`. The :meth:`query_batch` method
    has a default sequential implementation that can be overridden.

    The ``interest_tokens`` contract: when a caller passes a sequence of
    token strings, the backend reports probability mass for exactly those
    strings (summing over every vocabulary id that decodes to the string,
    normalized over the full vocabulary) instead of materializing a full or
    top-k distribution. This is semantics-free data — backends never learn
    what the tokens mean — and lets each backend skip decoding the rest of
    the vocabulary.
    """

    @abstractmethod
    def query(
        self,
        images: list[Image],
        prompt: str,
        system_prompt: str | None = None,
        generate: bool = False,
        interest_tokens: Sequence[str] | None = None,
        **inference_kwargs,
    ) -> VLMOutput:
        """Send a single query to the VLM."""
        ...

    def query_batch(
        self,
        images: list[Image],
        prompts: list[str],
        system_prompt: str | None = None,
        generate: bool = False,
        interest_tokens: Sequence[str] | None = None,
        **inference_kwargs,
    ) -> list[VLMOutput]:
        """Send multiple queries against the same set of images.

        Default implementation calls :meth:`query` sequentially.
        """
        return [
            self.query(
                images,
                p,
                system_prompt,
                generate,
                interest_tokens=interest_tokens,
                **inference_kwargs,
            )
            for p in prompts
        ]

    def unsupported_interest_tokens(self, tokens: Sequence[str]) -> list[str]:
        """Subset of ``tokens`` this backend cannot score as a single token.

        Default: assume everything is scorable (unknown strings already get
        0.0 mass under the interest-token contract). Backends with a reverse
        token index (HuggingFace, vLLM) should override this to report token
        strings that no single vocabulary id decodes to.
        """
        return []
