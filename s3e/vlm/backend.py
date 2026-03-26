"""VLM backend abstract base class and output types.

This module defines the interface that all VLM (Vision-Language Model)
backends must implement, plus the :class:`VLMOutput` dataclass that
standardizes their return values.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from PIL.Image import Image


@dataclass
class VLMOutput:
    """Result from a single VLM query.

    Attributes:
        token_probs: Mapping of token strings to their probabilities.
        text: The generated text response, if available.
    """

    token_probs: dict[str, float] = field(default_factory=dict)
    text: str | None = None


class VLMBackend(ABC):
    """Abstract base class for Vision-Language Model backends.

    Subclasses must implement :meth:`query`. The :meth:`query_batch` method
    has a default sequential implementation that can be overridden.
    """

    @abstractmethod
    def query(
        self,
        images: list[Image],
        prompt: str,
        system_prompt: str | None = None,
    ) -> VLMOutput:
        """Send a single query to the VLM."""
        ...

    def query_batch(
        self,
        images: list[Image],
        prompts: list[str],
        system_prompt: str | None = None,
    ) -> list[VLMOutput]:
        """Send multiple queries against the same set of images.

        Default implementation calls :meth:`query` sequentially.
        """
        return [self.query(images, p, system_prompt) for p in prompts]
