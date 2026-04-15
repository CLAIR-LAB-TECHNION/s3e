"""OpenAI API VLM backend.

This module provides a :class:`VLMBackend` implementation that uses the
OpenAI chat completions API with vision capabilities.
"""

import base64
from collections import defaultdict
from io import BytesIO

import numpy as np

from .backend import VLMBackend, VLMOutput
from ..constants import OPENAI_MODEL_IDENTIFIER

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]


MAX_ALLOWED_OPENAI_LOGPROBS = 20


def _check_openai_installed() -> None:
    if openai is None:
        raise ImportError(
            "The 'openai' package is required for OpenAIVLM. "
            "Install it with: pip install s3e[openai]"
        )


def _preprocess_image(image) -> str:
    """Convert a PIL image to a base64-encoded JPEG string."""
    buffered = BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class OpenAIVLM(VLMBackend):
    """VLM backend using the OpenAI chat completions API.

    Args:
        model_id: OpenAI model identifier (e.g. ``"gpt-4o"``).
            An ``"OpenAI/"`` prefix is stripped automatically.
        **client_kwargs: Additional keyword arguments for the OpenAI client constructor.
    """

    def __init__(self, model_id: str, **client_kwargs):
        _check_openai_installed()
        self.model_id = model_id.removeprefix(OPENAI_MODEL_IDENTIFIER)
        self._client = openai.OpenAI(**client_kwargs)

    def query(self, images, prompt, system_prompt=None, generate=False, **inference_kwargs):
        """Send a query to the OpenAI API."""
        # no need to differentiate between generation and logprobs modes here since the API can return both in one call
        del generate

        self._set_inference_kwargs_defaults(inference_kwargs)

        # Build image content
        image_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_preprocess_image(img)}"},
            }
            for img in images
        ]
        user_content = [{"type": "text", "text": prompt}] + image_content

        messages = []
        if system_prompt is not None:
            messages.append({"role": "developer", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        response = self._client.chat.completions.create(
            messages=messages,
            model=self.model_id,
            **inference_kwargs,
        )

        # Extract token probabilities from first generated token
        token_probs = self._extract_token_probs(response)
        text = response.choices[0].message.content

        return VLMOutput(token_probs=token_probs, text=text)
    
    def _set_inference_kwargs_defaults(self, inference_kwargs):
        # force inference loggprobs regardless of what the user passed.
        inference_kwargs["logprobs"] = True
        inference_kwargs["top_logprobs"] = MAX_ALLOWED_OPENAI_LOGPROBS

        # by default, use deterministic outputs. enable user override
        # Note that we do not set `max_completion_tokens` by default to enable reasoning to proceed as the model sees fit.
        inference_kwargs.setdefault("temperature", 0.0)

    @staticmethod
    def _extract_token_probs(response) -> dict[str, float]:
        """Extract token probabilities from an OpenAI response."""
        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs

        tok_to_prob: dict[str, float] = defaultdict(float)
        for item in top_logprobs:
            tok_to_prob[item.token] += float(np.exp(item.logprob))

        return dict(tok_to_prob)
