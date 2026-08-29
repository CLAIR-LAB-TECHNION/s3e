"""OpenAI API VLM backend.

This module provides a :class:`VLMBackend` implementation that uses the
OpenAI chat completions API with vision capabilities.
"""

import base64
from collections import defaultdict
from io import BytesIO

from .._deps import require

require("openai", "openai", "OpenAIVLM")

import numpy as np
import openai

from .backend import VLMBackend, VLMOutput
from .resolve import OPENAI_MODEL_IDENTIFIER

MAX_ALLOWED_OPENAI_LOGPROBS = 20


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
        self.model_id = model_id.removeprefix(OPENAI_MODEL_IDENTIFIER)
        self._client = openai.OpenAI(**client_kwargs)

    def query(
        self,
        images,
        prompt,
        system_prompt=None,
        generate=False,
        interest_tokens=None,
        **inference_kwargs,
    ):
        """Send a query to the OpenAI API.

        Logprobs mode forces ``logprobs``/``top_logprobs`` on the request;
        generate mode requests neither (models without logprob support, e.g.
        reasoning models, stay usable via ``scoring="text_match"``) and
        returns text only, like the other backends.
        """
        self._set_inference_kwargs_defaults(inference_kwargs, generate)

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
        text = response.choices[0].message.content

        if generate:
            return VLMOutput(token_probs=None, text=text)

        # Extract token probabilities from first generated token
        token_probs, argmax_in_interest = self._extract_token_probs(
            response, interest_tokens
        )

        return VLMOutput(
            token_probs=token_probs,
            text=text,
            argmax_in_interest=argmax_in_interest,
        )

    def _set_inference_kwargs_defaults(self, inference_kwargs, generate):
        # force inference logprobs in logprobs mode regardless of what the
        # user passed; generate mode requests none.
        if not generate:
            inference_kwargs["logprobs"] = True
            inference_kwargs["top_logprobs"] = MAX_ALLOWED_OPENAI_LOGPROBS

        # by default, use deterministic outputs. enable user override
        # Note that we do not set `max_completion_tokens` by default to enable reasoning to proceed as the model sees fit.
        inference_kwargs.setdefault("temperature", 0.0)

    @staticmethod
    def _extract_token_probs(
        response, interest_tokens=None
    ) -> tuple[dict[str, float], bool | None]:
        """Extract token probabilities from an OpenAI response.

        Without ``interest_tokens``, returns every returned token string and
        ``None`` for the argmax flag. With ``interest_tokens``, returns
        exactly those tokens (0.0 when absent from the returned top
        logprobs) plus whether the single highest-logprob entry is an
        interest token.

        Raises:
            ValueError: If the response carries no logprobs — some models
                (e.g. reasoning models) do not support them.
        """
        logprobs = response.choices[0].logprobs
        if logprobs is None or not logprobs.content:
            raise ValueError(
                "OpenAI response carries no logprobs; the model may not "
                "support them. Use scoring='text_match' or a model that "
                "returns logprobs."
            )
        top_logprobs = logprobs.content[0].top_logprobs

        tok_to_prob: dict[str, float] = defaultdict(float)
        for item in top_logprobs:
            tok_to_prob[item.token] += float(np.exp(item.logprob))

        if interest_tokens is None:
            return dict(tok_to_prob), None

        # dict.fromkeys dedups while preserving the caller's ordering.
        interest = list(dict.fromkeys(interest_tokens))
        token_probs = {token: tok_to_prob.get(token, 0.0) for token in interest}
        # max over entries, not list order: the API does not guarantee that
        # top_logprobs is sorted.
        best = max(top_logprobs, key=lambda item: item.logprob)
        return token_probs, best.token in set(interest)
