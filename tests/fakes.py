# tests/fakes.py
"""Shared fake VLM backend implementing the full VLMBackend contract.

Used by engine, estimator, calibration, and consumer tests. Honors the
interest_tokens contract: when interest tokens are requested, the returned
token_probs contains exactly those keys (absent tokens get 0.0).
"""

from s3e.backends import VLMBackend, VLMOutput


class FakeVLM(VLMBackend):
    """Deterministic fake backend.

    Args:
        token_probs: Default token-string -> probability mapping returned
            for every query (before interest-token filtering).
        text: Default generated text returned when ``generate=True``.
        argmax_in_interest: Value reported when interest tokens are given.
    """

    def __init__(self, token_probs=None, text=None, argmax_in_interest=True):
        self.token_probs = dict(token_probs or {"yes": 0.7, "no": 0.2})
        self.text = text
        self.argmax_in_interest = argmax_in_interest
        self.calls: list[dict] = []
        self._scripted: dict[str, dict] = {}

    def script_responses(self, mapping: dict[str, dict]) -> None:
        """Per-query overrides: prompt-substring -> token_probs mapping."""
        self._scripted.update(mapping)

    def _probs_for(self, prompt: str) -> dict[str, float]:
        for needle, probs in self._scripted.items():
            if needle in prompt:
                return dict(probs)
        return dict(self.token_probs)

    def query(self, images, prompt, system_prompt=None, generate=False,
              interest_tokens=None, **inference_kwargs):
        self.calls.append(
            {
                "images": list(images),
                "prompts": [prompt],
                "system_prompt": system_prompt,
                "generate": generate,
                "interest_tokens": (
                    None if interest_tokens is None else list(interest_tokens)
                ),
                "inference_kwargs": dict(inference_kwargs),
            }
        )
        probs = self._probs_for(prompt)
        if interest_tokens is not None:
            token_probs = {t: probs.get(t, 0.0) for t in interest_tokens}
            argmax = self.argmax_in_interest
        else:
            token_probs = probs
            argmax = None
        return VLMOutput(
            token_probs=token_probs,
            text=self.text if generate else None,
            argmax_in_interest=argmax,
        )
