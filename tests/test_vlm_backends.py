"""Tests for VLM backends."""

import pytest
from PIL import Image

from s3e.vlm.backend import VLMBackend, VLMOutput


class FakeVLM(VLMBackend):
    """A fake VLM that returns configurable token probabilities."""

    def __init__(self, token_probs: dict[str, float] | None = None, text: str | None = None):
        self.token_probs = token_probs or {"yes": 0.8, "no": 0.2}
        self.text = text
        self.call_count = 0

    def query(self, images, prompt, system_prompt=None):
        self.call_count += 1
        return VLMOutput(token_probs=dict(self.token_probs), text=self.text)


class TestVLMOutput:
    def test_creation(self):
        output = VLMOutput(token_probs={"yes": 0.9, "no": 0.1})
        assert output.token_probs["yes"] == 0.9
        assert output.text is None

    def test_with_text(self):
        output = VLMOutput(token_probs={"yes": 0.9}, text="yes")
        assert output.text == "yes"


class TestVLMBackend:
    def test_query_returns_vlm_output(self):
        vlm = FakeVLM()
        img = Image.new("RGB", (64, 64))
        result = vlm.query([img], "Is block A on block B?")
        assert isinstance(result, VLMOutput)
        assert "yes" in result.token_probs

    def test_query_batch_default_loops(self):
        vlm = FakeVLM()
        img = Image.new("RGB", (64, 64))
        results = vlm.query_batch([img], ["q1", "q2", "q3"])
        assert len(results) == 3
        assert vlm.call_count == 3
        assert all(isinstance(r, VLMOutput) for r in results)

    def test_query_batch_passes_system_prompt(self):
        class TrackingVLM(VLMBackend):
            def __init__(self):
                self.received_system_prompts = []

            def query(self, images, prompt, system_prompt=None):
                self.received_system_prompts.append(system_prompt)
                return VLMOutput(token_probs={"yes": 0.5})

        vlm = TrackingVLM()
        img = Image.new("RGB", (64, 64))
        vlm.query_batch([img], ["q1", "q2"], system_prompt="Be helpful.")
        assert vlm.received_system_prompts == ["Be helpful.", "Be helpful."]
