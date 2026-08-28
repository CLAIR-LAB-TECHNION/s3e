# tests/backends/test_contract.py
"""Contract tests every VLMBackend implementation must pass.

Concrete backend test modules subclass ``BackendContract`` and provide a
``make_backend`` fixture. This module applies the suite to FakeVLM so the
fake can never drift from the contract real backends implement.
"""

import pytest

from s3e.backends import VLMOutput

from conftest import make_blank_image
from fakes import FakeVLM


class BackendContract:
    """Behavioral contract for VLMBackend implementations."""

    @pytest.fixture
    def images(self):
        return [make_blank_image()]

    def test_query_returns_vlm_output(self, make_backend, images):
        out = make_backend().query(images, "Is it red?")
        assert isinstance(out, VLMOutput)

    def test_interest_tokens_keys_are_exactly_the_request(self, make_backend, images):
        out = make_backend().query(
            images, "Is it red?", interest_tokens=["yes", "no", "zzz_absent"]
        )
        assert set(out.token_probs) == {"yes", "no", "zzz_absent"}
        assert out.token_probs["zzz_absent"] == 0.0
        assert out.argmax_in_interest is not None

    def test_interest_masses_are_probabilities(self, make_backend, images):
        out = make_backend().query(images, "q", interest_tokens=["yes", "no"])
        for mass in out.token_probs.values():
            assert 0.0 <= mass <= 1.0
        assert sum(out.token_probs.values()) <= 1.0 + 1e-9

    def test_query_batch_matches_sequential_query(self, make_backend, images):
        backend = make_backend()
        batch = backend.query_batch(images, ["a", "b"], interest_tokens=["yes", "no"])
        singles = [
            make_backend().query(images, p, interest_tokens=["yes", "no"])
            for p in ("a", "b")
        ]
        assert [o.token_probs for o in batch] == [o.token_probs for o in singles]

    def test_generate_mode_returns_text(self, make_backend, images):
        out = make_backend().query(images, "q", generate=True)
        assert out.text is None or isinstance(out.text, str)

    def test_multi_image_scene_accepted(self, make_backend):
        scene = [make_blank_image(), make_blank_image()]
        out = make_backend().query(scene, "q", interest_tokens=["yes", "no"])
        assert isinstance(out, VLMOutput)


class TestFakeVLMContract(BackendContract):
    @pytest.fixture
    def make_backend(self):
        return lambda: FakeVLM(text="yes")


def test_query_batch_honors_subclass_overriding_only_query():
    """A FakeVLM subclass overriding only query() must be honored by
    query_batch too, not bypassed by a direct output shortcut -- the
    established idiom for one-off fakes elsewhere in this suite."""

    class QueryOnlyFake(FakeVLM):
        def query(self, images, prompt, system_prompt=None, generate=False,
                   interest_tokens=None, **inference_kwargs):
            return VLMOutput(token_probs={"custom": 1.0})

    images = [make_blank_image()]
    results = QueryOnlyFake().query_batch(images, ["a", "b"])

    assert [r.token_probs for r in results] == [
        {"custom": 1.0},
        {"custom": 1.0},
    ]


def test_query_batch_still_records_one_call_per_batch():
    """Routing per-prompt output through query() must not reintroduce a
    per-prompt call entry; query_batch still records exactly one call
    carrying the full batch's prompt list."""
    fake = FakeVLM()
    images = [make_blank_image()]

    fake.query_batch(images, ["a", "b"])

    assert [call["prompts"] for call in fake.calls] == [["a", "b"]]
