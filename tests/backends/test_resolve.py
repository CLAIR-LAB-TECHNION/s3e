"""Tests for the public backend factory."""

import sys

import pytest

from s3e.backends import VLMBackend, VLMOutput, resolve_backend


class DummyBackend(VLMBackend):
    def query(self, images, prompt, system_prompt=None, generate=False,
              interest_tokens=None, **inference_kwargs):
        return VLMOutput()


class TestResolveBackend:
    def test_instance_passes_through(self):
        backend = DummyBackend()
        assert resolve_backend(backend) is backend

    def test_instance_with_vlm_kwargs_rejected(self):
        with pytest.raises(ValueError, match="vlm_kwargs"):
            resolve_backend(DummyBackend(), device_map="auto")

    @pytest.mark.parametrize("vlm", [None, 123, object()])
    def test_invalid_backend_type_rejected(self, vlm):
        with pytest.raises(TypeError, match="model-id string or VLMBackend"):
            resolve_backend(vlm)

    def test_openai_prefix_selects_openai_backend(self, monkeypatch):
        from s3e.backends import openai as openai_module

        captured = {}

        class FakeOpenAIVLM:
            def __init__(self, model, **kwargs):
                captured["model"] = model
                captured["kwargs"] = kwargs

        monkeypatch.setattr(openai_module, "OpenAIVLM", FakeOpenAIVLM)
        resolve_backend("OpenAI/gpt-4o", api_key="k")
        assert captured == {"model": "gpt-4o", "kwargs": {"api_key": "k"}}

    def test_plain_string_selects_huggingface_backend(self, monkeypatch):
        from s3e.backends import huggingface as hf_module

        captured = {}

        class FakeHFVLM:
            def __init__(self, model, **kwargs):
                captured["model"] = model
                captured["kwargs"] = kwargs

        monkeypatch.setattr(hf_module, "HuggingFaceVLM", FakeHFVLM)
        resolve_backend("Qwen/Qwen2-VL-7B-Instruct", device_map="auto")
        assert captured["model"] == "Qwen/Qwen2-VL-7B-Instruct"
        assert captured["kwargs"] == {"device_map": "auto"}


class TestLazyExports:
    def test_base_import_does_not_pull_heavy_modules(self):
        # s3e.backends is already imported by this test module; the assertion
        # is that importing it never dragged torch in transitively.
        import s3e.backends  # noqa: F401
        assert "s3e.backends.base_marker" not in sys.modules  # sanity
        # HuggingFaceVLM is exposed lazily:
        import s3e.backends as b
        assert "HuggingFaceVLM" in b.__all__

    def test_unknown_attribute_raises(self):
        import s3e.backends as b
        with pytest.raises(AttributeError):
            b.NoSuchThing
