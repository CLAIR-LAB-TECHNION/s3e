"""Tests for QueryEngine."""

import pytest

from s3e.engine import BinaryAnswers, CategoricalAnswers, PredictionSet, QueryEngine

from conftest import make_blank_image
from fakes import FakeVLM


@pytest.fixture
def images():
    return [make_blank_image()]


class TestAsk:
    def test_returns_prediction_set_keyed_by_query(self, images):
        engine = QueryEngine(FakeVLM({"yes": 0.8, "no": 0.1}))
        results = engine.ask(images, ["Is a on b?", "Is b clear?"])
        assert isinstance(results, PredictionSet)
        assert list(results) == ["Is a on b?", "Is b clear?"]
        assert results["Is a on b?"].answer is True

    def test_interest_tokens_passed_in_logprobs_mode(self, images):
        fake = FakeVLM()
        QueryEngine(fake).ask(images, ["q"])
        call = fake.calls[0]
        assert call["generate"] is False
        assert "yes" in call["interest_tokens"]
        assert "no" in call["interest_tokens"]

    def test_text_match_mode_generates(self, images):
        fake = FakeVLM(text="Yes.")
        results = QueryEngine(fake, scoring="text_match").ask(images, ["q"])
        assert fake.calls[0]["generate"] is True
        assert fake.calls[0]["interest_tokens"] is None
        assert results["q"].masses["yes"] == 1.0

    def test_prompt_template_applied(self, images):
        fake = FakeVLM()
        QueryEngine(fake, prompt_template="Answer yes or no: {query}").ask(
            images, ["Is it red?"]
        )
        assert fake.calls[0]["prompts"] == ["Answer yes or no: Is it red?"]

    def test_prompt_template_must_contain_query(self):
        with pytest.raises(ValueError, match="{query}"):
            QueryEngine(FakeVLM(), prompt_template="no placeholder")

    def test_system_prompt_forwarded(self, images):
        fake = FakeVLM()
        QueryEngine(fake, system_prompt="Be terse.").ask(images, ["q"])
        assert fake.calls[0]["system_prompt"] == "Be terse."

    def test_per_call_answer_space_override(self, images):
        fake = FakeVLM({"red": 0.6, "green": 0.2})
        engine = QueryEngine(fake)
        results = engine.ask(
            images, ["color?"], answers=CategoricalAnswers(["red", "green"])
        )
        assert results["color?"].answer == "red"

    def test_keep_raw(self, images):
        engine = QueryEngine(FakeVLM())
        assert engine.ask(images, ["q"])["q"].raw is None
        assert engine.ask(images, ["q"], keep_raw=True)["q"].raw is not None

    def test_option_with_no_single_token_form_rejected(self, images):
        class PickyFake(FakeVLM):
            def unsupported_interest_tokens(self, tokens):
                return [t for t in tokens if " " in t.strip()]

        space = CategoricalAnswers(["red", "dark blue"])
        with pytest.raises(ValueError, match="dark blue"):
            QueryEngine(PickyFake()).ask(images, ["color?"], answers=space)


class TestInferenceKwargs:
    def test_defaults_merged_with_per_call(self, images):
        fake = FakeVLM()
        engine = QueryEngine(fake, inference_kwargs={"temperature": 0.2, "seed": 1})
        engine.ask(images, ["q"], inference_kwargs={"seed": 7})
        assert fake.calls[0]["inference_kwargs"] == {"temperature": 0.2, "seed": 7}


class TestBatching:
    def test_queries_chunked_by_batch_size(self, images):
        fake = FakeVLM()
        QueryEngine(fake, batch_size=2).ask(images, ["a", "b", "c", "d", "e"])
        assert [call["prompts"] for call in fake.calls] == [
            ["a", "b"], ["c", "d"], ["e"]
        ]


class TestBackendResolution:
    def test_string_resolves_via_resolve_backend(self, monkeypatch, images):
        import s3e.engine.engine as engine_module

        fake = FakeVLM()
        monkeypatch.setattr(
            engine_module, "resolve_backend", lambda vlm, **kw: fake
        )
        engine = QueryEngine("some/model", vlm_kwargs={"device_map": "auto"})
        assert engine.backend is fake

    def test_instance_used_directly(self):
        fake = FakeVLM()
        assert QueryEngine(fake).backend is fake


class TestAskEach:
    def test_one_prediction_set_per_scene(self):
        fake = FakeVLM()
        scenes = [[make_blank_image()], [make_blank_image()]]
        sets = QueryEngine(fake).ask_each(scenes, ["q"])
        assert len(sets) == 2
        assert all(isinstance(s, PredictionSet) for s in sets)
