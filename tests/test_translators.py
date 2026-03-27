"""Tests for query translators."""

import pytest

from s3e.translation.translator import QueryTranslator
from s3e.translation.identity import IdentityTranslator
from s3e.translation.prewritten import PrewrittenTranslator
from s3e.translation.template import TemplateTranslator


SAMPLE_DOMAIN = "(define (domain test) (:types block) (:predicates (on ?x - block ?y - block) (clear ?x - block)))"
SAMPLE_PROBLEM = "(define (problem p1) (:domain test) (:objects a b - block) (:init) (:goal (on a b)))"

SAMPLE_PREDICATES = ["on(a,b)", "on(b,a)", "clear(a)", "clear(b)"]


class TestIdentityTranslator:
    def test_returns_predicates_unchanged(self):
        translator = IdentityTranslator()
        result = translator.translate(SAMPLE_PREDICATES, SAMPLE_DOMAIN, SAMPLE_PROBLEM)
        assert result == {p: p for p in SAMPLE_PREDICATES}

    def test_empty_list(self):
        translator = IdentityTranslator()
        result = translator.translate([], SAMPLE_DOMAIN, SAMPLE_PROBLEM)
        assert result == {}


class TestPrewrittenTranslator:
    def test_returns_provided_queries(self):
        queries = {
            "on(a,b)": "Is A on B?",
            "on(b,a)": "Is B on A?",
            "clear(a)": "Is A clear?",
            "clear(b)": "Is B clear?",
        }
        translator = PrewrittenTranslator(queries)
        result = translator.translate(SAMPLE_PREDICATES, SAMPLE_DOMAIN, SAMPLE_PROBLEM)
        assert result == queries

    def test_raises_on_missing_predicate(self):
        queries = {"on(a,b)": "Is A on B?"}  # missing the rest
        translator = PrewrittenTranslator(queries)
        with pytest.raises(ValueError, match="Missing translations"):
            translator.translate(SAMPLE_PREDICATES, SAMPLE_DOMAIN, SAMPLE_PROBLEM)

    def test_extra_keys_ignored(self):
        queries = {
            "on(a,b)": "Is A on B?",
            "on(b,a)": "Is B on A?",
            "clear(a)": "Is A clear?",
            "clear(b)": "Is B clear?",
            "extra(x)": "Extra?",
        }
        translator = PrewrittenTranslator(queries)
        result = translator.translate(SAMPLE_PREDICATES, SAMPLE_DOMAIN, SAMPLE_PROBLEM)
        assert "extra(x)" not in result
        assert len(result) == 4


class TestTemplateTranslator:
    def test_fills_templates(self):
        templates = {
            "on": "Is {0} on top of {1}?",
            "clear": "Is {0} clear?",
        }
        translator = TemplateTranslator(templates)
        result = translator.translate(SAMPLE_PREDICATES, SAMPLE_DOMAIN, SAMPLE_PROBLEM)
        assert result["on(a,b)"] == "Is a on top of b?"
        assert result["clear(a)"] == "Is a clear?"

    def test_raises_on_missing_template(self):
        templates = {"on": "Is {0} on {1}?"}  # missing "clear"
        translator = TemplateTranslator(templates)
        with pytest.raises(ValueError, match="No template"):
            translator.translate(SAMPLE_PREDICATES, SAMPLE_DOMAIN, SAMPLE_PROBLEM)

    def test_single_arg_predicate(self):
        templates = {"clear": "Is {0} clear?"}
        translator = TemplateTranslator(templates)
        result = translator.translate(["clear(a)"], SAMPLE_DOMAIN, SAMPLE_PROBLEM)
        assert result["clear(a)"] == "Is a clear?"

    def test_no_arg_predicate(self):
        templates = {"done": "Is the task done?"}
        translator = TemplateTranslator(templates)
        result = translator.translate(["done()"], SAMPLE_DOMAIN, SAMPLE_PROBLEM)
        assert result["done()"] == "Is the task done?"


from unittest.mock import patch, MagicMock
from s3e.translation.llm import LLMTranslator


class TestLLMTranslatorMocked:
    """Test LLMTranslator with mocked model calls."""

    @patch("s3e.translation.llm._openai_translate")
    def test_openai_translation(self, mock_translate):
        mock_translate.side_effect = lambda model_id, pred, system, **kw: f"Is {pred} true?"

        translator = LLMTranslator("OpenAI/gpt-4o")
        result = translator.translate(
            ["on(a,b)", "clear(a)"], SAMPLE_DOMAIN, SAMPLE_PROBLEM
        )

        assert len(result) == 2
        assert result["on(a,b)"] == "Is on(a,b) true?"
        assert result["clear(a)"] == "Is clear(a) true?"

    @patch("s3e.translation.llm._openai_translate")
    def test_caching_skips_known_predicates(self, mock_translate, tmp_path):
        mock_translate.side_effect = lambda model_id, pred, system, **kw: f"Q: {pred}"

        cache_dir = str(tmp_path)
        translator = LLMTranslator("OpenAI/gpt-4o", cache_dir=cache_dir)

        # First call: translates both predicates
        result1 = translator.translate(
            ["on(a,b)", "clear(a)"], SAMPLE_DOMAIN, SAMPLE_PROBLEM
        )
        assert mock_translate.call_count == 2

        # Second call: should load from cache
        mock_translate.reset_mock()
        result2 = translator.translate(
            ["on(a,b)", "clear(a)"], SAMPLE_DOMAIN, SAMPLE_PROBLEM
        )
        assert mock_translate.call_count == 0
        assert result2 == result1

    @patch("s3e.translation.llm._openai_translate")
    def test_caching_translates_only_missing(self, mock_translate, tmp_path):
        mock_translate.side_effect = lambda model_id, pred, system, **kw: f"Q: {pred}"

        cache_dir = str(tmp_path)
        translator = LLMTranslator("OpenAI/gpt-4o", cache_dir=cache_dir)

        # First call: translate one predicate
        translator.translate(["on(a,b)"], SAMPLE_DOMAIN, SAMPLE_PROBLEM)
        assert mock_translate.call_count == 1

        # Second call: one cached, one new
        mock_translate.reset_mock()
        result = translator.translate(
            ["on(a,b)", "clear(a)"], SAMPLE_DOMAIN, SAMPLE_PROBLEM
        )
        assert mock_translate.call_count == 1  # only "clear(a)" is new
        assert "on(a,b)" in result
        assert "clear(a)" in result

    @patch("s3e.translation.llm._openai_translate")
    def test_no_cache_dir_skips_caching(self, mock_translate):
        mock_translate.side_effect = lambda model_id, pred, system, **kw: f"Q: {pred}"

        translator = LLMTranslator("OpenAI/gpt-4o", cache_dir=None)
        translator.translate(["on(a,b)"], SAMPLE_DOMAIN, SAMPLE_PROBLEM)

        # Call again — should translate again (no cache)
        mock_translate.reset_mock()
        translator.translate(["on(a,b)"], SAMPLE_DOMAIN, SAMPLE_PROBLEM)
        assert mock_translate.call_count == 1


@pytest.mark.slow
class TestLLMTranslatorIntegration:
    """Integration tests with a tiny real HuggingFace causal LM."""

    TINY_LLM_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    def test_huggingface_translate(self):
        translator = LLMTranslator(self.TINY_LLM_ID, cache_dir=None)
        result = translator.translate(
            ["on(a,b)", "clear(a)"], SAMPLE_DOMAIN, SAMPLE_PROBLEM
        )

        assert isinstance(result, dict)
        assert len(result) == 2
        assert all(isinstance(v, str) and len(v) > 0 for v in result.values())
