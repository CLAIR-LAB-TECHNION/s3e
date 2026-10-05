"""Tests for query translators."""

import pytest

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

    def test_fills_templates_with_named_placeholders(self):
        templates = {
            "on": "Is {x} on top of {y}?",
            "clear": "Is {x} clear?",
        }
        translator = TemplateTranslator(templates)
        result = translator.translate(SAMPLE_PREDICATES, SAMPLE_DOMAIN, SAMPLE_PROBLEM)
        assert result["on(a,b)"] == "Is a on top of b?"
        assert result["clear(a)"] == "Is a clear?"

    def test_fills_templates_with_custom_kwarg_name(self):
        domain = """
        (define (domain people)
          (:requirements :typing)
          (:types person)
          (:predicates (hasname ?x - person))
        )
        """
        problem = """
        (define (problem p1)
          (:domain people)
          (:objects alice - person)
          (:init)
          (:goal (hasname alice))
        )
        """
        translator = TemplateTranslator({"hasname": "my name is {name}"})

        result = translator.translate(["hasname(alice)"], domain, problem)
        assert result["hasname(alice)"] == "my name is alice"

    def test_hyphenated_predicate_name(self):
        domain = """
        (define (domain people)
          (:requirements :typing)
          (:types person)
          (:predicates (has-name ?person - person))
        )
        """
        problem = """
        (define (problem p1)
          (:domain people)
          (:objects alice - person)
          (:init)
          (:goal (has-name alice))
        )
        """
        translator = TemplateTranslator({"has-name": "is {person} registered?"})

        result = translator.translate(["has-name(alice)"], domain, problem)
        assert result["has-name(alice)"] == "is alice registered?"

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


from unittest.mock import patch
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


    @patch("s3e.translation.llm._openai_translate")
    def test_pddl_file_paths_put_domain_text_in_prompt(self, mock_translate, tmp_path):
        """from_pddl accepts .pddl paths; the translator's prompt must carry
        the domain's contents, not the path string."""
        mock_translate.side_effect = lambda model_id, pred, system, **kw: f"Q: {pred}"
        domain_file = tmp_path / "domain.pddl"
        problem_file = tmp_path / "problem.pddl"
        domain_file.write_text(SAMPLE_DOMAIN)
        problem_file.write_text(SAMPLE_PROBLEM)

        LLMTranslator("OpenAI/gpt-4o").translate(
            ["on(a,b)"], str(domain_file), str(problem_file)
        )

        system = mock_translate.call_args.args[2]
        assert "(define (domain" in system
        assert str(domain_file) not in system


class TestLLMTranslatorModelCalls:
    """The OpenAI and HuggingFace call paths, with the clients mocked."""

    def test_openai_translate_sends_predicate_and_instructions(self):
        pytest.importorskip("openai")
        from s3e.translation.llm import _openai_translate

        with patch("openai.OpenAI") as mock_client_cls:
            create = mock_client_cls.return_value.responses.create
            create.return_value.output_text = "Is a on b?"
            result = _openai_translate("gpt-4o", "on(a,b)", "SYSTEM", temperature=0)

        assert result == "Is a on b?"
        create.assert_called_once_with(
            input="on(a,b)", model="gpt-4o", instructions="SYSTEM", temperature=0
        )

    @staticmethod
    def _fake_hf(chat_template_error=None):
        """A tokenizer/model pair whose generate() appends tokens [7, 8]."""
        torch = pytest.importorskip("torch")
        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        if chat_template_error is not None:
            tokenizer.apply_chat_template.side_effect = chat_template_error
        else:
            tokenizer.apply_chat_template.return_value = "<chat>"
        encoded = {"input_ids": torch.tensor([[1, 2, 3]])}
        tokenizer.return_value.to.return_value = encoded
        tokenizer.decode.side_effect = lambda ids, **kw: f" decoded {ids.tolist()} "
        model = MagicMock()
        model.generate.return_value = torch.tensor([[1, 2, 3, 7, 8]])
        return tokenizer, model

    def test_huggingface_translate_decodes_only_new_tokens(self):
        from s3e.translation.llm import _huggingface_translate

        tokenizer, model = self._fake_hf()
        result = _huggingface_translate(model, tokenizer, ["on(a,b)"], "SYSTEM")

        assert result == ["decoded [7, 8]"]
        tokenizer.assert_called_once_with("<chat>", return_tensors="pt")
        assert model.generate.call_args.kwargs["max_new_tokens"] == 128

    def test_huggingface_translate_without_chat_template(self):
        from s3e.translation.llm import _huggingface_translate

        tokenizer, model = self._fake_hf(chat_template_error=ValueError("none"))
        _huggingface_translate(model, tokenizer, ["on(a,b)"], "SYSTEM")

        tokenizer.assert_called_once_with("SYSTEM\n\non(a,b)", return_tensors="pt")

    def test_huggingface_model_is_loaded_and_used(self):
        pytest.importorskip("transformers")

        with patch("transformers.AutoTokenizer.from_pretrained") as tok_load, patch(
            "transformers.AutoModelForCausalLM.from_pretrained"
        ) as model_load, patch(
            "s3e.translation.llm._huggingface_translate",
            side_effect=lambda model, tok, preds, system, **kw: [f"Q{p}" for p in preds],
        ) as translate:
            translator = LLMTranslator("org/tiny-llm", temperature=0.1)
            result = translator.translate(
                ["on(a,b)", "clear(a)"], SAMPLE_DOMAIN, SAMPLE_PROBLEM
            )

        tok_load.assert_called_once_with("org/tiny-llm")
        model_load.assert_called_once_with("org/tiny-llm", device_map="auto")
        assert translate.call_args.kwargs == {"temperature": 0.1}
        assert result == {"on(a,b)": "Qon(a,b)", "clear(a)": "Qclear(a)"}


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


class TestPddlFreeTranslation:
    def test_identity_without_domain(self):
        from s3e.translation import IdentityTranslator

        result = IdentityTranslator().translate(["on(a,b)"])
        assert result == {"on(a,b)": "on(a,b)"}

    def test_template_positional_without_domain(self):
        from s3e.translation import TemplateTranslator

        translator = TemplateTranslator({"on": "Is {0} on {1}?"})
        result = translator.translate(["on(a,b)"])
        assert result == {"on(a,b)": "Is a on b?"}

    def test_template_custom_names_without_domain(self):
        from s3e.translation import TemplateTranslator

        translator = TemplateTranslator({"on": "Is {top} on {bottom}?"})
        result = translator.translate(["on(a,b)"])
        assert result == {"on(a,b)": "Is a on b?"}

    def test_prewritten_without_domain(self):
        from s3e.translation import PrewrittenTranslator

        translator = PrewrittenTranslator({"on(a,b)": "Is a on b?"})
        result = translator.translate(["on(a,b)"])
        assert result == {"on(a,b)": "Is a on b?"}

    def test_llm_translator_requires_domain(self):
        from s3e.translation import LLMTranslator

        translator = LLMTranslator.__new__(LLMTranslator)  # skip client setup
        import pytest

        with pytest.raises(ValueError, match="domain"):
            translator.translate(["on(a,b)"])
