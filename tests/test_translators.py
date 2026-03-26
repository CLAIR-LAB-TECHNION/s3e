"""Tests for query translators."""

import pytest

from s3e.translation.translator import QueryTranslator
from s3e.translation.identity import IdentityTranslator
from s3e.translation.prewritten import PrewrittenTranslator
from s3e.translation.template import TemplateTranslator


SAMPLE_DOMAIN = "(define (domain test) (:predicates (on ?x - block ?y - block) (clear ?x - block)))"
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
