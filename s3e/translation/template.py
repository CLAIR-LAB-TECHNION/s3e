"""Template translator — per-predicate-type templates with positional placeholders."""

import re

from .translator import QueryTranslator


def _parse_predicate(predicate_str: str) -> tuple[str, list[str]]:
    """Parse a grounded predicate string into its name and arguments."""
    match = re.match(r"(\w+)\((.*)\)", predicate_str)
    if not match:
        raise ValueError(f"Cannot parse predicate string: {predicate_str!r}")
    name = match.group(1)
    args_str = match.group(2).strip()
    args = [a.strip() for a in args_str.split(",")] if args_str else []
    return name, args


class TemplateTranslator(QueryTranslator):
    """Translator using per-predicate-type templates with positional placeholders."""

    def __init__(self, templates: dict[str, str]):
        self.templates = templates

    def translate(self, predicates, domain, problem):
        result: dict[str, str] = {}
        for pred in predicates:
            name, args = _parse_predicate(pred)
            if name not in self.templates:
                raise ValueError(
                    f"No template for predicate type {name!r}. "
                    f"Available templates: {list(self.templates.keys())}"
                )
            result[pred] = self.templates[name].format(*args)
        return result
