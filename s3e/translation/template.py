"""Template translator — per-predicate-type templates with positional or keyword placeholders."""

import re
import string

from ..pddl.up_utils import create_up_problem
from .translator import QueryTranslator


_FORMATTER = string.Formatter()


def _parse_predicate(predicate_str: str) -> tuple[str, list[str]]:
    """Parse a grounded predicate string into its name and arguments."""
    match = re.match(r"(\w+)\((.*)\)", predicate_str)
    if not match:
        raise ValueError(f"Cannot parse predicate string: {predicate_str!r}")
    name = match.group(1)
    args_str = match.group(2).strip()
    args = [a.strip() for a in args_str.split(",")] if args_str else []
    return name, args


def _predicate_argument_names(domain: str, problem: str) -> dict[str, list[str]]:
    """Build predicate-name to argument-name mapping from the PDDL domain/problem."""
    up_problem = create_up_problem(domain, problem)
    return {
        fluent.name: [parameter.name for parameter in fluent.signature]
        for fluent in up_problem.fluents
    }


def _named_template_fields(template: str) -> list[str]:
    """Return unique named fields from a format template, preserving order."""
    fields: list[str] = []
    for _, field_name, _, _ in _FORMATTER.parse(template):
        if not field_name:
            continue
        root = re.split(r"[.\[]", field_name, maxsplit=1)[0]
        if not root or root.isdigit() or root in fields:
            continue
        fields.append(root)
    return fields


def _build_template_kwargs(
    template: str,
    args: list[str],
    signature_arg_names: list[str],
) -> dict[str, str]:
    """Build kwargs for template formatting.

    Argument names from the PDDL signature are always supported.
    For additional custom field names, remaining arguments are mapped in
    left-to-right placeholder order.
    """
    kwargs = {
        arg_name: arg_value
        for arg_name, arg_value in zip(signature_arg_names, args)
    }
    index_by_arg_name = {
        arg_name: index
        for index, arg_name in enumerate(signature_arg_names)
        if index < len(args)
    }

    named_fields = _named_template_fields(template)
    used_indices = {
        index_by_arg_name[field]
        for field in named_fields
        if field in index_by_arg_name
    }

    remaining_indices = [i for i in range(len(args)) if i not in used_indices]
    remaining_iter = iter(remaining_indices)

    for field in named_fields:
        if field in kwargs:
            continue
        try:
            kwargs[field] = args[next(remaining_iter)]
        except StopIteration:
            break

    return kwargs


class TemplateTranslator(QueryTranslator):
    """Translator using per-predicate templates.

    Supports positional placeholders (e.g., ``{0}``) and keyword placeholders
    (e.g., ``{x}``, ``{name}``). Keyword placeholders can match either the
    predicate argument names from the PDDL signature or custom names mapped
    left-to-right onto predicate arguments.
    """

    def __init__(self, templates: dict[str, str]):
        self.templates = templates

    def translate(self, predicates, domain, problem):
        result: dict[str, str] = {}
        predicate_arg_names = _predicate_argument_names(domain, problem)

        for pred in predicates:
            name, args = _parse_predicate(pred)
            if name not in self.templates:
                raise ValueError(
                    f"No template for predicate type {name!r}. "
                    f"Available templates: {list(self.templates.keys())}"
                )

            template = self.templates[name]
            kwargs = _build_template_kwargs(
                template,
                args,
                predicate_arg_names.get(name, []),
            )
            result[pred] = template.format(*args, **kwargs)

        return result
