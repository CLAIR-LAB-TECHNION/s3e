# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""Prewritten translator — user supplies a complete predicate-to-query mapping."""

from .translator import QueryTranslator


class PrewrittenTranslator(QueryTranslator):
    """Translator using a user-provided dictionary of queries.

    Example:
        >>> from s3e import PrewrittenTranslator
        >>> translator = PrewrittenTranslator({"holding(a)": "Is the robot holding block a?"})
        >>> translator.translate(["holding(a)"])
        {'holding(a)': 'Is the robot holding block a?'}
    """

    def __init__(self, queries: dict[str, str]):
        self.queries = queries

    def translate(self, predicates, domain=None, problem=None):
        """Look up each predicate's prewritten query.

        Raises:
            ValueError: If any predicate has no prewritten query.
        """
        missing = set(predicates) - set(self.queries)
        if missing:
            raise ValueError(
                f"Missing translations for the following predicates: {missing}"
            )
        return {p: self.queries[p] for p in predicates}
