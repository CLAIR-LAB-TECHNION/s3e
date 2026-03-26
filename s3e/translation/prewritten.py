"""Prewritten translator — user supplies a complete predicate-to-query mapping."""

from .translator import QueryTranslator


class PrewrittenTranslator(QueryTranslator):
    """Translator using a user-provided dictionary of queries."""

    def __init__(self, queries: dict[str, str]):
        self.queries = queries

    def translate(self, predicates, domain, problem):
        missing = set(predicates) - set(self.queries)
        if missing:
            raise ValueError(
                f"Missing translations for the following predicates: {missing}"
            )
        return {p: self.queries[p] for p in predicates}
