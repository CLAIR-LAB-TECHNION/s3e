"""Identity translator — passes predicates through unchanged."""

from .translator import QueryTranslator


class IdentityTranslator(QueryTranslator):
    """Translator that returns predicates as-is (no translation)."""

    def translate(self, predicates, domain, problem):
        return {pred: pred for pred in predicates}
