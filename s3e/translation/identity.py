# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""Identity translator — passes predicates through unchanged."""

from .translator import QueryTranslator


class IdentityTranslator(QueryTranslator):
    """Translator that returns predicates as-is (no translation)."""

    def translate(self, predicates, domain=None, problem=None):
        """Map each predicate to itself; ``domain`` and ``problem`` are ignored."""
        return {pred: pred for pred in predicates}
