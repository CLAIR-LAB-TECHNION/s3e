# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""Query translator abstract base class.

A query translator converts grounded PDDL predicate strings into
natural-language (or other) query strings that can be sent to a VLM.
"""

from abc import ABC, abstractmethod


class QueryTranslator(ABC):
    """Abstract base class for predicate-to-query translation."""

    @abstractmethod
    def translate(
        self,
        predicates: list[str],
        domain: str | None = None,
        problem: str | None = None,
    ) -> dict[str, str]:
        """Translate grounded predicates to query strings.

        ``domain``/``problem`` provide optional PDDL context; translators
        that can work without it must accept ``None``.
        """
        ...

    def close(self) -> None:  # noqa: B027 (deliberate no-op default)
        """Release any model the translator holds. It is unusable afterwards."""

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
