"""Answer spaces: what counts as an answer and how model output scores it.

An :class:`AnswerOption` is a label plus the token strings that express it.
An :class:`AnswerSpace` is an ordered set of options (plus an optional
explicit null/abstain option) that can score a :class:`VLMOutput` either
from token masses ("logprobs") or from generated text ("text_match").
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..backends import VLMOutput

SCORING_MODES = ("logprobs", "text_match")


def _is_word_in_text(word: str, text: str) -> bool:
    """Check if word appears as a complete word in text (with word boundaries)."""
    return bool(re.search(r"\b" + re.escape(word) + r"\b", text))


def expand_token_variants(label: str) -> tuple[str, ...]:
    """Case and leading-space variants of a label, label's own casing first."""
    seen: list[str] = []
    for base in (label, label.lower(), label.capitalize(), label.upper()):
        for variant in (base, " " + base):
            if variant not in seen:
                seen.append(variant)
    return tuple(seen)


@dataclass(frozen=True)
class AnswerOption:
    """One admissible answer: a label plus its accepted token strings."""

    label: str
    tokens: tuple[str, ...]

    @classmethod
    def make(cls, label: str, tokens: "Sequence[str] | None" = None) -> "AnswerOption":
        """Build an option, auto-expanding the label when tokens are omitted."""
        if tokens is None:
            return cls(label, expand_token_variants(label))
        return cls(label, tuple(tokens))


@dataclass(frozen=True)
class ScoredMasses:
    """Raw scoring result: per-option masses plus null and unassigned mass."""

    masses: dict[str, float]
    null_mass: float
    unassigned_mass: float


class AnswerSpace:
    """Ordered set of answer options with scoring behavior."""

    def __init__(
        self,
        options: Sequence[AnswerOption],
        null_option: "AnswerOption | None" = None,
    ):
        self.options = tuple(options)
        if not self.options:
            raise ValueError("An answer space needs at least one option.")
        self.null_option = null_option
        labels = [o.label for o in self.options]
        if len(set(labels)) != len(labels):
            raise ValueError(f"Duplicate answer labels: {labels}")
        all_options = self.options + ((null_option,) if null_option else ())
        seen_tokens: dict[str, str] = {}
        for option in all_options:
            for token in option.tokens:
                if token in seen_tokens and seen_tokens[token] != option.label:
                    raise ValueError(
                        f"Token {token!r} would overlap between options "
                        f"{seen_tokens[token]!r} and {option.label!r}"
                    )
                seen_tokens[token] = option.label

    @property
    def labels(self) -> list[str]:
        return [o.label for o in self.options]

    @property
    def interest_tokens(self) -> list[str]:
        tokens: list[str] = []
        for option in self.options + (
            (self.null_option,) if self.null_option else ()
        ):
            tokens.extend(option.tokens)
        return tokens

    def score(self, output: VLMOutput, scoring: str) -> ScoredMasses:
        """Score a backend output into per-option masses."""
        if scoring == "logprobs":
            return self._score_logprobs(output)
        if scoring == "text_match":
            return self._score_text(output)
        raise ValueError(
            f"Unknown scoring mode {scoring!r}; expected one of {SCORING_MODES}"
        )

    def _score_logprobs(self, output: VLMOutput) -> ScoredMasses:
        probs = output.token_probs
        masses = {
            o.label: sum(probs.get(t, 0.0) for t in o.tokens) for o in self.options
        }
        null_mass = (
            sum(probs.get(t, 0.0) for t in self.null_option.tokens)
            if self.null_option
            else 0.0
        )
        unassigned = max(0.0, 1.0 - sum(masses.values()) - null_mass)
        return ScoredMasses(masses=masses, null_mass=null_mass, unassigned_mass=unassigned)

    def _score_text(self, output: VLMOutput) -> ScoredMasses:
        text = output.text or ""
        matched: "str | None" = None
        null_matched = False
        candidates = list(self.options) + (
            [self.null_option] if self.null_option else []
        )
        for option in candidates:
            if any(token.strip() and _is_word_in_text(token.strip(), text) for token in option.tokens):
                if self.null_option and option is self.null_option:
                    null_matched = True
                else:
                    matched = option.label
                break
        masses = {o.label: (1.0 if o.label == matched else 0.0) for o in self.options}
        null_mass = 1.0 if null_matched else 0.0
        unassigned = 0.0 if (matched or null_matched) else 1.0
        return ScoredMasses(masses=masses, null_mass=null_mass, unassigned_mass=unassigned)

    def to_dict(self) -> dict:
        return {
            "type": "categorical",
            "options": [
                {"label": o.label, "tokens": list(o.tokens)} for o in self.options
            ],
            "null_option": (
                {
                    "label": self.null_option.label,
                    "tokens": list(self.null_option.tokens),
                }
                if self.null_option
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnswerSpace":
        if data["type"] == "binary":
            return BinaryAnswers._from_dict(data)
        if data["type"] == "categorical":
            return CategoricalAnswers._from_dict(data)
        raise ValueError(f"Unknown answer space type: {data['type']!r}")


def _null_option_from(
    null_label: str, null_tokens: "Sequence[str] | None"
) -> "AnswerOption | None":
    if null_tokens is None:
        return None
    return AnswerOption.make(null_label, null_tokens)


class BinaryAnswers(AnswerSpace):
    """Two-option answer space with boolean semantics (true first)."""

    def __init__(
        self,
        true_label: str = "yes",
        false_label: str = "no",
        *,
        true_tokens: "Sequence[str] | None" = None,
        false_tokens: "Sequence[str] | None" = None,
        null_label: str = "unknown",
        null_tokens: "Sequence[str] | None" = None,
    ):
        self.true_label = true_label
        self.false_label = false_label
        super().__init__(
            [
                AnswerOption.make(true_label, true_tokens),
                AnswerOption.make(false_label, false_tokens),
            ],
            null_option=_null_option_from(null_label, null_tokens),
        )

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["type"] = "binary"
        data["true_label"] = self.true_label
        data["false_label"] = self.false_label
        return data

    @classmethod
    def _from_dict(cls, data: dict) -> "BinaryAnswers":
        options = data["options"]
        null = data.get("null_option")
        return cls(
            data["true_label"],
            data["false_label"],
            true_tokens=options[0]["tokens"],
            false_tokens=options[1]["tokens"],
            null_label=null["label"] if null else "unknown",
            null_tokens=null["tokens"] if null else None,
        )


class CategoricalAnswers(AnswerSpace):
    """N-option answer space built from labels or explicit options."""

    def __init__(
        self,
        options: "Sequence[str | AnswerOption]",
        *,
        null_label: str = "unknown",
        null_tokens: "Sequence[str] | None" = None,
    ):
        built = [
            o if isinstance(o, AnswerOption) else AnswerOption.make(o)
            for o in options
        ]
        super().__init__(built, null_option=_null_option_from(null_label, null_tokens))

    @classmethod
    def _from_dict(cls, data: dict) -> "CategoricalAnswers":
        null = data.get("null_option")
        return cls(
            [AnswerOption(o["label"], tuple(o["tokens"])) for o in data["options"]],
            null_label=null["label"] if null else "unknown",
            null_tokens=null["tokens"] if null else None,
        )
