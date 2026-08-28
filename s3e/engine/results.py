"""Lazy prediction objects: store masses, derive everything else on demand."""

import math
from collections.abc import Iterator, Mapping, Sequence
from functools import cached_property

from .answers import AnswerSpace, BinaryAnswers

EPS = 1e-12
PREDICTION_SET_FORMAT_VERSION = 1


class Prediction:
    """One query's outcome. Immutable; derived values are cached lazily.

    Stores per-option probability masses, the explicit-null option's mass,
    and the unassigned remainder. ``raw`` holds the backend
    :class:`~s3e.backends.VLMOutput` only when the engine was asked to keep
    it and is never serialized.
    """

    def __init__(
        self,
        query: str,
        masses: Mapping[str, float],
        null_mass: float,
        unassigned_mass: float,
        answers: AnswerSpace,
        *,
        text: "str | None" = None,
        argmax_in_interest: "bool | None" = None,
        raw=None,
        probability_override: "float | None" = None,
    ):
        self.query = query
        self.masses = dict(masses)
        self.null_mass = null_mass
        self.unassigned_mass = unassigned_mass
        self.answers = answers
        self.text = text
        self.argmax_in_interest = argmax_in_interest
        self.raw = raw
        self.probability_override = probability_override

    @cached_property
    def null_dominated(self) -> bool:
        """True when the explicit null option out-masses every answer option."""
        if not self.masses:
            return False
        return self.null_mass > max(self.masses.values())

    @cached_property
    def answer(self):
        """Argmax label; a bool for binary spaces; None when null-dominated."""
        if self.null_dominated:
            return None
        label = max(self.masses, key=self.masses.__getitem__)
        if isinstance(self.answers, BinaryAnswers):
            return label == self.answers.true_label
        return label

    @cached_property
    def probability(self) -> float:
        """Normalized P(true) for binary spaces (override wins when set)."""
        if self.probability_override is not None:
            return self.probability_override
        self._require_binary("probability")
        true_mass = self.masses[self.answers.true_label]
        false_mass = self.masses[self.answers.false_label]
        return (true_mass + EPS) / (true_mass + false_mass + 2 * EPS)

    @cached_property
    def score(self) -> float:
        """Grouped log-odds log(true_mass / false_mass) for binary spaces."""
        self._require_binary("score")
        true_mass = self.masses[self.answers.true_label]
        false_mass = self.masses[self.answers.false_label]
        return math.log((true_mass + EPS) / (false_mass + EPS))

    def distribution(self) -> dict[str, float]:
        """Masses normalized over the answer options."""
        total = sum(self.masses.values())
        if total <= 0.0:
            uniform = 1.0 / len(self.masses)
            return {label: uniform for label in self.masses}
        return {label: mass / total for label, mass in self.masses.items()}

    def confident(self, threshold: float) -> bool:
        """Whether either boolean outcome reaches the threshold."""
        return self.probability >= threshold or (1.0 - self.probability) >= threshold

    def with_probability(self, probability: float) -> "Prediction":
        """Copy of this prediction with an overriding probability (calibration)."""
        return Prediction(
            self.query,
            self.masses,
            self.null_mass,
            self.unassigned_mass,
            self.answers,
            text=self.text,
            argmax_in_interest=self.argmax_in_interest,
            raw=self.raw,
            probability_override=probability,
        )

    def _require_binary(self, what: str) -> None:
        if not isinstance(self.answers, BinaryAnswers):
            raise ValueError(
                f"{what} is only defined for binary answer spaces; "
                f"this prediction uses {type(self.answers).__name__}"
            )

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "masses": dict(self.masses),
            "null_mass": self.null_mass,
            "unassigned_mass": self.unassigned_mass,
            "text": self.text,
            "argmax_in_interest": self.argmax_in_interest,
            "probability_override": self.probability_override,
            "answers": self.answers.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Prediction":
        return cls(
            query=data["query"],
            masses=data["masses"],
            null_mass=data["null_mass"],
            unassigned_mass=data["unassigned_mass"],
            answers=AnswerSpace.from_dict(data["answers"]),
            text=data.get("text"),
            argmax_in_interest=data.get("argmax_in_interest"),
            probability_override=data.get("probability_override"),
        )


class PredictionSet(Mapping):
    """Ordered mapping of query (or predicate) to :class:`Prediction`."""

    def __init__(self, predictions: Mapping[str, Prediction]):
        self._predictions = dict(predictions)

    def __getitem__(self, key: str) -> Prediction:
        return self._predictions[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._predictions)

    def __len__(self) -> int:
        return len(self._predictions)

    def probabilities(self) -> dict[str, float]:
        """Per-query P(true) (binary spaces)."""
        return {key: p.probability for key, p in self._predictions.items()}

    def to_state(self, confidence: float = 0.5) -> "dict[str, bool | None]":
        """Threshold probabilities into a three-valued boolean state.

        True when P(true) >= confidence, False when P(false) >= confidence,
        None otherwise or when the prediction is null-dominated.
        """
        state: dict[str, bool | None] = {}
        for key, p in self._predictions.items():
            if p.null_dominated:
                state[key] = None
            elif p.probability >= confidence:
                state[key] = True
            elif (1.0 - p.probability) >= confidence:
                state[key] = False
            else:
                state[key] = None
        return state

    def where(self, predicate) -> "PredictionSet":
        """Subset of predictions for which ``predicate(prediction)`` is true."""
        return PredictionSet(
            {k: p for k, p in self._predictions.items() if predicate(p)}
        )

    def to_dict(self) -> dict:
        return {
            "format_version": PREDICTION_SET_FORMAT_VERSION,
            "predictions": {k: p.to_dict() for k, p in self._predictions.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PredictionSet":
        version = data.get("format_version")
        if version != PREDICTION_SET_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported PredictionSet format_version: {version!r} "
                f"(expected {PREDICTION_SET_FORMAT_VERSION})"
            )
        return cls(
            {k: Prediction.from_dict(d) for k, d in data["predictions"].items()}
        )

    @classmethod
    def average(cls, sets: "Sequence[PredictionSet]") -> "PredictionSet":
        """Mean of stored masses across prediction sets over the same queries."""
        if not sets:
            raise ValueError("Expected at least one PredictionSet to average.")
        keys = list(sets[0])
        for other in sets[1:]:
            if list(other) != keys:
                raise ValueError("All PredictionSets must cover the same queries.")
        count = len(sets)
        averaged: dict[str, Prediction] = {}
        for key in keys:
            members = [s[key] for s in sets]
            first = members[0]
            averaged[key] = Prediction(
                query=first.query,
                masses={
                    label: sum(m.masses[label] for m in members) / count
                    for label in first.masses
                },
                null_mass=sum(m.null_mass for m in members) / count,
                unassigned_mass=sum(m.unassigned_mass for m in members) / count,
                answers=first.answers,
                argmax_in_interest=first.argmax_in_interest,
            )
        return cls(averaged)
