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

    A prediction has no answer (:attr:`has_answer` is False) when the null
    option is the model's top answer (:attr:`null_dominated`) or no option
    received any mass at all (:attr:`matched` is False). Such predictions are
    uninformative: binary :attr:`probability` is 0.5 and :meth:`distribution`
    is uniform.

    The decision rule never alters stored data. ``masses``, ``null_mass``,
    ``unassigned_mass`` and ``probability_override`` (a calibrator's
    yes-vs-no probability) keep their original values, so other rules can be
    derived from them. For example, counting null mass as half true and half
    false: ``(T + n / 2) / (T + F + n)`` from the raw masses, or
    ``w * probability_override + (1 - w) * 0.5`` with ``w = (T + F) / (T + F + n)``
    for calibrated predictions (fall back to 0.5 when ``T + F + n == 0``).
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
    def matched(self) -> bool:
        """True when any answer option or the null option received mass.

        In ``text_match`` scoring, False means the generated text started with
        none of the answer space's tokens. In ``logprobs`` scoring it means
        every interest token had zero probability, which happens with
        backends that return only the top-k tokens; for full-vocabulary
        backends, ``argmax_in_interest`` and ``unassigned_mass`` are the more
        useful signals of an off-target answer.
        """
        return sum(self.masses.values()) + self.null_mass > 0.0

    @cached_property
    def has_answer(self) -> bool:
        """False when the prediction is null-dominated or unmatched."""
        return self.matched and not self.null_dominated

    @cached_property
    def answer(self):
        """The decided answer; never None.

        For binary spaces, ``probability >= 0.5`` (the state at the default
        confidence). For other spaces, the most likely label in
        :meth:`distribution`; ties, including the uniform distribution of a
        prediction with no answer, go to the first listed option.
        """
        if isinstance(self.answers, BinaryAnswers):
            return bool(self.probability >= 0.5)
        distribution = self.distribution()
        return max(distribution, key=distribution.__getitem__)

    @cached_property
    def probability(self) -> float:
        """P(true) for binary spaces.

        0.5 when the prediction has no answer (see :attr:`has_answer`), even
        if calibrated. Otherwise the calibrated probability when an
        override is set, else ``T / (T + F)`` over the true and false masses.
        """
        self._require_binary("probability")
        if not self.has_answer:
            return 0.5
        if self.probability_override is not None:
            return self.probability_override
        true_mass = self.masses[self.answers.true_label]
        false_mass = self.masses[self.answers.false_label]
        return true_mass / (true_mass + false_mass)

    @cached_property
    def score(self) -> float:
        """Grouped log-odds log(true_mass / false_mass) for binary spaces.

        Computed from the raw masses only (null mass is ignored); this is the
        input calibrators fit on. ``EPS`` keeps it finite for zero masses.
        """
        self._require_binary("score")
        true_mass = self.masses[self.answers.true_label]
        false_mass = self.masses[self.answers.false_label]
        return math.log((true_mass + EPS) / (false_mass + EPS))

    def distribution(self) -> dict[str, float]:
        """Masses normalized over the answer options (uncalibrated).

        Uniform when the prediction has no answer (see :attr:`has_answer`).
        """
        if not self.has_answer:
            uniform = 1.0 / len(self.masses)
            return {label: uniform for label in self.masses}
        total = sum(self.masses.values())
        return {label: mass / total for label, mass in self.masses.items()}

    def confident(self, threshold: float) -> bool:
        """Whether either boolean outcome reaches the threshold (binary only).

        True when P(true) >= threshold or P(false) >= threshold. No range is
        enforced on ``threshold``. Useful for filtering, e.g.
        ``results.where(lambda p: p.confident(0.9))``.
        """
        return bool(
            self.probability >= threshold or (1.0 - self.probability) >= threshold
        )

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

    def to_state(self, confidence: float = 0.5) -> dict[str, bool]:
        """Threshold P(true) into a boolean state (binary answer spaces only).

        ``confidence`` is an acceptance threshold on P(true): a predicate is
        True when P(true) >= confidence and False otherwise, so every
        predicate gets a value. A prediction with no answer (null-dominated or
        unmatched) has P(true) = 0.5, so it is True for confidence <= 0.5 and
        False above. Raises ``ValueError`` for non-binary answer spaces.
        """
        return {
            key: bool(p.probability >= confidence)
            for key, p in self._predictions.items()
        }

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
        """Mean of stored data across prediction sets over the same queries.

        Probability overrides are stored data too: when *every* member of a
        key carries one (e.g. each scene was calibrated before averaging),
        the averaged prediction carries their mean. When only some — or no —
        members have an override, the averaged prediction has none and its
        probability is re-derived from the averaged masses.

        The decision rule is applied after averaging, to the averaged data: a
        member with no answer still contributes its raw masses, and the
        averaged prediction has no answer only if its averaged null mass
        dominates or every member was unmatched. A member's override is not in
        effect when it has no answer, so the override mean skips such members
        (if none has an answer, neither does the average).

        Non-numeric per-member data does not average: ``text`` is dropped,
        and ``argmax_in_interest`` is kept only when every member agrees
        (``None`` otherwise).
        """
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
            overrides = [m.probability_override for m in members]
            answered_overrides = [
                m.probability_override for m in members if m.has_answer
            ]
            argmax_flags = {m.argmax_in_interest for m in members}
            averaged[key] = Prediction(
                query=first.query,
                masses={
                    label: sum(m.masses[label] for m in members) / count
                    for label in first.masses
                },
                null_mass=sum(m.null_mass for m in members) / count,
                unassigned_mass=sum(m.unassigned_mass for m in members) / count,
                answers=first.answers,
                argmax_in_interest=(
                    argmax_flags.pop() if len(argmax_flags) == 1 else None
                ),
                probability_override=(
                    sum(answered_overrides) / len(answered_overrides)
                    if all(o is not None for o in overrides) and answered_overrides
                    else None
                ),
            )
        return cls(averaged)
