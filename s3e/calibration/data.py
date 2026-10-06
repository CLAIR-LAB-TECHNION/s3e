# SPDX-FileCopyrightText: CLAIR Lab Technion
# SPDX-License-Identifier: MIT

"""Calibration data: labeled examples, precomputed samples, and sample sets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from PIL.Image import Image

CALIBRATION_SET_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CalibrationExample:
    """A labeled scene used to collect calibration data.

    Attributes:
        images: The scene, as a list of images shown to the VLM together.
        state_dict: Ground-truth truth value for each grounded predicate to
            collect, e.g. ``{"on(a,b)": True, "clear(b)": False}``.
        problem: Optional PDDL problem string the scene belongs to. When set,
            :meth:`CalibrationSet.collect` re-grounds the estimator against
            this problem before querying it.
    """

    images: list[Image]
    state_dict: dict[str, bool]
    problem: str | None = None


@dataclass(frozen=True)
class CalibrationSample:
    """Precomputed score and label used to fit a calibrator.

    The score is the grouped log-odds value produced by the estimator's
    configured true and false token groups. `problem` should be set when
    the sample came from a problem instance other than the estimator's
    current problem, especially for lifted-scope calibration.
    """

    predicate: str
    score: float
    label: bool
    problem: str | None = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "predicate": self.predicate,
            "score": self.score,
            "label": self.label,
            "problem": self.problem,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationSample":
        """Rebuild a sample from the output of :meth:`to_dict`."""
        return cls(
            predicate=str(data["predicate"]),
            score=float(data["score"]),
            label=bool(data["label"]),
            problem=(
                None
                if data.get("problem") is None
                else str(data["problem"])
            ),
        )


@dataclass
class CalibrationSet:
    """Scores + labels collected once, reusable for any calibrator fit."""

    samples: list[CalibrationSample]
    meta: dict

    @classmethod
    def collect(cls, estimator, examples: list[CalibrationExample]) -> "CalibrationSet":
        """Query the estimator's VLM on labeled examples (the expensive step).

        Examples carrying their own ``problem`` are estimated under that
        problem; the estimator is restored to its original problem before
        returning (even on error).

        Predictions with no answer (``Prediction.has_answer`` is False) are
        skipped: their probability is fixed at 0.5 whatever a calibrator says,
        so they are not valid training points. There can therefore be fewer
        samples than labels.

        Raises:
            ValueError: If the estimator does not use ``scoring="logprobs"`` —
                grouped log-odds scores are not defined for text-match masses.
        """
        meta = estimator.calibration_meta()
        if meta.get("scoring") != "logprobs":
            raise ValueError(
                "CalibrationSet.collect requires an estimator with "
                f"scoring='logprobs'; got scoring={meta.get('scoring')!r} — "
                "grouped log-odds scores are not defined for text-match masses"
            )
        samples: list[CalibrationSample] = []
        original_problem = estimator.problem_pddl
        problem_changed = False
        try:
            for example in examples:
                if example.problem is not None:
                    estimator.set_problem(estimator.domain_pddl, example.problem)
                    problem_changed = True
                results = estimator.estimate(
                    example.images, predicates=list(example.state_dict)
                )
                for predicate, label in example.state_dict.items():
                    prediction = results[predicate]
                    if not prediction.has_answer:
                        continue
                    samples.append(
                        CalibrationSample(
                            predicate=predicate,
                            score=prediction.score,
                            label=bool(label),
                            problem=example.problem,
                        )
                    )
        finally:
            if problem_changed:
                estimator.set_problem(estimator.domain_pddl, original_problem)
        return cls(samples=samples, meta=meta)

    def to_dict(self) -> dict:
        """Serialize to a versioned, JSON-compatible dict."""
        return {
            "format_version": CALIBRATION_SET_FORMAT_VERSION,
            "meta": self.meta,
            "samples": [s.to_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationSet":
        """Rebuild a set from the output of :meth:`to_dict`.

        Raises:
            ValueError: If ``data`` has an unsupported ``format_version``.
        """
        version = data.get("format_version")
        if version != CALIBRATION_SET_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported CalibrationSet format_version: {version!r} "
                f"(expected {CALIBRATION_SET_FORMAT_VERSION})"
            )
        return cls(
            samples=[CalibrationSample.from_dict(s) for s in data["samples"]],
            meta=dict(data.get("meta", {})),
        )

    def save(self, path: "str | Path") -> None:
        """Write this set to a JSON file."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: "str | Path") -> "CalibrationSet":
        """Read a set written by :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
