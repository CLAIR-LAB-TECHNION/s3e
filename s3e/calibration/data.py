"""Calibration data: labeled examples, precomputed samples, and sample sets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from PIL.Image import Image

CALIBRATION_SET_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CalibrationExample:
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
        return {
            "predicate": self.predicate,
            "score": self.score,
            "label": self.label,
            "problem": self.problem,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationSample":
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
        """Query the estimator's VLM on labeled examples (the expensive step)."""
        samples: list[CalibrationSample] = []
        for example in examples:
            if example.problem is not None:
                estimator.set_problem(estimator.domain_pddl, example.problem)
            results = estimator.estimate(
                example.images, predicates=list(example.state_dict)
            )
            for predicate, label in example.state_dict.items():
                samples.append(
                    CalibrationSample(
                        predicate=predicate,
                        score=results[predicate].score,
                        label=bool(label),
                        problem=example.problem,
                    )
                )
        return cls(samples=samples, meta=estimator.calibration_meta())

    def to_dict(self) -> dict:
        return {
            "format_version": CALIBRATION_SET_FORMAT_VERSION,
            "meta": self.meta,
            "samples": [s.to_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationSet":
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
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: "str | Path") -> "CalibrationSet":
        return cls.from_dict(json.loads(Path(path).read_text()))
