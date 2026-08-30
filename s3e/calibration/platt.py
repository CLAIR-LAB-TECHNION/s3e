"""Platt scaling: fit a sigmoid over grouped log-odds scores, then apply it."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from ..engine.results import EPS
from .base import Calibrator

GLOBAL_CALIBRATION_KEY = "__global__"


@dataclass(frozen=True)
class PlattParameters:
    a: float
    b: float
    sample_count: int
    positive_count: int
    negative_count: int


def grouped_log_odds(
    token_probs: dict[str, float],
    true_tokens: list[str],
    false_tokens: list[str],
    eps: float = EPS,
) -> float:
    true_mass = sum(token_probs.get(token, 0.0) for token in true_tokens)
    false_mass = sum(token_probs.get(token, 0.0) for token in false_tokens)
    return math.log((true_mass + eps) / (false_mass + eps))


def apply_platt_scaling(score: float, params: PlattParameters) -> float:
    """Apply the fitted sigmoid without overflowing for extreme logits."""
    z = params.a * score + params.b
    if z >= 0.0:
        exp_neg_z = math.exp(-z)
        return exp_neg_z / (1.0 + exp_neg_z)
    return 1.0 / (1.0 + math.exp(z))


def fit_platt_parameters(scores: list[float], labels: list[bool]) -> PlattParameters:
    from .._deps import require

    require("sklearn", "calibration", "Platt scaling fitting")
    from sklearn.linear_model import LogisticRegression

    if not scores:
        raise ValueError("Expected at least one calibration sample.")

    positives = sum(bool(label) for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Platt scaling requires both positive and negative labels.")

    model = LogisticRegression(random_state=0)
    model.fit([[score] for score in scores], labels)

    coef = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    return PlattParameters(
        a=-coef,
        b=-intercept,
        sample_count=len(scores),
        positive_count=positives,
        negative_count=negatives,
    )


VALID_SCOPES = ("global", "lifted", "grounded")


def _group_key(predicate: str, scope: str) -> str:
    if scope == "global":
        return GLOBAL_CALIBRATION_KEY
    if scope == "lifted":
        return predicate.split("(", 1)[0]
    return predicate


class PlattCalibrator(Calibrator):
    """Per-group Platt scaling fitted on grouped log-odds scores."""

    PLATT_FORMAT_VERSION = 1

    def __init__(self, scope: str, groups: dict[str, PlattParameters], meta: dict):
        self.scope = scope
        self.groups = groups
        self.meta = meta

    @classmethod
    def fit(
        cls,
        data: "CalibrationSet",
        scope: str = "global",
        pass_through_single_class: bool = False,
    ) -> "PlattCalibrator":
        if scope not in VALID_SCOPES:
            raise ValueError(f"Unknown scope {scope!r}; expected one of {VALID_SCOPES}")
        scoring = data.meta.get("scoring")
        if scoring is not None and scoring != "logprobs":
            raise ValueError(
                f"CalibrationSet meta records scoring={scoring!r}; Platt "
                "scaling requires scores collected with scoring='logprobs'"
            )
        grouped: dict[str, list] = {}
        for sample in data.samples:
            grouped.setdefault(_group_key(sample.predicate, scope), []).append(sample)
        groups: dict[str, PlattParameters] = {}
        for key, samples in grouped.items():
            labels = [s.label for s in samples]
            if len(set(labels)) < 2:
                if pass_through_single_class:
                    continue
                raise ValueError(
                    f"Calibration group {key!r} has only "
                    f"{'positive' if all(labels) else 'negative'} samples; "
                    "Platt scaling requires both positive and negative labels "
                    "(or pass pass_through_single_class=True to leave such "
                    "groups uncalibrated)"
                )
            groups[key] = fit_platt_parameters([s.score for s in samples], labels)
        return cls(scope=scope, groups=groups, meta=dict(data.meta))

    def group_keys(self) -> list[str]:
        return list(self.groups)

    def apply(self, results: "PredictionSet") -> "PredictionSet":
        calibrated = {}
        for key, prediction in results.items():
            params = self.groups.get(_group_key(key, self.scope))
            if params is None:
                calibrated[key] = prediction
            else:
                calibrated[key] = prediction.with_probability(
                    apply_platt_scaling(prediction.score, params)
                )
        return type(results)(calibrated)

    def save(self, path: str | Path) -> None:
        payload = {
            "format_version": self.PLATT_FORMAT_VERSION,
            "kind": "platt",
            "scope": self.scope,
            "meta": self.meta,
            "groups": {
                key: {
                    "a": params.a,
                    "b": params.b,
                    "sample_count": params.sample_count,
                    "positive_count": params.positive_count,
                    "negative_count": params.negative_count,
                }
                for key, params in self.groups.items()
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "PlattCalibrator":
        data = json.loads(Path(path).read_text())
        version = data.get("format_version")
        if version != cls.PLATT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported PlattCalibrator format_version: {version!r} "
                f"(expected {cls.PLATT_FORMAT_VERSION})"
            )
        return cls(
            scope=data["scope"],
            groups={
                key: PlattParameters(
                    a=float(g["a"]),
                    b=float(g["b"]),
                    sample_count=int(g["sample_count"]),
                    positive_count=int(g["positive_count"]),
                    negative_count=int(g["negative_count"]),
                )
                for key, g in data["groups"].items()
            },
            meta=data.get("meta", {}),
        )
