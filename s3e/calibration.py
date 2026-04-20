"""Helpers and data structures for Platt scaling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from PIL.Image import Image

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None  # type: ignore[assignment]


CALIBRATION_SCHEMA_VERSION = 1
SCORE_EPS = 1e-12
GLOBAL_CALIBRATION_KEY = "__global__"


@dataclass(frozen=True)
class CalibrationExample:
    images: list[Image]
    state_dict: dict[str, bool]
    problem: str | None = None


@dataclass(frozen=True)
class PlattParameters:
    a: float
    b: float
    sample_count: int
    positive_count: int
    negative_count: int


@dataclass(frozen=True)
class PlattScalingProfile:
    scope: str
    probability_method: str
    true_tokens: list[str]
    false_tokens: list[str]
    domain_fingerprint: str
    score_kind: str
    groups: dict[str, PlattParameters]
    schema_version: int = CALIBRATION_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "probability_method": self.probability_method,
            "true_tokens": list(self.true_tokens),
            "false_tokens": list(self.false_tokens),
            "domain_fingerprint": self.domain_fingerprint,
            "score_kind": self.score_kind,
            "groups": {
                key: {
                    "a": value.a,
                    "b": value.b,
                    "sample_count": value.sample_count,
                    "positive_count": value.positive_count,
                    "negative_count": value.negative_count,
                }
                for key, value in self.groups.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlattScalingProfile":
        schema_version = int(data["schema_version"])
        if schema_version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported calibration schema version: {schema_version}")
        return cls(
            schema_version=schema_version,
            scope=data["scope"],
            probability_method=data["probability_method"],
            true_tokens=list(data["true_tokens"]),
            false_tokens=list(data["false_tokens"]),
            domain_fingerprint=data["domain_fingerprint"],
            score_kind=data["score_kind"],
            groups={
                key: PlattParameters(
                    a=float(value["a"]),
                    b=float(value["b"]),
                    sample_count=int(value["sample_count"]),
                    positive_count=int(value["positive_count"]),
                    negative_count=int(value["negative_count"]),
                )
                for key, value in data["groups"].items()
            },
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "PlattScalingProfile":
        return cls.from_dict(json.loads(Path(path).read_text()))


def compute_domain_fingerprint(domain: str) -> str:
    return hashlib.sha256(domain.encode("utf-8")).hexdigest()


def grouped_log_odds(
    token_probs: dict[str, float],
    true_tokens: list[str],
    false_tokens: list[str],
    eps: float = SCORE_EPS,
) -> float:
    true_mass = sum(token_probs.get(token, 0.0) for token in true_tokens)
    false_mass = sum(token_probs.get(token, 0.0) for token in false_tokens)
    return math.log((true_mass + eps) / (false_mass + eps))


def apply_platt_scaling(score: float, params: PlattParameters) -> float:
    return 1.0 / (1.0 + math.exp(params.a * score + params.b))
