"""Calibrator interface: fit offline, apply to prediction sets."""

from abc import ABC, abstractmethod
from pathlib import Path

from ..engine import PredictionSet


class Calibrator(ABC):
    """Transforms a PredictionSet's probabilities; never mutates inputs."""

    @abstractmethod
    def apply(self, results: PredictionSet) -> PredictionSet:
        """Return a new PredictionSet with calibrated probabilities."""

    @abstractmethod
    def save(self, path: "str | Path") -> None:
        """Persist this calibrator to a JSON file."""

    @classmethod
    @abstractmethod
    def load(cls, path: "str | Path") -> "Calibrator":
        """Restore a calibrator persisted by :meth:`save`."""
