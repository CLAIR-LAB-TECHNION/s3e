"""Base classes for state estimation.

This module provides abstract base classes for state estimation in
environments described by PDDL. State estimators convert visual
observations (images) into dictionaries of boolean predicate values.

Classes:
    StateEstimator: Abstract base class for state estimation.
    ProbabilisticStateEstimator: Adds probability-based estimation
        with confidence thresholding.
"""

from abc import ABC, abstractmethod

from PIL.Image import Image

from .pddl.up_utils import create_up_problem


class StateEstimator(ABC):
    """Abstract base class for state estimation.

    A state estimator takes a set of images and produces a dictionary
    mapping grounded PDDL predicate strings to boolean truth values.

    Attributes:
        up_problem: The Unified Planning problem instance.

    Args:
        domain: PDDL domain as a file path or string.
        problem: PDDL problem as a file path or string.
    """

    def __init__(self, domain: str, problem: str):
        self.up_problem = create_up_problem(domain, problem)

    def swap_problem(self, domain: str, problem: str) -> None:
        """Update the planning problem with a new domain and problem.

        Args:
            domain: New PDDL domain (file path or string).
            problem: New PDDL problem (file path or string).
        """
        self.up_problem = create_up_problem(domain, problem)

    @abstractmethod
    def __call__(self, images: list[Image]) -> dict[str, bool]:
        """Estimate the current state from images.

        Args:
            images: List of PIL images representing the current
                environment state.

        Returns:
            Dictionary mapping predicate strings (e.g. ``"on(a,b)"``)
            to boolean values.
        """
        ...


class ProbabilisticStateEstimator(StateEstimator, ABC):
    """Abstract base class for probabilistic state estimation.

    Extends :class:`StateEstimator` with probability-based estimation.
    Subclasses implement :meth:`estimate_probabilities` which returns
    per-predicate probabilities; the :meth:`__call__` method thresholds
    these into booleans.

    Attributes:
        confidence: Default confidence threshold for converting
            probabilities to boolean values.

    Args:
        domain: PDDL domain as a file path or string.
        problem: PDDL problem as a file path or string.
        confidence: Default confidence threshold (default: 0.5).
    """

    def __init__(self, domain: str, problem: str, confidence: float = 0.5):
        super().__init__(domain, problem)
        self.confidence = confidence

    def __call__(
        self, images: list[Image], confidence: float | None = None
    ) -> dict[str, bool]:
        """Estimate state as boolean predicates via thresholded probabilities.

        Args:
            images: List of PIL images.
            confidence: Optional confidence threshold override.

        Returns:
            Dictionary mapping predicate strings to boolean values.
        """
        probs = self.estimate_probabilities(images)
        threshold = confidence if confidence is not None else self.confidence
        return {pred: bool(prob >= threshold) for pred, prob in probs.items()}

    @abstractmethod
    def estimate_probabilities(self, images: list[Image]) -> dict[str, float]:
        """Estimate per-predicate probabilities from images.

        Args:
            images: List of PIL images.

        Returns:
            Dictionary mapping predicate strings to probability values
            in ``[0, 1]``.
        """
        ...
