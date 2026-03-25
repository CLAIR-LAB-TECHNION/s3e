"""Base classes for state estimation in semantic environments.

This module provides abstract base classes for state estimation in semantic environments.
State estimators are responsible for determining the current state of the environment
based on observations (typically images) and converting them into a format compatible
with planning systems.

Classes:
    StateEstimator: Abstract base class for state estimation.
    ProbabilisticStateEstimator: Abstract base class for probabilistic state estimation.
    PredFnStateEstimator: Abstract base class for predicate function-based state estimation.
"""

from abc import ABC, abstractmethod

from PIL.Image import Image
from unified_planning.model import FNode

from .utils.up_utils import create_up_problem, bool_constant_to_fnode


class StateEstimator(ABC):
    """Abstract base class for state estimation.
    
    This class provides the basic interface for state estimation in semantic environments.
    It handles the creation and management of the unified planning problem.

    Attributes:
        up_problem: The unified planning problem instance.

    Args:
        domain: The PDDL domain description.
        problem: The PDDL problem description.
    """

    def __init__(
            self,
            domain,
            problem,
    ):
        self.up_problem = create_up_problem(domain, problem)

    def swap_queries(self, domain, problem, *args, **kwargs):
        """Update the current planning problem with new domain and problem descriptions.
        
        Args:
            domain: New PDDL domain description.
            problem: New PDDL problem description.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.up_problem = create_up_problem(domain, problem)

    @abstractmethod
    def __call__(self, images: list[Image]) -> dict[str, bool]:
        """Estimate the current state from a list of images.
        
        Args:
            images: List of PIL Image objects representing the current environment state.

        Returns:
            Dictionary mapping predicate strings to boolean values representing the estimated state.
        """
        ...


class ProbabilisticStateEstimator(StateEstimator, ABC):
    """Abstract base class for probabilistic state estimation.
    
    This class extends StateEstimator to provide probabilistic state estimation capabilities.
    It allows for confidence-based thresholding of probabilistic estimates.

    Attributes:
        confidence: Default confidence threshold for converting probabilities to boolean values.

    Args:
        domain: The PDDL domain description.
        problem: The PDDL problem description.
        confidence: Default confidence threshold (default: 0.5).
    """

    def __init__(self, domain, problem, confidence: float = 0.5):
        super().__init__(domain, problem)
        self.confidence = confidence

    def __call__(self, images: list[Image], confidence=None) -> dict[str, bool]:
        """Estimate the current state from images with probabilistic confidence.
        
        Args:
            images: List of PIL Image objects representing the current environment state.
            confidence: Optional confidence threshold to override the default value.

        Returns:
            Dictionary mapping predicate strings to boolean values based on probability thresholding.
        """
        fluent_prob_map = self.estimate_state(images)

        # set predefined confidence if not provided
        if confidence is None:
            confidence = self.confidence

        state = {
            predicate: bool(prob >= confidence)  # force bool (not np.bool_)
            for predicate, prob in fluent_prob_map.items()
        }

        return state

    @abstractmethod
    def estimate_state(self, images: list[Image]) -> dict[str, float]:
        """Estimate the current state probabilities from images.
        
        Args:
            images: List of PIL Image objects representing the current environment state.

        Returns:
            Dictionary mapping predicate strings to probability values.
        """
        ...


class PredFnStateEstimator(StateEstimator, ABC):
    """Abstract base class for predicate function-based state estimation.
    
    This class extends StateEstimator to provide state estimation based on predicate functions.
    It automatically maps PDDL predicates to corresponding Python functions.

    Attributes:
        all_ground_literals: List of all grounded predicates in the current problem.

    Args:
        domain: The PDDL domain description.
        problem: The PDDL problem description.
    """

    def __init__(self, domain, problem):
        super().__init__(domain, problem)
        self.all_ground_literals = list(self.up_problem.initial_values.keys())

    def swap_queries(self, domain, problem, *args, **kwargs):
        """Update the current planning problem and ground literals.
        
        Args:
            domain: New PDDL domain description.
            problem: New PDDL problem description.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().swap_queries(domain, problem, *args, **kwargs)
        self.all_ground_literals = list(self.up_problem.initial_values.keys())

    def __call__(self, state):
        """Estimate the current state using predicate functions implemented in the class.
        
        Args:
            state: Current environment state.

        Returns:
            Dictionary mapping predicate strings to boolean values.

        Raises:
            AttributeError: If a required predicate function is not implemented.
        """
        out = {}

        for lit in self.all_ground_literals:
            predicate_name = lit.fluent().name
            pred_fn_name = predicate_name.replace('-', '_')
            predicate_args = [str(arg) for arg in lit.args]
            try:
                pred_fn = getattr(self, pred_fn_name)
            except AttributeError:
                raise AttributeError(
                    f"action mapper {self.__class__.__name__} does not support predicate '{predicate_name}'"
                )
            out[f'{predicate_name}({",".join(predicate_args)})'] = pred_fn(*predicate_args, state)

        return out

