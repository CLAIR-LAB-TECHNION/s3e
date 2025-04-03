"""Random state estimation for testing and benchmarking.

This module provides implementations of state estimators that randomly modify
the ground truth state with a specified success rate. This is useful for
testing and benchmarking purposes.

Classes:
    RandomStateEstimator: Random state estimation with configurable success rate.
    PseudoRandomStateEstimator: Deterministic random state estimation using state-based seeding.
"""

import numpy as np

from .state_estimator import StateEstimator
from .utils.up_utils import get_all_grounded_predicates_for_objects


class RandomStateEstimator(StateEstimator):
    """Random state estimation with configurable success rate.
    
    This class implements a state estimator that randomly modifies the ground truth
    state with a specified success rate. It is useful for testing and benchmarking
    purposes.

    Attributes:
        sr: Success rate for state estimation (probability of correct prediction).
        gt: Ground truth state estimator.

    Args:
        domain: The PDDL domain description.
        problem: The PDDL problem description.
        success_rate: Probability of correct state estimation.
        gt_state_estimator: Ground truth state estimator to base predictions on.
    """

    def __init__(self, domain, problem, success_rate, gt_state_estimator: StateEstimator):
        super().__init__(domain, problem)
        self.sr = success_rate
        self.gt = gt_state_estimator

    def swap_queries(self, domain, problem, *args, **kwargs):
        """Update the current planning problem and ground truth estimator.
        
        Args:
            domain: New PDDL domain description.
            problem: New PDDL problem description.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().swap_queries(domain, problem, *args, **kwargs)
        self.gt.swap_queries(domain, problem)

    def __call__(self, images):
        """Generate random state estimates based on ground truth.
        
        Args:
            images: List of PIL Image objects (not used).

        Returns:
            Dictionary mapping predicate strings to randomly modified boolean values.
        """
        true_state = self.gt(self.gt.env.get_state())
        return {
            predicate: value if np.random.rand() < self.sr else not value
            for predicate, value in true_state.items()
        }


class PseudoRandomStateEstimator(RandomStateEstimator):
    """Deterministic random state estimation using state-based seeding.
    
    This class extends RandomStateEstimator to provide deterministic random
    state estimation by using the current state as a seed for the random number
    generator. This ensures that the same state always produces the same random
    modifications.

    Args:
        domain: The PDDL domain description.
        problem: The PDDL problem description.
        success_rate: Probability of correct state estimation.
        gt_state_estimator: Ground truth state estimator to base predictions on.
    """

    def __call__(self, images):
        """Generate deterministic random state estimates based on ground truth.
        
        Args:
            images: List of PIL Image objects (not used).

        Returns:
            Dictionary mapping predicate strings to deterministically modified boolean values.
        """
        true_state = self.gt(self.gt.env.get_state())

        # set constant seed for a given state
        seed = hash(tuple(true_state.items()))
        np.random.seed(seed)
        
        return {
            predicate: value if np.random.rand() < self.sr else not value
            for predicate, value in true_state.items()
        }
