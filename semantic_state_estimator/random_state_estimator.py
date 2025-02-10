import numpy as np

from .state_estimator import StateEstimator
from .utils.up_utils import get_all_grounded_predicates_for_objects


class RandomStateEstimator(StateEstimator):
    def __init__(self, domain, problem, success_rate, gt_state_estimator):
        super().__init__(domain, problem)
        self.sr = success_rate
        self.gt = gt_state_estimator

    def __call__(self, images):
        true_state = self.gt(self.gt.env.get_state())
        return {
            predicate: value if np.random.rand() < self.sr else not value
            for predicate, value in true_state.items()
        }

class PseudoRandomStateEstimator(RandomStateEstimator):

    def __call__(self, images):
        true_state = self.gt(self.gt.env.get_state())

        # set constant seed for a given state
        seed = hash(tuple(true_state.items()))
        np.random.seed(seed)
        
        return {
            predicate: value if np.random.rand() < self.sr else not value
            for predicate, value in true_state.items()
        }
