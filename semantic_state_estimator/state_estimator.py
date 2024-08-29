from abc import ABC, abstractmethod
from sre_parse import State

from PIL.Image import Image
from unified_planning.model import FNode

from .utils.up_utils import create_up_problem, bool_constant_to_fnode


class StateEstimator(ABC):
    def __init__(
            self,
            domain,
            problem,
    ):
        self.up_problem = create_up_problem(domain, problem)

    @abstractmethod
    def __call__(self, images: list[Image]) -> dict[FNode, bool]:
        ...


class ProbabilisticStateEstimator(StateEstimator, ABC):
    def __call__(self, images: list[Image], confidence=0.5) -> dict[FNode, bool]:
        fluent_prob_map = self.estimate_state(images)
        state = {
            fluent: prob >= confidence
            for fluent, prob in fluent_prob_map.items()
        }

        return state

    @abstractmethod
    def estimate_state(self, images: list[Image]) -> dict[FNode, float]:
        ...


class PredFnStateEstimator(StateEstimator, ABC):
    def __init__(self, domain, problem):
        super().__init__(domain, problem)
        self.all_ground_literals = list(self.up_problem.initial_values.keys())

    def __call__(self, state):
        out = {}

        for lit in self.all_ground_literals:
            predicate_name = lit.fluent().name.replace('-', '_')
            predicate_args = [str(arg) for arg in lit.args]
            try:
                pred_fn = getattr(self, predicate_name)
            except AttributeError:
                raise AttributeError(
                    f"action mapper {self.__class__.__name__} does not support predicate '{predicate_name}'"
                )
            out[lit] = bool_constant_to_fnode(self.up_problem, pred_fn(*predicate_args, state))

        return out

