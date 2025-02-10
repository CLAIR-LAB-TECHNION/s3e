from abc import ABC, abstractmethod

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
    def __call__(self, images: list[Image]) -> dict[str, bool]:
        ...


class ProbabilisticStateEstimator(StateEstimator, ABC):
    def __init__(self, domain, problem, confidence: float = 0.5):
        super().__init__(domain, problem)
        self.confidence = confidence

    def __call__(self, images: list[Image], confidence=None) -> dict[str, bool]:
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
        ...


class PredFnStateEstimator(StateEstimator, ABC):
    def __init__(self, domain, problem):
        super().__init__(domain, problem)
        self.all_ground_literals = list(self.up_problem.initial_values.keys())

    def __call__(self, state):
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

