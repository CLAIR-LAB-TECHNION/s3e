from abc import ABC, abstractmethod
from typing import Any

from n_table_blocks_world.n_table_blocks_world import NTableBlocksWorld
from numpy.typing import NDArray

ActionReturnType = tuple[bool, list[NDArray]]


class SkillExecuter(ABC):
    def __init__(self, env: NTableBlocksWorld, render_freq=0):
        self.env = env
        self.render_freq = render_freq

    def execute_action(self, action: str, parameters: list[Any]) -> ActionReturnType:
        # replace PDDL legal hyphens to python legal underscores
        action = action.replace("-", "_")

        # get skill function from class attributes
        try:
            skill = getattr(self, action)
        except AttributeError:
            raise AttributeError(
                f"action mapper {self.__class__.__name__} does not support action '{action}'"
            )

        # execute skill
        return skill(*parameters)

    def reset_env(self):
        self.env.reset()

    @abstractmethod
    def go_home(self) -> ActionReturnType: ...

    @abstractmethod
    def wait(self, n_steps) -> ActionReturnType: ...
