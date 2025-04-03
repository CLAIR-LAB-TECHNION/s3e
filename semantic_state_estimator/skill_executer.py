"""Skill execution framework for semantic environments.

This module provides a base class for executing skills in semantic environments.
Skills are represented as actions that can be performed on the environment, with
parameters that define their behavior.

Classes:
    SkillExecuter: Abstract base class for skill execution.
"""

from abc import ABC, abstractmethod
from typing import Any

from n_table_blocks_world.n_table_blocks_world import NTableBlocksWorld
from numpy.typing import NDArray

ActionReturnType = tuple[bool, list[NDArray]]


class SkillExecuter(ABC):
    """Abstract base class for skill execution in semantic environments.
    
    This class provides the interface for executing skills (actions) in a semantic
    environment. It handles the mapping between PDDL actions and their corresponding
    Python implementations.

    Attributes:
        env: The semantic environment instance.
        render_freq: Frequency of environment rendering (0 for no rendering).

    Args:
        env: The semantic environment to execute skills in.
        render_freq: Frequency of environment rendering (default: 0).
    """

    def __init__(self, env: NTableBlocksWorld, render_freq=0):
        self.env = env
        self.render_freq = render_freq

    def execute_action(self, action: str, parameters: list[Any]) -> ActionReturnType:
        """Execute a skill action with the given parameters.
        
        Args:
            action: The name of the action to execute.
            parameters: List of parameters for the action.

        Returns:
            Tuple containing:
                - Boolean indicating success/failure
                - List of numpy arrays containing action results

        Raises:
            AttributeError: If the action is not supported by the skill executor.
        """
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
        """Reset the environment to its initial state."""
        self.env.reset()

    @abstractmethod
    def go_home(self) -> ActionReturnType:
        """Move the robot to its home position.
        
        Returns:
            Tuple containing:
                - Boolean indicating success/failure
                - List of numpy arrays containing action results
        """
        ...

    @abstractmethod
    def wait(self, n_steps) -> ActionReturnType:
        """Wait for a specified number of steps.
        
        Args:
            n_steps: Number of steps to wait.

        Returns:
            Tuple containing:
                - Boolean indicating success/failure
                - List of numpy arrays containing action results
        """
        ...
