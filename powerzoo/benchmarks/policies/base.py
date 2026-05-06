"""Abstract base class for all PowerZoo policies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BasePolicy(ABC):
    """Abstract policy interface.

    All policies must implement ``act(obs, info) -> action`` so they can be
    passed directly to ``evaluate()``.

    Parameters
    ----------
    action_space :
        A Gymnasium ``spaces.Space`` used to determine valid actions.
    """

    def __init__(self, action_space=None):
        self.action_space = action_space

    @abstractmethod
    def act(self, obs: Any, info: Optional[Dict] = None) -> Any:
        """Select an action given the current observation.

        Args:
            obs:  Observation array (or dict) from the environment.
            info: Optional info dict from the previous ``step()``.

        Returns:
            action compatible with ``self.action_space``.
        """

    def reset(self) -> None:
        """Called at the start of each episode (optional hook)."""
