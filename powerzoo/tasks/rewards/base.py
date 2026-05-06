"""Base abstractions for task-level rewards."""

from typing import Any, Dict

import numpy as np


class RewardFunction:
    """Base class for task-level reward functions."""

    def __init__(self, normalize: bool = False, **kwargs):
        self.params = kwargs
        self.normalize = normalize
        self._reward_min: float = float('inf')
        self._reward_max: float = float('-inf')

    def compute(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        raise NotImplementedError

    def _maybe_normalize(self, value: float) -> float:
        if not self.normalize:
            return value
        self._reward_min = min(self._reward_min, value)
        self._reward_max = max(self._reward_max, value)
        span = self._reward_max - self._reward_min
        if span < 1e-6:
            return 0.0
        return float(np.clip(2.0 * (value - self._reward_min) / span - 1.0, -1.0, 1.0))

    def reset(self):
        pass
