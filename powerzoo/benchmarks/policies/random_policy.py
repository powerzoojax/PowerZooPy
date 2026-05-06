"""Random policy — samples uniformly from the action space."""

from typing import Any, Dict, Optional
import numpy as np

from powerzoo.benchmarks.policies.base import BasePolicy


class RandomPolicy(BasePolicy):
    """Uniformly-random policy.

    Serves as the **lower bound** baseline: an agent that does no better than
    random should be considered non-functional.

    Args:
        action_space: A Gymnasium ``spaces.Space``.
        seed: Optional random seed for reproducibility.

    Example::

        from powerzoo.benchmarks.policies import RandomPolicy, evaluate
        from powerzoo.wrappers import GymnasiumWrapper
        from powerzoo.envs.grid.trans import TransGridEnv

        env = GymnasiumWrapper(TransGridEnv())
        policy = RandomPolicy(env.action_space, seed=0)
        result = evaluate(policy, env, n_episodes=5)
    """

    def __init__(self, action_space=None, seed: Optional[int] = None):
        super().__init__(action_space)
        if action_space is not None and seed is not None:
            action_space.seed(seed)

    def act(self, obs: Any, info: Optional[Dict] = None) -> Any:
        if self.action_space is None:
            raise ValueError("RandomPolicy requires an action_space.")
        return self.action_space.sample()
