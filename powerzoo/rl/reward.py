"""Reward wrappers for custom rewards and CMDP-to-MDP fallback shaping."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Union

import gymnasium as gym
import numpy as np

from powerzoo.tasks.rewards.base import RewardFunction
from powerzoo.tasks.rewards.registry import get_reward_function
from powerzoo.wrappers.safe_rl_wrapper import _annotate_constraint_info


class RewardWrapper(gym.Wrapper):
    """Replace the scalar reward returned by a single-agent Gymnasium env.

    The wrapper accepts either a :class:`~powerzoo.tasks.rewards.base.RewardFunction`
    instance (which receives the full ``state`` dict) or a plain callable
    ``fn(state, info) -> float``.

    The underlying env's ``_current_state`` is accessed via
    ``self.unwrapped`` (standard Gymnasium interface) so the wrapper
    operates correctly when stacked on top of ``FlattenWrapper`` or other
    intermediate wrappers.

    Args:
        env:       A single-agent Gymnasium env wrapping a ``PowerEnv``.
        reward_fn: Either a :class:`RewardFunction` instance, a reward-type
                   dict (e.g. ``{'type': 'lmp_arbitrage'}``), or any callable
                   ``(state, info) -> float``.

    Note:
        This wrapper is designed for **single-agent** environments only.
        For MARL envs reward is computed inside the task adapter and should
        be overridden at the Task / adapter level.

    Example::

        from powerzoo.rl import make_env, RewardWrapper

        env = make_env('battery_arbitrage')
        env = RewardWrapper(env, {'type': 'lmp_arbitrage', 'profit_weight': 2.0})

    """

    def __init__(
        self,
        env: gym.Env,
        reward_fn: Union[RewardFunction, Dict[str, Any], Callable],
    ):
        super().__init__(env)
        if isinstance(reward_fn, dict):
            self._reward_fn: Union[RewardFunction, Callable] = get_reward_function(reward_fn)
        else:
            self._reward_fn = reward_fn

    def step(self, action):
        obs, _orig_reward, terminated, truncated, info = self.env.step(action)
        state = self._get_state()
        if isinstance(self._reward_fn, RewardFunction):
            reward = self._reward_fn.compute(state, info)
        else:
            reward = float(self._reward_fn(state, info))
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(self._reward_fn, RewardFunction):
            self._reward_fn.reset()
        return result

    def _get_state(self) -> Dict[str, Any]:
        """Retrieve the current physical state dict from the underlying PowerEnv."""
        base = self.unwrapped
        state = getattr(base, '_current_state', None)
        return state if state is not None else {}


class MDPFallbackRewardWrapper(gym.Wrapper):
    """Project selected CMDP vector costs into a scalar fallback reward.

    ``reward_fallback = env_reward - w · selected_constraint_costs``

    The original env reward is preserved in ``info['env_reward']``.
    """

    def __init__(
        self,
        env: gym.Env,
        fallback_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__(env)
        if fallback_weights is None:
            spec = getattr(env, 'constraint_spec', None)
            spec = spec() if callable(spec) else spec
            fallback_weights = tuple(spec.fallback_weights) if spec is not None else ()
        self.fallback_weights = np.asarray(fallback_weights, dtype=np.float32).reshape(-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = _annotate_constraint_info(self.env, info)
        return obs, info

    def step(self, action):
        step_out = self.env.step(action)
        if len(step_out) == 6:
            obs, reward, _costs, terminated, truncated, info = step_out
        else:
            obs, reward, terminated, truncated, info = step_out

        info = _annotate_constraint_info(self.env, info)
        selected_costs = np.asarray(
            info.get('selected_constraint_costs', ()),
            dtype=np.float32,
        ).reshape(-1)
        if self.fallback_weights.size == 0:
            fallback_weights = np.ones_like(selected_costs)
        elif self.fallback_weights.size == 1 and selected_costs.size > 1:
            fallback_weights = np.full_like(selected_costs, float(self.fallback_weights[0]))
        else:
            fallback_weights = self.fallback_weights

        if fallback_weights.shape != selected_costs.shape:
            raise ValueError(
                f"fallback_weights shape {fallback_weights.shape} does not match "
                f"selected_constraint_costs shape {selected_costs.shape}."
            )

        env_reward = float(reward)
        reward_fallback = env_reward - float(np.dot(fallback_weights, selected_costs))
        info['env_reward'] = env_reward
        info['reward_fallback'] = reward_fallback
        return obs, reward_fallback, terminated, truncated, info
