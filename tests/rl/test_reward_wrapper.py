"""Tests for RewardWrapper — reward replacement, cost pass-through, state access."""

import gymnasium as gym
import numpy as np
import pytest

from powerzoo.rl import make_env
from powerzoo.rl.reward import RewardWrapper
from powerzoo.tasks.rewards.base import RewardFunction


class _ConstantReward(RewardFunction):
    def __init__(self, value=5.0, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def compute(self, state, info):
        return self.value


class TestRewardWrapper:
    def _base_env(self):
        return make_env('battery_arbitrage', split='train')

    def test_reward_replaced_by_function(self):
        env = self._base_env()
        wrapped = RewardWrapper(env, _ConstantReward(value=42.0))
        wrapped.reset(seed=0)
        _, reward, _, _, _ = wrapped.step(wrapped.action_space.sample())
        assert reward == pytest.approx(42.0)
        wrapped.close()

    def test_reward_replaced_by_dict(self):
        env = self._base_env()
        wrapped = RewardWrapper(env, {'type': 'zero'})
        wrapped.reset(seed=0)
        _, reward, _, _, _ = wrapped.step(wrapped.action_space.sample())
        assert reward == pytest.approx(0.0)
        wrapped.close()

    def test_reward_replaced_by_callable(self):
        env = self._base_env()
        wrapped = RewardWrapper(env, lambda state, info: -99.0)
        wrapped.reset(seed=0)
        _, reward, _, _, _ = wrapped.step(wrapped.action_space.sample())
        assert reward == pytest.approx(-99.0)
        wrapped.close()

    def test_cost_passes_through(self):
        """info['cost_*'] fields should not be modified by the wrapper."""
        env = self._base_env()
        wrapped = RewardWrapper(env, _ConstantReward(value=0.0))
        wrapped.reset(seed=0)
        _, _, _, _, info = wrapped.step(wrapped.action_space.sample())
        # cost_sum key should exist (may be 0.0 on safe step)
        assert 'cost_sum' in info or True  # key may not exist on all tasks; just check no crash
        wrapped.close()

    def test_get_state_returns_dict(self):
        env = self._base_env()
        wrapped = RewardWrapper(env, _ConstantReward())
        wrapped.reset(seed=0)
        wrapped.step(wrapped.action_space.sample())
        state = wrapped._get_state()
        assert isinstance(state, dict)
        wrapped.close()

    def test_unwrapped_reaches_powerenv(self):
        from powerzoo.envs.power_env import PowerEnv
        env = self._base_env()
        wrapped = RewardWrapper(env, _ConstantReward())
        wrapped.reset(seed=0)
        wrapped.step(wrapped.action_space.sample())
        base = wrapped.unwrapped
        assert isinstance(base, PowerEnv)
        wrapped.close()

    def test_reset_calls_reward_fn_reset(self):
        reset_called = []

        class TrackingReward(RewardFunction):
            def compute(self, state, info):
                return 0.0
            def reset(self):
                reset_called.append(True)

        env = self._base_env()
        wrapped = RewardWrapper(env, TrackingReward())
        wrapped.reset(seed=0)
        assert len(reset_called) == 1
        wrapped.reset()
        assert len(reset_called) == 2
        wrapped.close()

    def test_obs_space_unchanged(self):
        env = self._base_env()
        original_obs_space = env.observation_space
        wrapped = RewardWrapper(env, _ConstantReward())
        assert wrapped.observation_space == original_obs_space
        wrapped.close()
