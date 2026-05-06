"""Tests for make_env() — all dispatch paths and wrapper application."""

import warnings

import gymnasium as gym
import numpy as np
import pytest

from powerzoo.rl import make_env, RLConfig


# ── str task name path ───────────────────────────────────────────────────────

class TestMakeEnvStrPath:
    def test_returns_gymnasium_env(self):
        env = make_env('battery_arbitrage')
        assert isinstance(env, gym.Env)
        env.close()

    def test_obs_action_space_valid(self):
        env = make_env('battery_arbitrage')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
        env.close()

    def test_reset_step_cycle(self):
        env = make_env('battery_arbitrage')
        obs, info = env.reset(seed=0)
        assert obs is not None
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(action)
        assert obs2 is not None
        assert isinstance(reward, float)
        env.close()

    def test_split_train(self):
        env = make_env('battery_arbitrage', split='train')
        obs, _ = env.reset()
        assert obs is not None
        env.close()

    def test_seed_resets_env(self):
        env = make_env('battery_arbitrage', seed=42)
        # env already reset with seed=42 inside make_env, should be usable
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        env.close()


# ── dict config path ─────────────────────────────────────────────────────────

class TestMakeEnvDictPath:
    def test_anonymous_single_agent(self):
        config = {
            'grid': {'type': 'distribution', 'case': 'case33bw'},
            'resources': [
                {
                    'type': 'battery',
                    'bus_id': 6,
                    'capacity_mwh': 1.0,
                    'charge_power_kw': 500,
                }
            ],
            'reward': {'type': 'battery_arbitrage'},
            'episode': {'max_steps': 24},
        }
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            env = make_env(config)
        assert isinstance(env, gym.Env)
        obs, _ = env.reset()
        assert obs is not None
        env.close()


# ── RLConfig path ────────────────────────────────────────────────────────────

class TestMakeEnvRLConfigPath:
    def test_rlconfig_task_name(self):
        cfg = RLConfig(task_name='battery_arbitrage', split='train')
        env = make_env(cfg)
        assert isinstance(env, gym.Env)
        env.close()

    def test_rlconfig_normalize(self):
        cfg = RLConfig(task_name='battery_arbitrage', normalize=True)
        env = make_env(cfg)
        obs, _ = env.reset()
        # Normalization wrapper clips obs to [-1, 1] range typically
        assert obs is not None
        env.close()


# ── reward override ───────────────────────────────────────────────────────────

class TestMakeEnvRewardOverride:
    def test_dict_reward_override(self):
        env = make_env(
            'battery_arbitrage',
            reward={'type': 'battery_lmp_arbitrage'},
        )
        obs, _ = env.reset(seed=0)
        _, r, _, _, _ = env.step(env.action_space.sample())
        assert isinstance(r, float)
        env.close()

    def test_callable_reward_override(self):
        def my_reward(state, info):
            return -1.0

        env = make_env('battery_arbitrage', reward=my_reward)
        obs, _ = env.reset(seed=0)
        _, r, _, _, _ = env.step(env.action_space.sample())
        assert r == -1.0
        env.close()

    def test_marl_reward_override_warns(self):
        try:
            import pettingzoo  # noqa: F401
        except ImportError:
            pytest.skip("pettingzoo not installed")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            env = make_env(
                'marl_opf',
                reward={'type': 'economic_dispatch'},
                framework='pettingzoo',
            )
        assert any('multi-agent' in str(x.message).lower() for x in w)
        env.close()


# ── wrapper application ───────────────────────────────────────────────────────

class TestMakeEnvWrappers:
    def test_normalize_wrapper_applied(self):
        from powerzoo.wrappers.gym_wrappers import NormalizationWrapper
        env = make_env('battery_arbitrage', normalize=True)
        # check NormalizationWrapper is in the wrapper stack
        unwrapped = env
        found = False
        while unwrapped is not None:
            if isinstance(unwrapped, NormalizationWrapper):
                found = True
                break
            unwrapped = getattr(unwrapped, 'env', None)
        assert found, "NormalizationWrapper not found in stack"
        env.close()

    def test_forecast_wrapper_applied(self):
        from powerzoo.wrappers.forecast_wrapper import ForecastWrapper
        env = make_env('battery_arbitrage', forecast_horizon=4)
        unwrapped = env
        found = False
        while unwrapped is not None:
            if isinstance(unwrapped, ForecastWrapper):
                found = True
                break
            unwrapped = getattr(unwrapped, 'env', None)
        assert found, "ForecastWrapper not found in stack"
        env.close()

    def test_safe_rl_wrapper_applied(self):
        from powerzoo.wrappers.safe_rl_wrapper import GymnasiumSafeWrapper
        env = make_env('battery_arbitrage', safe_rl=True, cost_threshold=10.0)
        unwrapped = env
        found = False
        while unwrapped is not None:
            if isinstance(unwrapped, GymnasiumSafeWrapper):
                found = True
                break
            unwrapped = getattr(unwrapped, 'env', None)
        assert found, "GymnasiumSafeWrapper not found in stack"
        env.close()

    def test_marl_single_agent_wrappers_warn(self):
        try:
            import pettingzoo  # noqa: F401
        except ImportError:
            pytest.skip("pettingzoo not installed")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            env = make_env('marl_opf', normalize=True, framework='pettingzoo')
        assert any('multi-agent' in str(x.message).lower() for x in w)
        env.close()


# ── invalid input ─────────────────────────────────────────────────────────────

class TestMakeEnvInvalidInput:
    def test_unknown_task_name_raises(self):
        with pytest.raises((ValueError, KeyError)):
            make_env('nonexistent_task_xyz')

    def test_invalid_type_raises(self):
        with pytest.raises((ValueError, TypeError)):
            make_env(12345)
