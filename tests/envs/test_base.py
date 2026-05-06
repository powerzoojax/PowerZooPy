"""Tests for powerzoo.envs.base — BaseEnv.

BaseEnv is the root of the environment hierarchy, providing:
  - np_random: Gymnasium-managed seeded RNG
  - Gymnasium-compatible reset/step/obs stubs
  - delta_t_minutes time step configuration
"""
import numpy as np
import pytest

from powerzoo.envs.base import BaseEnv


# ── Helpers ──────────────────────────────────────────────────────────

class ConcreteEnv(BaseEnv):
    """Minimal concrete subclass for testing the abstract base."""

    def __init__(self, delta_t_minutes: float = 1.0):
        super().__init__(delta_t_minutes=delta_t_minutes)
        self._reset_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._reset_count += 1
        return np.zeros(2), {}

    def step(self, action):
        self.time_step += 1
        obs = np.zeros(2)
        return obs, 0.0, False, False, {}

    def obs(self, state=None):
        return np.zeros(2)


# ── Tests ────────────────────────────────────────────────────────────

class TestBaseEnvInit:
    """Constructor defaults and parameter handling."""

    def test_default_delta_t(self):
        env = ConcreteEnv()
        assert env.delta_t_minutes == 1.0

    def test_custom_delta_t(self):
        env = ConcreteEnv(delta_t_minutes=15.0)
        assert env.delta_t_minutes == 15.0

    def test_initial_time_step_zero(self):
        env = ConcreteEnv()
        assert env.time_step == 0

    def test_spaces_initially_none(self):
        env = ConcreteEnv()
        assert env.action_space is None
        assert env.observation_space is None

    def test_np_random_property_available(self):
        """Gymnasium lazily exposes a usable RNG via ``np_random``."""
        env = ConcreteEnv()
        assert env.np_random is not None


class TestResetSeedContract:
    """Gymnasium reset seeding contract."""

    def test_same_seed_same_sequence(self):
        env = ConcreteEnv()
        env.reset(seed=123)
        a = env.np_random.random(5)
        env.reset(seed=123)
        b = env.np_random.random(5)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_different_sequence(self):
        env = ConcreteEnv()
        env.reset(seed=1)
        a = env.np_random.random(5)
        env.reset(seed=2)
        b = env.np_random.random(5)
        assert not np.allclose(a, b)

    def test_seed_none_keeps_existing_generator_progress(self):
        env = ConcreteEnv()
        env.reset(seed=7)
        first = env.np_random.random(5)
        env.reset(seed=None)
        second = env.np_random.random(5)
        env.reset(seed=7)
        replay = env.np_random.random(5)

        assert not np.allclose(first, second)
        np.testing.assert_array_equal(first, replay)


class TestAbstractInterface:
    """BaseEnv is abstract — step() and obs() must be implemented."""

    def test_cannot_instantiate_base_env(self):
        with pytest.raises(TypeError):
            BaseEnv()

    def test_incomplete_subclass_missing_step(self):
        class MissingStep(BaseEnv):
            def obs(self, state=None):
                return np.zeros(2)
        with pytest.raises(TypeError):
            MissingStep()

    def test_incomplete_subclass_missing_obs(self):
        class MissingObs(BaseEnv):
            def step(self, action):
                return np.zeros(2), 0.0, False, False, {}
        with pytest.raises(TypeError):
            MissingObs()


class TestRewardHook:
    """Default reward hook is identity passthrough."""

    def test_reward_passthrough(self):
        env = ConcreteEnv()
        assert env.reward(3.14, {}, {}) == 3.14

    def test_reward_negative(self):
        env = ConcreteEnv()
        assert env.reward(-1.5, {}, {}) == -1.5


class TestCostHook:
    """Default cost hook returns 0.0."""

    def test_cost_default_zero(self):
        env = ConcreteEnv()
        assert env.cost({}, {}) == 0.0


class TestConcreteReset:
    """Reset on concrete subclass follows seed contract."""

    def test_reset_seeds_rng(self):
        env = ConcreteEnv()
        env.reset(seed=42)
        assert env.np_random is not None

    def test_reset_increments_counter(self):
        env = ConcreteEnv()
        env.reset(seed=0)
        env.reset(seed=1)
        assert env._reset_count == 2

    def test_reset_resets_time_step(self):
        env = ConcreteEnv()
        env.reset(seed=0)
        env.step(None)
        env.step(None)
        assert env.time_step == 2
        env.reset(seed=0)
        assert env.time_step == 0


class TestGymnasiumIntegration:
    """Verify Gymnasium base class detection."""

    def test_inherits_gym_env(self):
        import gymnasium as gym
        env = ConcreteEnv()
        assert isinstance(env, gym.Env)
