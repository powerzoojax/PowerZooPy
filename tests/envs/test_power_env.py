"""Tests for powerzoo.envs.power_env — PowerEnv.

PowerEnv is the unified benchmark-facing facade that composes:
  - One grid environment (transmission or distribution)
  - Zero or more attached DER resources
  - A reward function (injected by Task, or resolved from config)
  - Dict observation space: {grid, resources, time}
  - Dict action space: {unit_power_mw?, resource_id: resource_action, ...}

Domain notes:
  - Time encoding: cyclic [sin(2π·t/T), cos(2π·t/T)] where T = steps_per_day
  - Resource observations are concatenated into a flat 'resources' sub-space
  - Episode day window: required_days = ceil(max_steps / steps_per_day)
  - CMDP cost channel: resource-level safety costs (e.g. overtemp) merge into info
"""
import numpy as np
import pytest
from gymnasium import spaces

from powerzoo.envs.power_env import (
    PowerEnv,
    _collect_resource_violation_costs,
    _flatten_observation_value,
    _flattened_space_bounds,
    _flattened_space_size,
)


def _make_config(**overrides):
    """Minimal valid PowerEnv config."""
    config = {
        'name': 'TestScenario',
        'grid': {
            'type': 'transmission',
            'case': 'Case5',
            'delta_t_minutes': 30.0,
            'time_series': np.ones(48) * 100,
        },
        'resources': [],
        'episode': {'max_steps': 10},
        'reward': {'type': 'zero'},
    }
    config.update(overrides)
    return config


# ── Constructor ──────────────────────────────────────────────────────

class TestPowerEnvInit:
    """Constructor wiring and space setup."""

    def test_name_from_config(self):
        env = PowerEnv(_make_config(name='MyBenchmark'))
        assert env.name == 'MyBenchmark'

    def test_default_name(self):
        cfg = _make_config()
        cfg.pop('name', None)
        env = PowerEnv(cfg)
        assert env.name == 'CustomScenario'

    def test_grid_created(self):
        env = PowerEnv(_make_config())
        assert env.grid is not None

    def test_delta_t_from_grid(self):
        env = PowerEnv(_make_config())
        assert env.delta_t_minutes == env.grid.delta_t_minutes

    def test_max_steps_from_config(self):
        env = PowerEnv(_make_config())
        assert env.max_steps_per_episode == 10

    def test_observation_space_is_dict(self):
        env = PowerEnv(_make_config())
        from gymnasium import spaces
        assert isinstance(env.observation_space, spaces.Dict)
        assert 'grid' in env.observation_space.spaces
        assert 'resources' in env.observation_space.spaces
        assert 'time' in env.observation_space.spaces

    def test_time_space_shape(self):
        env = PowerEnv(_make_config())
        assert env.observation_space['time'].shape == (2,)

    def test_action_space_is_dict(self):
        env = PowerEnv(_make_config())
        from gymnasium import spaces
        assert isinstance(env.action_space, spaces.Dict)


# ── With Resources ───────────────────────────────────────────────────

class TestPowerEnvWithResources:
    """PowerEnv with attached resources."""

    def test_resources_attached(self):
        cfg = _make_config(resources=[
            {'type': 'battery', 'bus_id': 2, 'capacity_mwh': 50.0, 'power_mw': 10.0},
        ])
        env = PowerEnv(cfg)
        assert len(env.resources) == 1

    def test_resource_obs_dim(self):
        """Resource observation should be > 0 when resources are present."""
        cfg = _make_config(resources=[
            {'type': 'battery', 'bus_id': 2, 'capacity_mwh': 50.0, 'power_mw': 10.0},
        ])
        env = PowerEnv(cfg)
        res_space = env.observation_space['resources']
        assert res_space.shape[0] > 0

    def test_resource_obs_bounds_finite(self):
        """Resources Box bounds should come from observation_space (not ±inf)."""
        cfg = _make_config(resources=[
            {'type': 'battery', 'bus_id': 2, 'capacity_mwh': 50.0, 'power_mw': 10.0},
        ])
        env = PowerEnv(cfg)
        res_space = env.observation_space['resources']
        assert np.all(np.isfinite(res_space.low)), "resources Box low contains inf"
        assert np.all(np.isfinite(res_space.high)), "resources Box high contains inf"

    def test_resource_obs_order_is_sorted(self):
        """Obs vector order must match sorted resource key order."""
        cfg = _make_config(resources=[
            {'type': 'battery', 'bus_id': 2, 'capacity_mwh': 50.0, 'power_mw': 10.0},
            {'type': 'battery', 'bus_id': 3, 'capacity_mwh': 30.0, 'power_mw': 5.0},
        ])
        env = PowerEnv(cfg)
        env.reset(seed=0)
        obs_vec = env._build_resource_observation_vector()
        # Reconstruct expected order manually
        parts = []
        for res_id in sorted(env.resources.keys()):
            resource = env.resources[res_id]
            raw = resource.obs()
            if isinstance(raw, dict):
                parts.append(np.array([raw[k] for k in sorted(raw.keys())], dtype=np.float32))
            else:
                parts.append(np.asarray(raw, dtype=np.float32))
        expected = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        np.testing.assert_array_equal(obs_vec, expected)

    def test_resource_action_in_action_space(self):
        cfg = _make_config(resources=[
            {'type': 'battery', 'bus_id': 2, 'capacity_mwh': 50.0, 'power_mw': 10.0},
        ])
        env = PowerEnv(cfg)
        assert len(env.action_space.spaces) >= 1

    def test_resource_metadata(self):
        cfg = _make_config(resources=[
            {'type': 'battery', 'bus_id': 2, 'capacity_mwh': 50.0, 'power_mw': 10.0},
        ])
        env = PowerEnv(cfg)
        rid = list(env.resources.keys())[0]
        meta = env.get_resource_metadata(rid)
        assert meta['type'] == 'battery'

    def test_flattened_space_size_box(self):
        """_flattened_space_size handles Box correctly."""
        box = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        assert _flattened_space_size(box) == 3

    def test_flattened_space_size_dict(self):
        """_flattened_space_size handles Dict by summing sub-space dims."""
        d = spaces.Dict({
            'a': spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            'b': spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        })
        assert _flattened_space_size(d) == 6

    def test_flattened_space_bounds_box(self):
        """_flattened_space_bounds returns low/high arrays from a Box."""
        box = spaces.Box(low=-2.0, high=5.0, shape=(2,), dtype=np.float32)
        lo, hi = _flattened_space_bounds(box)
        np.testing.assert_array_equal(lo, [-2.0, -2.0])
        np.testing.assert_array_equal(hi, [5.0, 5.0])

    def test_flattened_space_bounds_dict(self):
        """_flattened_space_bounds concatenates sub-space bounds in sorted key order."""
        d = spaces.Dict({
            'z': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'a': spaces.Box(low=-3.0, high=3.0, shape=(2,), dtype=np.float32),
        })
        lo, hi = _flattened_space_bounds(d)
        # sorted order: 'a' first, then 'z'
        np.testing.assert_array_equal(lo, [-3.0, -3.0, 0.0])
        np.testing.assert_array_equal(hi, [3.0, 3.0, 1.0])

    def test_flattened_space_size_unsupported_raises(self):
        """_flattened_space_size raises ValueError for unsupported space types."""
        with pytest.raises(ValueError, match="Unsupported observation space type"):
            _flattened_space_size(spaces.Discrete(5))

    def test_flatten_obs_array(self):
        """_flatten_observation_value flattens a plain array to 1-D float32."""
        result = _flatten_observation_value(np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])
        assert result.dtype == np.float32

    def test_flatten_obs_shallow_dict(self):
        """_flatten_observation_value flattens a dict in sorted key order."""
        obs = {'z': np.array([3.0]), 'a': np.array([1.0, 2.0])}
        result = _flatten_observation_value(obs)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_flatten_obs_nested_dict(self):
        """_flatten_observation_value recursively flattens nested dicts."""
        obs = {'b': {'y': np.array([4.0]), 'x': np.array([3.0])}, 'a': np.array([1.0, 2.0])}
        result = _flatten_observation_value(obs)
        # sorted outer: 'a' then 'b'; sorted inner 'b': 'x' then 'y'
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])

    def test_flatten_obs_ragged_values(self):
        """_flatten_observation_value handles dict values with different lengths (ragged)."""
        obs = {'a': np.array([1.0, 2.0, 3.0]), 'b': np.array([4.0])}
        result = _flatten_observation_value(obs)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])

    def test_build_resource_observation_dim_mismatch_raises(self):
        """_build_resource_observation_vector raises RuntimeError on dimension mismatch."""
        from unittest.mock import patch

        cfg = _make_config(resources=[
            {'type': 'battery', 'bus_id': 2, 'capacity_mwh': 50.0, 'power_mw': 10.0},
        ])
        env = PowerEnv(cfg)
        env.reset(seed=0)
        rid = next(iter(env.resources))
        # Patch obs() to return a vector of the wrong size
        with patch.object(env.resources[rid], 'obs', return_value=np.zeros(999)):
            with pytest.raises(RuntimeError, match="dimension mismatch"):
                env._build_resource_observation_vector()


# ── Reset ────────────────────────────────────────────────────────────

class TestPowerEnvReset:
    """Reset lifecycle."""

    def test_reset_returns_obs_info(self):
        env = PowerEnv(_make_config())
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert 'grid' in obs
        assert 'resources' in obs
        assert 'time' in obs
        assert isinstance(info, dict)

    def test_reset_with_day_id(self):
        env = PowerEnv(_make_config())
        env.reset(seed=42, day_id=0)
        assert env._clock.start_day_id == 0

    def test_reset_episode_step_zero(self):
        env = PowerEnv(_make_config())
        env.reset(seed=42)
        assert env._clock.step == 0

    def test_reset_seeds_rng(self):
        env = PowerEnv(_make_config())
        env.reset(seed=42)
        assert env.np_random is not None

    def test_time_obs_shape(self):
        env = PowerEnv(_make_config())
        obs, _ = env.reset(seed=42)
        assert obs['time'].shape == (2,)
        assert obs['time'].dtype == np.float32


# ── Step ─────────────────────────────────────────────────────────────

class TestPowerEnvStep:
    """Step mechanics."""

    def test_step_returns_five_tuple(self):
        env = PowerEnv(_make_config())
        env.reset(seed=42)
        result = env.step({})
        assert len(result) == 5

    def test_step_increments_episode_step(self):
        env = PowerEnv(_make_config())
        env.reset(seed=42)
        env.step({})
        assert env._clock.step == 1

    def test_truncated_at_max_steps(self):
        env = PowerEnv(_make_config())
        env.reset(seed=42)
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step({})
            if terminated or truncated:
                break
        assert truncated

    def test_info_has_episode_step(self):
        env = PowerEnv(_make_config())
        env.reset(seed=42)
        _, _, _, _, info = env.step({})
        assert 'episode_step' in info

    def test_info_has_delta_t(self):
        env = PowerEnv(_make_config())
        env.reset(seed=42)
        _, _, _, _, info = env.step({})
        assert 'delta_t_minutes' in info

    def test_info_has_resources(self):
        env = PowerEnv(_make_config())
        env.reset(seed=42)
        _, _, _, _, info = env.step({})
        assert 'resources' in info

    def test_info_has_cost_fields(self):
        env = PowerEnv(_make_config())
        env.reset(seed=42)
        _, _, _, _, info = env.step({})
        assert 'cost_sum' in info
        assert 'is_safe' in info


# ── Observation Building ─────────────────────────────────────────────

class TestObservationBuilding:
    """Observation structure and time encoding."""

    def test_obs_before_reset_raises(self):
        env = PowerEnv(_make_config())
        with pytest.raises(RuntimeError, match="reset"):
            env.obs()

    def test_grid_obs_flat(self):
        env = PowerEnv(_make_config())
        obs, _ = env.reset(seed=42)
        assert obs['grid'].ndim == 1
        assert obs['grid'].dtype == np.float32

    def test_time_encoding_cyclic(self):
        """Time features should satisfy sin² + cos² = 1."""
        env = PowerEnv(_make_config())
        obs, _ = env.reset(seed=42)
        sin_t, cos_t = obs['time'][0], obs['time'][1]
        np.testing.assert_allclose(sin_t ** 2 + cos_t ** 2, 1.0, atol=1e-6)

    def test_resource_obs_empty_when_no_resources(self):
        env = PowerEnv(_make_config(resources=[]))
        obs, _ = env.reset(seed=42)
        assert obs['resources'].shape == (0,)


# ── Action Canonicalisation ──────────────────────────────────────────

class TestActionCanonicalize:
    """Action parsing and normalisation."""

    def test_none_action_accepted(self):
        env = PowerEnv(_make_config())
        env.reset(seed=42)
        # None is converted to empty dict
        env.step(None)

    def test_non_dict_action_raises(self):
        env = PowerEnv(_make_config())
        env.reset(seed=42)
        with pytest.raises(TypeError, match="dict"):
            env.step("invalid")

    def test_step_before_reset_raises(self):
        env = PowerEnv(_make_config())
        with pytest.raises(RuntimeError, match="reset"):
            env.step({})


# ── Day Validation ───────────────────────────────────────────────────

class TestDayValidation:
    """Episode day window validation."""

    def test_negative_day_raises(self):
        env = PowerEnv(_make_config())
        with pytest.raises(ValueError, match="day_id must be >= 0"):
            env._validate_start_day(-1)


# ── Resource Cost Merging ────────────────────────────────────────────

class TestResourceCostMerging:
    """CMDP cost channel merging from resources."""

    def test_no_cost_keys_zero_total(self):
        resource_statuses = {'batt_0': {'soc': 0.5}}
        costs = _collect_resource_violation_costs(resource_statuses)
        assert costs['cost_resource_violation'] == 0.0

    def test_cost_key_contributes(self):
        resource_statuses = {'dc_0': {'cost_overtemp': 2.5}}
        costs = _collect_resource_violation_costs(resource_statuses)
        assert costs['dc_0/cost_overtemp'] == 2.5
        assert costs['cost_resource_violation'] == 2.5


# ── Repr ─────────────────────────────────────────────────────────────

class TestRepr:
    def test_repr_contains_name(self):
        env = PowerEnv(_make_config(name='TestRepr'))
        r = repr(env)
        assert 'TestRepr' in r

    def test_repr_contains_case(self):
        env = PowerEnv(_make_config())
        r = repr(env)
        assert 'Case5' in r


# ── Close & Render ───────────────────────────────────────────────────

class TestCloseRender:
    def test_close_does_not_crash(self):
        env = PowerEnv(_make_config())
        env.close()

    def test_render_does_not_crash(self):
        env = PowerEnv(_make_config())
        env.reset(seed=42)
        # TransGridEnv.render requires matplotlib, just test no hard error
        try:
            env.render()
        except ImportError:
            pass  # matplotlib not installed is OK
