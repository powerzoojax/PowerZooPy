"""Tests for powerzoo.envs.grid.base — GridEnv.

GridEnv is the abstract base for all power grid environments.
Key responsibilities:
  - Time-series data loading & preprocessing (demand → node-level loads)
  - Resource management (register/unregister DER assets)
  - RL interface: reset(), step() with episode tracking
  - Node-level load distribution (via case topology)

Domain knowledge embedded:
  - Load scaling: raw MW demand is normalised by generator capacity × max_load_ratio
  - Node distribution: total system demand is split across buses proportional to d_max
  - steps_per_day = 1440 / delta_t_minutes (e.g., 48 steps at 30-min resolution)
  - day_id + time_step uniquely identify the operating point in the time series
"""
import numpy as np
import pandas as pd
import pytest

from powerzoo.envs.grid.base import GridEnv
from powerzoo.data import signals as S


# ── Helpers ──────────────────────────────────────────────────────────

class MinimalGridEnv(GridEnv):
    """Concrete subclass filling abstract methods with safe stubs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run_power_flow(self, action):
        return True

    def _get_state(self):
        return {'time_step': self.time_step}

    def _compute_reward(self, state):
        return 0.0

    def build_info(self, state):
        return {}

    def obs(self, state=None):
        return np.zeros(1, dtype=np.float32)

    def cal_pf(self, *args, **kwargs):
        return {}, {}

    def safety_check(self, *args, **kwargs):
        return True, {}


# ── Constructor & Time Configuration ─────────────────────────────────

class TestGridEnvInit:
    """Constructor defaults and time resolution."""

    def test_default_delta_t(self, case5):
        env = MinimalGridEnv(case=case5)
        assert env.delta_t_minutes == 30.0

    def test_steps_per_day_30min(self, case5):
        """30-min resolution → 48 steps per day."""
        env = MinimalGridEnv(case=case5, delta_t_minutes=30.0)
        assert env.steps_per_day == 48

    def test_steps_per_day_15min(self, case5):
        """15-min resolution → 96 steps per day."""
        env = MinimalGridEnv(case=case5, delta_t_minutes=15.0)
        assert env.steps_per_day == 96

    def test_steps_per_day_60min(self, case5):
        """60-min resolution → 24 steps per day."""
        env = MinimalGridEnv(case=case5, delta_t_minutes=60.0)
        assert env.steps_per_day == 24

    def test_max_episode_steps_defaults_to_one_day(self, case5):
        env = MinimalGridEnv(case=case5)
        assert env.max_episode_steps == env.steps_per_day

    def test_max_episode_steps_custom(self, case5):
        env = MinimalGridEnv(case=case5, max_episode_steps=100)
        assert env.max_episode_steps == 100

    def test_resource_containers_empty(self, case5):
        env = MinimalGridEnv(case=case5)
        assert env.sub_resources == {}
        assert env.nodes_resources_map is None


# ── User-supplied Time Series ────────────────────────────────────────

class TestUserTimeSeries:
    """Loading user-supplied time series (DataFrame or numpy)."""

    def test_load_dataframe(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        assert env._time_series_data is not None
        assert env.n_days >= 1

    def test_load_numpy_array(self, case5, simple_numpy_demand):
        env = MinimalGridEnv(case=case5, time_series=simple_numpy_demand)
        assert env._time_series_data is not None
        assert S.LOAD_ACTUAL_MW in env._time_series_data.columns

    def test_node_loads_p_precomputed(self, case5, simple_time_series):
        """Node-level P loads should be precomputed as (T, n_loads) matrix."""
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        assert env._node_loads_p is not None
        assert env._node_loads_p.ndim == 2
        T = len(simple_time_series)
        assert env._node_loads_p.shape[0] == T

    def test_node_loads_q_cached_when_case_has_reactive_baseline(self, case33, simple_time_series):
        """Cases with ``Qd`` should precompute a matching constant-power-factor Q cache."""
        env = MinimalGridEnv(case=case33, time_series=simple_time_series)
        assert env._node_loads_q is not None
        base_p = case33.loads['Pd'].to_numpy(dtype=float)
        base_q = case33.loads['Qd'].to_numpy(dtype=float)
        q_over_p = np.divide(
            base_q,
            base_p,
            out=np.zeros_like(base_q, dtype=float),
            where=base_p > 0,
        )
        np.testing.assert_allclose(env._node_loads_q[0], env._node_loads_p[0] * q_over_p)

    def test_load_scaling_respects_max_load_ratio(self, case5, simple_time_series):
        """Peak scaled demand should be ≈ total_gen_capacity × max_load_ratio."""
        ratio = 0.7
        env = MinimalGridEnv(case=case5, time_series=simple_time_series,
                             max_load_ratio=ratio)
        total_cap = float(case5.units['p_max'].sum())
        expected_peak = total_cap * ratio
        actual_peak = env._time_series_data[S.LOAD_ACTUAL_MW].max()
        np.testing.assert_allclose(actual_peak, expected_peak, rtol=0.01)

    def test_dataframe_without_demand_raises(self, case5):
        """DataFrame without 'ActualDemand' column should fail gracefully."""
        df = pd.DataFrame({'Other': [1, 2, 3]},
                          index=pd.date_range('2024-01-01', periods=3, freq='30min'))
        env = MinimalGridEnv(case=case5, time_series=df)
        # Should handle gracefully (warning, not crash)
        # _time_series_data may be None after failure
        assert True  # no exception raised

    def test_invalid_type_handled(self, case5):
        """Non-array, non-DataFrame types should not crash."""
        env = MinimalGridEnv(case=case5, time_series="invalid")
        # Should handle gracefully
        assert True


# ── Resource Management ──────────────────────────────────────────────

class TestResourceManagement:
    """register_resource / unregister_resource lifecycle."""

    def test_register_auto_names(self, case5, simple_time_series):
        """Auto-generated names follow '{type}_{counter}' convention."""
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        from powerzoo.envs.resource.battery import BatteryEnv
        b1 = BatteryEnv(normalize_actions=False)
        b2 = BatteryEnv(normalize_actions=False)
        rid1 = b1.attach(env, bus_id=1)
        rid2 = b2.attach(env, bus_id=2)
        assert 'battery' in rid1
        assert 'battery' in rid2
        assert rid1 != rid2

    def test_register_custom_name(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        from powerzoo.envs.resource.battery import BatteryEnv
        b = BatteryEnv(normalize_actions=False)
        rid = b.attach(env, bus_id=1, name='my_bess')
        assert rid == 'my_bess'

    def test_duplicate_name_raises(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        from powerzoo.envs.resource.battery import BatteryEnv
        b1 = BatteryEnv(normalize_actions=False)
        b2 = BatteryEnv(normalize_actions=False)
        b1.attach(env, bus_id=1, name='dup')
        with pytest.raises(ValueError, match='already exists'):
            b2.attach(env, bus_id=2, name='dup')

    def test_custom_name_reserves_auto_generated_suffix(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        from powerzoo.envs.resource.battery import BatteryEnv

        BatteryEnv(normalize_actions=False).attach(env, bus_id=1, name='battery_1')
        auto = BatteryEnv(normalize_actions=False)
        rid = auto.attach(env, bus_id=2)

        assert rid == 'battery_2'
        assert len(set(env.sub_resources)) == len(env.sub_resources)

    def test_unregister_removes_resource(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        from powerzoo.envs.resource.battery import BatteryEnv
        b = BatteryEnv(normalize_actions=False)
        rid = b.attach(env, bus_id=1)
        assert rid in env.sub_resources
        env.unregister_resource(rid)
        assert rid not in env.sub_resources

    def test_nodes_resources_map_updates(self, case5, simple_time_series):
        """Node-resource mapping matrix should update on register."""
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        from powerzoo.envs.resource.battery import BatteryEnv
        b = BatteryEnv(normalize_actions=False)
        b.attach(env, bus_id=2)
        assert env.nodes_resources_map is not None
        n_nodes = len(case5.nodes)
        assert env.nodes_resources_map.shape == (n_nodes, 1)

    def test_nodes_resources_map_cleared_on_last_unregister(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        from powerzoo.envs.resource.battery import BatteryEnv
        b = BatteryEnv(normalize_actions=False)
        rid = b.attach(env, bus_id=1)
        env.unregister_resource(rid)
        assert env.nodes_resources_map is None


# ── Reset & Episode Tracking ─────────────────────────────────────────

class TestReset:
    """Reset lifecycle and day selection."""

    def test_reset_returns_state_and_info(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        state, info = env.reset(seed=42)
        assert state is not None
        assert isinstance(info, dict)

    def test_reset_seeds_rng(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        env.reset(seed=42)
        assert env.np_random is not None

    def test_reset_with_day_id(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        env.reset(seed=0, day_id=0)
        assert env.day_id == 0

    def test_reset_random_day_reproducible(self, case5, simple_time_series):
        """Same seed → same random day selection."""
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        env.reset(seed=42)
        day1 = env.day_id
        env.reset(seed=42)
        day2 = env.day_id
        assert day1 == day2

    def test_reset_propagates_time_offset_to_sub_resources(self, case5, simple_time_series):
        env = MinimalGridEnv(
            case=case5,
            time_series=simple_time_series,
            randomize_start_time=True,
            max_episode_steps=10,
        )
        from powerzoo.envs.resource.battery import BatteryEnv

        battery = BatteryEnv(normalize_actions=False)
        battery.attach(env, bus_id=1)

        env.reset(seed=42, day_id=0)

        assert battery.day_id == env.day_id
        assert battery.time_step == env.time_step

    def test_episode_reward_resets(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        env.reset(seed=0)
        assert env._episode_reward == 0.0
        assert env._episode_steps == 0


# ── Step & Termination ───────────────────────────────────────────────

class TestStep:
    """Step mechanics and episode termination."""

    def test_step_increments_time(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        env.reset(seed=0, day_id=0)
        initial_time = env.time_step
        env.step({})
        assert env.time_step == initial_time + 1

    def test_step_returns_five_tuple(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        env.reset(seed=0, day_id=0)
        result = env.step({})
        assert len(result) == 5
        state, reward, terminated, truncated, info = result
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_truncated_at_max_steps(self, case5, simple_time_series):
        """After max_episode_steps, truncated should be True."""
        env = MinimalGridEnv(case=case5, time_series=simple_time_series,
                             max_episode_steps=3)
        env.reset(seed=0, day_id=0)
        for i in range(3):
            _, _, terminated, truncated, info = env.step({})
        assert truncated

    def test_truncated_uses_episode_steps_not_time_step(self, case5, simple_time_series):
        """Truncation counts steps taken, not time_step, so a random start offset
        does not shorten the episode below max_episode_steps."""
        env = MinimalGridEnv(
            case=case5,
            time_series=simple_time_series,
            max_episode_steps=5,
            randomize_start_time=True,
        )
        env.reset(seed=0, day_id=0)
        # Force a non-zero offset to expose the bug
        env.time_offset = 3
        env.time_step = 3
        env._episode_steps = 0

        for _ in range(4):
            _, _, _, truncated, _ = env.step({})
            assert not truncated, "episode truncated too early"
        _, _, _, truncated, _ = env.step({})
        assert truncated, "episode should be truncated after max_episode_steps"

    def test_get_state_failure_before_reset_raises_runtime_error(self, case5, simple_time_series):
        """step() before reset() (or _get_state crash with no prior valid state) raises RuntimeError."""
        class AlwaysRaisingGetStateGridEnv(MinimalGridEnv):
            def _get_state(self):
                raise RuntimeError("state unavailable")

        env = AlwaysRaisingGetStateGridEnv(case=case5, time_series=simple_time_series)
        # Manually bring env to a post-init state but skip reset() so
        # _last_valid_state is None, then trigger the fallback path.
        env._episode_steps = 0
        env.day_id = 0
        with pytest.raises(RuntimeError, match="Call reset\\(\\) before step\\(\\)"):
            env.step({})

    def test_pf_failure_terminates_with_penalty_reward(self, case5, simple_time_series):
        """When PF fails, the fixed penalty reward is returned and episode terminates."""
        class FailingPFGridEnv(MinimalGridEnv):
            def _run_power_flow(self, action):
                return False

        env = FailingPFGridEnv(case=case5, time_series=simple_time_series)
        env.reset(seed=0, day_id=0)
        state, reward, terminated, truncated, info = env.step({})

        assert terminated
        assert not truncated
        assert reward == env._PF_FAILURE_REWARD
        assert np.isfinite(reward)
        assert info['pf_converged'] is False
        assert info['cost_exception'] == 1.0
        assert 'episode' in info

    def test_pf_failure_reward_always_finite(self, case5, simple_time_series):
        """NaN reward from _compute_reward is clamped to _PF_FAILURE_REWARD."""
        class NaNRewardGridEnv(MinimalGridEnv):
            def _compute_reward(self, state):
                return float('nan')

        env = NaNRewardGridEnv(case=case5, time_series=simple_time_series)
        env.reset(seed=0, day_id=0)
        _, reward, _, _, _ = env.step({})
        assert np.isfinite(reward)
        assert reward == env._PF_FAILURE_REWARD

    def test_pf_failure_last_valid_state_used_as_fallback(self, case5, simple_time_series):
        """If _get_state() raises on PF failure, the last valid state is used."""
        call_count = {'n': 0}

        class FailingGetStateGridEnv(MinimalGridEnv):
            def _run_power_flow(self, action):
                call_count['n'] += 1
                return call_count['n'] <= 1  # succeed first call, fail thereafter

            def _get_state(self):
                if call_count['n'] > 1:
                    raise RuntimeError("solver state invalid")
                return super()._get_state()

        env = FailingGetStateGridEnv(case=case5, time_series=simple_time_series)
        env.reset(seed=0, day_id=0)
        first_state, _, _, _, _ = env.step({})  # succeeds → saves _last_valid_state
        second_state, _, terminated, _, _ = env.step({})  # _get_state raises → fallback

        assert terminated
        assert second_state == first_state

    def test_episode_summary_at_end(self, case5, simple_time_series):
        """info['episode'] should be present on final step."""
        env = MinimalGridEnv(case=case5, time_series=simple_time_series,
                             max_episode_steps=2)
        env.reset(seed=0, day_id=0)
        env.step({})
        _, _, _, _, info = env.step({})
        assert 'episode' in info
        assert 'r' in info['episode']
        assert 'l' in info['episode']

    def test_pf_converged_in_info(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        env.reset(seed=0, day_id=0)
        _, _, _, _, info = env.step({})
        assert 'pf_converged' in info

    def test_resource_auto_stepped(self, case5, simple_time_series):
        """Resources without explicit actions are auto-stepped (None)."""
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        from powerzoo.envs.resource.battery import BatteryEnv
        b = BatteryEnv(normalize_actions=False)
        b.attach(env, bus_id=1)
        env.reset(seed=0, day_id=0)
        initial_soc = b.soc
        # Step without providing battery action → auto-step with None
        env.step({})
        # Battery with action=None should remain at same SOC (no charge/discharge)
        assert b.soc == initial_soc


# ── Randomized Start Time ────────────────────────────────────────────

class TestRandomizeStartTime:
    """F5 fix: randomize intra-day start offset."""

    def test_default_no_offset(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        env.reset(seed=0, day_id=0)
        assert env.time_offset == 0
        assert env.time_step == 0

    def test_randomize_produces_offset(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series,
                             randomize_start_time=True, max_episode_steps=10)
        env.reset(seed=42, day_id=0)
        # With max_episode_steps=10 and steps_per_day=48, offset can be up to 38
        assert env.time_offset >= 0
        assert env.time_step == env.time_offset

    def test_episode_length_unaffected_by_offset(self, case5, simple_time_series):
        """An episode with a randomised offset should run for exactly max_episode_steps."""
        env = MinimalGridEnv(
            case=case5, time_series=simple_time_series,
            randomize_start_time=True, max_episode_steps=5,
        )
        env.reset(seed=99, day_id=0)
        steps = 0
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step({})
            steps += 1
            done = terminated or truncated
        assert steps == 5

    def test_current_time_index_matches_flat_regular_index(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        env.reset(seed=0, day_id=1)
        env.time_step = 5
        assert env._get_current_time_index() == env.steps_per_day + 5


# ── Day Profile ──────────────────────────────────────────────────────

class TestDayProfile:
    """_get_day_profile: normalised [0,1] profile extraction."""

    def test_profile_normalised(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        profile = env._get_day_profile(day_id=0, column=S.LOAD_ACTUAL_MW)
        assert profile.max() <= 1.0 + 1e-9
        assert profile.min() >= 0.0 - 1e-9

    def test_profile_length_matches_steps(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        profile = env._get_day_profile(day_id=0, column=S.LOAD_ACTUAL_MW)
        assert len(profile) == env.steps_per_day

    def test_missing_column_returns_zeros(self, case5, simple_time_series):
        env = MinimalGridEnv(case=case5, time_series=simple_time_series)
        profile = env._get_day_profile(day_id=0, column='NonexistentCol')
        np.testing.assert_array_equal(profile, np.zeros(env.steps_per_day))
