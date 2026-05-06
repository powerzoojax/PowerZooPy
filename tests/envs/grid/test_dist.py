"""Tests for powerzoo.envs.grid.dist — DistGridEnv.

DistGridEnv wraps a radial distribution network (default: IEEE 33-bus) with:
  - Backward/Forward Sweep (BFS) power flow (DistFlow equations)
  - Voltage limits check (v_min ≤ V ≤ v_max in p.u.)
  - Branch thermal limits (|S| ≤ cap MVA)
  - Loss-based reward for CMDP formulations

Domain knowledge:
  - Radial distribution: tree topology, single feeder, one slack bus
  - BFS converges in O(10) iterations for radial networks
  - Voltage drop: ΔV ≈ (PR + QX) / V  along feeder
  - Power loss: P_loss = I²R = (P² + Q²)/(V²) × R per branch
  - baseMVA / baseKV: per-unit conversion bases
  - Apparent power |S| = √(P² + Q²) used for thermal limit check
"""
import numpy as np
import pandas as pd
import pytest

from powerzoo.data import signals as S
from powerzoo.envs.grid.dist import DistGridEnv


# ── Constructor & Configuration ──────────────────────────────────────

class TestDistGridEnvInit:
    """Constructor defaults, difficulty presets, topology."""

    def test_default_case33(self):
        """Default distribution case is 33-bus (Case33bw)."""
        env = DistGridEnv(time_series=np.ones(48) * 3)
        assert env.n_nodes == 33 or 'Case33' in type(env.case).__name__

    def test_voltage_limits_default(self):
        env = DistGridEnv(time_series=np.ones(48) * 3)
        assert env.v_min == 0.90
        assert env.v_max == 1.10

    def test_difficulty_easy(self):
        env = DistGridEnv(difficulty='easy', time_series=np.ones(48) * 3)
        assert env.v_min == 0.88
        assert env.v_max == 1.12

    def test_difficulty_medium(self):
        env = DistGridEnv(difficulty='medium', time_series=np.ones(48) * 3)
        assert env.v_min == 0.90
        assert env.v_max == 1.10

    def test_difficulty_hard(self):
        env = DistGridEnv(difficulty='hard', time_series=np.ones(48) * 3)
        assert env.v_min == 0.93
        assert env.v_max == 1.07

    def test_invalid_difficulty_raises(self):
        with pytest.raises(ValueError, match="difficulty"):
            DistGridEnv(difficulty='extreme', time_series=np.ones(48) * 3)

    def test_topology_built(self, case33):
        env = DistGridEnv(case=case33, time_series=np.ones(48) * 3)
        assert env.topo is not None
        assert env.n_nodes == len(case33.nodes)
        assert env.n_lines > 0

    def test_slack_bus_default(self, case33):
        env = DistGridEnv(case=case33, time_series=np.ones(48) * 3)
        assert env.slack_bus_id == getattr(case33, 'slack_bus', 0)

    def test_v_slack_default(self):
        env = DistGridEnv(time_series=np.ones(48) * 3)
        assert env.v_slack == 1.0

    def test_base_values(self, case33):
        env = DistGridEnv(case=case33, time_series=np.ones(48) * 3)
        assert env.baseMVA > 0
        assert env.baseKV > 0

    def test_ref_bus_alias(self, case33):
        """ref_bus property should alias slack_bus_id."""
        env = DistGridEnv(case=case33, time_series=np.ones(48) * 3)
        assert env.ref_bus == env.slack_bus_id

    def test_v_ref_mag_alias(self, case33):
        """v_ref_mag property should alias v_slack."""
        env = DistGridEnv(case=case33, time_series=np.ones(48) * 3)
        assert env.v_ref_mag == env.v_slack


# ── Observation & Action Spaces ──────────────────────────────────────

class TestDistSpaces:
    """Space configuration for distribution grid."""

    def test_observation_space_shape(self, case33):
        """obs = [v(n)] + [p_flow(n_l)] + [q_flow(n_l)] + [p_load(n)] + [q_load(n)] + [time(2)]."""
        env = DistGridEnv(case=case33, time_series=np.ones(48) * 3)
        n = env.n_nodes
        n_l = env.n_lines
        expected = 3 * n + 2 * n_l + 2
        assert env.observation_space.shape == (expected,)

    def test_action_space_empty(self, case33):
        """Distribution grid has 0-dim action (pure observer mode)."""
        env = DistGridEnv(case=case33, time_series=np.ones(48) * 3)
        assert env.action_space.shape == (0,)

    def test_obs_names_count(self, case33):
        env = DistGridEnv(case=case33, time_series=np.ones(48) * 3)
        assert len(env.obs_names) == env.observation_space.shape[0]


# ── Reset & Power Flow ───────────────────────────────────────────────

class TestDistReset:
    """Reset triggers BFS and returns valid state."""

    def test_reset_returns_state(self, case33, simple_time_series):
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        state, info = env.reset(seed=42, day_id=0)
        assert isinstance(state, dict)
        assert isinstance(info, dict)
        assert 'nodes' in state
        assert 'lines' in state
        assert 'is_safe' in state
        assert 'p_loss_MW' in state
        assert 'p_slack_MW' in state
        assert 'q_slack_MVAr' in state
        assert 'is_diverged' in state
        assert 'voltage_collapse' in state

    def test_nodes_have_voltage(self, case33, simple_time_series):
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        assert env._nodes is not None
        assert 'v_mag' in env._nodes.columns
        # Voltage magnitudes should be near 1.0 p.u. for normal loading
        v = env._nodes['v_mag'].values
        assert np.all(v > 0.5)
        assert np.all(v < 1.5)

    def test_slack_bus_voltage(self, case33, simple_time_series):
        """Slack bus voltage should be exactly v_slack."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        v_slack_actual = env._nodes['v_mag'].iloc[env.slack_bus_id]
        np.testing.assert_allclose(v_slack_actual, env.v_slack, atol=1e-6)

    def test_state_slack_exchange_matches_root_branch_sum(self, case33, simple_time_series):
        """State should expose feeder-head active/reactive exchange."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        state, _ = env.reset(seed=42, day_id=0)
        slack_mask = env.topo.sending_nodes == env.slack_bus_id
        np.testing.assert_allclose(
            state['p_slack_MW'],
            env._lines.loc[slack_mask, 'p_flow_MW'].sum(),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            state['q_slack_MVAr'],
            env._lines.loc[slack_mask, 'q_flow_MVAr'].sum(),
            atol=1e-9,
        )
        assert state['is_diverged'] is False
        assert state['voltage_collapse'] is False


# ── BFS Power Flow ───────────────────────────────────────────────────

class TestBFSPowerFlow:
    """Backward/Forward Sweep convergence and physics."""

    def test_bfs_converges(self, case33, simple_time_series):
        """BFS should converge for normal loading conditions."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        assert env._converged is True

    def test_line_flows_exist(self, case33, simple_time_series):
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        assert env._lines is not None
        assert 'p_flow_MW' in env._lines.columns
        assert 'q_flow_MVAr' in env._lines.columns

    def test_losses_non_negative(self, case33, simple_time_series):
        """Active power loss (I²R) must be non-negative — resistive heating."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        assert env._p_loss >= 0.0
        assert env._q_loss >= 0.0  # Q loss (I²X) also non-negative

    def test_loss_columns_in_lines(self, case33, simple_time_series):
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        assert 'p_loss_MW' in env._lines.columns
        assert 'q_loss_MVAr' in env._lines.columns

    def test_voltage_drop_along_feeder(self, case33, simple_time_series):
        """Voltage should generally decrease along the radial feeder.

        In radial distribution networks, buses far from the slack bus tend
        to have lower voltage due to accumulated ΔV ≈ (PR + QX) / V drops.
        """
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        v = env._nodes['v_mag'].values
        # Slack bus should have the highest or near-highest voltage
        assert v[env.slack_bus_id] >= v.mean()

    def test_cal_pf_returns_tuple(self, case33, simple_time_series):
        """cal_pf(df=False) returns (v_mag, p_flow_MW)."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        v_mag, p_flow = env.cal_pf(df=False)
        assert v_mag.shape == (env.n_nodes,)

    def test_cal_pf_df_returns_dataframes(self, case33, simple_time_series):
        """cal_pf(df=True) returns (nodes_df, lines_df)."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        nodes, lines = env.cal_pf(df=True)
        assert isinstance(nodes, pd.DataFrame)
        assert isinstance(lines, pd.DataFrame)
        assert 'v_mag' in nodes.columns
        assert 'p_flow_MW' in lines.columns


# ── Observation ──────────────────────────────────────────────────────

class TestDistObs:
    """Observation construction and normalisation."""

    def test_obs_shape(self, case33, simple_time_series):
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        obs = env.obs()
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32

    def test_obs_voltage_normalisation(self, case33, simple_time_series):
        """Voltage part of obs is (V - 1.0) / 0.1, so ~0 at nominal."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        obs = env.obs()
        # First n_nodes entries are normalised voltages
        v_norm = obs[:env.n_nodes]
        # Should be near 0 for nominal loading (V ≈ 1.0 p.u.)
        assert np.abs(v_norm).max() < 5.0  # within ±0.5 p.u. of nominal

    def test_obs_q_flow_present(self, case33, simple_time_series):
        """Q_flow slice (index n_nodes+n_lines .. n_nodes+2*n_lines) should be non-trivial."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        obs = env.obs()
        n, n_l = env.n_nodes, env.n_lines
        q_flow_norm = obs[n + n_l : n + 2 * n_l]
        assert q_flow_norm.shape == (n_l,)
        # For a loaded network, Q flows should be non-zero
        assert np.any(q_flow_norm != 0.0)

    def test_obs_q_load_present(self, case33, simple_time_series):
        """Q_load slice (index 2*n_nodes+2*n_lines .. 3*n_nodes+2*n_lines) should be positive."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        obs = env.obs()
        n, n_l = env.n_nodes, env.n_lines
        q_load_norm = obs[2 * n_l + 2 * n : 2 * n_l + 3 * n]
        assert q_load_norm.shape == (n,)
        # Q loads should be non-negative (lagging load)
        assert np.all(q_load_norm >= 0.0)

    def test_obs_names_include_q(self, case33):
        """obs_names should contain q_norm entries for lines and nodes."""
        env = DistGridEnv(case=case33, time_series=np.ones(48) * 3)
        names = env.obs_names
        assert any('q_norm' in name for name in names), "Missing q_flow entries"
        assert any('q_load_norm' in name for name in names), "Missing q_load entries"

    def test_obs_nan_free_after_divergence(self, case33, simple_time_series):
        """obs() must never emit NaN, even when power flow diverged.

        NaN in the observation vector causes neural network weight poisoning
        (NaN propagation through gradients), crashing RL training instantly.
        """
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        # Simulate diverged power flow: inject NaN into cached results
        env._nodes = env._nodes.copy()
        env._nodes['v_mag'] = np.nan
        env._lines = env._lines.copy()
        env._lines['p_flow_MW'] = np.nan
        env._lines['q_flow_MVAr'] = np.nan
        env._pf_failed = True

        obs = env.obs()
        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs)), (
            f"obs() contains non-finite values after PF divergence: "
            f"NaN={np.isnan(obs).sum()}, Inf={np.isinf(obs).sum()}"
        )

    def test_obs_failed_state_uses_penalty_voltage_and_zero_flows(self, case33, simple_time_series):
        """Failed states should map to an explicit catastrophe observation.

        GymnasiumWrapper calls ``obs(state)`` on the step result, so the
        explicit-state path must also expose the PF-failure sentinel rather
        than a cleaned-up near-boundary voltage profile.
        """
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)

        failed_nodes = env._nodes.copy()
        failed_nodes['v_mag'] = np.nan
        failed_lines = env._lines.copy()
        failed_lines['p_flow_MW'] = np.nan
        failed_lines['q_flow_MVAr'] = np.nan
        failed_state = {
            'nodes': failed_nodes,
            'lines': failed_lines,
            'time_step': env.time_step,
            'safety_info': {'converged': False},
        }

        obs = env.obs(failed_state)
        n = env.n_nodes
        n_l = env.n_lines
        expected_v = (max(env.v_min - env._PF_FAILURE_VOLTAGE_DROP_PU, 0.0) - 1.0) / 0.1

        np.testing.assert_allclose(obs[:n], expected_v, atol=1e-6)
        np.testing.assert_allclose(obs[n:n + n_l], 0.0, atol=1e-9)
        np.testing.assert_allclose(obs[n + n_l:n + 2 * n_l], 0.0, atol=1e-9)

    def test_obs_time_encoding_uses_state_day_and_actual_clock(self, case33):
        """time_sin/time_cos should follow the real clock encoded by the state."""
        idx = pd.date_range('2024-01-01 12:00', periods=96, freq='30min', tz='UTC')
        time_series = pd.DataFrame(
            {S.LOAD_ACTUAL_MW: np.linspace(100.0, 200.0, len(idx))},
            index=idx,
        )
        env = DistGridEnv(case=case33, time_series=time_series)
        state, _ = env.reset(seed=42, day_id=0)

        # Disturb the live env clock: obs(state) must still use the state's noon timestamp.
        env.day_id = 1
        env.time_step = 17
        obs = env.obs(state)

        np.testing.assert_allclose(
            obs[-2:],
            np.array([0.0, -1.0], dtype=np.float32),
            atol=1e-6,
        )


# ── Safety Check ─────────────────────────────────────────────────────

class TestDistSafetyCheck:
    """Voltage and thermal limit checks."""

    def test_safety_check_structure(self, case33, simple_time_series):
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        is_safe, info = env.safety_check(env._nodes, env._lines, with_info=True)
        assert isinstance(is_safe, (bool, np.bool_))
        assert isinstance(info, dict)
        assert 'v_min_actual' in info
        assert 'v_max_actual' in info

    def test_voltage_violation_detected(self, case33, simple_time_series):
        """Artificially low voltage triggers violation."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        # Create fake nodes with very low voltage
        nodes = env._nodes.copy()
        nodes['v_mag'] = 0.80  # below v_min=0.90
        is_safe, info = env.safety_check(nodes, env._lines, with_info=True)
        assert not is_safe
        assert len(info['v_violation_nodes']) > 0

    def test_tight_limits_more_violations(self, case33, simple_time_series):
        """Tighter voltage limits should produce more violations."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        _, info_loose = env.safety_check(env._nodes, env._lines,
                                         v_min=0.80, v_max=1.20, with_info=True)
        _, info_tight = env.safety_check(env._nodes, env._lines,
                                         v_min=0.99, v_max=1.01, with_info=True)
        assert len(info_tight['v_violation_nodes']) >= len(info_loose['v_violation_nodes'])

    def test_nan_voltages_detected_as_unsafe(self, case33, simple_time_series):
        """NaN voltages must NOT slip through as 'safe' (np.nan < v_min == False).

        This is the critical NaN-comparison pitfall: if BFS diverges, v_mag
        can be NaN, and np.nan < 0.9 evaluates to False in NumPy, which would
        incorrectly make a diverged state appear safe.
        """
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        nodes = env._nodes.copy()
        nodes['v_mag'] = np.nan
        env._converged = False

        is_safe, info = env.safety_check(nodes, env._lines, with_info=True)
        assert not is_safe, "NaN voltages were misclassified as safe"
        assert len(info['v_violation_nodes']) == env.n_nodes, (
            "All nodes should be flagged as violating when PF diverged"
        )
        assert info['converged'] is False

    def test_inf_voltages_detected_as_unsafe(self, case33, simple_time_series):
        """Inf voltages must be flagged unsafe even if _converged is True."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        nodes = env._nodes.copy()
        nodes['v_mag'] = np.inf
        # _converged is True, but data has Inf → still unsafe
        env._converged = True

        is_safe, info = env.safety_check(nodes, env._lines, with_info=True)
        assert not is_safe, "Inf voltages were misclassified as safe"

    def test_safety_check_rejects_ndarray_lines(self, case33, simple_time_series):
        """safety_check must raise TypeError when lines_result is a raw ndarray.

        Passing ndarray silently skips all thermal limit checks, producing
        a falsely-safe result without any error or warning.
        """
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        v_mag, p_branch = env.cal_pf(df=False)  # returns ndarrays
        with pytest.raises(TypeError, match="DataFrame"):
            env.safety_check(env._nodes, p_branch)

    def test_safety_check_rejects_ndarray_nodes(self, case33, simple_time_series):
        """safety_check must raise TypeError when nodes_result is a raw ndarray."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        v_mag, _ = env.cal_pf(df=False)
        with pytest.raises(TypeError, match="DataFrame"):
            env.safety_check(v_mag, env._lines)


# ── Reward & CMDP Cost ───────────────────────────────────────────────

class TestDistReward:
    """Loss-based reward and CMDP cost channels."""

    def test_reward_is_negative_loss(self, case33, simple_time_series):
        """Default reward = -0.1 × P_loss_MW (no violation penalty)."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        state, reward, _, _, info = env.step({})
        expected = -0.1 * state['p_loss_MW']
        np.testing.assert_allclose(reward, expected, atol=1e-9)

    def test_violation_penalty_weight_zero(self, case33, simple_time_series):
        """violation_penalty_weight=0 keeps reward purely loss-based."""
        env = DistGridEnv(case=case33, time_series=simple_time_series,
                          violation_penalty_weight=0.0)
        env.reset(seed=42, day_id=0)
        state, reward, _, _, info = env.step({})
        assert 'violation_penalty' not in state['reward_components']
        np.testing.assert_allclose(reward, -0.1 * state['p_loss_MW'], atol=1e-9)

    def test_violation_penalty_weight_positive(self, case33, simple_time_series):
        """violation_penalty_weight > 0 folds violations into scalar reward."""
        env = DistGridEnv(case=case33, time_series=simple_time_series,
                          violation_penalty_weight=1.0)
        env.reset(seed=42, day_id=0)
        state, reward, _, _, info = env.step({})
        components = state['reward_components']
        assert 'violation_penalty' in components
        expected = components['loss_penalty'] + components['violation_penalty']
        np.testing.assert_allclose(reward, expected, atol=1e-9)

    def test_info_has_cmdp_fields(self, case33, simple_time_series):
        """CMDP cost fields exposed in info."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        _, _, _, _, info = env.step({})
        assert 'cost_voltage_violation' in info
        assert 'cost_thermal_overload' in info
        assert 'cost_sum' in info
        assert 'goal_met' in info

    def test_cost_sum_is_violation_sum(self, case33, simple_time_series):
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        _, _, _, _, info = env.step({})
        expected_sum = info['cost_voltage_violation'] + info['cost_thermal_overload']
        np.testing.assert_allclose(info['cost_sum'], expected_sum, atol=1e-9)

    def test_v_dev_penalty_weight_zero_default(self, case33, simple_time_series):
        """v_dev_penalty_weight=0 (default): reward is unchanged, no v_dev_penalty component."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        state, reward, _, _, _ = env.step({})
        assert 'v_dev_penalty' not in state['reward_components']
        np.testing.assert_allclose(reward, -0.1 * state['p_loss_MW'], atol=1e-9)

    def test_v_dev_penalty_weight_positive(self, case33, simple_time_series):
        """v_dev_penalty_weight > 0 adds a continuous voltage-deviation term."""
        env = DistGridEnv(case=case33, time_series=simple_time_series,
                          v_dev_penalty_weight=1.0)
        env.reset(seed=42, day_id=0)
        state, reward, _, _, _ = env.step({})
        components = state['reward_components']
        assert 'v_dev_penalty' in components
        assert components['v_dev_penalty'] <= 0.0, "v_dev_penalty must be non-positive"
        expected = components['loss_penalty'] + components['v_dev_penalty']
        np.testing.assert_allclose(reward, expected, atol=1e-9)

    def test_v_dev_penalty_is_negative_when_voltages_deviate(self, case33, simple_time_series):
        """Voltage deviation penalty is strictly negative when any v ≠ 1.0."""
        env = DistGridEnv(case=case33, time_series=simple_time_series,
                          v_dev_penalty_weight=1.0)
        env.reset(seed=42, day_id=0)
        state, _, _, _, _ = env.step({})
        # In a loaded network some nodes deviate from 1.0 p.u.
        v_mag = state['nodes']['v_mag'].values
        if np.any(v_mag != 1.0):
            assert state['reward_components']['v_dev_penalty'] < 0.0

    def test_v_dev_penalty_scales_with_weight(self, case33, simple_time_series):
        """Doubling v_dev_penalty_weight should double the penalty magnitude."""
        env1 = DistGridEnv(case=case33, time_series=simple_time_series,
                           v_dev_penalty_weight=1.0)
        env2 = DistGridEnv(case=case33, time_series=simple_time_series,
                           v_dev_penalty_weight=2.0)
        env1.reset(seed=42, day_id=0)
        env2.reset(seed=42, day_id=0)
        state1, _, _, _, _ = env1.step({})
        state2, _, _, _, _ = env2.step({})
        p1 = state1['reward_components']['v_dev_penalty']
        p2 = state2['reward_components']['v_dev_penalty']
        np.testing.assert_allclose(p2, 2.0 * p1, rtol=1e-6)

    def test_loss_penalty_weight_scales_reward(self, case33, simple_time_series):
        """loss_penalty_weight should rescale the loss-only reward component."""
        env1 = DistGridEnv(case=case33, time_series=simple_time_series,
                           loss_penalty_weight=0.1)
        env2 = DistGridEnv(case=case33, time_series=simple_time_series,
                           loss_penalty_weight=0.5)
        env1.reset(seed=42, day_id=0)
        env2.reset(seed=42, day_id=0)

        state1, reward1, _, _, _ = env1.step({})
        state2, reward2, _, _, _ = env2.step({})

        np.testing.assert_allclose(state1['p_loss_MW'], state2['p_loss_MW'], atol=1e-9)
        np.testing.assert_allclose(
            state1['reward_components']['loss_penalty'],
            -0.1 * state1['p_loss_MW'],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            state2['reward_components']['loss_penalty'],
            -0.5 * state2['p_loss_MW'],
            atol=1e-9,
        )
        np.testing.assert_allclose(reward2, 5.0 * reward1, atol=1e-9)


# ── Step Cycle ───────────────────────────────────────────────────────

class TestDistStep:
    """Step mechanics."""

    def test_step_returns_five_tuple(self, case33, simple_time_series):
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        result = env.step({})
        assert len(result) == 5

    def test_multi_step_episode(self, case33, simple_time_series):
        """Multiple steps should work without errors."""
        env = DistGridEnv(case=case33, time_series=simple_time_series,
                          max_episode_steps=10)
        env.reset(seed=42, day_id=0)
        for _ in range(10):
            state, reward, terminated, truncated, info = env.step({})
            if terminated or truncated:
                break

    def test_episode_metrics_loss(self, case33, simple_time_series):
        env = DistGridEnv(case=case33, time_series=simple_time_series,
                          max_episode_steps=3)
        env.reset(seed=42, day_id=0)
        total_loss = 0.0
        for _ in range(3):
            state, _, terminated, truncated, info = env.step({})
            total_loss += state['p_loss_MW']
            if terminated or truncated:
                break
        if 'episode' in info:
            np.testing.assert_allclose(
                info['episode']['metrics']['total_loss_mw'],
                total_loss, atol=1e-6,
            )

    def test_step_terminates_and_marks_exception_on_pf_failure(
        self, case33, simple_time_series, monkeypatch
    ):
        """PF failure should terminate the episode and expose cost_exception."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)

        failed_nodes = env._nodes.copy()
        failed_nodes['v_mag'] = np.nan
        failed_lines = env._lines.copy()
        failed_lines['p_flow_MW'] = np.nan
        failed_lines['q_flow_MVAr'] = np.nan
        failed_lines['p_loss_MW'] = np.nan
        failed_lines['q_loss_MVAr'] = np.nan

        def fake_run_power_flow(action):
            env._prev_nodes = env._nodes
            env._prev_lines = env._lines
            env._nodes = failed_nodes
            env._lines = failed_lines
            env._converged = False
            env._iterations = env.max_iter
            env._pf_failed = True
            env._is_safe = False
            env._safety_info = {
                'v_violation_nodes': list(range(env.n_nodes)),
                'line_violation_ids': list(range(env.n_lines)),
                'converged': False,
                'iterations': env.max_iter,
            }
            env._p_loss, env._q_loss = env.get_total_loss(env._lines)
            return False

        monkeypatch.setattr(env, '_run_power_flow', fake_run_power_flow)

        state, reward, terminated, truncated, info = env.step({})
        assert terminated is True
        assert truncated is False
        assert info['pf_converged'] is False
        assert info['cost_exception'] == 1.0
        assert np.isfinite(reward)

        obs = env.obs(state)
        expected_v = (max(env.v_min - env._PF_FAILURE_VOLTAGE_DROP_PU, 0.0) - 1.0) / 0.1
        np.testing.assert_allclose(obs[:env.n_nodes], expected_v, atol=1e-6)

    def test_step_terminates_on_voltage_collapse(self, case33, simple_time_series):
        """Severe low-voltage states should surface as PF failure, not clamped success."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)

        # Under the scaled time-series fixture, collapse starts around 12x.
        p_load = env._nodes['p_load_MW'].values * 12.0
        q_load = env._nodes['q_load_MVAr'].values * 12.0

        with pytest.warns(RuntimeWarning, match="voltage collapse"):
            state, reward, terminated, truncated, info = env.step({
                'p_load': p_load,
                'q_load': q_load,
            })

        assert terminated is True
        assert truncated is False
        assert info['pf_converged'] is False
        assert info['cost_exception'] == 1.0
        assert info['voltage_collapse'] is True
        assert 'is_diverged' in info
        assert 'is_diverged' in state
        assert 'is_diverged' in state['safety_info']
        assert state['voltage_collapse'] is True
        assert state['safety_info']['voltage_collapse'] is True
        assert np.isfinite(reward)


# ── Load scaling helpers ─────────────────────────────────────────────

class TestDistLoadScaling:
    """Distribution load helpers should preserve node-level reactive shape."""

    def test_get_node_loads_q_scales_per_node_and_handles_zero_base_p(
        self, case33, simple_time_series
    ):
        """Reactive demand should follow node-level P scaling, not a system-wide ratio."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)

        base_p = np.zeros(env.n_nodes)
        base_q = np.zeros(env.n_nodes)
        current_p = np.zeros(env.n_nodes)
        base_p[:3] = [1.0, 2.0, 0.0]
        base_q[:3] = [0.4, 0.8, 0.3]
        current_p[:3] = [0.5, 4.0, 1.0]

        env._cached_case_p = base_p
        env._cached_case_q = base_q
        env._nodes_loads_map = np.eye(env.n_nodes)
        env._node_loads_p = np.array([current_p], dtype=float)
        env._node_loads_q = None
        env.time_step = 0

        q_load = env._get_node_loads_q()

        expected = np.zeros(env.n_nodes)
        expected[:3] = [0.2, 1.6, 0.0]
        np.testing.assert_allclose(q_load, expected, atol=1e-9)

    def test_explicit_reactive_time_series_overrides_pf_scaled_q(self, case33):
        """load.reactive_mvar should take precedence over inferred Q scaling."""
        idx = pd.date_range('2024-01-01', periods=2, freq='30min', tz='UTC')
        time_series = pd.DataFrame(
            {
                S.LOAD_ACTUAL_MW: [100.0, 200.0],
                S.LOAD_REACTIVE_MVAR: [80.0, 80.0],
            },
            index=idx,
        )
        env = DistGridEnv(case=case33, time_series=time_series)
        env.reset(seed=42, day_id=0)
        env.time_step = 1

        q_load = env._get_node_loads_q()

        q_base = case33.loads['Qd'].to_numpy(dtype=float)
        q_ratio = q_base / q_base.sum()
        explicit_q = float(env._time_series_data[S.LOAD_REACTIVE_MVAR].iloc[1])
        expected_q = env._get_loads_map().dot(explicit_q * q_ratio)
        np.testing.assert_allclose(q_load, expected_q, atol=1e-9)

        base_p = env._get_node_loads('Pd')
        base_q = env._get_node_loads('Qd')
        current_p = env._get_node_loads_p()
        inferred_q = base_q * np.divide(
            current_p,
            base_p,
            out=np.zeros_like(current_p, dtype=float),
            where=base_p > 0,
        )
        assert not np.allclose(q_load, inferred_q)


# ── Resource Integration ─────────────────────────────────────────────

class TestDistResourceIntegration:
    """DER resources modify net load in BFS power flow."""

    def test_battery_reduces_net_load(self, case33, simple_time_series):
        """A discharging battery injects power → reduces net load at its bus."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)

        # Baseline loss
        baseline_state = env._get_state()
        baseline_loss = baseline_state['p_loss_MW']

        # Attach a battery that will discharge
        from powerzoo.envs.resource.battery import BatteryEnv
        batt = BatteryEnv(normalize_actions=False, capacity_mwh=10.0, power_mw=5.0, initial_soc=0.9)
        batt.attach(env, bus_id=2)
        batt.reset()

        # Discharge: positive current_p = injection
        batt.step({'p_mw': 5.0})

        # Re-run power flow with battery
        env._run_power_flow({})
        batt_state = env._get_state()
        # Battery injection should change the power flow result
        assert batt_state is not None

    def test_update_action_space_reflects_resource(self, case33, simple_time_series):
        """After registering a resource, update_action_space reflects new dimension."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        assert env.action_space.shape == (0,), "Initial action_space should be empty"

        from powerzoo.envs.resource.battery import BatteryEnv
        batt = BatteryEnv(normalize_actions=True, capacity_mwh=5.0, power_mw=2.0, initial_soc=0.5)
        # register_resource automatically calls update_action_space
        batt.attach(env, bus_id=5)
        batt.reset()

        # action_space should now have at least 1 dimension
        assert env.action_space.shape[0] >= 1, (
            "action_space should expand when resource is registered"
        )

    def test_update_action_space_empty_after_init(self, case33):
        """update_action_space on no-resource env keeps shape (0,)."""
        env = DistGridEnv(case=case33, time_series=np.ones(48) * 3)
        env.update_action_space()
        assert env.action_space.shape == (0,)

    def test_update_action_space_uses_grid_action_bounds(self, case33, simple_time_series):
        """update_action_space should delegate bounds to res.grid_action_bounds()."""
        from unittest.mock import MagicMock

        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)

        mock_res = MagicMock()
        mock_res.grid_action_bounds.return_value = (-2.0, 2.0)
        env.sub_resources = {'bat': mock_res}

        env.update_action_space()

        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == pytest.approx(-2.0)
        assert env.action_space.high[0] == pytest.approx(2.0)
        assert env.action_names == ['bat']

    def test_resource_p_array_does_not_broadcast(self, case33, simple_time_series):
        """cal_pf must not trigger NumPy broadcasting when current_p_mw is 1D.

        If current_p_mw is a 1-element array [val] (common in RL action
        pipelines), the old np.array([...]) construction produced a (n,1)
        column vector.  nodes_resources_map.dot((n,1)) then returned (n_nodes,1)
        instead of (n_nodes,), and p_load_mw -= (n_nodes,1) silently expanded
        to a (n_nodes, n_nodes) matrix, corrupting the power flow inputs.
        """
        from unittest.mock import MagicMock
        import gymnasium.spaces as _spaces

        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)

        # Mock resource whose current_p_mw is a 1-element ndarray (RL convention)
        mock_res = MagicMock()
        mock_res.current_p_mw = np.array([2.0])   # 1D array, shape (1,)
        mock_res.current_q_mvar = np.array([0.5])
        mock_res.action_space = _spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)

        env.sub_resources = {'mock': mock_res}
        # Build a valid nodes_resources_map (bus 5 → resource index 0)
        import scipy.sparse as sp
        n = env.n_nodes
        mat = np.zeros((n, 1), dtype=float)
        mat[5, 0] = 1.0
        env.nodes_resources_map = mat
        env._resource_col_index = {'mock': 0}

        # If broadcasting occurs, cal_pf will raise an error or return wrong shapes.
        nodes_df, lines_df = env.cal_pf(df=True)
        assert nodes_df.shape[0] == n, "nodes_df has wrong row count — broadcasting likely occurred"
        assert lines_df.shape[0] == env.n_lines, "lines_df has wrong row count"

    def test_resource_injections_follow_mapping_column_order_not_dict_order(
        self, case33, simple_time_series
    ):
        """cal_pf should align resource power by map column index, not dict iteration order."""
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)

        class _Res:
            def __init__(self, bus_id, p_mw, q_mvar):
                self.bus_id = bus_id
                self.current_p_mw = np.array([p_mw], dtype=float)
                self.current_q_mvar = np.array([q_mvar], dtype=float)

            def grid_obs(self):
                return np.zeros(1, dtype=np.float32)

            def grid_obs_names(self, rid):
                return [f'{rid}_p_norm']

            def grid_action_bounds(self):
                return (0.0, 1.0)

            def status(self):
                return {}

        env.register_resource(_Res(bus_id=2, p_mw=0.8, q_mvar=0.10), bus_id=2, name='res_a')
        env.register_resource(_Res(bus_id=3, p_mw=0.2, q_mvar=0.05), bus_id=3, name='res_b')

        # Reverse plain dict order to emulate debugging / mutation of sub_resources.
        env.sub_resources = {
            'res_b': env.sub_resources['res_b'],
            'res_a': env.sub_resources['res_a'],
        }

        nodes_df, _ = env.cal_pf(df=True)

        expected_p = env._get_node_loads_p()
        expected_q = env._get_node_loads_q()
        expected_p[env._get_internal_bus_id(2)] -= 0.8
        expected_p[env._get_internal_bus_id(3)] -= 0.2
        expected_q[env._get_internal_bus_id(2)] -= 0.10
        expected_q[env._get_internal_bus_id(3)] -= 0.05

        np.testing.assert_allclose(nodes_df['p_load_MW'].values, expected_p, atol=1e-9)
        np.testing.assert_allclose(nodes_df['q_load_MVAr'].values, expected_q, atol=1e-9)


# ── DER Observation Integration ─────────────────────────────────────────────

class TestDistDERObs:
    """DER states appear in DistGridEnv obs / observation_space after registration."""

    def test_obs_dim_grows_with_battery(self, case33, simple_time_series):
        """Registering a BatteryEnv adds 4 obs dims (soc, p_discharge_max, p_charge_max, p_mw)."""
        from powerzoo.envs.resource.battery import BatteryEnv
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        dim_base = env.observation_space.shape[0]
        BatteryEnv(parent=env, bus_id=5, capacity_mwh=10.0, power_mw=5.0)
        assert env.observation_space.shape[0] == dim_base + 4

    def test_obs_dim_grows_with_solar(self, case33, simple_time_series):
        """Registering a SolarEnv adds 2 obs dims (available_cf, p_mw_norm)."""
        from powerzoo.envs.resource.renewable import SolarEnv
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        dim_base = env.observation_space.shape[0]
        SolarEnv(parent=env, bus_id=3, capacity_mw=50.0)
        assert env.observation_space.shape[0] == dim_base + 2

    def test_obs_names_length_matches_obs_space(self, case33, simple_time_series):
        """obs_names length must equal observation_space dim after resource registration."""
        from powerzoo.envs.resource.battery import BatteryEnv
        from powerzoo.envs.resource.renewable import SolarEnv
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        SolarEnv(parent=env, bus_id=3, capacity_mw=30.0)
        BatteryEnv(parent=env, bus_id=5, capacity_mwh=10.0, power_mw=5.0)
        assert len(env.obs_names) == env.observation_space.shape[0]

    def test_obs_shape_matches_space_after_reset(self, case33, simple_time_series):
        """After reset, obs() shape must equal observation_space.shape."""
        from powerzoo.envs.resource.battery import BatteryEnv
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        BatteryEnv(parent=env, bus_id=5, capacity_mwh=10.0, power_mw=5.0)
        env.reset(seed=0, day_id=0)
        obs = env.obs()
        assert obs.shape == env.observation_space.shape

    def test_obs_dim_shrinks_after_unregister(self, case33, simple_time_series):
        """Unregistering a resource reduces obs_dim back to baseline."""
        from powerzoo.envs.resource.battery import BatteryEnv
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        dim_base = env.observation_space.shape[0]
        batt = BatteryEnv(parent=env, bus_id=5, capacity_mwh=10.0, power_mw=5.0)
        assert env.observation_space.shape[0] == dim_base + 4
        env.unregister_resource(batt.resource_id)
        assert env.observation_space.shape[0] == dim_base

    def test_der_obs_values_in_range(self, case33, simple_time_series):
        """Battery grid_obs values (SOC, bounds, p_norm) must be in [0, 1] or [-1, 1]."""
        from powerzoo.envs.resource.battery import BatteryEnv
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        batt = BatteryEnv(parent=env, bus_id=5, capacity_mwh=10.0, power_mw=5.0,
                          initial_soc=0.5)
        env.reset(seed=0, day_id=0)
        obs = env.obs()
        # DER segment is the last 4 elements before the 2 time-encoding values
        der_segment = obs[-6:-2]
        assert np.all(der_segment >= -1.0 - 1e-6)
        assert np.all(der_segment <= 1.0 + 1e-6)


# ── Total Loss Calculation ───────────────────────────────────────────

class TestGetTotalLoss:
    """get_total_loss from line DataFrame."""

    def test_total_loss_from_lines(self, case33, simple_time_series):
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        p_loss, q_loss = env.get_total_loss(env._lines)
        assert p_loss >= 0.0
        assert q_loss >= 0.0
        # Sum of branch losses should match per-branch data
        np.testing.assert_allclose(p_loss, env._lines['p_loss_MW'].sum(), atol=1e-9)

    def test_nan_loss_does_not_propagate_to_reward(self, case33, simple_time_series):
        """get_total_loss must not return NaN even when lines_df contains NaN.

        When BFS diverges, branch currents are undefined and p_loss_MW
        becomes NaN.  Propagating NaN into _compute_reward causes
        loss_penalty = -0.1 * NaN = NaN, which crashes RL training.
        When _pf_failed is True the method returns the baseMVA sentinel.
        """
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        env._pf_failed = True  # divergence flag must be set

        lines_nan = env._lines.copy()
        lines_nan['p_loss_MW'] = np.nan
        lines_nan['q_loss_MVAr'] = np.nan

        p_loss, q_loss = env.get_total_loss(lines_nan)
        assert np.isfinite(p_loss), f"p_loss is not finite: {p_loss}"
        assert np.isfinite(q_loss), f"q_loss is not finite: {q_loss}"
        assert p_loss == env.baseMVA, "divergence sentinel should equal baseMVA"

    def test_inf_loss_does_not_propagate_to_reward(self, case33, simple_time_series):
        """np.nansum does NOT filter Inf — get_total_loss must intercept it.

        np.nansum([np.inf]) == inf, so without the _pf_failed early-exit guard
        Inf propagates into reward as -inf, causing gradient explosion.
        """
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        env._pf_failed = True

        lines_inf = env._lines.copy()
        lines_inf['p_loss_MW'] = np.inf
        lines_inf['q_loss_MVAr'] = np.inf

        p_loss, q_loss = env.get_total_loss(lines_inf)
        assert np.isfinite(p_loss), f"p_loss is not finite when lines contain Inf: {p_loss}"
        assert np.isfinite(q_loss), f"q_loss is not finite when lines contain Inf: {q_loss}"

    def test_divergence_sentinel_larger_than_normal_loss(self, case33, simple_time_series):
        """The divergence sentinel (baseMVA) must exceed any normal operating loss.

        If the sentinel were 0 or smaller than typical losses, an agent in CMDP
        mode could learn to deliberately crash the grid to reduce apparent losses
        (reward hacking).
        """
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        normal_p_loss, _ = env.get_total_loss(env._lines)

        env._pf_failed = True
        sentinel_p, _ = env.get_total_loss(env._lines)

        assert sentinel_p > normal_p_loss, (
            f"Divergence sentinel ({sentinel_p}) should exceed normal loss ({normal_p_loss})"
        )

    def test_get_total_loss_rejects_ndarray(self, case33, simple_time_series):
        """get_total_loss must raise TypeError when passed a raw ndarray.

        cal_pf(df=False) returns ndarrays that lack p_loss_MW / q_loss_MVAr;
        silently accepting them would return 0.0 and hide real losses.
        """
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        v_mag, p_branch = env.cal_pf(df=False)
        with pytest.raises(TypeError, match="DataFrame"):
            env.get_total_loss(p_branch)  # ndarray, not DataFrame

    def test_reward_finite_after_divergence(self, case33, simple_time_series):
        """End-to-end: _compute_reward must return finite value even after PF divergence."""
        env = DistGridEnv(
            case=case33, time_series=simple_time_series,
            violation_penalty_weight=1.0,
        )
        env.reset(seed=42, day_id=0)
        env._pf_failed = True
        env._p_loss, env._q_loss = env.get_total_loss(env._lines)

        state = env._get_state()
        reward = env._compute_reward(state)
        assert np.isfinite(reward), f"reward is not finite after PF divergence: {reward}"
        assert reward < 0.0, "reward must be negative (loss penalty) after divergence"


# ── obs(state) stateless contract ─────────────────────────────────────────────────────

class TestObsStateContract:
    """obs(state) must be stateless w.r.t. the live env — replay-buffer safety."""

    def test_obs_with_explicit_state_ignores_pf_failed(self, case33, simple_time_series):
        """obs(state) must not overwrite an explicitly-passed historic state
        with the env's _prev_nodes when _pf_failed is True.

        Scenario: replay-buffer wrapper calls obs(historic_state) to recompute
        an old observation *after* the env has diverged.  The live flag
        _pf_failed must not corrupt that historic state's obs.
        """
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)

        # Capture a clean historic state
        historic_nodes = env._nodes.copy()
        historic_nodes['v_mag'] = 1.01  # known, well-defined value
        historic_state = {
            'nodes': historic_nodes,
            'lines': env._lines.copy(),
            'time_step': env.time_step,
        }
        obs_historic = env.obs(historic_state).copy()

        # Now simulate a divergence in the live env
        env._pf_failed = True
        env._prev_nodes = env._nodes.copy()
        env._prev_nodes['v_mag'] = 0.80  # different from historic 1.01

        # Re-requesting obs with the same historic state must give the same result
        obs_after_fail = env.obs(historic_state)
        np.testing.assert_array_equal(
            obs_historic, obs_after_fail,
            err_msg="obs(state) was contaminated by live _pf_failed flag"
        )


# ── _ep_violations accumulation ───────────────────────────────────────────────────────

class TestEpViolationsAccumulation:
    """_ep_violations must accumulate in every mode, not only when violation_penalty_weight > 0."""

    def test_ep_violations_accumulates_in_cmdp_mode(self, case33, simple_time_series):
        """In CMDP mode (violation_penalty_weight=0), _ep_violations must still count violations.

        Bug: safety_info was only parsed inside `if violation_penalty_weight > 0.0`,
        so _ep_violations was always 0 in CMDP mode regardless of actual violations.
        """
        env = DistGridEnv(
            case=case33, time_series=simple_time_series,
            violation_penalty_weight=0.0,   # CMDP mode
            v_min=0.999,                    # extremely tight → guaranteed violations
            v_max=1.001,
        )
        env.reset(seed=42, day_id=0)
        assert env._ep_violations == 0, "must start at 0"

        # Drive a step so _compute_reward is called
        state = env._get_state()
        # Manually inject a safety_info with violations so the test is deterministic
        state['safety_info'] = {
            'v_violation_nodes': [1, 2, 3],
            'line_violation_ids': [0],
        }
        state['p_loss_MW'] = env._p_loss

        env._compute_reward(state)
        assert env._ep_violations == 4, (
            f"Expected 4 violations (3 voltage + 1 line), got {env._ep_violations}"
        )

    def test_ep_violations_accumulates_in_penalty_mode(self, case33, simple_time_series):
        """In soft-penalty mode (violation_penalty_weight > 0), _ep_violations must also accumulate."""
        env = DistGridEnv(
            case=case33, time_series=simple_time_series,
            violation_penalty_weight=1.0,
        )
        env.reset(seed=42, day_id=0)
        assert env._ep_violations == 0

        state = env._get_state()
        state['safety_info'] = {
            'v_violation_nodes': [5],
            'line_violation_ids': [],
        }
        state['p_loss_MW'] = env._p_loss

        env._compute_reward(state)
        assert env._ep_violations == 1, (
            f"Expected 1 violation, got {env._ep_violations}"
        )

    def test_ep_violations_zero_when_no_violations(self, case33, simple_time_series):
        """When no violations occur, _ep_violations must remain 0."""
        env = DistGridEnv(
            case=case33, time_series=simple_time_series,
            violation_penalty_weight=0.0,
        )
        env.reset(seed=42, day_id=0)

        state = env._get_state()
        state['safety_info'] = {
            'v_violation_nodes': [],
            'line_violation_ids': [],
        }
        state['p_loss_MW'] = env._p_loss

        env._compute_reward(state)
        assert env._ep_violations == 0


# ── _on_resource_changed hook ────────────────────────────────────────
class TestOnResourceChangedHook:
    """Verify that register_resource / unregister_resource trigger space rebuilds
    via the _on_resource_changed hook."""

    def test_register_updates_obs_and_action_spaces(self, case33, simple_time_series):
        """Both observation_space and action_space grow after resource registration."""
        from powerzoo.envs.resource.battery import BatteryEnv
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        obs_dim_before = env.observation_space.shape[0]
        act_dim_before = env.action_space.shape[0]  # 0 (no resources yet)

        BatteryEnv(parent=env, bus_id=5, capacity_mwh=10.0, power_mw=5.0)

        # obs grows by the battery's grid_obs() width (4 features)
        assert env.observation_space.shape[0] == obs_dim_before + 4
        # action gains one DER dim
        assert env.action_space.shape[0] == act_dim_before + 1

    def test_unregister_restores_spaces(self, case33, simple_time_series):
        """Unregistering a resource restores both spaces to their original dims."""
        from powerzoo.envs.resource.battery import BatteryEnv
        env = DistGridEnv(case=case33, time_series=simple_time_series)
        obs_dim_base = env.observation_space.shape[0]

        batt = BatteryEnv(parent=env, bus_id=5, capacity_mwh=10.0, power_mw=5.0)
        env.unregister_resource(batt.resource_id)

        assert env.observation_space.shape[0] == obs_dim_base
        assert env.action_space.shape[0] == 0

