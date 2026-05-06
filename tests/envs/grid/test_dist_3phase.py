"""Tests for powerzoo.envs.grid.dist_3phase — DistGrid3PhaseEnv.

DistGrid3PhaseEnv wraps the IEEE 123-bus three-phase radial distribution
network. Inherits from DistGridEnv but overrides:
  - _build_topology → builds ThreePhaseTopology with 3×3 impedance matrices
  - _build_spaces / obs → per-phase observation (V_A/V_B/V_C, P_A/P_B/P_C, …)
  - cal_pf → runs three-phase BIBC/BCBV BFS with phase-aware resource injection
  - safety_check → checks per-phase voltage limits + VUF + thermal limit
  - build_info → adds cost_vuf_violation to CMDP cost channel
  - calculate_vuf → Fortescue-based Voltage Unbalance Factor

Domain knowledge:
  - Case123: 123 nodes, 122 lines, 4.16 kV, 10 MVA base
  - Zbase = baseKV² / baseMVA = 4.16² / 10 ≈ 1.7306
  - 3-phase BFS: convergence in O(10) iterations
  - Per-phase columns: V_A/V_B/V_C, angle_A/angle_B/angle_C
  - Obs vector: V_A, V_B, V_C per node (not averaged v_mag)
  - VUF = |V_neg|/|V_pos| × 100% (Fortescue transform)
  - Angle convention: degrees (from np.angle(…, deg=True))
"""

import numpy as np
import pandas as pd
import pytest

from powerzoo.envs.grid.dist_3phase import DistGrid3PhaseEnv


# ── Physical constants ───────────────────────────────────────────────
V_REF_MAG = 1.05


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture(scope="module")
def env():
    """DistGrid3PhaseEnv with default Case123, no time series."""
    return DistGrid3PhaseEnv()


@pytest.fixture(scope="module")
def env_ts():
    """DistGrid3PhaseEnv with simple time series for step/reset."""
    ts = np.ones(48) * 3.0  # 48 half-hours, 3 MW constant
    return DistGrid3PhaseEnv(time_series=ts)


# =====================================================================
# Constructor & Configuration
# =====================================================================

class TestInit:
    """Constructor defaults and difficulty presets."""

    def test_default_case123(self, env):
        assert 'Case123' in type(env.case).__name__ or env.n_nodes == 123

    def test_n_nodes(self, env):
        """Case123 has 114 active nodes (some bus IDs are not contiguous)."""
        assert env.n_nodes == 114

    def test_n_lines(self, env):
        """n_lines = n_nodes - 1 in a radial tree."""
        assert env.n_lines == env.n_nodes - 1

    def test_v_slack_105(self, env):
        """Case123 uses V_ref = 1.05 p.u."""
        np.testing.assert_allclose(env.v_ref_mag, V_REF_MAG, atol=1e-10)

    def test_baseMVA(self, env):
        assert env.baseMVA == 10.0

    def test_baseKV(self, env):
        assert env.baseKV == 4.16

    def test_Zbase(self, env):
        """Zbase = kV² / MVA = 4.16² / 10 ≈ 1.7306."""
        expected = 4.16 ** 2 / 10.0
        np.testing.assert_allclose(env.Zbase, expected, rtol=1e-4)

    def test_voltage_limits_default(self, env):
        assert env.v_min == 0.90
        assert env.v_max == 1.10

    def test_difficulty_easy(self):
        e = DistGrid3PhaseEnv(difficulty='easy')
        assert e.v_min == 0.88
        assert e.v_max == 1.12

    def test_difficulty_medium(self):
        e = DistGrid3PhaseEnv(difficulty='medium')
        assert e.v_min == 0.90
        assert e.v_max == 1.10

    def test_difficulty_hard(self):
        e = DistGrid3PhaseEnv(difficulty='hard')
        assert e.v_min == 0.93
        assert e.v_max == 1.07

    def test_invalid_difficulty_raises(self):
        with pytest.raises(ValueError, match="difficulty"):
            DistGrid3PhaseEnv(difficulty='nightmare')

    def test_has_topo3ph(self, env):
        """Three-phase topology must be built at init."""
        assert hasattr(env, 'topo3ph')
        assert env.topo3ph is not None

    def test_slack_bus_id(self, env):
        assert env.slack_bus_id == 0

    def test_topology_node_mapping_matches_case_ids(self, env):
        expected = env.case.nodes.index[env._non_ref_mask()].to_numpy()
        np.testing.assert_array_equal(env.topo3ph.non_ref_node_ids, expected)
        assert env.topo3ph.vector_layout == 'node_major_abc'

    def test_topology_lookup_excludes_reference_bus(self, env):
        ref_id = env.case.nodes.index[env.slack_bus_id]
        assert ref_id not in env.topo3ph.node_id_to_matrix_index


# =====================================================================
# Observation & Action Spaces
# =====================================================================

class TestSpaces:

    def test_obs_space_shape(self, env_ts):
        """obs = [V_A(n) + V_B(n) + V_C(n)] + [P_A/B/C(nl) + Q_A/B/C(nl)]
              + [p_load_A/B/C(n) + q_load_A/B/C(n)] + [time(2)]."""
        n = env_ts.n_nodes
        nl = env_ts.n_lines
        expected = 3 * n + 6 * nl + 6 * n + 2
        assert env_ts.observation_space.shape == (expected,)

    def test_obs_space_bounds_are_finite(self, env_ts):
        """observation_space bounds must be ±20.0 to match the obs() clip.
        Infinite bounds would misrepresent the actual observation range to
        algorithms that rely on space bounds for normalisation."""
        assert np.all(np.isfinite(env_ts.observation_space.low))
        assert np.all(np.isfinite(env_ts.observation_space.high))
        np.testing.assert_array_equal(env_ts.observation_space.low, -20.0)
        np.testing.assert_array_equal(env_ts.observation_space.high, 20.0)

    def test_obs_within_space_bounds_after_reset(self, env_ts):
        """obs() values must lie within observation_space after a reset."""
        env_ts.reset(seed=0, day_id=0)
        o = env_ts.obs()
        assert np.all(o >= env_ts.observation_space.low)
        assert np.all(o <= env_ts.observation_space.high)

    def test_action_space_empty(self, env_ts):
        """Pure observer mode: 0-dim action."""
        assert env_ts.action_space.shape == (0,)

    def test_action_space_updates_after_resource_registration(self):
        """Three-phase env inherits DistGridEnv.update_action_space()."""
        e = DistGrid3PhaseEnv()
        obs_shape_before = e.observation_space.shape
        obs_names_before = list(e.obs_names)

        class _Res:
            bus_id = 1
            current_p_mw = 0.0
            current_q_mvar = 0.0
            def grid_action_bounds(self): return (-1.0, 1.0)
            def grid_obs(self): return np.zeros(1, dtype=np.float32)
            def grid_obs_names(self, rid): return [f'{rid}_p_norm']
            def status(self): return {}
            def obs(self): return {}
            def reset(self, *args, **kwargs): return {}
            def step(self, action): return None

        rid = e.register_resource(_Res(), bus_id=1)

        assert e.action_space.shape == (1,)
        assert e.action_names == [rid]
        # Observation contract must not be corrupted by single-phase _update_obs_space
        assert e.observation_space.shape == obs_shape_before, (
            "register_resource() must not overwrite the three-phase observation schema"
        )
        assert e.obs_names == obs_names_before

    def test_obs_names_count(self, env_ts):
        assert len(env_ts.obs_names) == env_ts.observation_space.shape[0]

    def test_obs_names_contain_per_phase(self, env_ts):
        """Observation names should reference per-phase features."""
        names = env_ts.obs_names
        assert any('V_A' in n for n in names)
        assert any('V_B' in n for n in names)
        assert any('V_C' in n for n in names)
        assert any('P_A' in n for n in names)

    def test_obs_names_contain_per_phase_loads(self, env_ts):
        """Per-phase load names (p_load_A/B/C, q_load_A/B/C) must appear; total load must not."""
        names = env_ts.obs_names
        for ph in 'ABC':
            assert any(f'p_load_{ph}' in n for n in names), f"Missing p_load_{ph} in obs_names"
            assert any(f'q_load_{ph}' in n for n in names), f"Missing q_load_{ph} in obs_names"
        # Total-load names should not be present (replaced by per-phase)
        assert not any(n.endswith('p_load_norm') for n in names)
        assert not any(n.endswith('q_load_norm') for n in names)


# =====================================================================
# Power Flow (cal_pf)
# =====================================================================

class TestCalPf:
    """Raw power flow API."""

    def test_cal_pf_df_returns_dataframes(self, env):
        nodes, lines = env.cal_pf(df=True)
        assert isinstance(nodes, pd.DataFrame)
        assert isinstance(lines, pd.DataFrame)

    def test_nodes_has_3phase_columns(self, env):
        nodes, _ = env.cal_pf(df=True)
        for ph in 'ABC':
            assert f'V_{ph}' in nodes.columns, f'Missing V_{ph}'
            assert f'angle_{ph}' in nodes.columns, f'Missing angle_{ph}'

    def test_nodes_has_vmag(self, env):
        """v_mag = mean(V_A, V_B, V_C)."""
        nodes, _ = env.cal_pf(df=True)
        assert 'v_mag' in nodes.columns
        expected = nodes[['V_A', 'V_B', 'V_C']].mean(axis=1)
        np.testing.assert_allclose(nodes['v_mag'].values, expected.values, atol=1e-10)

    def test_nodes_has_per_phase_load_columns(self, env):
        """nodes_df must contain per-phase load columns for obs() to consume."""
        nodes, _ = env.cal_pf(df=True)
        for ph in 'ABC':
            assert f'p_load_{ph}_MW' in nodes.columns, f"Missing p_load_{ph}_MW"
            assert f'q_load_{ph}_MVAr' in nodes.columns, f"Missing q_load_{ph}_MVAr"

    def test_per_phase_loads_sum_to_total(self, env):
        """p_load_A + p_load_B + p_load_C must equal p_load_MW (total)."""
        nodes, _ = env.cal_pf(df=True)
        phase_sum = (nodes['p_load_A_MW'] + nodes['p_load_B_MW'] + nodes['p_load_C_MW'])
        np.testing.assert_allclose(phase_sum.values, nodes['p_load_MW'].values, atol=1e-9)

    def test_lines_has_phase_columns(self, env):
        _, lines = env.cal_pf(df=True)
        for ph in 'ABC':
            assert f'P_{ph}_MW' in lines.columns
            assert f'p_loss_{ph}_MW' in lines.columns

    def test_lines_has_totals(self, env):
        _, lines = env.cal_pf(df=True)
        assert 'p_flow_MW' in lines.columns
        assert 'p_loss_MW' in lines.columns

    def test_cal_pf_no_df_returns_arrays(self, env):
        v_mag, p_branch = env.cal_pf(df=False)
        assert isinstance(v_mag, np.ndarray)
        assert isinstance(p_branch, np.ndarray)

    def test_convergence(self, env):
        env.cal_pf(df=True)
        assert env._converged


# =====================================================================
# Safety Check
# =====================================================================

class TestSafetyCheck:

    def test_safe_default_limits(self, env):
        nodes, lines = env.cal_pf(df=True)
        is_safe, info = env.safety_check(nodes, lines, with_info=True)
        assert info['converged']
        assert is_safe

    def test_info_keys(self, env):
        nodes, lines = env.cal_pf(df=True)
        _, info = env.safety_check(nodes, lines, with_info=True)
        required = ('v_min_actual', 'v_max_actual', 'v_violation_nodes',
                    'line_violation_ids', 'max_vuf_percent',
                    'vuf_violation', 'vuf_violation_nodes',
                    'converged', 'iterations')
        for key in required:
            assert key in info, f"Missing: {key}"

    def test_vuf_in_info(self, env):
        nodes, lines = env.cal_pf(df=True)
        _, info = env.safety_check(nodes, lines, with_info=True)
        assert info['max_vuf_percent'] >= 0
        assert isinstance(info['vuf_violation'], bool)
        assert isinstance(info['vuf_violation_nodes'], list)

    def test_tight_limits_trigger_violation(self, env):
        """Very tight voltage band should produce violations."""
        nodes, lines = env.cal_pf(df=True)
        is_safe, info = env.safety_check(nodes, lines,
                                         v_min=1.04, v_max=1.06,
                                         with_info=True)
        assert not is_safe
        assert len(info['v_violation_nodes']) > 0

    def test_vuf_max_affects_safety(self):
        """Setting vuf_max=0 should force VUF violation on any unbalanced case."""
        e = DistGrid3PhaseEnv(vuf_max=0.0)
        nodes, lines = e.cal_pf(df=True)
        is_safe, info = e.safety_check(nodes, lines, with_info=True)
        # With vuf_max=0, any non-zero VUF makes it unsafe
        if info['max_vuf_percent'] > 0:
            assert not is_safe
            assert info['vuf_violation'] is True

    def test_vuf_max_default(self, env):
        assert env.vuf_max == 2.0

    def test_no_info_returns_none(self, env):
        """with_info=False returns None for info."""
        nodes, lines = env.cal_pf(df=True)
        is_safe, info = env.safety_check(nodes, lines, with_info=False)
        assert isinstance(is_safe, bool)
        assert info is None

    def test_numpy_array_input_raises_typeerror(self, env):
        """Passing numpy arrays (from cal_pf(df=False)) must raise TypeError.

        Three-phase safety checks depend on per-phase columns (V_A, V_B, V_C,
        angle_A/B/C) that are absent in flat numpy array output.  Silently
        accepting arrays would skip all voltage/VUF checks and return a
        false-safe result, so the method now enforces DataFrame input.
        """
        v_mag, p_branch = env.cal_pf(df=False)
        with pytest.raises(TypeError, match="pd.DataFrame"):
            env.safety_check(v_mag, p_branch)


# =====================================================================
# VUF Calculation
# =====================================================================

class TestVUF:

    def test_calculate_vuf_api(self, env):
        nodes, _ = env.cal_pf(df=True)
        vuf_arr, max_vuf = env.calculate_vuf(nodes)
        assert isinstance(vuf_arr, np.ndarray)
        assert isinstance(max_vuf, float)
        assert len(vuf_arr) == env.n_nodes

    def test_vuf_non_negative(self, env):
        nodes, _ = env.cal_pf(df=True)
        vuf_arr, _ = env.calculate_vuf(nodes)
        assert np.all(vuf_arr >= 0)

    def test_vuf_below_ieee_limit(self, env):
        nodes, _ = env.cal_pf(df=True)
        _, max_vuf = env.calculate_vuf(nodes)
        assert max_vuf < 2.0, f"Max VUF {max_vuf:.4f}% exceeds IEEE 2%"

    def test_calculate_vuf_rejects_non_dataframe(self, env):
        """Passing a non-DataFrame should raise TypeError."""
        with pytest.raises(TypeError, match="DataFrame"):
            env.calculate_vuf(np.zeros((env.n_nodes, 6)))

    def test_calculate_vuf_rejects_missing_columns(self, env):
        """DataFrame missing required columns should raise KeyError."""
        df = pd.DataFrame({'V_A': [1.0], 'V_B': [1.0]})
        with pytest.raises(KeyError, match="V_C"):
            env.calculate_vuf(df)


# =====================================================================
# Observation (per-phase)
# =====================================================================

class TestObs:

    def test_obs_shape_matches_space(self, env_ts):
        env_ts.reset(seed=42, day_id=0)
        o = env_ts.obs()
        assert o.shape == env_ts.observation_space.shape

    def test_obs_dtype(self, env_ts):
        env_ts.reset(seed=42, day_id=0)
        o = env_ts.obs()
        assert o.dtype == np.float32

    def test_obs_no_nan(self, env_ts):
        env_ts.reset(seed=42, day_id=0)
        o = env_ts.obs()
        assert not np.any(np.isnan(o))

    def test_obs_reflects_phase_differences(self, env_ts):
        """Per-phase voltages in obs should not all be identical (unbalanced load)."""
        env_ts.reset(seed=42, day_id=0)
        o = env_ts.obs()
        n = env_ts.n_nodes
        v_a = o[:n]
        v_b = o[n:2*n]
        v_c = o[2*n:3*n]
        # At least one node must have different per-phase voltages
        # (Case123 is unbalanced)
        assert not np.allclose(v_a, v_b, atol=1e-6) or not np.allclose(v_b, v_c, atol=1e-6)

    def test_obs_state_not_replaced_when_pf_failed(self, env_ts):
        """Regression: explicit caller-provided state must not be overwritten by
        _prev_nodes even when _pf_failed is True (Replay Buffer use-case)."""
        env_ts.reset(seed=42, day_id=0)
        # Capture nodes produced by a *valid* power flow
        nodes_good, lines_good = env_ts.cal_pf(df=True)
        state_good = {
            'nodes': nodes_good,
            'lines': lines_good,
            'time_step': env_ts.time_step,
        }
        obs_good = env_ts.obs(state_good)

        # Simulate a diverged step (force _pf_failed = True)
        env_ts._pf_failed = True
        # Now calling obs() with an explicit state must still yield the same obs
        obs_with_state = env_ts.obs(state_good)
        np.testing.assert_array_equal(obs_good, obs_with_state)

        # Restore
        env_ts._pf_failed = False

    def test_live_obs_uses_penalty_observation_when_pf_failed(self, env_ts):
        """Live obs() should emit a collapse observation after PF divergence."""
        env_ts.reset(seed=42, day_id=0)
        env_ts._prev_nodes = env_ts._nodes.copy()
        env_ts._prev_lines = env_ts._lines.copy()

        # Make the cached previous state obviously different from the penalty obs
        env_ts._prev_nodes.loc[:, ['V_A', 'V_B', 'V_C']] = 1.05
        env_ts._prev_lines.loc[:, [f'P_{ph}_MW' for ph in 'ABC']] = 9.0
        env_ts._prev_lines.loc[:, [f'Q_{ph}_MVAr' for ph in 'ABC']] = 9.0

        env_ts._pf_failed = True
        obs = env_ts.obs()

        n = env_ts.n_nodes
        nl = env_ts.n_lines
        expected_v = np.full(3 * n, (env_ts.v_min - 1.0) / 0.1, dtype=np.float32)

        np.testing.assert_allclose(obs[:3 * n], expected_v, atol=1e-6)
        np.testing.assert_allclose(obs[3 * n:3 * n + 6 * nl], 0.0)
        assert obs.dtype == np.float32
        assert not np.any(np.isnan(obs))

        env_ts._pf_failed = False

    def test_obs_clipped_to_bounds(self, env_ts):
        """obs() must return values in [-20.0, 20.0] even when nodes contain
        extreme finite voltages (near-diverged BFS edge case)."""
        env_ts.reset(seed=42, day_id=0)
        nodes_bad = env_ts._nodes.copy()
        # Inject extreme finite voltages that nan_to_num would not catch
        for ph in ('V_A', 'V_B', 'V_C'):
            nodes_bad.loc[:, ph] = 9999.0
        state_bad = {
            'nodes': nodes_bad,
            'lines': env_ts._lines,
            'time_step': env_ts.time_step,
        }
        o = env_ts.obs(state_bad)
        assert np.all(o >= -20.0), "obs below -20 clip bound"
        assert np.all(o <= 20.0), "obs above +20 clip bound"

    def test_live_obs_uses_penalty_when_not_converged(self, env_ts):
        """Live obs() should use penalty observation when _converged is False,
        even if _pf_failed was not updated yet (double-guard)."""
        env_ts.reset(seed=42, day_id=0)
        env_ts._prev_nodes = env_ts._nodes.copy()
        env_ts._prev_lines = env_ts._lines.copy()
        env_ts._prev_nodes.loc[:, ['V_A', 'V_B', 'V_C']] = 1.05

        # Set _converged=False but leave _pf_failed=False to test the second guard
        env_ts._converged = False
        env_ts._pf_failed = False
        obs = env_ts.obs()

        n = env_ts.n_nodes
        expected_v = np.full(3 * n, (env_ts.v_min - 1.0) / 0.1, dtype=np.float32)
        np.testing.assert_allclose(obs[:3 * n], expected_v, atol=1e-6)

        # Restore
        env_ts._converged = True


# =====================================================================
# Build Info (CMDP cost channel)
# =====================================================================

class TestBuildInfo:

    def test_build_info_has_vuf_cost(self, env_ts):
        env_ts.reset(seed=42, day_id=0)
        state, _, _, _, info = env_ts.step({})
        assert 'cost_vuf_violation' in info
        assert 'cost_voltage_violation' in info
        assert 'cost_thermal_overload' in info
        assert 'cost_sum' in info

    def test_cost_sum_includes_vuf(self, env_ts):
        """cost_sum = cost_voltage + cost_thermal + cost_vuf."""
        env_ts.reset(seed=42, day_id=0)
        _, _, _, _, info = env_ts.step({})
        expected = (info['cost_voltage_violation']
                    + info['cost_thermal_overload']
                    + info['cost_vuf_violation'])
        assert info['cost_sum'] == expected

    def test_build_info_exposes_pf_converged_without_step(self, env_ts):
        env_ts.reset(seed=42, day_id=0)
        info = env_ts.build_info(env_ts._get_state())
        assert info['pf_converged'] is True

    def test_build_info_missing_loss_keys_uses_fallbacks(self):
        """build_info must not raise KeyError when loss keys are absent from state
        (e.g. external wrapper returns a truncated dict)."""
        e = DistGrid3PhaseEnv()
        e.reset(seed=0, day_id=0)
        state = e._get_state()
        # Remove loss keys to simulate a truncated wrapper state
        state.pop('p_loss_MW', None)
        state.pop('q_loss_MVAr', None)
        info = e.build_info(state)
        assert info['p_loss_MW'] == e.baseMVA
        assert info['q_loss_MVAr'] == 0.0


# =====================================================================
# Phase-Aware Resource Injection
# =====================================================================

class TestPhaseInjection:
    """Verify resource.phase attribute is respected in cal_pf."""

    def test_phase_alloc_table_exists(self, env):
        """Environment must have _PHASE_ALLOC lookup."""
        assert hasattr(env, '_PHASE_ALLOC')
        for key in ('A', 'B', 'C', 'AB', 'ABC'):
            assert key in env._PHASE_ALLOC
            np.testing.assert_allclose(env._PHASE_ALLOC[key].sum(), 1.0, atol=1e-10)

    def test_phase_alloc_single_phase(self, env):
        """Single-phase allocation puts all power on one phase."""
        np.testing.assert_array_equal(env._PHASE_ALLOC['A'], [1, 0, 0])
        np.testing.assert_array_equal(env._PHASE_ALLOC['B'], [0, 1, 0])
        np.testing.assert_array_equal(env._PHASE_ALLOC['C'], [0, 0, 1])

    def test_route_b_vs_route_a_equivalent_balanced(self):
        """Route B with equal 3-element array must match Route A with phase='ABC' scalar.

        For a balanced three-phase resource (p_A == p_B == p_C == P/3), Route B
        with [P/3, P/3, P/3] must produce the same net injection as Route A with
        scalar P and phase='ABC' (_PHASE_ALLOC['ABC'] = [1/3, 1/3, 1/3]).
        """
        import types

        env_b = DistGrid3PhaseEnv()

        # Build a minimal mock resource at node 1
        class _Res:
            phase = 'ABC'
            current_p_mw = np.array([1.0])
            current_q_mvar = np.array([0.0])
            def status(self): return {}

        res_a = _Res()
        res_b = _Res()
        res_b.current_p_mw = np.array([1/3, 1/3, 1/3])

        n = env_b.n_nodes
        nrm = np.zeros((n, 1))
        nrm[1, 0] = 1.0

        # Route A: scalar [1.0] with phase='ABC'
        env_b.sub_resources = {'r': res_a}
        env_b.nodes_resources_map = nrm
        nodes_a, _ = env_b.cal_pf(df=True)

        # Route B: 3-element [1/3, 1/3, 1/3]
        env_b.sub_resources = {'r': res_b}
        nodes_b, _ = env_b.cal_pf(df=True)

        # Voltages must be identical to within solver tolerance
        for ph in 'ABC':
            np.testing.assert_allclose(
                nodes_a[f'V_{ph}'].values, nodes_b[f'V_{ph}'].values,
                atol=1e-6, err_msg=f"Route A vs Route B mismatch for phase {ph}"
            )

    def test_route_b_invalid_q_length_raises(self):
        env_b = DistGrid3PhaseEnv()

        class _Res:
            phase = 'ABC'
            current_p_mw = np.array([1/3, 1/3, 1/3], dtype=float)
            current_q_mvar = np.array([0.1, 0.2], dtype=float)
            def status(self): return {}

        nrm = np.zeros((env_b.n_nodes, 1))
        nrm[1, 0] = 1.0
        env_b.sub_resources = {'r': _Res()}
        env_b.nodes_resources_map = nrm

        with pytest.raises(ValueError, match="current_q_mvar"):
            env_b.cal_pf(df=True)

    def test_route_b_row_vector_is_flattened(self):
        env_b = DistGrid3PhaseEnv()

        class _Res:
            phase = 'ABC'
            current_p_mw = np.array([1/3, 1/3, 1/3], dtype=float)
            current_q_mvar = np.zeros(3, dtype=float)
            def status(self): return {}

        res_flat = _Res()
        res_row = _Res()
        res_row.current_p_mw = np.array([[1/3, 1/3, 1/3]], dtype=float)
        res_row.current_q_mvar = np.array([[0.0, 0.0, 0.0]], dtype=float)

        nrm = np.zeros((env_b.n_nodes, 1))
        nrm[1, 0] = 1.0

        env_b.sub_resources = {'r': res_flat}
        env_b.nodes_resources_map = nrm
        nodes_flat, _ = env_b.cal_pf(df=True)

        env_b.sub_resources = {'r': res_row}
        nodes_row, _ = env_b.cal_pf(df=True)

        for ph in 'ABC':
            np.testing.assert_allclose(
                nodes_flat[f'V_{ph}'].values,
                nodes_row[f'V_{ph}'].values,
                atol=1e-6,
                err_msg=f"Route B row-vector flattening mismatch for phase {ph}",
            )


# =====================================================================
# Reset & Step Lifecycle
# =====================================================================

class TestResetStep:

    def test_reset_returns_dict(self, env_ts):
        state, info = env_ts.reset(seed=42, day_id=0)
        assert isinstance(state, dict)
        assert isinstance(info, dict)
        assert 'nodes' in state
        assert 'lines' in state
        assert 'is_safe' in state

    def test_reset_runs_power_flow(self, env_ts):
        env_ts.reset(seed=42, day_id=0)
        assert env_ts._nodes is not None
        assert env_ts._lines is not None
        assert env_ts._converged is True

    def test_3phase_columns_after_reset(self, env_ts):
        env_ts.reset(seed=42, day_id=0)
        for ph in 'ABC':
            assert f'V_{ph}' in env_ts._nodes.columns

    def test_step_empty_action(self, env_ts):
        env_ts.reset(seed=42, day_id=0)
        state, reward, done, truncated, info = env_ts.step({})
        assert isinstance(state, dict)
        assert 'nodes' in state

    def test_episode_progresses(self, env_ts):
        """Multiple steps without error; timestep advances."""
        env_ts.reset(seed=42, day_id=0)
        t0 = env_ts.time_step
        for _ in range(3):
            env_ts.step({})
        assert env_ts.time_step == t0 + 3


# =====================================================================
# Determinism
# =====================================================================

class TestDeterminism:

    def test_reset_deterministic(self, env_ts):
        s1, _ = env_ts.reset(seed=42, day_id=0)
        s2, _ = env_ts.reset(seed=42, day_id=0)
        np.testing.assert_array_equal(s1['nodes']['v_mag'].values,
                                      s2['nodes']['v_mag'].values)


# =====================================================================

class TestComputeReward:
    """Tests for _compute_reward override that includes VUF violations."""

    def test_loss_penalty_weight_default(self, env):
        assert env.loss_penalty_weight == 0.1

    def test_loss_penalty_weight_custom(self):
        e = DistGrid3PhaseEnv(loss_penalty_weight=0.25)
        assert e.loss_penalty_weight == 0.25

    def test_loss_penalty_uses_configured_weight(self):
        e = DistGrid3PhaseEnv(loss_penalty_weight=0.25)
        state = {'p_loss_MW': 2.0, 'safety_info': {}}
        reward = e._compute_reward(state)
        assert state['reward_components']['loss_penalty'] == pytest.approx(-0.5)
        assert reward == pytest.approx(-0.5)

    @pytest.mark.parametrize('bad_loss', [np.nan, np.inf, -np.inf])
    def test_non_finite_loss_uses_base_mva_fallback(self, bad_loss):
        e = DistGrid3PhaseEnv(loss_penalty_weight=0.25)
        state = {'p_loss_MW': bad_loss, 'safety_info': {}}
        reward = e._compute_reward(state)
        expected = -0.25 * e.baseMVA
        assert np.isfinite(reward)
        assert state['reward_components']['loss_penalty'] == pytest.approx(expected)
        assert reward == pytest.approx(expected)

    def test_violation_penalty_weight_default(self, env):
        assert env.violation_penalty_weight == 0.0

    def test_violation_penalty_weight_custom(self):
        e = DistGrid3PhaseEnv(violation_penalty_weight=1.5)
        assert e.violation_penalty_weight == 1.5

    def test_no_violation_penalty_when_weight_zero(self, env):
        """With weight=0 (default), reward_components must not contain violation_penalty."""
        state = {
            'p_loss_MW': 0.1,
            'safety_info': {'v_violation_nodes': [0], 'line_violation_ids': [1],
                            'vuf_violation_nodes': [2, 3]},
        }
        reward = env._compute_reward(state)
        assert 'violation_penalty' not in state['reward_components']
        assert reward == pytest.approx(-env.loss_penalty_weight * 0.1)

    def test_vuf_violations_included_in_penalty(self):
        """With weight>0, VUF violation nodes must contribute to violation_penalty."""
        e = DistGrid3PhaseEnv(violation_penalty_weight=2.0)
        state = {
            'p_loss_MW': 0.0,
            'safety_info': {
                'v_violation_nodes': [],
                'line_violation_ids': [],
                'vuf_violation_nodes': [0, 1, 2],   # 3 VUF-violating nodes
            },
        }
        reward = e._compute_reward(state)
        assert state['reward_components']['violation_penalty'] == pytest.approx(-6.0)
        assert reward == pytest.approx(-6.0)

    def test_violation_penalty_counts_all_types(self):
        """n_v + n_l + n_vuf must all enter the penalty sum."""
        e = DistGrid3PhaseEnv(violation_penalty_weight=1.0)
        state = {
            'p_loss_MW': 0.0,
            'safety_info': {
                'v_violation_nodes': [0],          # 1
                'line_violation_ids': [0, 1],      # 2
                'vuf_violation_nodes': [2, 3, 4],  # 3   → total 6
            },
        }
        e._compute_reward(state)
        assert state['reward_components']['violation_penalty'] == pytest.approx(-6.0)

    def test_vuf_dense_penalty_weight_default(self, env):
        assert env.vuf_dense_penalty_weight == 0.0

    def test_vuf_dense_penalty_weight_custom(self):
        e = DistGrid3PhaseEnv(vuf_dense_penalty_weight=0.5)
        assert e.vuf_dense_penalty_weight == 0.5

    def test_no_dense_vuf_when_weight_zero(self, env):
        """When vuf_dense_penalty_weight=0, reward_components must not contain vuf_dense_penalty."""
        state = {
            'p_loss_MW': 0.0,
            'safety_info': {'v_violation_nodes': [], 'line_violation_ids': [],
                            'vuf_violation_nodes': [], 'max_vuf_percent': 3.0},
        }
        env._compute_reward(state)
        assert 'vuf_dense_penalty' not in state['reward_components']

    def test_dense_vuf_penalty_zero_inside_deadband(self):
        e = DistGrid3PhaseEnv(vuf_dense_penalty_weight=10.0)
        state = {
            'p_loss_MW': 0.0,
            'safety_info': {'v_violation_nodes': [], 'line_violation_ids': [],
                            'vuf_violation_nodes': [], 'max_vuf_percent': 1.4},
        }
        reward = e._compute_reward(state)
        assert state['reward_components']['vuf_dense_penalty'] == pytest.approx(0.0)
        assert reward == pytest.approx(0.0)

    def test_dense_vuf_penalty_proportional(self):
        """Dense VUF penalty activates only above the deadband."""
        e = DistGrid3PhaseEnv(vuf_dense_penalty_weight=10.0)
        state = {
            'p_loss_MW': 0.0,
            'safety_info': {'v_violation_nodes': [], 'line_violation_ids': [],
                            'vuf_violation_nodes': [], 'max_vuf_percent': 1.8},
        }
        reward = e._compute_reward(state)
        expected_dense = -10.0 * (1.8 - max(0.0, 0.75 * e.vuf_max)) / 100.0
        assert state['reward_components']['vuf_dense_penalty'] == pytest.approx(expected_dense)
        assert reward == pytest.approx(expected_dense)

    def test_dense_vuf_penalty_zero_when_balanced(self):
        """Dense VUF penalty is zero when max_vuf_percent = 0."""
        e = DistGrid3PhaseEnv(vuf_dense_penalty_weight=5.0)
        state = {
            'p_loss_MW': 0.0,
            'safety_info': {'v_violation_nodes': [], 'line_violation_ids': [],
                            'vuf_violation_nodes': [], 'max_vuf_percent': 0.0},
        }
        reward = e._compute_reward(state)
        assert state['reward_components']['vuf_dense_penalty'] == pytest.approx(0.0)

    def test_ep_violations_accumulated_regardless_of_weight(self):
        """_ep_violations must be incremented even when violation_penalty_weight=0.

        Regression: previously the violation count was only computed inside
        the ``if violation_penalty_weight > 0`` branch, so CMDP-mode runs
        (weight=0) always reported total_violations=0 to TensorBoard.
        """
        # weight=0 (CMDP mode) — violations must still be counted
        e = DistGrid3PhaseEnv(violation_penalty_weight=0.0)
        e._ep_violations = 0  # reset explicitly in case fixture reuse
        state = {
            'p_loss_MW': 0.0,
            'safety_info': {
                'v_violation_nodes': [0, 1],       # 2
                'line_violation_ids': [0],          # 1
                'vuf_violation_nodes': [2, 3, 4],  # 3   → total 6
            },
        }
        e._compute_reward(state)
        assert e._ep_violations == 6

        # Second call accumulates on top
        e._compute_reward(state)
        assert e._ep_violations == 12

    def test_missing_p_loss_mw_key_uses_base_mva(self):
        """_compute_reward must not raise KeyError when p_loss_MW is absent
        (e.g. external wrapper truncated the state dict)."""
        e = DistGrid3PhaseEnv(loss_penalty_weight=0.25)
        state = {'safety_info': {}}  # p_loss_MW intentionally omitted
        reward = e._compute_reward(state)
        expected = -0.25 * e.baseMVA
        assert np.isfinite(reward)
        assert reward == pytest.approx(expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
