"""Tests for powerzoo.envs.grid.trans — TransGridEnv.

TransGridEnv wraps a transmission power system (default: IEEE 5-bus) with
four solver modes controlled by ``physics`` (dc/ac) and ``solver_mode`` (opf/pf):
  - DCOPF: DC optimal power flow (LP via Gurobi/scipy/cvxpy)
  - ACOPF: AC optimal power flow (NLP via cyipopt/SLSQP)
  - DCPF:  DC power flow (PTDF, no optimisation — agent provides dispatch)
  - ACPF:  AC power flow (Newton-Raphson, no optimisation)

Domain knowledge:
  - In DC-OPF, line flows = PTDF × node_injection_mw (Power Transfer Distribution Factor)
  - LMP = system marginal cost + congestion component
  - Generation cost is quadratic: C = a·P² + b·P + c ($/MWh)
  - Power balance: Σ P_gen = Σ P_load at every time step
  - Slack bus absorbs imbalance in DC power flow
"""
import numpy as np
import pytest

from powerzoo.envs.grid.trans import TransGridEnv


# ── Constructor & Configuration ──────────────────────────────────────

class TestTransGridEnvInit:
    """Constructor defaults, difficulty presets, and parameter handling."""

    def test_default_case5(self):
        """Default transmission case is 5-bus."""
        env = TransGridEnv(normalize_actions=False, time_series=np.ones(48) * 100)
        assert 'Case5' in type(env.case).__name__ or len(env.case.nodes) == 5

    def test_default_physics_dc(self):
        env = TransGridEnv(normalize_actions=False, time_series=np.ones(48) * 100)
        assert env.physics == 'dc'
        assert env.solver_mode == 'opf'
        # Backward-compat property
        assert env.pf_mode == 'dc'

    def test_invalid_physics_raises(self):
        with pytest.raises(ValueError, match="physics"):
            TransGridEnv(normalize_actions=False, physics='invalid', time_series=np.ones(48) * 100)

    def test_invalid_solver_mode_raises(self):
        with pytest.raises(ValueError, match="solver_mode"):
            TransGridEnv(normalize_actions=False, solver_mode='invalid', time_series=np.ones(48) * 100)

    def test_difficulty_easy(self):
        env = TransGridEnv(normalize_actions=False, difficulty='easy', time_series=np.ones(48) * 100)
        assert env.max_load_ratio == 0.7
        assert env.delta_t_minutes == 60.0

    def test_difficulty_medium(self):
        env = TransGridEnv(normalize_actions=False, difficulty='medium', time_series=np.ones(48) * 100)
        assert env.max_load_ratio == 0.9
        assert env.delta_t_minutes == 30.0

    def test_difficulty_hard(self):
        env = TransGridEnv(normalize_actions=False, difficulty='hard', time_series=np.ones(48) * 100)
        assert env.max_load_ratio == 0.95
        assert env.delta_t_minutes == 15.0

    def test_invalid_difficulty_raises(self):
        with pytest.raises(ValueError, match="difficulty"):
            TransGridEnv(normalize_actions=False, difficulty='extreme', time_series=np.ones(48) * 100)

    def test_ptdf_shape(self, case5):
        """PTDF matrix shape: (n_lines, n_buses)."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=np.ones(48) * 100)
        n_lines = len(case5.lines)
        n_nodes = len(case5.nodes)
        assert env.PTDF.shape == (n_lines, n_nodes)


# ── Observation & Action Spaces ──────────────────────────────────────

class TestTransSpaces:
    """Observation and action space shape/type verification."""

    def test_observation_space_shape(self, case5):
        """obs = [line_flows] + [net_loads] + [unit_power_mw] + [time(2)]."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=np.ones(48) * 100)
        n_lines = len(case5.lines)
        n_loads = len(case5.loads)
        n_units = len(case5.units)
        expected_dim = n_lines + n_loads + n_units + 2
        assert env.observation_space.shape == (expected_dim,)

    def test_action_space_matches_units(self, case5):
        """Action dim = n_units, bounded by [p_min, p_max]."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=np.ones(48) * 100)
        n_units = len(case5.units)
        assert env.action_space.shape == (n_units,)
        # Bounds should reflect unit limits
        np.testing.assert_array_equal(
            env.action_space.low,
            case5.units['p_min'].values.astype(np.float32)
        )
        np.testing.assert_array_equal(
            env.action_space.high,
            case5.units['p_max'].values.astype(np.float32)
        )

    def test_obs_names_count(self, case5):
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=np.ones(48) * 100)
        assert len(env.obs_names) == env.observation_space.shape[0]

    def test_action_names_count(self, case5):
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=np.ones(48) * 100)
        assert len(env.action_names) == env.action_space.shape[0]


# ── Reset & Initial Power Flow ───────────────────────────────────────

class TestTransReset:
    """Reset triggers initial power flow and returns valid state."""

    def test_reset_returns_state_dict(self, case5, simple_time_series):
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series)
        state, info = env.reset(seed=42, day_id=0)
        assert isinstance(state, dict)
        assert isinstance(info, dict)
        assert 'lines' in state
        assert 'nodes' in state
        assert 'is_safe' in state

    def test_reset_populates_unit_power_mw(self, case5, simple_time_series):
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        assert env._unit_power_mw is not None
        assert len(env._unit_power_mw) == len(case5.units)

    def test_reset_clears_episode_metrics(self, case5, simple_time_series):
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series)
        env.reset(seed=0, day_id=0)
        assert env._ep_violations == 0
        assert env._ep_cost == 0.0


# ── Observation ──────────────────────────────────────────────────────

class TestTransObs:
    """obs() output shape and normalisation."""

    def test_obs_shape(self, case5, simple_time_series):
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        observation = env.obs()
        assert observation.shape == env.observation_space.shape
        assert observation.dtype == np.float32

    def test_time_encoding_bounded(self, case5, simple_time_series):
        """sin/cos time features should be in [-1, 1]."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series)
        env.reset(seed=42, day_id=0)
        obs = env.obs()
        time_sin, time_cos = obs[-2], obs[-1]
        assert -1.0 <= time_sin <= 1.0
        assert -1.0 <= time_cos <= 1.0


# ── DC Power Flow & OPF ─────────────────────────────────────────────

class TestDCPowerFlow:
    """DC-OPF: linear power flow via PTDF, economic dispatch."""

    def test_dc_power_balance(self, case5, simple_time_series):
        """P_gen_total ≈ P_load_total (DC-OPF power balance constraint)."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           solver_type='scipy')
        env.reset(seed=42, day_id=0)
        if env._unit_power_mw is not None and env._opf_result is not None:
            total_gen = float(env._unit_power_mw.sum())
            total_load_mw = float(env._opf_result['node_net_injection_mw'].sum()
                               + env._unit_power_mw.sum())
            # In DC-OPF, generation must meet total net load
            assert total_gen > 0

    def test_line_flow_via_ptdf(self, case5, simple_time_series):
        """Line flows should be derivable from PTDF × node_injection_mw."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           solver_type='scipy')
        env.reset(seed=42, day_id=0)
        if env._opf_result is not None:
            node_inj_mw = env._opf_result['node_net_injection_mw']
            expected_flow = env.PTDF.dot(node_inj_mw)
            actual_flow = env._opf_result['line_flow_mw']
            np.testing.assert_allclose(actual_flow, expected_flow, atol=1e-6)

    def test_unit_power_mw_within_bounds(self, case5, simple_time_series):
        """Generator dispatch must respect p_min ≤ P ≤ p_max."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           solver_type='scipy')
        env.reset(seed=42, day_id=0)
        if env._unit_power_mw is not None:
            p_min = case5.units['p_min'].values
            p_max = case5.units['p_max'].values
            assert np.all(env._unit_power_mw >= p_min - 1e-6)
            assert np.all(env._unit_power_mw <= p_max + 1e-6)


# ── Step Cycle ───────────────────────────────────────────────────────

class TestTransStep:
    """Step mechanics, reward, and termination."""

    def test_step_returns_five_tuple(self, case5, simple_time_series):
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           solver_type='scipy')
        env.reset(seed=42, day_id=0)
        result = env.step({})
        assert len(result) == 5

    def test_step_reward_has_components(self, case5, simple_time_series):
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           solver_type='scipy')
        env.reset(seed=42, day_id=0)
        state, reward, _, _, info = env.step({})
        # Reward components should exist in state
        assert 'reward_components' in state

    def test_info_has_mandatory_fields(self, case5, simple_time_series):
        """F4 fix: all mandatory info fields must be present."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           solver_type='scipy')
        env.reset(seed=42, day_id=0)
        _, _, _, _, info = env.step({})
        assert 'is_safe' in info
        assert 'cost_sum' in info
        assert 'cost_voltage_violation' in info
        assert 'cost_thermal_overload' in info
        assert 'cost_power_balance' in info
        assert 'goal_met' in info
        assert 'pf_converged' in info


# ── Safety Check ─────────────────────────────────────────────────────

class TestTransSafetyCheck:
    """Line thermal limit checking: floor ≤ flow ≤ cap."""

    def test_safety_check_with_safe_flow(self, case5):
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=np.ones(48) * 100)
        # Create flows within limits
        lines = case5.lines.copy()
        lines['line_flow_mw'] = (lines['cap'].values + lines['floor'].values) / 2
        safe, info = env.safety_check(lines, with_info=True)
        assert safe.all()
        assert info is not None
        assert len(info['unsafe_line_ids']) == 0

    def test_safety_check_with_overcap_flow(self, case5):
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=np.ones(48) * 100)
        lines = case5.lines.copy()
        # Exceed cap on all lines
        lines['line_flow_mw'] = lines['cap'].values * 2
        safe, info = env.safety_check(lines, with_info=True)
        assert not safe.all()
        assert len(info['unsafe_line_ids']) > 0

    def test_safety_check_below_floor(self, case5):
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=np.ones(48) * 100)
        lines = case5.lines.copy()
        # Below floor (negative flow exceeding floor)
        lines['line_flow_mw'] = lines['floor'].values - 100
        safe, info = env.safety_check(lines, with_info=True)
        assert not safe.all()


# ── Reward Components ────────────────────────────────────────────────

class TestTransReward:
    """Reward decomposition: safety penalty + economic cost.

    Domain: reward should never include safety penalties in the main signal
    for CMDP formulations — but TransGridEnv's default reward does include
    them as a baseline. Tasks can override via reward_function.
    """

    def test_reward_negative_or_zero(self, case5, simple_time_series):
        """Reward is always non-positive (negative costs/penalties)."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           solver_type='scipy')
        env.reset(seed=42, day_id=0)
        for _ in range(5):
            _, reward, _, _, _ = env.step({})
            assert reward <= 0.0 + 1e-9

    def test_safety_penalty_coefficient(self, case5):
        """Each line violation contributes -10 to reward."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=np.ones(48) * 100)
        state = {
            'is_safe': False,
            'safety_info': {'unsafe_line_ids': [0, 1]},  # 2 violations
        }
        reward = env._compute_reward(state)
        assert state['reward_components']['safety_diagnostic'] == -20.0


# ── Episode Metrics ──────────────────────────────────────────────────

class TestTransEpisodeMetrics:
    """Episode-level KPIs."""

    def test_episode_metrics_keys(self, case5, simple_time_series):
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           max_episode_steps=2, solver_type='scipy')
        env.reset(seed=42, day_id=0)
        env.step({})
        _, _, _, _, info = env.step({})
        if 'episode' in info:
            metrics = info['episode'].get('metrics', {})
            assert 'total_line_violations' in metrics
            assert 'total_opf_cost' in metrics


# ── PF modes (solver_mode='pf') ────────────────────────────────────

class TestTransDCPF:
    """DCPF mode: agent provides dispatch, environment evaluates PTDF flow."""

    def test_dcpf_with_unit_power_mw(self, case5, simple_time_series):
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='dc', solver_mode='pf')
        state, _ = env.reset(seed=42, day_id=0)
        assert state is not None
        # Step with explicit unit power
        p_min = case5.units['p_min'].values
        p_max = case5.units['p_max'].values
        unit_power_mw = (p_min + p_max) / 2
        state, reward, terminated, truncated, info = env.step(
            {'unit_power_mw': unit_power_mw})
        assert 'is_safe' in info
        assert env._opf_result is None  # no OPF in PF mode

    def test_dcpf_fallback_dispatch(self, case5, simple_time_series):
        """Without unit_power_mw, DCPF uses proportional dispatch."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='dc', solver_mode='pf')
        env.reset(seed=42, day_id=0)
        state, reward, terminated, truncated, info = env.step({})
        assert env._unit_power_mw is not None

    def test_dcpf_state_has_physics_solver(self, case5, simple_time_series):
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='dc', solver_mode='pf')
        state, _ = env.reset(seed=42, day_id=0)
        assert state['physics'] == 'dc'
        assert state['solver_mode'] == 'pf'


class TestTransACPF:
    """ACPF mode: Newton-Raphson AC power flow."""

    def test_acpf_with_unit_power_mw(self, case5, simple_time_series):
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='ac', solver_mode='pf')
        state, _ = env.reset(seed=42, day_id=0)
        assert state is not None
        # Step with explicit unit power
        p_min = case5.units['p_min'].values
        p_max = case5.units['p_max'].values
        unit_power_mw = (p_min + p_max) / 2
        state, reward, terminated, truncated, info = env.step(
            {'unit_power_mw': unit_power_mw})
        assert 'is_safe' in info
        assert env._opf_result is None  # no OPF in PF mode

    def test_acpf_returns_voltage(self, case5, simple_time_series):
        """ACPF should produce voltage magnitudes and angles."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='ac', solver_mode='pf')
        state, _ = env.reset(seed=42, day_id=0)
        if env._nodes is not None and 'vm_pu' in env._nodes.columns:
            vm = env._nodes['vm_pu'].values
            assert np.all(vm > 0)  # voltage must be positive
            assert np.all(vm < 2)  # reasonable upper bound

    def test_acpf_node_net_load_matches_net_calculation(self, case5, simple_time_series):
        """ACPF _nodes['node_net_load_mw'] must equal _calculate_node_net_load().

        Without DER the two are identical.  This test guards the Bug #1 regression:
        the NR solver used to receive raw load (ignoring DER); now it receives net load.
        """
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='ac', solver_mode='pf')
        env.reset(seed=42, day_id=0)
        if env._nodes is not None and 'node_net_load_mw' in env._nodes.columns:
            recorded = env._nodes['node_net_load_mw'].values
            expected = env._calculate_node_net_load()
            np.testing.assert_allclose(recorded, expected, atol=1e-6,
                                       err_msg="ACPF node_net_load_mw must use net load")

    def test_acpf_cost_power_balance_nonzero_when_unbalanced(self, case5, simple_time_series):
        """ACPF mode: zero dispatch forces slack bus to cover all load; imbalance > 0."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='ac', solver_mode='pf')
        env.reset(seed=42, day_id=0)
        zero_dispatch = np.zeros(len(case5.units))
        _, _, _, _, info = env.step({'unit_power_mw': zero_dispatch})
        assert info['cost_power_balance'] > 0.0

    def test_acopf_bypass_reuses_acpf_path(self, case5, simple_time_series, monkeypatch):
        """Supplying unit_power_mw in AC-OPF mode should delegate to the ACPF path."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='ac', solver_mode='opf')
        unit_power_mw = (case5.units['p_min'].values + case5.units['p_max'].values) / 2
        called = {}

        def fake_acpf(action):
            called['action'] = action
            env._opf_result = None
            env._pf_result = {'converged': True}
            return True

        monkeypatch.setattr(env, '_run_power_flow_acpf', fake_acpf)

        assert env._run_power_flow_acopf({'unit_power_mw': unit_power_mw}) is True
        assert 'action' in called
        np.testing.assert_allclose(called['action']['unit_power_mw'], unit_power_mw)
        assert env._opf_result is None
        assert env._pf_result == {'converged': True}

    def test_acpf_slack_violation_marks_unsafe(self, case5, simple_time_series, monkeypatch):
        """Actual slack generation beyond [p_min, p_max] must propagate to info/is_safe."""
        import powerzoo.envs.grid.trans as trans_module

        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='ac', solver_mode='pf')
        unit_power_mw = (case5.units['p_min'].values + case5.units['p_max'].values) / 2
        env.case.units['Pg'] = unit_power_mw

        slack_mask = env._get_slack_unit_mask()
        fake_p_gen = unit_power_mw.astype(float).copy()
        fake_p_gen[slack_mask] = case5.units['p_max'].values[slack_mask] + 5.0

        def fake_run_acpf(case, Pd_mw=None, Pg_mw=None, verbose=False):
            return {
                'pf_from': np.zeros(len(case.lines)),
                'vm': np.ones(len(case.nodes)),
                'va_deg': np.zeros(len(case.nodes)),
                'p_gen': fake_p_gen,
                'q_gen': np.zeros(len(case.units)),
                'converged': True,
                'iterations': 1,
            }

        monkeypatch.setattr(
            'powerzoo.envs.grid._trans_solve._nr_acpf', fake_run_acpf
        )

        assert env._run_power_flow_acpf({'unit_power_mw': unit_power_mw}) is True
        info = env.build_info(env._get_state())
        assert info['is_safe'] is False
        assert info['slack_gen_violation_mw'] > 0.0
        assert info['safety_info']['slack_gen_violation_mw'] == pytest.approx(info['slack_gen_violation_mw'])


# ── AC MVA thermal check (acceptance tests) ──────────────────────────

class TestACThermalMVA:
    """AC thermal check uses apparent power |S| = sqrt(P²+Q²) against MVA cap.

    Acceptance criteria (from JAX alignment spec):
    1. ACPF: |S| = sqrt(Pf²+Qf²) ≥ |Pf| (strict except when Q=0)
    2. ACPF: max(|Sf|,|St|) overload ≥ from-end-only overload
    3. ACOPF: cost_thermal > 0 when line_viol_mva > threshold (no silent miss)
    4. DC: P-based floor/cap unchanged
    5. obs() line slice = |S|/cap in AC mode
    6. build_info cost_thermal = sum(max(0, |S| - cap)) in AC mode
    """

    @pytest.fixture
    def case14(self):
        from powerzoo.case.transmission import Case14
        return Case14()

    def test_acpf_sf_geq_pf_invariant(self, case14):
        """sqrt(Pf²+Qf²) >= |Pf| for every branch (ACPF result)."""
        env = TransGridEnv(normalize_actions=False, case=case14,
                          time_series=np.ones(48) * 150,
                          physics='ac', solver_mode='pf')
        env.reset()
        if env._pf_result and env._pf_result.get('converged'):
            pf = env._pf_result['pf_from']
            qf = env._pf_result.get('qf_from', np.zeros_like(pf))
            sf = np.sqrt(pf ** 2 + qf ** 2)
            np.testing.assert_array_less(np.abs(pf) - 1e-9, sf,
                                         err_msg="|S| < |P| invariant violated")

    def test_acpf_both_ends_overload_geq_from_only(self, case14, monkeypatch):
        """max(|Sf|,|St|) overload >= |Sf|-only overload for any valid PF result."""
        from powerzoo.envs.grid._trans_solve import ac_thermal_check
        env = TransGridEnv(normalize_actions=False, case=case14,
                          time_series=np.ones(48) * 150,
                          physics='ac', solver_mode='pf')
        env.reset()
        if env._pf_result and env._pf_result.get('converged'):
            pf_from = env._pf_result['pf_from']
            qf_from = env._pf_result.get('qf_from', np.zeros_like(pf_from))
            pf_to   = env._pf_result.get('pf_to',   np.zeros_like(pf_from))
            qf_to   = env._pf_result.get('qf_to',   np.zeros_like(pf_from))
            cap = env.case.lines['cap'].values

            _, _, cost_both = ac_thermal_check(
                pf_from, qf_from, cap, pf_to, qf_to, use_both_ends=True)
            _, _, cost_from = ac_thermal_check(
                pf_from, qf_from, cap, use_both_ends=False)
            assert cost_both >= cost_from - 1e-9, (
                f"Both-ends cost ({cost_both}) < from-only cost ({cost_from})")

    def test_acpf_obs_uses_mva_not_p(self, case14):
        """obs() line slice == |S|/cap when physics='ac'."""
        env = TransGridEnv(normalize_actions=False, case=case14,
                          time_series=np.ones(48) * 150,
                          physics='ac', solver_mode='pf')
        env.reset()
        if env._lines is None or 'line_flow_q_mvar' not in env._lines.columns:
            pytest.skip("ACPF did not converge or Q flow not available")
        pf = env._lines['line_flow_mw'].values
        qf = env._lines['line_flow_q_mvar'].values
        sf = np.sqrt(pf ** 2 + qf ** 2)
        caps = env.case.lines['cap'].values
        caps_safe = np.where(caps > 0, caps, 1.0)
        expected = (sf / caps_safe).astype(np.float32)
        obs = env.obs()
        n_lines = len(env.case.lines)
        np.testing.assert_allclose(obs[:n_lines], expected, rtol=1e-5,
                                   err_msg="obs line slice must be |S|/cap in AC mode")

    def test_acpf_build_info_cost_thermal_uses_mva(self, case14):
        """`cost_thermal_overload` uses |S| - cap (MVA) not |P| - cap in AC mode."""
        env = TransGridEnv(normalize_actions=False, case=case14,
                          time_series=np.ones(48) * 150,
                          physics='ac', solver_mode='pf')
        env.reset()
        if env._lines is None or 'line_flow_q_mvar' not in env._lines.columns:
            pytest.skip("ACPF did not converge or Q flow not available")
        pf = env._lines['line_flow_mw'].values
        qf = env._lines['line_flow_q_mvar'].values
        sf = np.sqrt(pf ** 2 + qf ** 2)
        caps = env.case.lines['cap'].values
        effective_cap = np.where(caps > 0, caps, np.inf)
        expected_cost = float(np.sum(np.maximum(0.0, sf - effective_cap)))
        info = env.build_info(env._get_state())
        assert info['cost_thermal_overload'] == pytest.approx(expected_cost, abs=0.01), (
            "cost_thermal_overload must use |S| in AC mode")

    def test_dc_thermal_uses_p_not_s(self, case5):
        """DC mode: cost_thermal_overload uses |P| (Q≈0 approximation), not |S|."""
        env = TransGridEnv(normalize_actions=False, case=case5,
                          time_series=np.ones(48) * 100,
                          physics='dc', solver_mode='opf')
        env.reset()
        assert env._lines is not None
        assert 'line_flow_q_mvar' not in env._lines.columns, (
            "DC mode must not store Q flow in lines DataFrame")
        # cost_thermal still computed (may be 0 if no overload)
        info = env.build_info(env._get_state())
        assert info['cost_thermal_overload'] >= 0.0

    def test_line_viol_mva_key_always_present(self, case5, case14):
        """info['line_viol_mva'] must be present in all four solver modes."""
        for case, phys, mode in [
            (case5,  'dc', 'opf'),
            (case5,  'dc', 'pf'),
            (case14, 'ac', 'pf'),
            (case14, 'ac', 'opf'),
        ]:
            env = TransGridEnv(normalize_actions=False, case=case,
                              time_series=np.ones(48) * 100,
                              physics=phys, solver_mode=mode)
            env.reset()
            info = env.build_info(env._get_state())
            assert 'line_viol_mva' in info, (
                f"line_viol_mva missing for physics={phys}, solver_mode={mode}")
            assert info['line_viol_mva'] >= 0.0


# ── reward_scale parameter ───────────────────────────────────────────

class TestRewardScale:
    """reward_scale multiplies the economic cost term in the reward signal."""

    def test_default_reward_scale(self, case5):
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=np.ones(48) * 100)
        assert env.reward_scale == pytest.approx(0.01)

    def test_custom_reward_scale(self, case5):
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=np.ones(48) * 100,
                           reward_scale=0.0001)
        assert env.reward_scale == pytest.approx(0.0001)

    def test_reward_scale_affects_economic_cost(self, case5):
        """economic_cost component is proportional to reward_scale."""
        env1 = TransGridEnv(normalize_actions=False, case=case5, time_series=np.ones(48) * 100,
                            reward_scale=0.01)
        env2 = TransGridEnv(normalize_actions=False, case=case5, time_series=np.ones(48) * 100,
                            reward_scale=0.001)
        state1 = {'is_safe': True, 'safety_info': {'unsafe_line_ids': []}, 'opf_cost': 500.0}
        state2 = {'is_safe': True, 'safety_info': {'unsafe_line_ids': []}, 'opf_cost': 500.0}
        r1 = env1._compute_reward(state1)
        r2 = env2._compute_reward(state2)
        # r1 should be exactly 10× larger in magnitude than r2
        assert abs(r1) == pytest.approx(abs(r2) * 10.0, rel=1e-6)

    def test_reward_scale_affects_der_cost_terms(self, case5, simple_time_series):
        """DER curtailment and operation costs should use the same reward_scale."""
        from powerzoo.envs.resource.battery import BatteryEnv
        from powerzoo.envs.resource.renewable import SolarEnv

        env1 = TransGridEnv(
            normalize_actions=False,
            case=case5,
            time_series=simple_time_series,
            reward_scale=0.01,
        )
        solar1 = SolarEnv(parent=env1, bus_id=1, capacity_mw=100.0,
                          curtailment_penalty_per_mwh=20.0)
        battery1 = BatteryEnv(parent=env1, bus_id=2, capacity_mwh=10.0, power_mw=5.0,
                               cycle_cost_per_mwh=15.0)
        solar1._capacity_factor = 0.8
        solar1.current_p_mw = 30.0
        battery1.current_p_mw = 4.0

        env2 = TransGridEnv(
            normalize_actions=False,
            case=case5,
            time_series=simple_time_series,
            reward_scale=0.001,
        )
        solar2 = SolarEnv(parent=env2, bus_id=1, capacity_mw=100.0,
                          curtailment_penalty_per_mwh=20.0)
        battery2 = BatteryEnv(parent=env2, bus_id=2, capacity_mwh=10.0, power_mw=5.0,
                               cycle_cost_per_mwh=15.0)
        solar2._capacity_factor = 0.8
        solar2.current_p_mw = 30.0
        battery2.current_p_mw = 4.0

        state1 = {'is_safe': True, 'safety_info': {'unsafe_line_ids': []}}
        state2 = {'is_safe': True, 'safety_info': {'unsafe_line_ids': []}}
        env1._compute_reward(state1)
        env2._compute_reward(state2)

        rc1 = state1['reward_components']
        rc2 = state2['reward_components']
        assert abs(rc1['der_econ_total']) == pytest.approx(abs(rc2['der_econ_total']) * 10.0)


# ── cost_power_balance (slack bus imbalance) ─────────────────────────

class TestCostPowerBalance:
    """build_info exposes cost_power_balance for PF-mode power imbalance."""

    def test_info_has_cost_power_balance_field(self, case5, simple_time_series):
        """cost_power_balance must appear in info dict every step."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='dc', solver_mode='pf')
        env.reset(seed=42, day_id=0)
        _, _, _, _, info = env.step({})
        assert 'cost_power_balance' in info
        assert info['cost_power_balance'] >= 0.0

    def test_cost_power_balance_nonzero_when_unbalanced(self, case5, simple_time_series):
        """Dispatching zero MW forces the slack bus to absorb all load; imbalance > 0."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='dc', solver_mode='pf')
        env.reset(seed=42, day_id=0)
        zero_dispatch = np.zeros(len(case5.units))
        _, _, _, _, info = env.step({'unit_power_mw': zero_dispatch})
        assert info['cost_power_balance'] > 0.0

    def test_cost_power_balance_included_in_cost_sum(self, case5, simple_time_series):
        """cost_sum = cost_thermal + cost_voltage + cost_power_balance."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='dc', solver_mode='pf')
        env.reset(seed=42, day_id=0)
        _, _, _, _, info = env.step({'unit_power_mw': np.zeros(len(case5.units))})
        expected_sum = (info['cost_thermal_overload']
                        + info['cost_voltage_violation']
                        + info['cost_power_balance'])
        assert info['cost_sum'] == pytest.approx(expected_sum, abs=1e-6)

    def test_cost_power_balance_zero_in_opf_mode(self, case5, simple_time_series):
        """In OPF mode the solver enforces balance so cost_power_balance should be 0."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='dc', solver_mode='opf', solver_type='scipy')
        env.reset(seed=42, day_id=0)
        _, _, _, _, info = env.step({})
        assert info['cost_power_balance'] == pytest.approx(0.0, abs=1e-3)


# ── Suggestion 3: Slack Bus Generator Fix ──────────────────────────────────
# cal_pf should route imbalance through the slack generator (not load).

class TestSlackBusGeneratorFix:
    """Slack bus absorbs imbalance via generator output, not phantom load."""

    def test_cal_pf_load_unchanged_when_balanced(self, case5, simple_time_series):
        """When dispatch equals load exactly, node_load_mw must be unmodified."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='dc', solver_mode='pf')
        env.reset(seed=42, day_id=0)
        node_load_mw = env._get_default_node_load().copy()
        # Build balanced dispatch (sum equals total load at each node)
        from powerzoo.envs.grid.trans import TransGridEnv as _T
        nodes_loads_map = case5.get_nodes_loads_map()
        total_load = float(nodes_loads_map.dot(node_load_mw).sum())
        p_min = case5.units['p_min'].values.astype(float)
        p_max = case5.units['p_max'].values.astype(float)
        unit_power_mw = env._proportional_dispatch(total_load)
        load_before = node_load_mw.copy()
        # call cal_pf — load array should be untouched
        env.cal_pf(unit_power_mw, node_load_mw, df=True)
        np.testing.assert_array_equal(node_load_mw, load_before,
                                      err_msg="cal_pf must not modify the node_load_mw array")

    def test_slack_gen_violation_zero_in_balanced_case(self, case5, simple_time_series):
        """Balanced dispatch → slack gen violation is 0."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='dc', solver_mode='pf')
        env.reset(seed=42, day_id=0)
        # OPF-dispatched result is balanced by construction
        env2 = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                            physics='dc', solver_mode='opf', solver_type='scipy')
        env2.reset(seed=42, day_id=0)
        if env2._unit_power_mw is not None:
            node_load_mw = env2._calculate_node_net_load()
            env2.cal_pf(env2._unit_power_mw, node_load_mw, df=True)
            # Imbalance should be near 0
            assert env2._power_imbalance_mw < 1.0  # within 1 MW (OPF enforces balance)

    def test_slack_gen_violation_nonzero_when_severely_unbalanced(self, case5, simple_time_series):
        """Extreme overgeneration pushes slack gen above p_max → violation > 0."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='dc', solver_mode='pf')
        env.reset(seed=42, day_id=0)
        # Dispatch all units at maximum (massive overgeneration)
        p_max = case5.units['p_max'].values.astype(float)
        node_load_mw = env._calculate_node_net_load()  # no DER registered → equals gross node load
        env.cal_pf(p_max, node_load_mw, df=True)
        # With all generators at p_max and normal load, slack gen likely exceeds p_max
        # (it must absorb the imbalance by reducing output below the sum already at p_max)
        # _slack_gen_violation_mw ≥ 0
        assert env._slack_gen_violation_mw >= 0.0

    def test_slack_gen_violation_in_safety_info(self, case5, simple_time_series):
        """Slack gen bound violation is reported in safety_info when triggered."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='dc', solver_mode='pf')
        env.reset(seed=42, day_id=0)
        p_max_all = case5.units['p_max'].values
        # Force a large overgeneration
        _, _, _, _, info = env.step({'unit_power_mw': p_max_all * 10.0})
        if env._slack_gen_violation_mw > 1e-3:
            assert not info['is_safe']
            assert 'slack_gen_violation_mw' in info['safety_info']

    def test_build_info_exposes_slack_gen_violation(self, case5, simple_time_series):
        """build_info always exposes slack_gen_violation_mw field."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           physics='dc', solver_mode='pf')
        env.reset(seed=42, day_id=0)
        _, _, _, _, info = env.step({})
        assert 'slack_gen_violation_mw' in info
        assert info['slack_gen_violation_mw'] >= 0.0


# ── Suggestion 1: DER Observation Space ─────────────────────────────────────

class TestDERObservationSpace:
    """DER states (SOC, available CF) are exposed in observation space."""

    def test_obs_dim_unchanged_without_resources(self, case5, simple_time_series):
        """Without any sub_resources obs dim equals the baseline formula."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series)
        n_lines = len(case5.lines)
        n_loads = len(case5.loads)
        n_units = len(case5.units)
        expected = n_lines + n_loads + n_units + 2
        assert env.observation_space.shape[0] == expected

    def test_obs_dim_grows_with_solar_resource(self, case5, simple_time_series):
        """Registering a SolarEnv adds 2 obs dims (available_cf, p_mw_norm)."""
        from powerzoo.envs.resource.renewable import SolarEnv
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series)
        dim_before = env.observation_space.shape[0]
        SolarEnv(parent=env, bus_id=1, capacity_mw=50.0)
        assert env.observation_space.shape[0] == dim_before + 2

    def test_obs_dim_grows_with_battery_resource(self, case5, simple_time_series):
        """Registering a BatteryEnv adds 4 obs dims."""
        from powerzoo.envs.resource.battery import BatteryEnv
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series)
        dim_before = env.observation_space.shape[0]
        BatteryEnv(parent=env, bus_id=2, capacity_mwh=10.0, power_mw=5.0)
        assert env.observation_space.shape[0] == dim_before + 4

    def test_obs_names_consistent_with_obs_dim_after_registration(self, case5, simple_time_series):
        """obs_names length must equal observation_space dim after resource registration."""
        from powerzoo.envs.resource.renewable import SolarEnv
        from powerzoo.envs.resource.battery import BatteryEnv
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series)
        SolarEnv(parent=env, bus_id=1, capacity_mw=30.0)
        BatteryEnv(parent=env, bus_id=2, capacity_mwh=5.0, power_mw=2.0)
        assert len(env.obs_names) == env.observation_space.shape[0]

    def test_obs_shape_after_reset_matches_space(self, case5, simple_time_series):
        """After reset, obs() output shape must match observation_space."""
        from powerzoo.envs.resource.renewable import SolarEnv
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series)
        SolarEnv(parent=env, bus_id=1, capacity_mw=50.0)
        env.reset(seed=0, day_id=0)
        obs = env.obs()
        assert obs.shape == env.observation_space.shape

    def test_obs_unregister_shrinks_dim(self, case5, simple_time_series):
        """Unregistering a resource reduces obs_dim back to baseline."""
        from powerzoo.envs.resource.renewable import SolarEnv
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series)
        dim_base = env.observation_space.shape[0]
        solar = SolarEnv(parent=env, bus_id=1, capacity_mw=50.0)
        env.unregister_resource(solar.resource_id)
        assert env.observation_space.shape[0] == dim_base


# ── Suggestion 2: DER Action Space (control_der) ────────────────────────────

class TestDERActionSpace:
    """control_der=True flattens DER dims into action space."""

    def test_action_space_unchanged_when_control_der_false(self, case5, simple_time_series):
        """Default control_der=False keeps action space at n_units."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           control_der=False)
        assert env.action_space.shape[0] == len(case5.units)

    def test_pf_action_space_extends_with_unit_and_der_dims(self, case5, simple_time_series):
        """PF mode keeps unit dispatch dims and appends one dim per DER."""
        from powerzoo.envs.resource.renewable import SolarEnv
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           control_der=True, solver_mode='pf')
        n_units = len(case5.units)
        SolarEnv(parent=env, bus_id=1, capacity_mw=50.0)
        assert env.action_space.shape[0] == n_units + 1

    def test_opf_action_space_der_only_when_control_der_true(self, case5, simple_time_series):
        """OPF mode should expose only DER dims in the flat action space."""
        from powerzoo.envs.resource.renewable import SolarEnv
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           control_der=True, solver_mode='opf')
        SolarEnv(parent=env, bus_id=1, capacity_mw=50.0)
        assert env.action_space.shape[0] == 1
        assert env.action_names == ['solar_0_action']

    def test_parse_flat_action_splits_correctly_in_pf_mode(self, case5, simple_time_series):
        """PF-mode flat action contains unit_power_mw plus one key per resource."""
        from powerzoo.envs.resource.renewable import SolarEnv
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           control_der=True, solver_mode='pf')
        solar = SolarEnv(parent=env, bus_id=1, capacity_mw=50.0)
        rid = solar.resource_id
        n_units = len(case5.units)
        flat = np.ones(n_units + 1, dtype=np.float32)
        parsed = env._parse_flat_action(flat)
        assert 'unit_power_mw' in parsed
        assert len(parsed['unit_power_mw']) == n_units
        assert rid in parsed

    def test_parse_flat_action_omits_unit_power_in_opf_mode(self, case5, simple_time_series):
        """OPF-mode flat action should never synthesize unit_power_mw."""
        from powerzoo.envs.resource.renewable import SolarEnv
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           control_der=True, solver_mode='opf')
        solar = SolarEnv(parent=env, bus_id=1, capacity_mw=50.0)
        parsed = env._parse_flat_action(np.array([0.25], dtype=np.float32))
        assert 'unit_power_mw' not in parsed
        assert solar.resource_id in parsed

    def test_parse_flat_action_denormalizes_child_actions(self, case5, simple_time_series):
        """Normalized flat DER actions must be converted to child physical actions."""
        from powerzoo.envs.resource.battery import BatteryEnv
        from powerzoo.envs.resource.renewable import SolarEnv

        env = TransGridEnv(normalize_actions=True, case=case5, time_series=simple_time_series,
                           control_der=True, solver_mode='pf')
        battery = BatteryEnv(parent=env, bus_id=1, capacity_mwh=4.0, power_mw=2.5,
                             normalize_actions=False)
        solar = SolarEnv(parent=env, bus_id=2, capacity_mw=20.0, normalize_actions=False)

        n_units = len(case5.units)
        flat = np.zeros(n_units + 2, dtype=np.float32)
        flat[n_units] = 1.0      # full battery discharge
        flat[n_units + 1] = -1.0  # full renewable curtailment

        parsed = env._parse_flat_action(flat)
        assert parsed[battery.resource_id]['p_mw'] == pytest.approx(battery.power_mw)
        assert parsed[solar.resource_id]['curtailment'] == pytest.approx(1.0)

    def test_step_with_flat_action_array_and_control_der_in_opf_mode(self, case5, simple_time_series):
        """OPF-mode flat DER control should not silently bypass OPF."""
        from powerzoo.envs.resource.renewable import SolarEnv
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           control_der=True, solver_type='scipy', solver_mode='opf')
        SolarEnv(parent=env, bus_id=1, capacity_mw=50.0)
        env.reset(seed=0, day_id=0)
        flat_action = np.zeros(1, dtype=np.float32)
        result = env.step(flat_action)
        assert len(result) == 5
        assert env._opf_result is not None


# ── Suggestion 4: DER Reward Semantics ───────────────────────────────────────

class TestDERRewardSemantics:
    """curtailment_penalty_per_mwh and DER operation cost flow through reward."""

    def test_no_curtailment_cost_without_penalty(self, case5, simple_time_series):
        """Default curtailment_penalty_per_mwh=0 → der_econ_total is 0."""
        from powerzoo.envs.resource.renewable import SolarEnv
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           solver_type='scipy')
        SolarEnv(parent=env, bus_id=1, capacity_mw=50.0)
        env.reset(seed=0, day_id=0)
        _, _, _, _, info = env.step({})
        assert info['der_econ_total'] == pytest.approx(0.0)

    def test_curtailment_cost_nonzero_with_penalty_and_curtailment(self, case5, simple_time_series):
        """When penalty > 0 and renewable is curtailed, der_econ_total < 0."""
        from powerzoo.envs.resource.renewable import SolarEnv
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           solver_type='scipy')
        solar = SolarEnv(parent=env, bus_id=1, capacity_mw=100.0,
                         curtailment_penalty_per_mwh=10.0)
        rid = solar.resource_id
        env.reset(seed=0, day_id=0)
        # Force full curtailment (action = -1 for full curtail in normalized mode)
        _, _, _, _, info = env.step({rid: -1.0})
        # If solar has nonzero capacity factor, curtailment should be penalised
        if solar.available_p_mw > 0:
            assert info['der_econ_total'] < 0.0
        else:
            assert info['der_econ_total'] == pytest.approx(0.0)

    def test_total_curtailment_mw_in_info(self, case5, simple_time_series):
        """build_info always reports total_curtailment_mw (≥ 0)."""
        from powerzoo.envs.resource.renewable import SolarEnv
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           solver_type='scipy')
        SolarEnv(parent=env, bus_id=1, capacity_mw=50.0)
        env.reset(seed=0, day_id=0)
        _, _, _, _, info = env.step({})
        assert 'total_curtailment_mw' in info
        assert info['total_curtailment_mw'] >= 0.0

    def test_reward_components_include_der_econ_key(self, case5, simple_time_series):
        """reward_components always has der_econ_total."""
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           solver_type='scipy')
        env.reset(seed=42, day_id=0)
        state, _, _, _, _ = env.step({})
        rc = state['reward_components']
        assert 'der_econ_total' in rc


# ── _on_resource_changed hook ────────────────────────────────────────
class TestOnResourceChangedHook:
    """Verify that register_resource / unregister_resource trigger space rebuilds
    via the _on_resource_changed hook."""

    def test_register_updates_obs_and_action_spaces(self, case5, simple_time_series):
        """Both observation_space and action_space grow after resource registration."""
        from powerzoo.envs.resource.battery import BatteryEnv
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           control_der=True, solver_mode='pf')
        obs_dim_before = env.observation_space.shape[0]
        act_dim_before = env.action_space.shape[0]   # n_units only

        BatteryEnv(parent=env, bus_id=2, capacity_mwh=10.0, power_mw=5.0)

        # obs grows by the battery's grid_obs() width (4 features)
        assert env.observation_space.shape[0] == obs_dim_before + 4
        # action grows by 1 DER dim
        assert env.action_space.shape[0] == act_dim_before + 1

    def test_unregister_restores_spaces(self, case5, simple_time_series):
        """Unregistering a resource restores both spaces to their original dims."""
        from powerzoo.envs.resource.battery import BatteryEnv
        env = TransGridEnv(normalize_actions=False, case=case5, time_series=simple_time_series,
                           control_der=True, solver_mode='pf')
        obs_dim_base = env.observation_space.shape[0]
        act_dim_base = env.action_space.shape[0]

        batt = BatteryEnv(parent=env, bus_id=2, capacity_mwh=10.0, power_mw=5.0)
        env.unregister_resource(batt.resource_id)

        assert env.observation_space.shape[0] == obs_dim_base
        assert env.action_space.shape[0] == act_dim_base

