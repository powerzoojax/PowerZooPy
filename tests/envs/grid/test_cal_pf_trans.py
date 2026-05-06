"""Solver-level tests for transmission power flow (cal_pf_trans).

Tests cover:
  - DC power flow (PTDF formulation): lossless, linear, B-theta
  - AC power flow (Newton-Raphson): convergence, Vm/Va, loss accounting
  - Ybus / PTDF matrix physical properties
  - Cross-validation against MATPOWER 8.0 reference outputs (Case14)

Domain knowledge applied:
  - DC PF assumes flat voltage, zero reactive power, lossless branches
      → P_injection = B_bus × θ  → line flow = PTDF × P_net
  - AC NR iterates Jacobian until ΔP/ΔQ < tol; losses = I²R
  - Ybus is symmetric for untapped branches; diagonal > off-diagonal (row sum)
  - PTDF column sums to ~0 for slack column (singular direction removed)
  - Power balance: ΣPg = ΣPd + P_loss  (AC)    ΣPg = ΣPd  (DC)
"""

import numpy as np
import pytest

from powerzoo.case.transmission import Case5, Case14
from powerzoo.envs.grid.cal_pf_trans import (
    build_dcpf_ptdf,
    build_ybus,
    run_acpf,
    run_dcpf,
)

# =====================================================================
# MATPOWER 8.0 reference data — Case14 (same as test_case14_pf.py)
# =====================================================================

DC_VA_DEG = np.array([
    0.000, -5.012, -12.954, -10.584, -9.094,
    -14.852, -13.907, -13.907, -15.695, -15.974,
    -15.619, -15.967, -16.140, -17.188,
])

DC_PF_FROM = np.array([
    147.84, 71.16, 70.01, 55.15, 40.97,
    -24.19, -61.75, 28.36, 16.55, 42.79,
    6.73, 7.61, 17.25, 0.00,
    28.36, 5.77, 9.64, -3.23, 1.51, 5.26,
])

AC_VM = np.array([
    1.060, 1.045, 1.010, 1.018, 1.020,
    1.070, 1.062, 1.090, 1.056, 1.051,
    1.057, 1.055, 1.050, 1.036,
])

AC_VA_DEG = np.array([
    0.000, -4.983, -12.725, -10.313, -8.774,
    -14.221, -13.360, -13.360, -14.939, -15.097,
    -14.791, -15.076, -15.156, -16.034,
])

AC_PGEN = np.array([232.39, 40.00, 0.00, 0.00, 0.00])
AC_QGEN = np.array([-16.55, 43.56, 25.08, 12.73, 17.62])

AC_PF_FROM = np.array([
    156.88, 75.51, 73.24, 56.13, 41.52,
    -23.29, -61.16, 28.07, 16.08, 44.09,
    7.35, 7.79, 17.75, -0.00,
    28.07, 5.23, 9.43, -3.79, 1.61, 5.64,
])

AC_P_LOSS = np.array([
    4.298, 2.763, 2.323, 1.677, 0.904,
    0.373, 0.514, 0.000, 0.000, 0.000,
    0.055, 0.072, 0.212, 0.000,
    0.000, 0.013, 0.116, 0.013, 0.006, 0.054,
])

AC_TOTAL_P_LOSS = 13.393
AC_TOTAL_LOAD = 259.0


# =====================================================================
# Fixtures (module-scoped to avoid redundant PF solves)
# =====================================================================

@pytest.fixture(scope="module")
def case5():
    c = Case5()
    c.init()
    return c


@pytest.fixture(scope="module")
def case14():
    return Case14()


@pytest.fixture(scope="module")
def dc14(case14):
    return run_dcpf(case14)


@pytest.fixture(scope="module")
def ac14(case14):
    return run_acpf(case14)


@pytest.fixture(scope="module")
def ptdf14(case14):
    return build_dcpf_ptdf(case14)


# =====================================================================
# Ybus matrix properties
# =====================================================================

class TestYbusProperties:
    """Ybus = G + jB is the network admittance matrix.

    Physical properties:
      - Ybus is square (n_bus × n_bus)
      - Symmetric for networks without phase-shifting transformers
        (Y_ij = Y_ji due to reciprocal admittances)
      - Diagonal dominant: |Y_ii| ≥ Σ_{j≠i} |Y_ij|
        (shunt admittances add to diagonal only)
      - Off-diagonal elements are negative real part for resistive lines
    """

    @staticmethod
    def _build_ybus_from_case(case):
        from powerzoo.envs.grid.cal_pf_trans import _extract_branch_arrays
        (n_bus, n_branch, n_gen, bus_id_to_idx, bus_type,
         br_from, br_to, br_r, br_x, br_b, br_ratio, br_angle, br_status
         ) = _extract_branch_arrays(case)
        nodes = case.nodes
        bus_gs = nodes['Gs'].values if 'Gs' in nodes.columns else np.zeros(n_bus)
        bus_bs = nodes['Bs'].values if 'Bs' in nodes.columns else np.zeros(n_bus)
        baseMVA = getattr(case, 'baseMVA', 100.0)
        return build_ybus(n_bus, br_from, br_to, br_r, br_x, br_b,
                          br_ratio, br_angle, br_status, bus_gs, bus_bs, baseMVA)

    def test_ybus_shape(self, case14):
        n_bus = len(case14.nodes)
        Ybus = self._build_ybus_from_case(case14)
        assert Ybus.shape == (n_bus, n_bus)

    def test_ybus_symmetry_no_phase_shifter(self, case14):
        """Case14 has transformers with tap ratio but no phase shift angle.
        Ybus should still be symmetric because angle=0 for all branches."""
        Ybus = self._build_ybus_from_case(case14)
        np.testing.assert_allclose(Ybus, Ybus.T, atol=1e-10,
                                   err_msg="Ybus should be symmetric for zero phase shift")

    def test_ybus_diagonal_nonzero(self, case14):
        """Every connected bus has nonzero self-admittance."""
        Ybus = self._build_ybus_from_case(case14)
        diag = np.abs(np.diag(Ybus))
        assert np.all(diag > 0), "Connected buses must have nonzero Y_ii"


# =====================================================================
# PTDF matrix properties
# =====================================================================

class TestPTDFProperties:
    """PTDF (Power Transfer Distribution Factor) maps bus injections to line flows.

    Physical properties:
      - Shape (n_line, n_bus); column for slack bus is effectively zero
      - flow = PTDF @ P_net_injection (in MW)
      - Rank = n_bus - 1 (one redundant equation removed for slack)
      - Zero net injection yields zero flow
    """

    def test_ptdf_shape(self, ptdf14, case14):
        n_bus = len(case14.nodes)
        n_line = len(case14.lines)
        assert ptdf14['PTDF'].shape == (n_line, n_bus)

    def test_zero_injection_zero_flow(self, ptdf14, case14):
        """Zero power injection at all buses → zero line flow."""
        flow = ptdf14['PTDF'] @ np.zeros(len(case14.nodes))
        np.testing.assert_allclose(flow, 0.0, atol=1e-12)

    def test_ptdf_slack_column_near_zero(self, ptdf14, case14):
        """Slack bus column in PTDF is the reference; should be near zero."""
        slack = 0  # Case14 slack bus is bus 1 (index 0)
        col = ptdf14['PTDF'][:, slack]
        np.testing.assert_allclose(col, 0.0, atol=1e-10)

    def test_ptdf_rank(self, ptdf14, case14):
        """PTDF rank should be n_bus - 1 (one equation removed for slack)."""
        n_bus = len(case14.nodes)
        rank = np.linalg.matrix_rank(ptdf14['PTDF'], tol=1e-8)
        assert rank == n_bus - 1

    def test_ptdf_unit_injection(self, ptdf14, case14):
        """1 MW injection at bus 2 (remove 1 MW at slack) → nonzero line flow."""
        n_bus = len(case14.nodes)
        P_inj = np.zeros(n_bus)
        P_inj[1] = 1.0   # inject at bus 2
        P_inj[0] = -1.0  # withdraw at slack
        flow = ptdf14['PTDF'] @ P_inj
        assert np.any(np.abs(flow) > 0.01), "1 MW inj should produce nonzero flow"

    def test_ptdf_sensitivity_sum_check(self, ptdf14, case14):
        """Sum across all line sensitivities for a unit injection should
        reflect the network position (not necessarily zero or one, but finite)."""
        n_bus = len(case14.nodes)
        for bus in range(1, n_bus):
            P_inj = np.zeros(n_bus)
            P_inj[bus] = 1.0
            P_inj[0] = -1.0
            flow = ptdf14['PTDF'] @ P_inj
            assert np.all(np.isfinite(flow))


# =====================================================================
# DC Power Flow vs MATPOWER — Case14
# =====================================================================

class TestDCPFCase14:
    """DC PF: P = B × θ, lossless, flat voltage.

    Properties:
      - All voltage magnitudes = 1.0 p.u. (flat voltage assumption)
      - Branch flow is antisymmetric: Pf_to = -Pf_from (no losses)
      - Total generation = total load (no losses)
      - Slack bus absorbs mismatch
    """

    def test_converged(self, dc14):
        assert dc14['converged']

    def test_vm_all_one(self, dc14):
        """DC PF: flat voltage profile V_m = 1.0 everywhere."""
        np.testing.assert_array_equal(dc14['vm'], 1.0)

    def test_voltage_angles_vs_matpower(self, dc14):
        np.testing.assert_allclose(dc14['va_deg'], DC_VA_DEG, atol=0.01)

    def test_branch_flows_vs_matpower(self, dc14):
        np.testing.assert_allclose(dc14['pf_from'], DC_PF_FROM, atol=0.01)

    def test_branch_flows_antisymmetric(self, dc14):
        """DC: lossless → Pf_to = -Pf_from exactly."""
        np.testing.assert_allclose(dc14['pf_to'], -dc14['pf_from'], atol=1e-10)

    def test_zero_losses(self, dc14):
        """DC PF is lossless: P_loss = 0 by construction."""
        np.testing.assert_allclose(dc14['p_loss'], 0.0, atol=1e-10)

    def test_total_gen_equals_load(self, dc14):
        """DC: no losses → ΣPg = ΣPd = 259 MW."""
        np.testing.assert_allclose(np.sum(dc14['p_gen']), AC_TOTAL_LOAD, atol=0.01)

    def test_slack_generation(self, dc14):
        """Slack bus picks up the residual: Pg_slack = Pd_total - ΣPg(non-slack)."""
        np.testing.assert_allclose(dc14['p_gen'][0], 219.0, atol=0.01)

    def test_non_slack_gen_unchanged(self, dc14):
        """Non-slack generators keep their scheduled dispatch (Pg from case data)."""
        np.testing.assert_allclose(dc14['p_gen'][1], 40.0, atol=0.01)
        np.testing.assert_allclose(dc14['p_gen'][2:], 0.0, atol=0.01)

    def test_angle_at_slack_is_zero(self, dc14):
        """Bus 1 (slack) angle = 0° by convention."""
        np.testing.assert_allclose(dc14['va_deg'][0], 0.0, atol=1e-10)

    def test_dc_flow_via_ptdf(self, dc14, ptdf14, case14):
        """Cross-check: PTDF @ P_net should equal the DCPF branch flow."""
        n_bus = len(case14.nodes)
        gen_bus_idx = case14.units['bus_id'].values.astype(int) - 1
        Pg_bus = np.zeros(n_bus)
        np.add.at(Pg_bus, gen_bus_idx, dc14['p_gen'])
        Pd = case14.nodes['Pd'].values.astype(float)
        P_net = Pg_bus - Pd  # net injection per bus
        flow_ptdf = ptdf14['PTDF'] @ P_net
        np.testing.assert_allclose(flow_ptdf, dc14['pf_from'], atol=0.01)


# =====================================================================
# AC Power Flow vs MATPOWER — Case14  (Newton-Raphson)
# =====================================================================

class TestACPFCase14:
    """AC NR: S = V × I*, iterative Jacobian solve.

    Properties:
      - Converges in ≤ 5 iterations for well-conditioned 14-bus case
      - Voltage magnitudes at PV buses match generator setpoints
      - Losses > 0 (resistive lines)
      - Power balance: Pg_total = Pd_total + P_loss_total
    """

    def test_converged(self, ac14):
        assert ac14['converged']

    def test_converged_fast(self, ac14):
        assert ac14['iterations'] <= 5, f"took {ac14['iterations']} iterations"

    def test_voltage_magnitudes_vs_matpower(self, ac14):
        np.testing.assert_allclose(ac14['vm'], AC_VM, atol=1e-3)

    def test_voltage_angles_vs_matpower(self, ac14):
        np.testing.assert_allclose(ac14['va_deg'], AC_VA_DEG, atol=0.01)

    def test_gen_p_vs_matpower(self, ac14):
        np.testing.assert_allclose(ac14['p_gen'], AC_PGEN, atol=0.05)

    def test_gen_q_vs_matpower(self, ac14):
        np.testing.assert_allclose(ac14['q_gen'], AC_QGEN, atol=0.05)

    def test_branch_pf_from_vs_matpower(self, ac14):
        np.testing.assert_allclose(ac14['pf_from'], AC_PF_FROM, atol=0.05)

    def test_branch_loss_per_line_vs_matpower(self, ac14):
        np.testing.assert_allclose(ac14['p_loss'], AC_P_LOSS, atol=0.005)

    def test_total_p_loss_vs_matpower(self, ac14):
        np.testing.assert_allclose(
            np.sum(ac14['p_loss']), AC_TOTAL_P_LOSS, atol=0.01
        )

    def test_power_balance(self, ac14):
        """Pg_total - Pd_total = P_loss_total."""
        total_gen = np.sum(ac14['p_gen'])
        total_loss = np.sum(ac14['p_loss'])
        np.testing.assert_allclose(total_gen - AC_TOTAL_LOAD, total_loss, atol=0.1)

    def test_pv_bus_voltages_at_setpoint(self, ac14, case14):
        """PV buses: |V| should match the generator voltage setpoint (Vg)."""
        gen_bus_idx = case14.units['bus_id'].values.astype(int) - 1
        gen_vg = case14.units['Vg'].values.astype(float)
        for i, (bus, vsp) in enumerate(zip(gen_bus_idx, gen_vg)):
            np.testing.assert_allclose(
                ac14['vm'][bus], vsp, atol=1e-3,
                err_msg=f"Gen at bus {bus+1}: Vm={ac14['vm'][bus]:.4f}, Vg={vsp:.4f}"
            )

    def test_losses_positive(self, ac14):
        """All branch losses must be ≥ 0 (resistive lines dissipate energy)."""
        assert np.all(ac14['p_loss'] >= -1e-10)

    def test_voltage_bounded(self, ac14):
        """All bus voltages within [0.9, 1.1] p.u. (normal operating range)."""
        assert np.all(ac14['vm'] >= 0.9)
        assert np.all(ac14['vm'] <= 1.1)


# =====================================================================
# DC vs AC consistency — Case14
# =====================================================================

class TestDCvsACConsistency:
    """DC is a linear approximation of AC; compare their outputs.

    DC approximation is valid when:
      - V ≈ 1.0 p.u. (flat profile)
      - θ_ij is small (sin θ ≈ θ, cos θ ≈ 1)
      - R << X (lossless approximation)

    For Case14 the DC approximation is reasonable:
    angles are within ~17°, voltages within [1.0, 1.09] p.u.
    """

    def test_slack_gen_dc_leq_ac(self, dc14, ac14):
        """DC slack gen ≤ AC slack gen because DC ignores losses."""
        assert dc14['p_gen'][0] <= ac14['p_gen'][0] + 0.01

    def test_angles_similar(self, dc14, ac14):
        """DC and AC angles should agree within ~1° for well-conditioned case."""
        np.testing.assert_allclose(dc14['va_deg'], ac14['va_deg'], atol=1.5)

    def test_branch_flows_similar(self, dc14, ac14):
        """DC and AC branch P flows should agree within ~10 MW
        (high tolerance because DC ignores losses and reactive power)."""
        np.testing.assert_allclose(dc14['pf_from'], ac14['pf_from'], atol=10.0)

    def test_dc_gen_total_leq_ac_gen_total(self, dc14, ac14):
        """DC total gen < AC total gen (DC is lossless)."""
        assert np.sum(dc14['p_gen']) <= np.sum(ac14['p_gen']) + 0.01


# =====================================================================
# Case5 smoke tests — minimal system
# =====================================================================

class TestCase5Smoke:
    """Minimal IEEE 5-bus case: PTDF and Ybus only.

    Case5 is an OPF case (no Pg/Qg columns), so run_acpf/run_dcpf
    are not directly applicable. Test structural properties instead.
    """

    def test_ptdf_shape_case5(self, case5):
        cache = build_dcpf_ptdf(case5)
        n_bus = len(case5.nodes)
        n_line = len(case5.lines)
        assert cache['PTDF'].shape == (n_line, n_bus)

    def test_ptdf_slack_column_zero(self, case5):
        cache = build_dcpf_ptdf(case5)
        np.testing.assert_allclose(cache['PTDF'][:, 0], 0.0, atol=1e-10)

    def test_ybus_shape_case5(self, case5):
        Ybus = TestYbusProperties._build_ybus_from_case(case5)
        assert Ybus.shape == (len(case5.nodes), len(case5.nodes))


# =====================================================================
# Determinism / reproducibility
# =====================================================================

class TestDeterminism:
    """Solver outputs must be deterministic given same case input."""

    def test_dcpf_deterministic(self, case14):
        r1 = run_dcpf(case14)
        r2 = run_dcpf(case14)
        np.testing.assert_array_equal(r1['va_deg'], r2['va_deg'])
        np.testing.assert_array_equal(r1['pf_from'], r2['pf_from'])

    def test_acpf_deterministic(self, case14):
        r1 = run_acpf(case14)
        r2 = run_acpf(case14)
        np.testing.assert_allclose(r1['vm'], r2['vm'], atol=1e-12)
        np.testing.assert_allclose(r1['va_deg'], r2['va_deg'], atol=1e-12)


# =====================================================================
# RL robustness — singular Jacobian must not crash the solver
# =====================================================================

class TestRLRobustness:
    """Verify that the solver degrades gracefully under extreme inputs."""

    def test_acpf_singular_jacobian_returns_converged_false(self, case14, monkeypatch):
        """Simulate a singular Jacobian (as can occur with extreme RL actions).

        The solver must return converged=False, not raise LinAlgError.
        """
        import powerzoo.envs.grid.cal_pf_trans as _mod

        original_build = _mod._build_jacobian

        def _singular_jacobian(*args, **kwargs):
            J = original_build(*args, **kwargs)
            # Zero out a row/column to make J singular
            J[0, :] = 0.0
            J[:, 0] = 0.0
            return J

        monkeypatch.setattr(_mod, '_build_jacobian', _singular_jacobian)

        result = run_acpf(case14)
        assert result['converged'] is False

    def test_acpf_diverged_returns_zero_placeholders(self, case14, monkeypatch):
        """When the solver diverges, all physical arrays must be zero-filled.

        Ensures that downstream RL code never processes physically meaningless
        intermediate NR states as if they were valid solutions.
        """
        import powerzoo.envs.grid.cal_pf_trans as _mod

        def _singular_jacobian(*args, **kwargs):
            J = _mod._build_jacobian.__wrapped__(*args, **kwargs) \
                if hasattr(_mod._build_jacobian, '__wrapped__') \
                else _mod._build_jacobian(*args, **kwargs)
            J[0, :] = 0.0
            J[:, 0] = 0.0
            return J

        original_build = _mod._build_jacobian

        def _make_singular(*args, **kwargs):
            J = original_build(*args, **kwargs)
            J[0, :] = 0.0
            J[:, 0] = 0.0
            return J

        monkeypatch.setattr(_mod, '_build_jacobian', _make_singular)

        result = run_acpf(case14)
        assert result['converged'] is False

        n_bus = len(case14.nodes)
        n_br  = len(case14.lines)
        n_gen = len(case14.units)

        # vm must be the flat-start placeholder (all ones), not an intermediate state
        np.testing.assert_array_equal(result['vm'], np.ones(n_bus))
        np.testing.assert_array_equal(result['va'], np.zeros(n_bus))

        for key in ('p_gen', 'q_gen'):
            assert result[key].shape == (n_gen,)
            np.testing.assert_array_equal(result[key], 0.0)

        for key in ('pf_from', 'qf_from', 'pf_to', 'qf_to',
                    'p_loss', 'q_loss', 'q_loss_net'):
            assert result[key].shape == (n_br,)
            np.testing.assert_array_equal(result[key], 0.0)

    def test_acpf_diverged_all_keys_present(self, case14, monkeypatch):
        """Diverged result dict must contain every expected key so callers
        can safely index it regardless of convergence status."""
        import powerzoo.envs.grid.cal_pf_trans as _mod

        original_build = _mod._build_jacobian

        def _make_singular(*args, **kwargs):
            J = original_build(*args, **kwargs)
            J[0, :] = 0.0
            J[:, 0] = 0.0
            return J

        monkeypatch.setattr(_mod, '_build_jacobian', _make_singular)

        result = run_acpf(case14)
        expected_keys = {
            'vm', 'va', 'va_deg', 'p_gen', 'q_gen',
            'pf_from', 'qf_from', 'pf_to', 'qf_to',
            'p_loss', 'q_loss', 'q_loss_net',
            'converged', 'iterations', 'Ybus',
        }
        assert expected_keys.issubset(result.keys())


# =====================================================================
# _distribute_gen_power: multi-gen bus splitting
# =====================================================================

class TestDistributeGenPower:
    """Unit-test the "base + deviation" splitting logic for multi-gen buses.

    Core invariants:
        sum(p_gen) == P_total  for all active-power device combinations.
        sum(q_gen) == Q_total  for all reactive-power device combinations.
    P and Q use independent weight vectors so that pure reactive devices
    (SVG, Pg=0, Qg≠0) absorb reactive deviations correctly.
    """

    @staticmethod
    def _call_distribute(pg_sched, p_total_bus, qg_sched=None, q_total_bus=0.0):
        """Call _distribute_gen_power with a 1-bus, N-generator setup.

        Args:
            pg_sched:      list of length N — scheduled P (MW).
            p_total_bus:   float — solved bus-level net P (MW).
            qg_sched:      optional list of length N — scheduled Q (MVAr).
            q_total_bus:   float — solved bus-level net Q (MVAr).
        """
        import pandas as pd
        from powerzoo.envs.grid.cal_pf_trans import _distribute_gen_power

        n = len(pg_sched)
        units = pd.DataFrame({'bus_id': [1] * n,
                              'Pg': pg_sched,
                              'Qg': qg_sched if qg_sched is not None else [0.0] * n})
        gen_bus_idx = np.zeros(n, dtype=int)
        S_bus = np.array([p_total_bus + q_total_bus * 1j])
        Pd_mw = np.array([0.0])
        Qd_mw = np.array([0.0])
        return _distribute_gen_power(S_bus, Pd_mw, Qd_mw, units, gen_bus_idx, n)

    def test_same_sign_gen_proportional(self):
        """Two generators, +60 and +40 MW scheduled, bus P_total = 100 MW.

        abs-sum = 100 → frac = [0.6, 0.4] → p_gen = [60, 40].
        """
        p_gen, _ = self._call_distribute([60.0, 40.0], p_total_bus=100.0)
        np.testing.assert_allclose(p_gen, [60.0, 40.0], atol=1e-9)

    def test_mixed_gen_storage_preserves_direction(self):
        """Gen (+10 MW) + storage (−10 MW) scheduled, bus P_total = 5 MW.

        "Base + deviation" logic:
            weights = [0.5, 0.5]   (from |Pg_sched| / abs_sum = 10/20, 10/20)
            P_delta = 5 − 0 = 5
            p_gen   = [10 + 2.5, −10 + 2.5] = [+12.5, −7.5]

        Crucially, sum(p_gen) = 5 = P_total  ← power is conserved.
        Storage remains negative (charging); the deviation is shared equally
        between the two equally-sized devices.
        """
        p_gen, _ = self._call_distribute([10.0, -10.0], p_total_bus=5.0)
        np.testing.assert_allclose(p_gen, [12.5, -7.5], atol=1e-9)

    def test_all_zero_scheduled_equal_split(self):
        """When Σ|Pg_sched| ≈ 0, fall back to equal split."""
        p_gen, _ = self._call_distribute([0.0, 0.0], p_total_bus=10.0)
        # Equal weights [0.5, 0.5]; Pg_sum = 0; P_delta = 10 → p_gen = [5, 5]
        np.testing.assert_allclose(p_gen, [5.0, 5.0], atol=1e-9)

    def test_svg_device_absorbs_q_deviation(self):
        """Gen (Pg=10, Qg=3) + SVG (Pg=0, Qg=5) on same bus, Q_total = 9 MVAr.

        With P-only weights, SVG gets weight_q = 0 (Pg=0) and can never absorb
        Q deviation — physically wrong for a reactive compensation device.

        With independent Q weights (|Qg|-based):
            weights_q = [3/8, 5/8]
            Q_delta   = 9 − 8 = 1
            q_gen     = [3 + 0.375,  5 + 0.625] = [3.375, 5.625],  sum = 9  ✓

        The SVG now absorbs more of the deviation than the generator (higher
        reactive capacity), which is the expected physical behaviour.
        """
        _, q_gen = self._call_distribute(
            pg_sched=[10.0, 0.0],
            qg_sched=[3.0, 5.0],
            p_total_bus=10.0,
            q_total_bus=9.0,
        )
        np.testing.assert_allclose(q_gen, [3.375, 5.625], atol=1e-9)
        np.testing.assert_allclose(q_gen.sum(), 9.0, atol=1e-9)

    def test_power_conservation_invariant(self):
        """sum(p_gen) and sum(q_gen) must equal their respective totals.

        Verifies the "base + deviation" invariant for both P and Q across
        mixed device mixes.
        """
        test_cases = [
            # (pg_sched, qg_sched, P_total, Q_total)
            ([60.0,  40.0],  [20.0, 10.0],   100.0, 30.0),   # same-sign, delta = 0
            ([10.0, -10.0],  [ 3.0,  5.0],     5.0,  9.0),   # gen + storage, SVG-like Q
            ([10.0, -10.0],  [ 3.0,  5.0],     0.0,  0.0),   # zero totals
            ([ 5.0,  -3.0, 8.0], [2.0, 1.0, 3.0], 15.0, 7.0),  # three devices
            ([0.0,    0.0],  [ 0.0,  0.0],    10.0,  4.0),   # fallback equal-split
        ]
        for pg_sched, qg_sched, p_total, q_total in test_cases:
            p_gen, q_gen = self._call_distribute(
                pg_sched, p_total_bus=p_total,
                qg_sched=qg_sched, q_total_bus=q_total,
            )
            np.testing.assert_allclose(
                p_gen.sum(), p_total, atol=1e-9,
                err_msg=f"P not conserved: pg={pg_sched}, P_total={p_total}"
            )
            np.testing.assert_allclose(
                q_gen.sum(), q_total, atol=1e-9,
                err_msg=f"Q not conserved: qg={qg_sched}, Q_total={q_total}"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
