"""Solver-level tests for the three-phase BIBC/BCBV power flow (cal_pf_dist_3phase).

Tests cover:
  - ThreePhaseTopology construction (BIBC, BCBV, DLF matrices)
  - Case123 convergence and iteration bounds
  - Slack bus 3-phase voltage (V_A = V_B = V_C = 1.05, angles 0°/-120°/+120°)
  - Passive feeder invariant: all per-phase voltages ≤ V_ref
  - Voltage Unbalance Factor (VUF) < 2% per IEEE Std 1159
  - Per-phase P_loss summation = total loss
  - Non-negative losses on each branch
  - Regression vs stored reference (tests/ref_data/*.csv, *.json)

Domain knowledge:
  - BIBC maps bus injections to branch currents (topology matrix)
  - BCBV maps branch currents to bus voltage drops (impedance matrix)
  - DLF = BCBV @ BIBC: combined sensitivity (pre-computed for speed)
  - IEEE 123-bus system: 3-phase unbalanced distribution, ref 4.16 kV, 10 MVA base
  - Fortescue VUF = |V_neg| / |V_pos| × 100%
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from powerzoo.envs.grid.cal_pf_dist_3phase import (
    ThreePhaseTopology,
    build_3phase_topology,
    run_3phase_bfs_power_flow,
    calculate_3phase_losses,
    get_phase_results,
    reshape_3phase_to_per_node,
)

# ── Reference paths ──────────────────────────────────────────────────
REF_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'ref_data')
REF_NODES_CSV = os.path.join(REF_DIR, 'case123_nodes.csv')
REF_LINES_CSV = os.path.join(REF_DIR, 'case123_lines.csv')
REF_SUMMARY_JSON = os.path.join(REF_DIR, 'case123_summary.json')
HAS_REFERENCE = os.path.exists(REF_SUMMARY_JSON)

# ── Physical constants ───────────────────────────────────────────────
V_REF_MAG = 1.05
V_REF_ANGLES_DEG = [0.0, -120.0, 120.0]
VUF_LIMIT = 2.0  # IEEE Std 1159


def _make_one_line_topology(z_3ph: np.ndarray | None = None,
                            v_ref_mag: float = 1.0) -> ThreePhaseTopology:
    """Build a tiny 2-bus / 1-line three-phase topology for solver unit tests."""
    if z_3ph is None:
        z_3ph = np.array([[
            [0.08 + 0.15j, 0.01 + 0.03j, 0.00 + 0.00j],
            [0.01 + 0.03j, 0.09 + 0.18j, 0.02 + 0.01j],
            [0.00 + 0.00j, 0.02 + 0.01j, 0.07 + 0.14j],
        ]], dtype=np.complex128)
    return build_3phase_topology(
        n_nodes=2,
        from_nodes=np.array([0], dtype=int),
        to_nodes=np.array([1], dtype=int),
        Z_3ph_pu=z_3ph,
        ref_bus=0,
        v_ref_mag=v_ref_mag,
    )


def _make_two_branch_topology() -> ThreePhaseTopology:
    """Build a tiny 3-bus radial topology with explicit physical node IDs."""
    z_3ph = np.array([
        [
            [0.10 + 0.20j, 0.00 + 0.00j, 0.00 + 0.00j],
            [0.00 + 0.00j, 0.11 + 0.22j, 0.00 + 0.00j],
            [0.00 + 0.00j, 0.00 + 0.00j, 0.12 + 0.24j],
        ],
        [
            [0.00 + 0.00j, 0.00 + 0.00j, 0.00 + 0.00j],
            [0.00 + 0.00j, 0.00 + 0.00j, 0.00 + 0.00j],
            [0.00 + 0.00j, 0.00 + 0.00j, 0.13 + 0.26j],
        ],
    ], dtype=np.complex128)
    return build_3phase_topology(
        n_nodes=3,
        from_nodes=np.array([0, 1], dtype=int),
        to_nodes=np.array([1, 2], dtype=int),
        Z_3ph_pu=z_3ph,
        ref_bus=0,
        v_ref_mag=1.0,
        node_ids=np.array([101, 205, 309], dtype=int),
    )

# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture(scope="module")
def env3ph():
    """DistGrid3PhaseEnv for Case123."""
    from powerzoo.envs.grid.dist_3phase import DistGrid3PhaseEnv
    return DistGrid3PhaseEnv()


@pytest.fixture(scope="module")
def pf(env3ph):
    """Run 3-phase PF once and share results."""
    nodes, lines = env3ph.cal_pf(df=True)
    return env3ph, nodes, lines


@pytest.fixture(scope="module")
def topo3ph(env3ph):
    return env3ph.topo3ph


# =====================================================================
# Topology construction
# =====================================================================

class TestThreePhaseTopology:
    """BIBC/BCBV matrix structural properties."""

    def test_topology_type(self, topo3ph):
        assert isinstance(topo3ph, ThreePhaseTopology)

    def test_tree_edge_count(self, topo3ph):
        """Radial network: n_lines = n_nodes - 1."""
        assert topo3ph.n_lines == topo3ph.n_nodes - 1

    def test_Z_3ph_shape(self, topo3ph):
        """Each line has a 3×3 complex impedance matrix."""
        assert topo3ph.Z_3ph.shape == (topo3ph.n_lines, 3, 3)

    def test_Z_3ph_dtype(self, topo3ph):
        assert np.iscomplexobj(topo3ph.Z_3ph)

    def test_BIBC_shape(self, topo3ph):
        """BIBC: (3*n_lines) × (3*n_lines)."""
        n3 = 3 * topo3ph.n_lines
        assert topo3ph.BIBC.shape == (n3, n3)

    def test_BCBV_shape(self, topo3ph):
        n3 = 3 * topo3ph.n_lines
        assert topo3ph.BCBV.shape == (n3, n3)

    def test_DLF_equals_BCBV_times_BIBC(self, topo3ph):
        """DLF must equal BCBV @ BIBC (precomputed product)."""
        expected = topo3ph.BCBV @ topo3ph.BIBC
        np.testing.assert_allclose(topo3ph.DLF, expected, atol=1e-12)

    def test_V_ref_3ph_magnitudes(self, topo3ph):
        """Reference phasors all have magnitude V_REF_MAG."""
        np.testing.assert_allclose(
            np.abs(topo3ph.V_ref_3ph), V_REF_MAG, atol=1e-10
        )

    def test_V_ref_3ph_angles(self, topo3ph):
        """Reference phasor angles: 0°, -120°, +120°."""
        angles = np.angle(topo3ph.V_ref_3ph, deg=True)
        np.testing.assert_allclose(angles, V_REF_ANGLES_DEG, atol=1e-8)

    def test_Z_3ph_positive_diagonal_resistance(self, topo3ph):
        """Self-impedance diagonal: at least some phases have R > 0 per line.
        (Not all phases may be connected on every line in Case123.)"""
        R_diag = np.real(np.diagonal(topo3ph.Z_3ph, axis1=1, axis2=2))
        assert np.all(R_diag >= 0), "Diagonal resistances must be non-negative"
        # Each line must have at least one connected phase with R > 0
        assert np.all(R_diag.max(axis=1) > 0), \
            "Every line must have at least one phase with positive resistance"

    def test_solver_vector_mapping_metadata(self):
        topo = _make_two_branch_topology()
        np.testing.assert_array_equal(topo.non_ref_node_ids, np.array([205, 309]))
        assert topo.ref_node_id == 101
        assert topo.node_id_to_matrix_index[205] == 0
        assert topo.node_id_to_matrix_index[309] == 1
        assert topo.vector_layout == 'node_major_abc'
        assert topo.phase_order == ('A', 'B', 'C')

    def test_precomputed_sending_bus_indices(self):
        topo = _make_two_branch_topology()
        np.testing.assert_array_equal(topo.sending_bus_is_ref, np.array([True, False]))
        np.testing.assert_array_equal(topo.sending_bus_gather_indices, np.array([0, 0]))

    def test_non_ref_phase_mask_follows_parent_branch(self):
        topo = _make_two_branch_topology()
        expected = np.array([
            [True, True, True],
            [False, False, True],
        ])
        np.testing.assert_array_equal(topo.non_ref_phase_mask, expected)
        np.testing.assert_array_equal(
            topo.non_ref_phase_mask_flat,
            np.array([True, True, True, False, False, True]),
        )


# =====================================================================
# Convergence
# =====================================================================

class TestConvergence:

    def test_converges(self, pf):
        env, nodes, lines = pf
        assert env._converged, "3-phase BFS should converge for Case123"

    def test_iterations_reasonable(self, pf):
        env, nodes, lines = pf
        assert 1 <= env._iterations <= 15, \
            f"iterations={env._iterations}, expected 1–15"

    def test_max_iter_exhaustion_returns_diagnostic_last_iter(self):
        topo = _make_one_line_topology(v_ref_mag=1.0)
        p_load = np.array([0.18, 0.12, 0.09], dtype=float)
        q_load = np.array([0.06, 0.04, 0.03], dtype=float)

        result = run_3phase_bfs_power_flow(
            topo3ph=topo,
            P_3ph_pu=p_load,
            Q_3ph_pu=q_load,
            v_ref_mag=1.0,
            max_iter=1,
            tol=0.0,
        )

        assert result['converged'] is False
        assert result['iterations'] == 1
        assert result['convergence_status'] == 'max_iter_exhausted'
        assert 'diagnostics only' in result['convergence_message']
        assert result['max_voltage_update_pu'] >= 0.0
        assert result['V'].shape == (3,)
        assert result['I_branch'].shape == (3,)
        assert np.all(np.isfinite(result['V_mag']))
        assert np.all(np.isfinite(result['I_branch']))

    def test_result_exposes_convergence_metadata_for_flat_start_skip(self):
        topo = _make_one_line_topology(v_ref_mag=1.0)
        result = run_3phase_bfs_power_flow(
            topo3ph=topo,
            P_3ph_pu=np.zeros(3, dtype=float),
            Q_3ph_pu=np.zeros(3, dtype=float),
            v_ref_mag=1.0,
            max_iter=0,
            tol=1e-9,
        )

        assert result['converged'] is False
        assert result['iterations'] == 0
        assert result['convergence_status'] == 'not_started'
        assert result['vector_layout'] == 'node_major_abc'
        assert result['phase_order'] == ('A', 'B', 'C')

    def test_missing_phase_inputs_are_clamped_to_zero(self):
        z_3ph = np.array([[
            [0.12 + 0.25j, 0.00 + 0.00j, 0.00 + 0.00j],
            [0.00 + 0.00j, 0.00 + 0.00j, 0.00 + 0.00j],
            [0.00 + 0.00j, 0.00 + 0.00j, 0.00 + 0.00j],
        ]], dtype=np.complex128)
        topo = _make_one_line_topology(z_3ph=z_3ph, v_ref_mag=1.0)

        result = run_3phase_bfs_power_flow(
            topo3ph=topo,
            P_3ph_pu=np.array([0.20, 0.30, 0.40], dtype=float),
            Q_3ph_pu=np.array([0.05, 0.06, 0.07], dtype=float),
            v_ref_mag=1.0,
        )

        i_branch = result['I_branch'].reshape(1, 3)[0]
        p_branch = result['P_branch'].reshape(1, 3)[0]
        q_branch = result['Q_branch'].reshape(1, 3)[0]

        assert result['inactive_phase_inputs_clamped'] is True
        assert result['inactive_phase_input_count'] == 2
        np.testing.assert_allclose(i_branch[1:], 0.0, atol=1e-12)
        np.testing.assert_allclose(p_branch[1:], 0.0, atol=1e-12)
        np.testing.assert_allclose(q_branch[1:], 0.0, atol=1e-12)


# =====================================================================
# Slack bus
# =====================================================================

class TestSlackBus:
    """Reference bus must hold its Set voltage at all three phases."""

    def test_per_phase_magnitude(self, pf):
        _, nodes, _ = pf
        for ph in 'ABC':
            v = float(nodes.iloc[0][f'V_{ph}'])
            np.testing.assert_allclose(v, V_REF_MAG, atol=1e-10,
                                       err_msg=f'Phase {ph}')

    def test_per_phase_angle(self, pf):
        _, nodes, _ = pf
        for ph, expected in zip('ABC', V_REF_ANGLES_DEG):
            ang = float(nodes.iloc[0][f'angle_{ph}'])
            np.testing.assert_allclose(ang, expected, atol=1e-10,
                                       err_msg=f'Phase {ph}')

    def test_average_voltage_equals_vref(self, pf):
        _, nodes, _ = pf
        np.testing.assert_allclose(nodes['v_mag'].iloc[0], V_REF_MAG, atol=1e-10)


# =====================================================================
# Bus voltages
# =====================================================================

class TestBusVoltages:

    def test_per_phase_in_safe_range(self, pf):
        """All per-phase voltages within [0.80, 1.20] p.u."""
        _, nodes, _ = pf
        for ph in 'ABC':
            v = nodes[f'V_{ph}'].dropna().values
            assert np.all(v >= 0.80), f'Phase {ph} below 0.80'
            assert np.all(v <= 1.20), f'Phase {ph} above 1.20'

    def test_passive_feeder_constraint(self, pf):
        """Passive network: non-slack buses ≤ V_ref."""
        _, nodes, _ = pf
        for ph in 'ABC':
            v = nodes[f'V_{ph}'].iloc[1:].dropna().values
            assert np.all(v <= V_REF_MAG + 1e-6), \
                f'Phase {ph} exceeds V_ref at slack'

    def test_average_voltage_range(self, pf):
        _, nodes, _ = pf
        assert np.all(nodes['v_mag'].values >= 0.80)
        assert np.all(nodes['v_mag'].values <= 1.20)


# =====================================================================
# Voltage Unbalance Factor
# =====================================================================

class TestVoltageUnbalance:
    """IEEE Std 1159: VUF = |V_neg|/|V_pos| × 100% < 2% in normal ops."""

    def test_max_vuf_below_ieee(self, pf):
        env, nodes, _ = pf
        _, max_vuf = env.calculate_vuf(nodes)
        assert max_vuf < VUF_LIMIT, f"Max VUF {max_vuf:.4f}% exceeds {VUF_LIMIT}%"

    def test_vuf_array_length(self, pf):
        env, nodes, _ = pf
        vuf_arr, _ = env.calculate_vuf(nodes)
        assert len(vuf_arr) == env.n_nodes

    def test_all_vuf_non_negative(self, pf):
        env, nodes, _ = pf
        vuf_arr, _ = env.calculate_vuf(nodes)
        assert np.all(vuf_arr >= 0)

    def test_slack_bus_vuf_near_zero(self, pf):
        """Balanced reference bus: VUF ≈ 0."""
        env, nodes, _ = pf
        vuf_arr, _ = env.calculate_vuf(nodes)
        assert vuf_arr[0] < 0.01, f"Slack bus VUF={vuf_arr[0]:.6f}%"


# =====================================================================
# Line losses
# =====================================================================

class TestLineLosses:

    def test_total_p_loss_positive(self, pf):
        _, _, lines = pf
        assert lines['p_loss_MW'].sum() > 0

    def test_total_q_loss_positive(self, pf):
        _, _, lines = pf
        assert lines['q_loss_MVAr'].sum() > 0

    def test_per_phase_p_loss_sum_equals_total(self, pf):
        _, _, lines = pf
        p_phases = sum(lines[f'p_loss_{ph}_MW'].sum() for ph in 'ABC')
        p_total = lines['p_loss_MW'].sum()
        np.testing.assert_allclose(p_phases, p_total, rtol=1e-6)

    def test_per_phase_q_loss_sum_equals_total(self, pf):
        _, _, lines = pf
        q_phases = sum(lines[f'q_loss_{ph}_MVAr'].sum() for ph in 'ABC')
        q_total = lines['q_loss_MVAr'].sum()
        np.testing.assert_allclose(q_phases, q_total, rtol=1e-6)

    def test_all_line_p_losses_non_negative(self, pf):
        _, _, lines = pf
        assert np.all(lines['p_loss_MW'].values >= -1e-8)

    def test_mutual_coupling_losses_use_full_impedance_matrix(self):
        z_3ph = np.array([[
            [0.12 + 0.25j, 0.03 + 0.04j, 0.01 + 0.02j],
            [0.03 + 0.04j, 0.11 + 0.21j, 0.02 + 0.05j],
            [0.01 + 0.02j, 0.02 + 0.05j, 0.10 + 0.20j],
        ]], dtype=np.complex128)
        topo = _make_one_line_topology(z_3ph=z_3ph)
        i_branch = np.array([1.0 + 0.4j, -0.6 + 1.1j, 0.8 - 0.2j], dtype=np.complex128)
        result = {'I_branch': i_branch}

        p_loss, q_loss = calculate_3phase_losses(topo, result)

        i_matrix = i_branch.reshape(1, 3)
        delta_v = np.einsum('bij,bj->bi', z_3ph, i_matrix, optimize=True)
        s_loss = delta_v * np.conj(i_matrix)
        expected_p = np.real(s_loss).reshape(-1)
        expected_q = np.imag(s_loss).reshape(-1)

        diag_only = np.real(np.diagonal(z_3ph, axis1=1, axis2=2)).reshape(-1) * np.abs(i_branch) ** 2

        np.testing.assert_allclose(p_loss, expected_p, atol=1e-12)
        np.testing.assert_allclose(q_loss, expected_q, atol=1e-12)
        assert not np.allclose(p_loss, diag_only)


# =====================================================================
# Utility functions
# =====================================================================

class TestUtilFunctions:
    """reshape & get_phase_results utilities."""

    def test_reshape_round_trip(self, topo3ph):
        flat = np.arange(3 * topo3ph.n_lines, dtype=float)
        mat = reshape_3phase_to_per_node(flat, topo3ph.n_lines)
        assert mat.shape == (topo3ph.n_lines, 3)
        np.testing.assert_array_equal(flat, mat.flatten())

    def test_get_phase_results_keys(self, pf):
        env, _, _ = pf
        # Run raw solver to get result dict
        P, Q = env._get_3phase_loads()
        result = run_3phase_bfs_power_flow(env.topo3ph, P, Q, V_REF_MAG)
        pr = get_phase_results(result, env.n_lines)
        expected_keys = {'V_A', 'V_B', 'V_C', 'angle_A', 'angle_B', 'angle_C',
                         'P_A', 'P_B', 'P_C', 'Q_A', 'Q_B', 'Q_C'}
        assert set(pr.keys()) == expected_keys

    def test_get_phase_results_shapes(self, pf):
        env, _, _ = pf
        P, Q = env._get_3phase_loads()
        result = run_3phase_bfs_power_flow(env.topo3ph, P, Q, V_REF_MAG)
        pr = get_phase_results(result, env.n_lines)
        for k, v in pr.items():
            assert v.shape == (env.n_lines,), f"{k} shape mismatch"


# =====================================================================
# Determinism
# =====================================================================

class TestDeterminism:

    def test_repeated_run_identical(self, env3ph):
        n1, l1 = env3ph.cal_pf(df=True)
        n2, l2 = env3ph.cal_pf(df=True)
        for ph in 'ABC':
            np.testing.assert_array_equal(
                n1[f'V_{ph}'].values, n2[f'V_{ph}'].values
            )
        np.testing.assert_array_equal(
            l1['p_loss_MW'].values, l2['p_loss_MW'].values
        )


# =====================================================================
# Regression (optional — requires ref_data)
# =====================================================================

@pytest.mark.skipif(not HAS_REFERENCE,
                    reason="Reference data not found; run draft/IEEE123_data_transform.py")
class TestVsReference:

    def test_per_phase_voltage_regression(self, pf):
        _, nodes, _ = pf
        ref = pd.read_csv(REF_NODES_CSV, index_col=0)
        for ph in 'ABC':
            col = f'V_{ph}'
            np.testing.assert_allclose(
                nodes[col].values, ref[col].values, atol=1e-6,
                err_msg=f'Phase {ph} voltage regression failed'
            )

    def test_p_loss_regression(self, pf):
        _, _, lines = pf
        with open(REF_SUMMARY_JSON) as f:
            ref = json.load(f)
        np.testing.assert_allclose(
            lines['p_loss_MW'].sum(), ref['p_loss_MW'], atol=1e-6
        )

    def test_q_loss_regression(self, pf):
        _, _, lines = pf
        with open(REF_SUMMARY_JSON) as f:
            ref = json.load(f)
        np.testing.assert_allclose(
            lines['q_loss_MVAr'].sum(), ref['q_loss_MVAr'], atol=1e-6
        )

    def test_max_vuf_regression(self, pf):
        env, nodes, _ = pf
        with open(REF_SUMMARY_JSON) as f:
            ref = json.load(f)
        _, max_vuf = env.calculate_vuf(nodes)
        np.testing.assert_allclose(max_vuf, ref['max_vuf_percent'], atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
