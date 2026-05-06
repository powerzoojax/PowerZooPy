"""Solver-level tests for distribution BFS power flow (cal_pf_dist).

Tests cover:
  - RadialTopology construction (tree validation, parent/child sets)
  - Backward/forward sweep convergence and iteration counts
  - Case33bw vs Baran & Wu (1989) literature reference
  - Physical invariants: voltage drop along feeder, I²R losses ≥ 0,
    power conservation (feeder head = load + loss)
  - Optional MATPOWER NR cross-validation (CSV reference data)

Domain knowledge:
  - Radial distribution: tree topology with single slack bus at root
  - BFS: backward sweep accumulates loads → branch currents;
          forward sweep computes voltage drops from root
  - Passive network: no bus can exceed slack voltage (no DG in base case)
  - Main feeder (bus 0→17): voltage drops monotonically as impedance accumulates
  - Baran & Wu 1989 benchmarks: P_loss≈202.67 kW, Q_loss≈135.14 kVAr,
    V_min≈0.9131 p.u. at bus 18 (0-indexed: 17)
"""

import os

import numpy as np
import pandas as pd
import pytest

from powerzoo.envs.grid.cal_pf_dist import (
    RadialTopology,
    build_radial_topology,
    run_bfs_power_flow,
    calculate_line_losses,
)

# ── Reference file paths ──────────────────────────────────────────────
REF_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'ref_data')
MATPOWER_BUS_CSV = os.path.join(REF_DIR, 'case33bw_bus.csv')
MATPOWER_BR_CSV = os.path.join(REF_DIR, 'case33bw_branch.csv')
HAS_MATPOWER = os.path.exists(MATPOWER_BUS_CSV)

# ── Literature constants (Baran & Wu, 1989) ──────────────────────────
LIT_P_LOSS_MW = 0.20267    # 202.67 kW
LIT_Q_LOSS_MVAR = 0.13514  # 135.14 kVAr
LIT_V_MIN_PU = 0.9131      # bus 18 (0-indexed: 17)
LIT_V_MIN_BUS = 17         # 0-based index


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture(scope="module")
def dist_env():
    """Full DistGridEnv for Case33bw — provides topology + power flow."""
    from powerzoo.envs.grid.dist import DistGridEnv
    return DistGridEnv()


@pytest.fixture(scope="module")
def pf_result(dist_env):
    """Run BFS power flow once and share across module."""
    nodes, lines = dist_env.cal_pf(df=True)
    return dist_env, nodes, lines


@pytest.fixture(scope="module")
def bfs_result(dist_env):
    """Direct solver result for the default Case33bw net load."""
    p_load_pu = dist_env._get_node_loads_p() / dist_env.baseMVA
    q_load_pu = dist_env._get_node_loads_q() / dist_env.baseMVA
    return run_bfs_power_flow(
        topo=dist_env.topo,
        p_load_pu=p_load_pu,
        q_load_pu=q_load_pu,
        v_slack=dist_env.v_slack,
        slack_bus_id=dist_env.slack_bus_id,
        max_iter=dist_env.max_iter,
        tol=dist_env.tol,
    )


@pytest.fixture(scope="module")
def topo_33(dist_env):
    """RadialTopology from 33-bus case."""
    return dist_env.topo


# =====================================================================
# Topology construction
# =====================================================================

class TestRadialTopology:
    """Tree structure for radial distribution feeders.

    Physical properties:
      - n_lines = n_nodes - 1 (tree connectivity, no loops)
      - Every non-root node has exactly one parent
      - DFS ordering: parent index < child index (topological sort)
      - Root (slack bus) has no parent; is starting point for forward sweep
    """

    def test_topology_type(self, topo_33):
        assert isinstance(topo_33, RadialTopology)

    def test_tree_edge_count(self, topo_33):
        """Radial network: #edges = #nodes - 1."""
        assert topo_33.n_lines == topo_33.n_nodes - 1

    def test_from_to_arrays_length(self, topo_33):
        assert len(topo_33.from_nodes) == topo_33.n_lines
        assert len(topo_33.to_nodes) == topo_33.n_lines

    def test_ref_bus_stored(self, topo_33, dist_env):
        """Root node is bus 0 (BFS starts from slack bus)."""
        assert dist_env.slack_bus_id == 0

    def test_impedance_arrays_shape(self, topo_33):
        """Per-branch impedance: r + jx in p.u."""
        assert len(topo_33.r_pu) == topo_33.n_lines
        assert len(topo_33.x_pu) == topo_33.n_lines

    def test_positive_impedance(self, topo_33):
        """All branch resistance and reactance must be positive (physical)."""
        assert np.all(topo_33.r_pu > 0)
        assert np.all(topo_33.x_pu > 0)

    def test_loop_warning_mentions_first_visit_pruning(self):
        """Non-radial input should warn that BFS keeps a first-visit tree."""
        with pytest.warns(UserWarning, match="first-visit spanning tree"):
            build_radial_topology(
                n_nodes=3,
                from_nodes=np.array([0, 1, 0]),
                to_nodes=np.array([1, 2, 2]),
                r_pu=np.array([0.01, 0.01, 0.01]),
                x_pu=np.array([0.01, 0.01, 0.01]),
                slack_bus_id=0,
            )

    def test_loop_can_be_rejected_explicitly(self):
        """Callers can fail fast instead of accepting BFS mesh pruning."""
        with pytest.raises(ValueError, match="allow_mesh_pruning=True"):
            build_radial_topology(
                n_nodes=3,
                from_nodes=np.array([0, 1, 0]),
                to_nodes=np.array([1, 2, 2]),
                r_pu=np.array([0.01, 0.01, 0.01]),
                x_pu=np.array([0.01, 0.01, 0.01]),
                slack_bus_id=0,
                allow_mesh_pruning=False,
            )


# =====================================================================
# BFS convergence
# =====================================================================

class TestBFSConvergence:
    """Forward-backward sweep convergence properties."""

    def test_converges(self, pf_result):
        env, nodes, lines = pf_result
        assert env._converged, "BFS should converge for Case33bw with base-case load"

    def test_iterations_reasonable(self, pf_result):
        """Well-connected radial: expect 2–8 iterations."""
        env, nodes, lines = pf_result
        assert 1 <= env._iterations <= 10, \
            f"iterations={env._iterations}, expected 1–10"


# =====================================================================
# Bus voltages
# =====================================================================

class TestBusVoltages:
    """Voltage profile for passive radial feeder."""

    def test_slack_bus_1pu(self, pf_result):
        """Root bus held at reference voltage: V_slack = 1.0 p.u."""
        _, nodes, _ = pf_result
        np.testing.assert_allclose(nodes['v_mag'].iloc[0], 1.0, atol=1e-10)

    def test_all_voltages_positive(self, pf_result):
        _, nodes, _ = pf_result
        assert np.all(nodes['v_mag'].values > 0)

    def test_no_voltage_exceeds_slack(self, pf_result):
        """Passive feeder (no DG): V ≤ V_slack = 1.0 everywhere."""
        _, nodes, _ = pf_result
        assert np.all(nodes['v_mag'].values <= 1.0 + 1e-6)

    def test_min_voltage_bus(self, pf_result):
        """Worst bus: bus 18 (0-based: 17) per Baran & Wu."""
        _, nodes, _ = pf_result
        assert nodes['v_mag'].values.argmin() == LIT_V_MIN_BUS

    def test_min_voltage_vs_literature(self, pf_result):
        """V_min within ±0.002 p.u. of 0.9131."""
        _, nodes, _ = pf_result
        v_min = nodes['v_mag'].min()
        assert abs(v_min - LIT_V_MIN_PU) < 0.002, \
            f"V_min={v_min:.6f}, literature={LIT_V_MIN_PU}"

    def test_main_feeder_monotonic_drop(self, pf_result):
        """Buses 0→17 (main feeder): voltage decreases monotonically
        because impedance accumulates from root to end-of-feeder."""
        _, nodes, _ = pf_result
        v_main = nodes['v_mag'].values[:18]
        diffs = np.diff(v_main)
        assert np.all(diffs <= 1e-8), \
            f"Non-monotonic voltage on main feeder: offending diffs={diffs[diffs>1e-8]}"

    def test_voltage_range_nominal(self, pf_result):
        """All voltages within [0.85, 1.05] p.u. for typical distribution."""
        _, nodes, _ = pf_result
        v = nodes['v_mag'].values
        assert np.all(v >= 0.85)
        assert np.all(v <= 1.05)


# =====================================================================
# Line losses — power conservation
# =====================================================================

class TestLineLosses:
    """Resistive losses in a radial network.

    Physics:
      - Active loss: P_loss = I² × R ≥ 0 (Ohmic heating)
      - Reactive loss: Q_loss = I² × X ≥ 0 (inductive)
      - Feeder head flow = total load + total loss (conservation)
    """

    def test_p_loss_vs_literature(self, pf_result):
        """Total P_loss within ±2 kW of literature (202.67 kW)."""
        _, _, lines = pf_result
        p_loss = lines['p_loss_MW'].sum()
        assert abs(p_loss - LIT_P_LOSS_MW) < 0.002, \
            f"P_loss={p_loss*1e3:.2f} kW, lit={LIT_P_LOSS_MW*1e3:.2f} kW"

    def test_q_loss_vs_literature(self, pf_result):
        """Total Q_loss within ±2 kVAr of literature (135.14 kVAr)."""
        _, _, lines = pf_result
        q_loss = lines['q_loss_MVAr'].sum()
        assert abs(q_loss - LIT_Q_LOSS_MVAR) < 0.002, \
            f"Q_loss={q_loss*1e3:.2f} kVAr, lit={LIT_Q_LOSS_MVAR*1e3:.2f} kVAr"

    def test_all_line_p_losses_non_negative(self, pf_result):
        """Each branch: P_loss = I²R ≥ 0 (resistive dissipation)."""
        _, _, lines = pf_result
        assert np.all(lines['p_loss_MW'].values >= -1e-8)

    def test_all_line_q_losses_non_negative(self, pf_result):
        """Each branch: Q_loss = I²X ≥ 0 (inductive)."""
        _, _, lines = pf_result
        assert np.all(lines['q_loss_MVAr'].values >= -1e-8)

    def test_power_conservation(self, pf_result):
        """Feeder head flow ≈ total load + total loss (within 1%)."""
        env, _, lines = pf_result
        p_head = lines['p_flow_MW'].iloc[0]
        p_load = env._get_node_loads_p().sum()
        p_loss = lines['p_loss_MW'].sum()
        np.testing.assert_allclose(
            p_head, p_load + p_loss, rtol=0.01,
            err_msg=f"Conservation: head={p_head:.4f}, load={p_load:.4f}, loss={p_loss:.4f}"
        )

    def test_total_loss_fraction(self, pf_result):
        """Distribution loss typically 2–8% of total load for Case33bw."""
        env, _, lines = pf_result
        p_load = env._get_node_loads_p().sum()
        p_loss = lines['p_loss_MW'].sum()
        loss_pct = p_loss / p_load * 100 if p_load > 0 else 0
        assert 1.0 < loss_pct < 15.0, f"Loss fraction {loss_pct:.1f}% outside typical range"

    def test_slack_power_matches_root_branch_sum_plus_local_load(self, dist_env, bfs_result):
        """Slack injection = first-layer branch flows + local load at slack node.

        For Case33bw the slack bus carries no local load (Pd=0, Qd=0), so the
        two terms happen to be equal.  The assertion uses the general formula so
        it remains correct for cases where the slack bus does have a local load.
        """
        slack_mask = dist_env.topo.sending_nodes == dist_env.slack_bus_id
        p_load_pu = dist_env._get_node_loads_p() / dist_env.baseMVA
        q_load_pu = dist_env._get_node_loads_q() / dist_env.baseMVA
        np.testing.assert_allclose(
            bfs_result['p_slack'],
            bfs_result['p_branch'][slack_mask].sum() + p_load_pu[dist_env.slack_bus_id],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            bfs_result['q_slack'],
            bfs_result['q_branch'][slack_mask].sum() + q_load_pu[dist_env.slack_bus_id],
            atol=1e-12,
        )

    def test_slack_power_balances_total_load_plus_loss(self, dist_env, bfs_result):
        """Feeder-head exchange should close the P/Q balance tightly."""
        p_loss_pu, q_loss_pu = calculate_line_losses(
            dist_env.topo, bfs_result['p_branch'], bfs_result['q_branch'], bfs_result['v_sq']
        )
        p_load_total = dist_env._get_node_loads_p().sum() / dist_env.baseMVA
        q_load_total = dist_env._get_node_loads_q().sum() / dist_env.baseMVA
        np.testing.assert_allclose(
            bfs_result['p_slack'],
            p_load_total + p_loss_pu.sum(),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            bfs_result['q_slack'],
            q_load_total + q_loss_pu.sum(),
            atol=1e-8,
        )


class TestWarmStart:
    """Warm-start (v_sq_init) reduces iteration count in RL rollout scenarios."""

    def test_warm_start_fewer_or_equal_iterations(self, dist_env):
        """Seeding v_sq from a previous converged solve should not need more iterations."""
        p_load_pu = dist_env._get_node_loads_p() / dist_env.baseMVA
        q_load_pu = dist_env._get_node_loads_q() / dist_env.baseMVA

        cold = run_bfs_power_flow(
            topo=dist_env.topo,
            p_load_pu=p_load_pu,
            q_load_pu=q_load_pu,
            v_slack=dist_env.v_slack,
            slack_bus_id=dist_env.slack_bus_id,
        )
        warm = run_bfs_power_flow(
            topo=dist_env.topo,
            p_load_pu=p_load_pu,
            q_load_pu=q_load_pu,
            v_slack=dist_env.v_slack,
            slack_bus_id=dist_env.slack_bus_id,
            v_sq_init=cold['v_sq'],
        )
        assert warm['converged']
        assert warm['iterations'] <= cold['iterations'], (
            f"Warm start used more iterations ({warm['iterations']}) "
            f"than cold start ({cold['iterations']})"
        )

    def test_warm_start_same_solution(self, dist_env):
        """Warm start must produce the same converged solution as cold start."""
        p_load_pu = dist_env._get_node_loads_p() / dist_env.baseMVA
        q_load_pu = dist_env._get_node_loads_q() / dist_env.baseMVA

        cold = run_bfs_power_flow(
            topo=dist_env.topo, p_load_pu=p_load_pu, q_load_pu=q_load_pu,
            v_slack=dist_env.v_slack, slack_bus_id=dist_env.slack_bus_id,
        )
        warm = run_bfs_power_flow(
            topo=dist_env.topo, p_load_pu=p_load_pu, q_load_pu=q_load_pu,
            v_slack=dist_env.v_slack, slack_bus_id=dist_env.slack_bus_id,
            v_sq_init=cold['v_sq'],
        )
        # Both solves must agree to within the convergence tolerance (1e-6).
        # p_branch can differ by one iteration's refinement between cold and
        # warm start, so a tolerance matching the BFS tol is appropriate.
        np.testing.assert_allclose(warm['v_mag'], cold['v_mag'], atol=1e-6)
        np.testing.assert_allclose(warm['p_branch'], cold['p_branch'], atol=1e-6)


class TestDtypeSafety:
    """float32 inputs (typical from RL frameworks) must not break the solver."""

    def test_float32_inputs_accepted(self, dist_env):
        """Passing float32 loads should produce the same result as float64."""
        p64 = dist_env._get_node_loads_p() / dist_env.baseMVA
        q64 = dist_env._get_node_loads_q() / dist_env.baseMVA
        p32 = p64.astype(np.float32)
        q32 = q64.astype(np.float32)

        result64 = run_bfs_power_flow(
            topo=dist_env.topo, p_load_pu=p64, q_load_pu=q64,
            v_slack=dist_env.v_slack, slack_bus_id=dist_env.slack_bus_id,
        )
        result32 = run_bfs_power_flow(
            topo=dist_env.topo, p_load_pu=p32, q_load_pu=q32,
            v_slack=dist_env.v_slack, slack_bus_id=dist_env.slack_bus_id,
        )
        assert result32['converged']
        # float32 inputs introduce ~1e-7 round-trip error; use a loose tolerance
        np.testing.assert_allclose(result32['v_mag'], result64['v_mag'], atol=1e-5)


class TestSlackBusCalibration:
    """Slack bus voltage is pinned to v_slack**2 before the first iteration."""

    def test_non_unity_v_slack_correct_result(self, dist_env):
        """v_slack=1.05 should produce a higher voltage profile than v_slack=1.0."""
        p_load_pu = dist_env._get_node_loads_p() / dist_env.baseMVA
        q_load_pu = dist_env._get_node_loads_q() / dist_env.baseMVA

        r10 = run_bfs_power_flow(
            topo=dist_env.topo, p_load_pu=p_load_pu, q_load_pu=q_load_pu,
            v_slack=1.0, slack_bus_id=dist_env.slack_bus_id,
        )
        r105 = run_bfs_power_flow(
            topo=dist_env.topo, p_load_pu=p_load_pu, q_load_pu=q_load_pu,
            v_slack=1.05, slack_bus_id=dist_env.slack_bus_id,
        )
        assert r105['converged']
        # Slack bus must be exactly at the setpoint
        np.testing.assert_allclose(r105['v_mag'][dist_env.slack_bus_id], 1.05, atol=1e-10)
        # Higher source voltage lifts all bus voltages
        assert np.all(r105['v_mag'] >= r10['v_mag'] - 1e-8)

    def test_non_unity_v_slack_warm_start_correct(self, dist_env):
        """Warm-start with non-unity v_slack must also pin the slack bus correctly."""
        p_load_pu = dist_env._get_node_loads_p() / dist_env.baseMVA
        q_load_pu = dist_env._get_node_loads_q() / dist_env.baseMVA

        cold = run_bfs_power_flow(
            topo=dist_env.topo, p_load_pu=p_load_pu, q_load_pu=q_load_pu,
            v_slack=1.05, slack_bus_id=dist_env.slack_bus_id,
        )
        warm = run_bfs_power_flow(
            topo=dist_env.topo, p_load_pu=p_load_pu, q_load_pu=q_load_pu,
            v_slack=1.05, slack_bus_id=dist_env.slack_bus_id,
            v_sq_init=cold['v_sq'],
        )
        assert warm['converged']
        np.testing.assert_allclose(warm['v_mag'][dist_env.slack_bus_id], 1.05, atol=1e-10)


class TestDenseMatrices:
    """Small networks should store dense matrices for faster RL rollouts."""

    def test_small_network_uses_dense_matrices(self, topo_33):
        """Case33bw (33 nodes) is below the dense threshold."""
        assert isinstance(topo_33.path_matrix, np.ndarray)
        assert isinstance(topo_33.downstream_matrix, np.ndarray)
        assert isinstance(topo_33.loss_matrix, np.ndarray)

    def test_large_network_uses_sparse_matrices(self):
        """Networks above the threshold keep sparse storage."""
        from scipy import sparse as sp
        n = 300
        # Build a simple chain topology with 300 nodes
        from_nodes = np.arange(0, n - 1)
        to_nodes = np.arange(1, n)
        r_pu = np.ones(n - 1) * 0.01
        x_pu = np.ones(n - 1) * 0.01
        topo = build_radial_topology(n, from_nodes, to_nodes, r_pu, x_pu)
        assert sp.issparse(topo.path_matrix)
        assert sp.issparse(topo.downstream_matrix)
        assert sp.issparse(topo.loss_matrix)

    def test_dense_matrices_same_result_as_sparse(self):
        """Dense and sparse topology must produce identical power flow results."""
        from scipy import sparse as sp
        n = 10
        from_nodes = np.arange(0, n - 1)
        to_nodes = np.arange(1, n)
        r_pu = np.ones(n - 1) * 0.05
        x_pu = np.ones(n - 1) * 0.05

        topo_dense = build_radial_topology(n, from_nodes, to_nodes, r_pu, x_pu)
        assert isinstance(topo_dense.path_matrix, np.ndarray)

        # Build a sparse version by manually converting back to sparse
        import copy
        topo_sparse = copy.copy(topo_dense)
        topo_sparse.path_matrix = sp.csr_matrix(topo_dense.path_matrix)
        topo_sparse.downstream_matrix = sp.csr_matrix(topo_dense.downstream_matrix)
        topo_sparse.loss_matrix = sp.csr_matrix(topo_dense.loss_matrix)

        p_load = np.array([0.0, 0.1, 0.05, 0.08, 0.12, 0.06, 0.09, 0.07, 0.04, 0.11])
        q_load = p_load * 0.3
        r_dense = run_bfs_power_flow(topo_dense, p_load, q_load)
        r_sparse = run_bfs_power_flow(topo_sparse, p_load, q_load)

        np.testing.assert_allclose(r_dense['v_mag'], r_sparse['v_mag'], atol=1e-12)
        np.testing.assert_allclose(r_dense['p_branch'], r_sparse['p_branch'], atol=1e-12)


class TestFailureFlags:
    """Solver should expose severe low-voltage states explicitly."""

    def test_voltage_collapse_flag_for_clamped_low_voltage_case(self, dist_env):
        """Numerical clamping must not hide a physically collapsed feeder."""
        p_load_pu = 4.0 * dist_env._get_node_loads_p() / dist_env.baseMVA
        q_load_pu = 4.0 * dist_env._get_node_loads_q() / dist_env.baseMVA
        result = run_bfs_power_flow(
            topo=dist_env.topo,
            p_load_pu=p_load_pu,
            q_load_pu=q_load_pu,
            v_slack=dist_env.v_slack,
            slack_bus_id=dist_env.slack_bus_id,
            max_iter=dist_env.max_iter,
            tol=dist_env.tol,
        )
        assert result['voltage_collapse'] is True
        assert result['is_diverged'] is False
        assert result['converged'] is False
        assert np.isclose(result['v_mag'].min(), 0.5)

    def test_is_diverged_without_voltage_collapse_when_iterations_are_cut_short(self, dist_env):
        """Missing the iteration tolerance is distinct from physical collapse."""
        p_load_pu = dist_env._get_node_loads_p() / dist_env.baseMVA
        q_load_pu = dist_env._get_node_loads_q() / dist_env.baseMVA
        result = run_bfs_power_flow(
            topo=dist_env.topo,
            p_load_pu=p_load_pu,
            q_load_pu=q_load_pu,
            v_slack=dist_env.v_slack,
            slack_bus_id=dist_env.slack_bus_id,
            max_iter=1,
            tol=1e-12,
        )
        assert result['is_diverged'] is True
        assert result['voltage_collapse'] is False
        assert result['converged'] is False


# =====================================================================
# Safety check API
# =====================================================================

class TestSafetyCheck:
    """Test the DistGridEnv.safety_check voltage/thermal constraint checker."""

    def test_safe_with_standard_limits(self, pf_result):
        """Case33bw should satisfy [0.9, 1.1] p.u. limits."""
        env, nodes, lines = pf_result
        is_safe, info = env.safety_check(nodes, lines, with_info=True)
        assert info['converged']
        assert is_safe

    def test_violations_with_tight_limits(self, pf_result):
        """Bus 18 (V≈0.913) violates [0.95, 1.05] limits."""
        env, nodes, lines = pf_result
        is_safe, info = env.safety_check(nodes, lines, v_min=0.95, v_max=1.05,
                                         with_info=True)
        assert not is_safe
        assert LIT_V_MIN_BUS in info['v_violation_nodes']

    def test_info_dict_completeness(self, pf_result):
        env, nodes, lines = pf_result
        _, info = env.safety_check(nodes, lines, with_info=True)
        required = ('v_min_actual', 'v_max_actual',
                    'v_violation_nodes', 'line_violation_ids',
                    'converged', 'iterations')
        for key in required:
            assert key in info, f"Missing key: {key}"


# =====================================================================
# MATPOWER NR cross-validation (optional — requires ref_data CSVs)
# =====================================================================

@pytest.mark.skipif(not HAS_MATPOWER, reason="MATPOWER ref CSVs not found")
class TestVsMatpowerNR:
    """Compare BFS to MATPOWER Newton-Raphson (exact for distribution).

    BFS is an iterative approximate method; NR is the industrial standard.
    Expected discrepancies are small for well-conditioned 33-bus case:
      Vm: ±5e-4 p.u.  |  Branch P: ±0.5%  |  Total loss: ±2 kW
    """

    def test_bus_voltage_magnitudes(self, pf_result):
        _, nodes, _ = pf_result
        ref = pd.read_csv(MATPOWER_BUS_CSV).sort_values('bus_i').reset_index(drop=True)
        np.testing.assert_allclose(
            nodes['v_mag'].values, ref['Vm'].values, atol=5e-4,
            err_msg="Vm differs from MATPOWER NR by > 5e-4 p.u."
        )

    def test_active_branch_flows(self, pf_result):
        _, _, lines = pf_result
        ref = pd.read_csv(MATPOWER_BR_CSV)
        active_ref = ref[ref['status'] == 1].reset_index(drop=True)
        np.testing.assert_allclose(
            np.abs(lines['p_flow_MW'].values),
            np.abs(active_ref['Pf_MW'].values),
            rtol=0.005,
            err_msg="Branch P flows differ from MATPOWER by > 0.5%"
        )

    def test_total_losses_vs_matpower(self, pf_result):
        _, _, lines = pf_result
        ref = pd.read_csv(MATPOWER_BR_CSV)
        active_ref = ref[ref['status'] == 1]
        p_loss_mp = (active_ref['Pf_MW'] + active_ref['Pt_MW']).sum()
        p_loss_ours = lines['p_loss_MW'].sum()
        assert abs(p_loss_ours - p_loss_mp) < 0.002, \
            f"P_loss: ours={p_loss_ours*1e3:.2f} kW, MP={p_loss_mp*1e3:.2f} kW"


# =====================================================================
# Determinism
# =====================================================================

class TestDeterminism:
    """BFS should produce identical results for identical inputs."""

    def test_bfs_deterministic(self, dist_env):
        # Clear warm-start cache so both runs start from flat voltage profile
        dist_env._last_v_sq = None
        n1, l1 = dist_env.cal_pf(df=True)
        dist_env._last_v_sq = None
        n2, l2 = dist_env.cal_pf(df=True)
        np.testing.assert_array_equal(n1['v_mag'].values, n2['v_mag'].values)
        np.testing.assert_array_equal(l1['p_loss_MW'].values, l2['p_loss_MW'].values)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
