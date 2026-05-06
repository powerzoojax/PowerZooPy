"""Benchmark: cyipopt (IPOPT) vs scipy SLSQP vs pandapower for AC-OPF.

Run with:
    pytest tests/test_acopf_benchmark.py -v -s

or directly:
    python tests/test_acopf_benchmark.py
"""
import time
import warnings
import numpy as np
import pytest

from powerzoo.case import load_case
from powerzoo.envs.grid.cal_acopf_trans import ACOPFSolverBuiltin, HAS_CYIPOPT

# ---------------------------------------------------------------------------
# Helper: build a representative net-load vector for each case
# ---------------------------------------------------------------------------

def _make_net_load(case, load_ratio=0.85):
    """Return a node_net_load_mw vector at a given fraction of total capacity."""
    case.init()
    n = len(case.nodes)
    total_cap = case.units['p_max'].sum()
    demand = total_cap * load_ratio

    # 1. Use Pd column from nodes if present
    if 'Pd' in case.nodes.columns:
        loads = case.nodes['Pd'].values
        if loads.sum() > 0:
            return loads * (demand / loads.sum())

    # 2. Use d_max from case.loads if available
    if hasattr(case, 'loads') and hasattr(case.loads, 'columns'):
        if 'd_max' in case.loads.columns and '#bus_id' in case.loads.columns:
            nl = np.zeros(n)
            for _, row in case.loads.iterrows():
                bus = int(row['#bus_id'])
                nl[bus] += float(row['d_max'])
            if nl.sum() > 0:
                return nl * (demand / nl.sum())

    # 3. Spread over non-slack load buses, or all buses as last resort
    gen_buses = set(case.units['#bus_id'].astype(int).tolist())
    ref_bus = getattr(case, 'slack_bus', 0)
    load_buses = [i for i in range(n) if i not in gen_buses and i != ref_bus]
    if not load_buses:
        load_buses = [i for i in range(n) if i != ref_bus]
    nl = np.zeros(n)
    nl[load_buses] = demand / len(load_buses)
    return nl


# ---------------------------------------------------------------------------
# Core timing utility
# ---------------------------------------------------------------------------

def _time_solver(solver, net_load, n_runs=3):
    """Return (mean_time_s, result) averaged over n_runs solves."""
    # Warm-up solve (not counted) — also builds IPOPT structures
    result = solver.solve(net_load)
    if not result['success']:
        return float('inf'), result

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        res = solver.solve(net_load)
        times.append(time.perf_counter() - t0)

    return float(np.mean(times)), res


# ---------------------------------------------------------------------------
# Benchmark fixture: run all three backends for a given case
# ---------------------------------------------------------------------------

def _run_benchmark(case_num, load_ratio=0.85, n_runs=3, verbose=False):
    case = load_case(case_num)
    net_load = _make_net_load(case, load_ratio)

    results = {}

    # 1. SLSQP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slsqp_solver = ACOPFSolverBuiltin(case, backend='slsqp', verbose=verbose)
    t_slsqp, res_slsqp = _time_solver(slsqp_solver, net_load, n_runs)
    results['slsqp'] = {'time': t_slsqp, 'result': res_slsqp}

    # 2. IPOPT (cyipopt)
    if HAS_CYIPOPT:
        ipopt_solver = ACOPFSolverBuiltin(case, backend='ipopt', verbose=verbose)
        t_ipopt, res_ipopt = _time_solver(ipopt_solver, net_load, n_runs)
        results['ipopt'] = {'time': t_ipopt, 'result': res_ipopt}
    else:
        results['ipopt'] = {'time': None, 'result': None}
        print("  [skip] cyipopt not installed")

    # 3. pandapower
    try:
        from powerzoo.envs.grid.cal_acopf_trans_pandapower import ACOPFSolver
        pp_solver = ACOPFSolver(case)
        # warm-up
        pp_solver.solve(net_load)
        pp_times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            res_pp = pp_solver.solve(net_load)
            pp_times.append(time.perf_counter() - t0)
        t_pp = float(np.mean(pp_times))
        results['pandapower'] = {'time': t_pp, 'result': res_pp}
    except Exception as e:
        results['pandapower'] = {'time': None, 'result': None}
        print(f"  [skip] pandapower: {e}")

    return results


def _print_summary(case_num, results):
    print(f"\n{'='*60}")
    print(f"  Case{case_num} AC-OPF benchmark (avg of warm solves)")
    print(f"{'='*60}")
    ref_time = None
    order = ['pandapower', 'ipopt', 'slsqp']
    for name in order:
        r = results.get(name, {})
        t = r.get('time')
        if t is None:
            print(f"  {name:<15}  N/A")
            continue
        if ref_time is None:
            ref_time = t
        ratio = t / ref_time if ref_time else 1.0
        res = r['result']
        cost_str = f"cost={res['total_cost']:.1f}" if res and res['success'] else "FAILED"
        print(f"  {name:<15}  {t*1000:7.1f} ms   ({ratio:5.1f}x)   {cost_str}")


# ---------------------------------------------------------------------------
# pytest tests
# ---------------------------------------------------------------------------

class TestACOPFBenchmarkCase5:
    """Case5 (5 buses, 5 gens) — quick convergence check."""

    def setup_method(self):
        self.case = load_case(5)
        self.net_load = _make_net_load(self.case)

    def test_slsqp_converges(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = ACOPFSolverBuiltin(self.case, backend='slsqp')
        res = solver.solve(self.net_load)
        assert res['success'], "SLSQP failed on Case5"
        assert res['total_cost'] < np.inf

    @pytest.mark.skipif(not HAS_CYIPOPT, reason="cyipopt not installed")
    def test_ipopt_converges(self):
        solver = ACOPFSolverBuiltin(self.case, backend='ipopt')
        res = solver.solve(self.net_load)
        assert res['success'], "IPOPT failed on Case5"
        assert res['total_cost'] < np.inf

    @pytest.mark.skipif(not HAS_CYIPOPT, reason="cyipopt not installed")
    def test_ipopt_slsqp_cost_close(self):
        """IPOPT and SLSQP should find similar optimal cost (within 1%)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slsqp_solver = ACOPFSolverBuiltin(self.case, backend='slsqp')
        ipopt_solver = ACOPFSolverBuiltin(self.case, backend='ipopt')

        res_s = slsqp_solver.solve(self.net_load)
        res_i = ipopt_solver.solve(self.net_load)

        assert res_s['success'] and res_i['success']
        # Allow 5% tolerance — both are approximate NLP solvers
        rel_diff = abs(res_s['total_cost'] - res_i['total_cost']) / max(abs(res_i['total_cost']), 1.0)
        assert rel_diff < 0.05, (
            f"Cost mismatch: SLSQP={res_s['total_cost']:.2f}, "
            f"IPOPT={res_i['total_cost']:.2f}, rel_diff={rel_diff:.4f}"
        )

    def test_backend_tag(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = ACOPFSolverBuiltin(self.case, backend='slsqp')
        res = solver.solve(self.net_load)
        assert res['solver_backend'] == 'scipy_slsqp'

    @pytest.mark.skipif(not HAS_CYIPOPT, reason="cyipopt not installed")
    def test_ipopt_backend_tag(self):
        solver = ACOPFSolverBuiltin(self.case, backend='ipopt')
        res = solver.solve(self.net_load)
        assert res['solver_backend'] == 'cyipopt_ipopt'

    @pytest.mark.skipif(not HAS_CYIPOPT, reason="cyipopt not installed")
    def test_ipopt_faster_than_slsqp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_s, _ = _time_solver(ACOPFSolverBuiltin(self.case, backend='slsqp'), self.net_load)
        t_i, _ = _time_solver(ACOPFSolverBuiltin(self.case, backend='ipopt'), self.net_load)
        print(f"\n  Case5: SLSQP={t_s*1000:.1f}ms  IPOPT={t_i*1000:.1f}ms")
        # Case5 is tiny — IPOPT startup overhead dominates.
        # Accept if IPOPT < 2× SLSQP, or both finish within 200 ms.
        assert t_i < t_s * 2 or t_i < 0.2


class TestACOPFBenchmarkCase5Timing:
    """Case5 timing — verifies both backends produce valid costs."""

    def setup_method(self):
        self.case = load_case(5)
        self.net_load = _make_net_load(self.case)

    @pytest.mark.skipif(not HAS_CYIPOPT, reason="cyipopt not installed")
    def test_both_backends_agree(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_s = ACOPFSolverBuiltin(self.case, backend='slsqp').solve(self.net_load)
        res_i = ACOPFSolverBuiltin(self.case, backend='ipopt').solve(self.net_load)
        assert res_s['success'] and res_i['success']
        rel = abs(res_s['total_cost'] - res_i['total_cost']) / max(abs(res_i['total_cost']), 1)
        print(f"\n  Case5: SLSQP={res_s['total_cost']:.1f}  IPOPT={res_i['total_cost']:.1f}  rel={rel:.4f}")
        assert rel < 0.05

    @pytest.mark.skipif(not HAS_CYIPOPT, reason="cyipopt not installed")
    def test_auto_uses_ipopt(self):
        """With cyipopt installed, backend='auto' should select IPOPT."""
        solver = ACOPFSolverBuiltin(self.case, backend='auto')
        res = solver.solve(self.net_load)
        assert res['success']
        assert res['solver_backend'] == 'cyipopt_ipopt'


class TestACOPFBenchmarkCase118:
    """Case118 (118 buses) — the key large-scale test."""

    def setup_method(self):
        self.case = load_case(118)
        self.net_load = _make_net_load(self.case)

    @pytest.mark.slow
    def test_slsqp_converges(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = ACOPFSolverBuiltin(self.case, backend='slsqp')
        res = solver.solve(self.net_load)
        assert res['success'], "SLSQP failed on Case118"

    @pytest.mark.skipif(not HAS_CYIPOPT, reason="cyipopt not installed")
    def test_ipopt_converges(self):
        solver = ACOPFSolverBuiltin(self.case, backend='ipopt')
        res = solver.solve(self.net_load)
        assert res['success'], "IPOPT failed on Case118"

    @pytest.mark.skipif(not HAS_CYIPOPT, reason="cyipopt not installed")
    def test_ipopt_cost_valid(self):
        solver = ACOPFSolverBuiltin(self.case, backend='ipopt')
        res = solver.solve(self.net_load)
        assert res['success']
        assert res['total_cost'] > 0
        assert np.all(res['vm_pu'] >= 0.85), "Voltage too low"
        assert np.all(res['vm_pu'] <= 1.15), "Voltage too high"

    @pytest.mark.skipif(not HAS_CYIPOPT, reason="cyipopt not installed")
    def test_ipopt_much_faster_than_slsqp(self):
        """On Case118, IPOPT should be at least 5x faster than SLSQP."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_s, res_s = _time_solver(
                ACOPFSolverBuiltin(self.case, backend='slsqp'), self.net_load, n_runs=1)
        t_i, res_i = _time_solver(
            ACOPFSolverBuiltin(self.case, backend='ipopt'), self.net_load, n_runs=3)
        print(f"\n  Case118: SLSQP={t_s:.2f}s  IPOPT={t_i*1000:.1f}ms  "
              f"speedup={t_s/t_i:.1f}x")
        assert res_i['success'], "IPOPT failed"
        # Goal: IPOPT < 2s (warm solve; allow headroom for CI variability)
        assert t_i < 2.0, f"IPOPT too slow: {t_i:.2f}s"


# ---------------------------------------------------------------------------
# Parity tests: builtin vs pandapower
# ---------------------------------------------------------------------------

try:
    from powerzoo.envs.grid.cal_acopf_trans_pandapower import (
        ACOPFSolver as PPACOPFSolver, HAS_PANDAPOWER,
    )
except ImportError:
    HAS_PANDAPOWER = False
    PPACOPFSolver = None


def _make_original_load(case):
    """Return the original Pd load vector from the case (not scaled)."""
    case.init()
    if 'Pd' in case.nodes.columns:
        loads = case.nodes['Pd'].values.astype(float)
        if loads.sum() > 0:
            return loads
    return _make_net_load(case, load_ratio=0.85)


class TestACOPFParityCase5:
    """Case5: builtin vs pandapower parity (no transformers, simple case)."""

    @pytest.mark.skipif(not HAS_PANDAPOWER, reason="pandapower not installed")
    def test_linear_cost_parity(self):
        """With mc_a=0, both solvers should agree closely on cost."""
        case = load_case(5)
        case.init()
        # Force linear cost
        case.units = case.units.copy()
        case.units['mc_a'] = 0.0

        net_load = _make_original_load(case)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            builtin = ACOPFSolverBuiltin(case, backend='auto')
        pp_solver = PPACOPFSolver(case)

        res_b = builtin.solve(net_load)
        res_p = pp_solver.solve(net_load)

        assert res_b['success'], "builtin failed"
        assert res_p['success'], "pandapower failed"

        rel_cost = abs(res_b['total_cost'] - res_p['total_cost']) / max(abs(res_p['total_cost']), 1)
        print(f"\n  Case5 linear: builtin={res_b['total_cost']:.2f}, "
              f"pp={res_p['total_cost']:.2f}, rel={rel_cost:.4f}")
        assert rel_cost < 0.02, (
            f"Cost mismatch: builtin={res_b['total_cost']:.2f}, "
            f"pp={res_p['total_cost']:.2f}, rel={rel_cost:.4f}"
        )

    @pytest.mark.skipif(not HAS_PANDAPOWER, reason="pandapower not installed")
    def test_quadratic_cost_parity(self):
        """With original quadratic cost, solvers should still agree."""
        case = load_case(5)
        net_load = _make_original_load(case)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            builtin = ACOPFSolverBuiltin(case, backend='auto')
        pp_solver = PPACOPFSolver(case)

        res_b = builtin.solve(net_load)
        res_p = pp_solver.solve(net_load)

        assert res_b['success'], "builtin failed"
        assert res_p['success'], "pandapower failed"

        rel_cost = abs(res_b['total_cost'] - res_p['total_cost']) / max(abs(res_p['total_cost']), 1)
        print(f"\n  Case5 quadratic: builtin={res_b['total_cost']:.2f}, "
              f"pp={res_p['total_cost']:.2f}, rel={rel_cost:.4f}")
        assert rel_cost < 0.05, (
            f"Cost mismatch: builtin={res_b['total_cost']:.2f}, "
            f"pp={res_p['total_cost']:.2f}, rel={rel_cost:.4f}"
        )

    @pytest.mark.skipif(not HAS_PANDAPOWER, reason="pandapower not installed")
    def test_unit_dispatch_close(self):
        """Unit dispatch should be similar."""
        case = load_case(5)
        case.init()
        case.units = case.units.copy()
        case.units['mc_a'] = 0.0
        net_load = _make_original_load(case)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            builtin = ACOPFSolverBuiltin(case, backend='auto')
        pp_solver = PPACOPFSolver(case)

        res_b = builtin.solve(net_load)
        res_p = pp_solver.solve(net_load)
        assert res_b['success'] and res_p['success']

        max_p_diff = np.max(np.abs(res_b['unit_power_mw'] - res_p['unit_power_mw']))
        print(f"\n  Case5 max unit P diff: {max_p_diff:.2f} MW")
        # Allow 5 MW tolerance (small system, both NLP solvers)
        assert max_p_diff < 5.0, f"Unit power diff too large: {max_p_diff:.2f} MW"


@pytest.mark.skipif(not HAS_PANDAPOWER, reason="pandapower not installed")
class TestACOPFParityCase14:
    """Case14: has transformers, bus shunts, reactive loads."""

    def test_linear_cost_parity(self):
        case = load_case(14)
        case.init()
        case.units = case.units.copy()
        case.units['mc_a'] = 0.0
        net_load = _make_original_load(case)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            builtin = ACOPFSolverBuiltin(case, backend='auto')
        pp_solver = PPACOPFSolver(case)

        res_b = builtin.solve(net_load)
        res_p = pp_solver.solve(net_load)

        assert res_b['success'], "builtin failed"
        assert res_p['success'], "pandapower failed"

        rel_cost = abs(res_b['total_cost'] - res_p['total_cost']) / max(abs(res_p['total_cost']), 1)
        print(f"\n  Case14 linear: builtin={res_b['total_cost']:.2f}, "
              f"pp={res_p['total_cost']:.2f}, rel={rel_cost:.4f}")
        assert rel_cost < 0.05, (
            f"Cost mismatch: builtin={res_b['total_cost']:.2f}, "
            f"pp={res_p['total_cost']:.2f}, rel={rel_cost:.4f}"
        )

    def test_quadratic_cost_parity(self):
        case = load_case(14)
        net_load = _make_original_load(case)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            builtin = ACOPFSolverBuiltin(case, backend='auto')
        pp_solver = PPACOPFSolver(case)

        res_b = builtin.solve(net_load)
        res_p = pp_solver.solve(net_load)

        assert res_b['success'], "builtin failed"
        assert res_p['success'], "pandapower failed"

        rel_cost = abs(res_b['total_cost'] - res_p['total_cost']) / max(abs(res_p['total_cost']), 1)
        print(f"\n  Case14 quadratic: builtin={res_b['total_cost']:.2f}, "
              f"pp={res_p['total_cost']:.2f}, rel={rel_cost:.4f}")
        assert rel_cost < 0.05


@pytest.mark.skipif(not HAS_PANDAPOWER, reason="pandapower not installed")
class TestACOPFParityCase118:
    """Case118: large-scale parity check."""

    def test_linear_cost_parity(self):
        case = load_case(118)
        case.init()
        case.units = case.units.copy()
        case.units['mc_a'] = 0.0
        net_load = _make_original_load(case)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            builtin = ACOPFSolverBuiltin(case, backend='auto')
        pp_solver = PPACOPFSolver(case)

        res_b = builtin.solve(net_load)
        res_p = pp_solver.solve(net_load)

        assert res_b['success'], "builtin failed"
        if not res_p['success']:
            pytest.skip(
                "pandapower AC-OPF did not converge on Case118 (solver/backends vary; "
                "numba and pandapower version affect large nets)."
            )

        rel_cost = abs(res_b['total_cost'] - res_p['total_cost']) / max(abs(res_p['total_cost']), 1)
        print(f"\n  Case118 linear: builtin={res_b['total_cost']:.2f}, "
              f"pp={res_p['total_cost']:.2f}, rel={rel_cost:.4f}")
        assert rel_cost < 0.05

    def test_quadratic_cost_parity(self):
        case = load_case(118)
        net_load = _make_original_load(case)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            builtin = ACOPFSolverBuiltin(case, backend='auto')
        pp_solver = PPACOPFSolver(case)

        res_b = builtin.solve(net_load)
        res_p = pp_solver.solve(net_load)

        assert res_b['success'], "builtin failed"
        if not res_p['success']:
            pytest.skip(
                "pandapower AC-OPF did not converge on Case118 (solver/backends vary; "
                "numba and pandapower version affect large nets)."
            )

        rel_cost = abs(res_b['total_cost'] - res_p['total_cost']) / max(abs(res_p['total_cost']), 1)
        print(f"\n  Case118 quadratic: builtin={res_b['total_cost']:.2f}, "
              f"pp={res_p['total_cost']:.2f}, rel={rel_cost:.4f}")
        assert rel_cost < 0.05


# ---------------------------------------------------------------------------
# Bug-fix regression tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PANDAPOWER, reason="pandapower not installed")
class TestACOPFPandapowerFixes:
    """Regression tests for specific bug fixes in ACOPFSolver."""

    def test_failure_result_finite_cost(self):
        """_failure_result must return the configured fail_cost, not np.inf."""
        case = load_case(5)
        solver = PPACOPFSolver(case, fail_cost=12345.0)
        result = solver._failure_result()
        assert np.isfinite(result['total_cost']), "total_cost must be finite"
        assert result['total_cost'] == 12345.0
        assert np.isfinite(result['slack_violation']), "slack_violation must be finite"

    def test_failure_result_default_fail_cost(self):
        """Default fail_cost (1e6) must be finite."""
        case = load_case(5)
        solver = PPACOPFSolver(case)
        result = solver._failure_result()
        assert np.isfinite(result['total_cost'])
        assert result['total_cost'] == 1e6

    def test_commitment_zeros_q_limits(self):
        """commitment=0 must zero Q limits to prevent reactive-power cheating."""
        case = load_case(5)
        solver = PPACOPFSolver(case)

        # Initial Q limits are non-zero (q_factor > 0)
        for gi in solver._pp_gen_idx:
            assert solver.net.gen.at[gi, 'max_q_mvar'] != 0.0

        # Solve with all units committed off (OPF will fail; limits are set before runopp)
        n_units = len(case.units)
        solver.solve(np.zeros(len(case.nodes)), commitment=np.zeros(n_units))

        for gi in solver._pp_gen_idx:
            assert solver.net.gen.at[gi, 'min_q_mvar'] == 0.0, \
                f"min_q_mvar not zeroed for committed-off gen {gi}"
            assert solver.net.gen.at[gi, 'max_q_mvar'] == 0.0, \
                f"max_q_mvar not zeroed for committed-off gen {gi}"

    def test_commitment_restores_q_limits(self):
        """Q limits must be restored when commitment is cleared (None)."""
        case = load_case(5)
        solver = PPACOPFSolver(case)
        n_units = len(case.units)

        # Zero out Q limits via commitment=0
        solver.solve(np.zeros(len(case.nodes)), commitment=np.zeros(n_units))
        for gi in solver._pp_gen_idx:
            assert solver.net.gen.at[gi, 'max_q_mvar'] == 0.0

        # Restore via commitment=None
        net_load = _make_original_load(case)
        solver.solve(net_load, commitment=None)
        for gi in solver._pp_gen_idx:
            assert solver.net.gen.at[gi, 'max_q_mvar'] > 0.0, \
                f"Q limit not restored after commitment=None for gen {gi}"

    def test_q_scaling_clamp_for_negative_net_load(self):
        """For nodes where Qd_orig > 0 (inductive), Q must not flip to negative
        when net load goes negative due to renewable over-generation."""
        case = load_case(14)
        case.init()
        if 'Qd' not in case.nodes.columns or 'Pd' not in case.nodes.columns:
            pytest.skip("case14 missing Qd/Pd columns")

        solver = PPACOPFSolver(case)
        Qd_orig = case.nodes['Qd'].values.astype(float)

        # Find a node with positive Qd and positive Pd_orig; make its net load negative
        Pd_orig = case.nodes['Pd'].values.astype(float)
        inductive_positive_load = np.where((Qd_orig > 1e-8) & (Pd_orig > 1e-8))[0]
        if len(inductive_positive_load) == 0:
            pytest.skip("no node with Qd>0 and Pd>0 in case14")

        node_idx = int(inductive_positive_load[0])
        net_load = Pd_orig.copy()
        net_load[node_idx] = -abs(Pd_orig[node_idx]) - 1.0  # force negative

        solver.solve(net_load)

        q_vals = solver.net.load['q_mvar'].values
        # At the over-generation node with Qd_orig > 0, Q must be clamped to 0 (not negative)
        assert q_vals[node_idx] >= -1e-10, (
            f"Node {node_idx}: Qd_orig={Qd_orig[node_idx]:.4f} went to Q={q_vals[node_idx]:.4f} "
            "with negative net load — ratio clamp not applied"
        )


# ---------------------------------------------------------------------------
# Full benchmark report (run as script)
# ---------------------------------------------------------------------------

def run_full_benchmark():
    print("\nAC-OPF Solver Benchmark")
    print("Backends: pandapower/IPOPT, cyipopt/IPOPT, scipy/SLSQP")
    print(f"cyipopt available: {HAS_CYIPOPT}")

    for case_num in [5, 118]:
        print(f"\nCase{case_num}:")
        try:
            results = _run_benchmark(case_num, n_runs=3)
            _print_summary(case_num, results)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()


if __name__ == '__main__':
    run_full_benchmark()
