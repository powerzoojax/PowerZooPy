"""Performance Comparison: Loop-based vs Vectorized Three-Phase Power Flow

This script compares the performance of two implementations:
1. Old version: Python for-loops (cal_pf_dist_3phase_old_by_loop.py)
2. New version: Vectorized operations (powerzoo/envs/grid/cal_pf_dist_3phase.py)

Both implementations are tested using Case123 with three-phase power flow.
"""
import time
import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the new optimized version
from powerzoo.envs.grid import cal_pf_dist_3phase as new_impl

# Import the old version from local file
from examples.x13_compare_dist_3phase_speed import cal_pf_dist_3phase_old_by_loop as old_impl

# Import environment and case
from powerzoo.envs.grid.dist_3phase import DistGrid3PhaseEnv
from powerzoo.case.distribution.Case123 import Case123


def prepare_test_data():
    """Prepare test data from Case123"""
    case = Case123()
    case.init()
    
    n_nodes = len(case.nodes)
    lines = case.lines
    
    # Filter active lines (status=1)
    if 'status' in lines.columns:
        active_lines = lines[lines['status'] == 1].copy()
    else:
        active_lines = lines.copy()
    
    active_line_indices = active_lines.index.tolist()
    from_nodes = active_lines['#from'].values.astype(int)
    to_nodes = active_lines['#to'].values.astype(int)
    
    # Build impedance matrices from line_config
    baseMVA = getattr(case, 'baseMVA', 10.0)
    baseKV = getattr(case, 'baseKV', 4.16)
    Zbase = (baseKV ** 2) / baseMVA
    
    n_lines = len(active_lines)
    Z_3ph_pu = np.zeros((n_lines, 3, 3), dtype=np.complex128)
    
    for i, (_, line) in enumerate(active_lines.iterrows()):
        config_id = int(line['config_name'])
        length = line['length']
        config = case.line_config.loc[config_id]
        
        Z_ohms = np.array([
            [config['Z11'], config['Z12'], config['Z13']],
            [config['Z21'], config['Z22'], config['Z23']],
            [config['Z31'], config['Z32'], config['Z33']]
        ], dtype=np.complex128) * length
        
        Z_3ph_pu[i] = Z_ohms / Zbase
    
    # Get three-phase loads
    nodes = case.nodes
    n_non_ref = n_nodes - 1
    
    P_3ph_pu = np.zeros(3 * n_non_ref)
    Q_3ph_pu = np.zeros(3 * n_non_ref)
    
    if 'Pd_A' in nodes.columns:
        non_ref_nodes = nodes.iloc[1:] if 0 == 0 else nodes.drop(index=0)
        for i, (_, node) in enumerate(non_ref_nodes.iterrows()):
            P_3ph_pu[i*3 + 0] = node['Pd_A']
            P_3ph_pu[i*3 + 1] = node['Pd_B']
            P_3ph_pu[i*3 + 2] = node['Pd_C']
            Q_3ph_pu[i*3 + 0] = node['Qd_A']
            Q_3ph_pu[i*3 + 1] = node['Qd_B']
            Q_3ph_pu[i*3 + 2] = node['Qd_C']
    else:
        # Fallback to single-phase format
        non_ref_nodes = nodes.iloc[1:] if 0 == 0 else nodes.drop(index=0)
        for i, (_, node) in enumerate(non_ref_nodes.iterrows()):
            Pd = node.get('Pd', 0.0) / 3.0
            Qd = node.get('Qd', 0.0) / 3.0
            P_3ph_pu[i*3:(i+1)*3] = Pd
            Q_3ph_pu[i*3:(i+1)*3] = Qd
    
    return {
        'n_nodes': n_nodes,
        'from_nodes': from_nodes,
        'to_nodes': to_nodes,
        'Z_3ph_pu': Z_3ph_pu,
        'active_line_indices': active_line_indices,
        'P_3ph_pu': P_3ph_pu,
        'Q_3ph_pu': Q_3ph_pu,
        'baseMVA': baseMVA,
        'v_ref_mag': 1.05
    }


def benchmark_old_version(data, n_runs=100, warmup=10):
    """Benchmark the old for-loop version"""
    # Build topology
    topo3ph = old_impl.build_3phase_topology(
        n_nodes=data['n_nodes'],
        from_nodes=data['from_nodes'],
        to_nodes=data['to_nodes'],
        Z_3ph_pu=data['Z_3ph_pu'],
        ref_bus=0,
        v_ref_mag=data['v_ref_mag']
    )
    
    # Warmup
    for _ in range(warmup):
        result = old_impl.run_3phase_bfs_power_flow(
            topo3ph, data['P_3ph_pu'], data['Q_3ph_pu'],
            v_ref_mag=data['v_ref_mag']
        )
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        result = old_impl.run_3phase_bfs_power_flow(
            topo3ph, data['P_3ph_pu'], data['Q_3ph_pu'],
            v_ref_mag=data['v_ref_mag']
        )
    end = time.perf_counter()
    
    avg_time = (end - start) / n_runs * 1000  # ms
    
    # Calculate losses
    P_loss, Q_loss = old_impl.calculate_3phase_losses(topo3ph, result)
    
    # Get phase results
    phase_results = old_impl.get_phase_results(result, data['n_nodes'] - 1)
    
    return {
        'avg_time_ms': avg_time,
        'throughput': n_runs / (end - start),
        'V_min': min(phase_results['V_A'].min(), phase_results['V_B'].min(), phase_results['V_C'].min()),
        'V_max': max(phase_results['V_A'].max(), phase_results['V_B'].max(), phase_results['V_C'].max()),
        'P_loss': np.sum(P_loss) * data['baseMVA'],
        'Q_loss': np.sum(Q_loss) * data['baseMVA'],
        'iterations': result['iterations'],
        'converged': result['converged']
    }


def benchmark_new_version(data, n_runs=100, warmup=10):
    """Benchmark the new vectorized version"""
    # Build topology
    topo3ph = new_impl.build_3phase_topology(
        n_nodes=data['n_nodes'],
        from_nodes=data['from_nodes'],
        to_nodes=data['to_nodes'],
        Z_3ph_pu=data['Z_3ph_pu'],
        ref_bus=0,
        v_ref_mag=data['v_ref_mag']
    )
    
    # Warmup
    for _ in range(warmup):
        result = new_impl.run_3phase_bfs_power_flow(
            topo3ph, data['P_3ph_pu'], data['Q_3ph_pu'],
            v_ref_mag=data['v_ref_mag']
        )
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        result = new_impl.run_3phase_bfs_power_flow(
            topo3ph, data['P_3ph_pu'], data['Q_3ph_pu'],
            v_ref_mag=data['v_ref_mag']
        )
    end = time.perf_counter()
    
    avg_time = (end - start) / n_runs * 1000  # ms
    
    # Calculate losses
    P_loss, Q_loss = new_impl.calculate_3phase_losses(topo3ph, result)
    
    # Get phase results
    phase_results = new_impl.get_phase_results(result, data['n_nodes'] - 1)
    
    return {
        'avg_time_ms': avg_time,
        'throughput': n_runs / (end - start),
        'V_min': min(phase_results['V_A'].min(), phase_results['V_B'].min(), phase_results['V_C'].min()),
        'V_max': max(phase_results['V_A'].max(), phase_results['V_B'].max(), phase_results['V_C'].max()),
        'P_loss': np.sum(P_loss) * data['baseMVA'],
        'Q_loss': np.sum(Q_loss) * data['baseMVA'],
        'iterations': result['iterations'],
        'converged': result['converged']
    }


def benchmark_env_version(n_runs=100, warmup=10):
    """Benchmark the full environment version (cal_pf with df=True)"""
    env = DistGrid3PhaseEnv(case=Case123())
    
    # Warmup
    for _ in range(warmup):
        nodes_df, lines_df = env.cal_pf(df=False)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        nodes_df, lines_df = env.cal_pf(df=False)
    end = time.perf_counter()
    
    avg_time = (end - start) / n_runs * 1000  # ms

    nodes_df, lines_df = env.cal_pf(df=True)
    # Get results from last run
    V_min = min(nodes_df['V_A'].min(), nodes_df['V_B'].min(), nodes_df['V_C'].min())
    V_max = max(nodes_df['V_A'].max(), nodes_df['V_B'].max(), nodes_df['V_C'].max())
    p_loss, q_loss = env.get_total_loss(lines_df)
    
    return {
        'avg_time_ms': avg_time,
        'throughput': n_runs / (end - start),
        'V_min': V_min,
        'V_max': V_max,
        'P_loss': p_loss,
        'Q_loss': q_loss,
        'iterations': env._iterations,
        'converged': env._converged
    }


def main():
    print("=" * 80)
    print("Three-Phase Power Flow Performance Comparison")
    print("Loop-based Version vs Vectorized Version")
    print("=" * 80)
    
    # Prepare data
    print("\nPreparing test data (Case123)...")
    data = prepare_test_data()
    print(f"  Nodes: {data['n_nodes']}")
    print(f"  Lines: {len(data['from_nodes'])}")
    print(f"  Base MVA: {data['baseMVA']}")
    
    n_runs = 1000
    warmup = 10
    
    # Benchmark old version
    print(f"\n{'='*80}")
    print(f"Benchmarking OLD version (for-loops)... ({n_runs} runs, {warmup} warmup)")
    print("=" * 80)
    old_result = benchmark_old_version(data, n_runs, warmup)
    
    print(f"\n  Average time: {old_result['avg_time_ms']:.3f} ms")
    print(f"  Throughput:   {old_result['throughput']:.1f} runs/second")
    print(f"  Converged:    {old_result['converged']} in {old_result['iterations']} iterations")
    print(f"  Min voltage:  {old_result['V_min']:.4f} p.u.")
    print(f"  Max voltage:  {old_result['V_max']:.4f} p.u.")
    print(f"  P loss:       {old_result['P_loss']:.4f} MW")
    
    # Benchmark new version
    print(f"\n{'='*80}")
    print(f"Benchmarking NEW version (vectorized)... ({n_runs} runs, {warmup} warmup)")
    print("=" * 80)
    new_result = benchmark_new_version(data, n_runs, warmup)
    
    print(f"\n  Average time: {new_result['avg_time_ms']:.3f} ms")
    print(f"  Throughput:   {new_result['throughput']:.1f} runs/second")
    print(f"  Converged:    {new_result['converged']} in {new_result['iterations']} iterations")
    print(f"  Min voltage:  {new_result['V_min']:.4f} p.u.")
    print(f"  Max voltage:  {new_result['V_max']:.4f} p.u.")
    print(f"  P loss:       {new_result['P_loss']:.4f} MW")
    
    # Benchmark environment version (full cal_pf with df=True)
    print(f"\n{'='*80}")
    print(f"Benchmarking ENV version (full cal_pf with df=True)... ({n_runs} runs, {warmup} warmup)")
    print("=" * 80)
    env_result = benchmark_env_version(n_runs, warmup)
    
    print(f"\n  Average time: {env_result['avg_time_ms']:.3f} ms")
    print(f"  Throughput:   {env_result['throughput']:.1f} runs/second")
    print(f"  Converged:    {env_result['converged']} in {env_result['iterations']} iterations")
    print(f"  Min voltage:  {env_result['V_min']:.4f} p.u.")
    print(f"  Max voltage:  {env_result['V_max']:.4f} p.u.")
    print(f"  P loss:       {env_result['P_loss']:.4f} MW")
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    speedup = old_result['avg_time_ms'] / new_result['avg_time_ms']
    v_diff = abs(old_result['V_min'] - new_result['V_min'])
    loss_diff = abs(old_result['P_loss'] - new_result['P_loss'])
    
    print(f"\n{'Metric':<25} {'Old (loop)':<20} {'New (vectorized)':<20} {'Improvement':<15}")
    print("-" * 80)
    print(f"{'Average Time':<25} {old_result['avg_time_ms']:.3f} ms{'':<12} {new_result['avg_time_ms']:.3f} ms{'':<12} {speedup:.2f}x faster")
    print(f"{'Throughput':<25} {old_result['throughput']:.1f} /s{'':<12} {new_result['throughput']:.1f} /s{'':<12} {new_result['throughput']/old_result['throughput']:.2f}x")
    print(f"{'Min Voltage':<25} {old_result['V_min']:.4f} p.u.{'':<8} {new_result['V_min']:.4f} p.u.{'':<8} Δ={v_diff:.6f}")
    print(f"{'P Loss':<25} {old_result['P_loss']:.4f} MW{'':<10} {new_result['P_loss']:.4f} MW{'':<10} Δ={loss_diff:.6f}")
    print(f"{'Iterations':<25} {old_result['iterations']:<20} {new_result['iterations']:<20}")
    
    # Environment overhead
    env_overhead = env_result['avg_time_ms'] - new_result['avg_time_ms']
    env_overhead_pct = (env_overhead / new_result['avg_time_ms']) * 100
    
    print(f"\n{'='*80}")
    print("ENVIRONMENT OVERHEAD")
    print("=" * 80)
    print(f"  Core function time: {new_result['avg_time_ms']:.3f} ms")
    print(f"  Full env time:      {env_result['avg_time_ms']:.3f} ms")
    print(f"  Overhead:           {env_overhead:.3f} ms ({env_overhead_pct:.1f}%)")
    print(f"  (Overhead includes DataFrame construction and result formatting)")
    
    # Assessment
    print(f"\n{'='*80}")
    print("ASSESSMENT")
    print("=" * 80)
    
    if speedup >= 2.0:
        print(f"\n✓ Excellent speedup! Vectorized version is {speedup:.1f}x faster")
    elif speedup >= 1.5:
        print(f"\n○ Good speedup. Vectorized version is {speedup:.1f}x faster")
    elif speedup >= 1.1:
        print(f"\n△ Moderate speedup ({speedup:.1f}x). Vectorization helps but may vary with system size.")
    elif speedup >= 0.8:
        print(f"\n△ Similar performance ({speedup:.1f}x). For small systems, loops may be competitive.")
        print(f"  Note: Vectorization typically shows benefits on larger systems (>1000 nodes)")
    else:
        print(f"\n△ Loop version is faster ({1/speedup:.1f}x) for this system size.")
        print(f"  Note: Vectorization overhead may outweigh benefits for small systems.")
        print(f"  For larger systems, vectorized version should perform better.")
    
    if v_diff < 0.001:
        print("✓ Results match well (voltage difference < 0.001 p.u.)")
    else:
        print(f"△ Results differ slightly (voltage difference = {v_diff:.4f} p.u.)")
    
    if env_overhead_pct < 50:
        print(f"✓ Environment overhead is reasonable ({env_overhead_pct:.1f}%)")
    else:
        print(f"△ Environment overhead is significant ({env_overhead_pct:.1f}%) - consider optimization")
    
    print(f"\n{'='*80}")
    print("Benchmark completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

