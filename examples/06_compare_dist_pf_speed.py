"""Performance Comparison: For-Loop vs Matrix-based Forward-Backward Sweep

This script compares the performance of two implementations:
1. Original version: Python for-loops (cal_df_dist.bak.py)
2. Optimized version: Sparse matrix operations (cal_pf_dist.py)
"""
import time
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, '.')

# Import the new optimized version
from powerzoo.envs.grid import cal_pf_dist as new_impl

# Import the old version from local backup
from examples.x06_compare_dist_pf_speed import cal_pf_dist_old as old_impl


def prepare_test_data():
    """Prepare test data from Case33bw"""
    from powerzoo.case.distribution.Case33bw import Case33bw
    
    case = Case33bw()
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
    r_pu = active_lines['r'].values.copy()
    x_pu = active_lines['x'].values.copy()
    
    # Get loads
    baseMVA = getattr(case, 'baseMVA', 100.0)
    nodes_loads_map = case.get_nodes_loads_map()
    p_load = nodes_loads_map.dot(case.loads['Pd'].values)
    q_load = nodes_loads_map.dot(case.loads['Qd'].values)
    
    p_load_pu = p_load / baseMVA
    q_load_pu = q_load / baseMVA
    
    return {
        'n_nodes': n_nodes,
        'from_nodes': from_nodes,
        'to_nodes': to_nodes,
        'r_pu': r_pu,
        'x_pu': x_pu,
        'active_line_indices': active_line_indices,
        'p_load_pu': p_load_pu,
        'q_load_pu': q_load_pu,
        'baseMVA': baseMVA
    }


def benchmark_old_version(data, n_runs=100, warmup=10):
    """Benchmark the old for-loop version"""
    # Build topology
    topo = old_impl.build_radial_topology(
        n_nodes=data['n_nodes'],
        from_nodes=data['from_nodes'],
        to_nodes=data['to_nodes'],
        r_pu=data['r_pu'],
        x_pu=data['x_pu'],
        slack_bus_id=0,
        active_line_indices=data['active_line_indices']
    )
    
    # Warmup
    for _ in range(warmup):
        result = old_impl.run_fbs_power_flow(
            topo, data['p_load_pu'], data['q_load_pu']
        )
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        result = old_impl.run_fbs_power_flow(
            topo, data['p_load_pu'], data['q_load_pu']
        )
    end = time.perf_counter()
    
    avg_time = (end - start) / n_runs * 1000  # ms
    
    # Calculate losses
    p_loss, q_loss = old_impl.calculate_line_losses(
        topo, result['p_branch'], result['q_branch'], result['v_sq']
    )
    
    return {
        'avg_time_ms': avg_time,
        'throughput': n_runs / (end - start),
        'v_min': result['v_mag'].min(),
        'v_max': result['v_mag'].max(),
        'p_loss': np.sum(p_loss) * data['baseMVA'],
        'q_loss': np.sum(q_loss) * data['baseMVA'],
        'iterations': result['iterations'],
        'converged': result['converged']
    }


def benchmark_new_version(data, n_runs=100, warmup=10):
    """Benchmark the new matrix version"""
    # Build topology
    topo = new_impl.build_radial_topology(
        n_nodes=data['n_nodes'],
        from_nodes=data['from_nodes'],
        to_nodes=data['to_nodes'],
        r_pu=data['r_pu'],
        x_pu=data['x_pu'],
        slack_bus_id=0,
        active_line_indices=data['active_line_indices']
    )
    
    # Warmup
    for _ in range(warmup):
        result = new_impl.run_fbs_power_flow(
            topo, data['p_load_pu'], data['q_load_pu']
        )
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        result = new_impl.run_fbs_power_flow(
            topo, data['p_load_pu'], data['q_load_pu']
        )
    end = time.perf_counter()
    
    avg_time = (end - start) / n_runs * 1000  # ms
    
    # Calculate losses
    p_loss, q_loss = new_impl.calculate_line_losses(
        topo, result['p_branch'], result['q_branch'], result['v_sq']
    )
    
    return {
        'avg_time_ms': avg_time,
        'throughput': n_runs / (end - start),
        'v_min': result['v_mag'].min(),
        'v_max': result['v_mag'].max(),
        'p_loss': np.sum(p_loss) * data['baseMVA'],
        'q_loss': np.sum(q_loss) * data['baseMVA'],
        'iterations': result['iterations'],
        'converged': result['converged']
    }


def main():
    print("=" * 80)
    print("Forward-Backward Sweep Performance Comparison")
    print("For-Loop Version vs Matrix Version")
    print("=" * 80)
    
    # Prepare data
    print("\nPreparing test data (Case33bw)...")
    data = prepare_test_data()
    print(f"  Nodes: {data['n_nodes']}")
    print(f"  Lines: {len(data['from_nodes'])}")
    
    n_runs = 500
    warmup = 50
    
    # Benchmark old version
    print(f"\n{'='*80}")
    print(f"Benchmarking OLD version (for-loops)... ({n_runs} runs, {warmup} warmup)")
    print("=" * 80)
    old_result = benchmark_old_version(data, n_runs, warmup)
    
    print(f"\n  Average time: {old_result['avg_time_ms']:.3f} ms")
    print(f"  Throughput:   {old_result['throughput']:.1f} runs/second")
    print(f"  Converged:    {old_result['converged']} in {old_result['iterations']} iterations")
    print(f"  Min voltage:  {old_result['v_min']:.4f} p.u.")
    print(f"  P loss:       {old_result['p_loss']:.4f} MW")
    
    # Benchmark new version
    print(f"\n{'='*80}")
    print(f"Benchmarking NEW version (sparse matrix)... ({n_runs} runs, {warmup} warmup)")
    print("=" * 80)
    new_result = benchmark_new_version(data, n_runs, warmup)
    
    print(f"\n  Average time: {new_result['avg_time_ms']:.3f} ms")
    print(f"  Throughput:   {new_result['throughput']:.1f} runs/second")
    print(f"  Converged:    {new_result['converged']} in {new_result['iterations']} iterations")
    print(f"  Min voltage:  {new_result['v_min']:.4f} p.u.")
    print(f"  P loss:       {new_result['p_loss']:.4f} MW")
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    speedup = old_result['avg_time_ms'] / new_result['avg_time_ms']
    v_diff = abs(old_result['v_min'] - new_result['v_min'])
    loss_diff = abs(old_result['p_loss'] - new_result['p_loss'])
    
    print(f"\n{'Metric':<25} {'Old (for-loop)':<20} {'New (matrix)':<20} {'Improvement':<15}")
    print("-" * 80)
    print(f"{'Average Time':<25} {old_result['avg_time_ms']:.3f} ms{'':<12} {new_result['avg_time_ms']:.3f} ms{'':<12} {speedup:.2f}x faster")
    print(f"{'Throughput':<25} {old_result['throughput']:.1f} /s{'':<12} {new_result['throughput']:.1f} /s{'':<12} {new_result['throughput']/old_result['throughput']:.2f}x")
    print(f"{'Min Voltage':<25} {old_result['v_min']:.4f} p.u.{'':<8} {new_result['v_min']:.4f} p.u.{'':<8} Δ={v_diff:.6f}")
    print(f"{'P Loss':<25} {old_result['p_loss']:.4f} MW{'':<10} {new_result['p_loss']:.4f} MW{'':<10} Δ={loss_diff:.6f}")
    print(f"{'Iterations':<25} {old_result['iterations']:<20} {new_result['iterations']:<20}")
    
    # Assessment
    print(f"\n{'='*80}")
    print("ASSESSMENT")
    print("=" * 80)
    
    if speedup >= 2.0:
        print(f"\n✓ Excellent speedup! Matrix version is {speedup:.1f}x faster")
    elif speedup >= 1.5:
        print(f"\n○ Good speedup. Matrix version is {speedup:.1f}x faster")
    else:
        print(f"\n△ Marginal speedup ({speedup:.1f}x). May vary with system size.")
    
    if v_diff < 0.001:
        print("✓ Results match well (voltage difference < 0.001 p.u.)")
    else:
        print(f"△ Results differ slightly (voltage difference = {v_diff:.4f} p.u.)")
    
    print(f"\n{'='*80}")
    print("Benchmark completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

