"""Profile Three-Phase Environment Performance

Identify bottlenecks in dist_3phase.py cal_pf method.
"""
import time
import sys
import os
import cProfile
import pstats
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from powerzoo.envs.grid.dist_3phase import DistGrid3PhaseEnv
from powerzoo.case.distribution.Case123 import Case123


def profile_cal_pf():
    """Profile the cal_pf method"""
    env = DistGrid3PhaseEnv(case=Case123())
    
    # Warmup
    for _ in range(5):
        nodes_df, lines_df = env.cal_pf(df=True)
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    for _ in range(50):
        nodes_df, lines_df = env.cal_pf(df=True)
    
    profiler.disable()
    
    # Print results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    print(s.getvalue())


def time_components():
    """Time individual components"""
    env = DistGrid3PhaseEnv(case=Case123())
    
    # Warmup
    for _ in range(5):
        nodes_df, lines_df = env.cal_pf(df=True)
    
    n_runs = 100
    
    # Time _get_3phase_loads
    times_loads = []
    for _ in range(n_runs):
        start = time.perf_counter()
        P_3ph_pu, Q_3ph_pu = env._get_3phase_loads()
        times_loads.append((time.perf_counter() - start) * 1000)
    avg_loads = sum(times_loads) / len(times_loads)
    
    # Time power flow core
    P_3ph_pu, Q_3ph_pu = env._get_3phase_loads()
    times_pf = []
    for _ in range(n_runs):
        start = time.perf_counter()
        from powerzoo.envs.grid.cal_pf_dist_3phase import run_3phase_bfs_power_flow
        result = run_3phase_bfs_power_flow(
            env.topo3ph, P_3ph_pu, Q_3ph_pu, v_ref_mag=env.v_ref_mag
        )
        times_pf.append((time.perf_counter() - start) * 1000)
    avg_pf = sum(times_pf) / len(times_pf)
    
    # Time loss calculation
    times_loss = []
    for _ in range(n_runs):
        start = time.perf_counter()
        from powerzoo.envs.grid.cal_pf_dist_3phase import calculate_3phase_losses
        P_loss, Q_loss = calculate_3phase_losses(env.topo3ph, result)
        times_loss.append((time.perf_counter() - start) * 1000)
    avg_loss = sum(times_loss) / len(times_loss)
    
    # Time DataFrame building
    from powerzoo.envs.grid.cal_pf_dist_3phase import calculate_3phase_losses, get_phase_results
    P_loss, Q_loss = calculate_3phase_losses(env.topo3ph, result)
    times_nodes_df = []
    times_lines_df = []
    for _ in range(n_runs):
        start = time.perf_counter()
        nodes_df = env._build_nodes_df(result, P_3ph_pu, Q_3ph_pu)
        times_nodes_df.append((time.perf_counter() - start) * 1000)
        
        start = time.perf_counter()
        lines_df = env._build_lines_df(result, P_loss, Q_loss)
        times_lines_df.append((time.perf_counter() - start) * 1000)
    avg_nodes_df = sum(times_nodes_df) / len(times_nodes_df)
    avg_lines_df = sum(times_lines_df) / len(times_lines_df)
    
    # Time full cal_pf
    times_full = []
    for _ in range(n_runs):
        start = time.perf_counter()
        nodes_df, lines_df = env.cal_pf(df=True)
        times_full.append((time.perf_counter() - start) * 1000)
    avg_full = sum(times_full) / len(times_full)
    
    print("=" * 80)
    print("Component Timing Analysis (100 runs each)")
    print("=" * 80)
    print(f"\n{'Component':<30} {'Time (ms)':<15} {'% of Total':<15}")
    print("-" * 80)
    print(f"{'_get_3phase_loads()':<30} {avg_loads:>10.3f} ms  {avg_loads/avg_full*100:>10.1f}%")
    print(f"{'run_3phase_bfs_power_flow()':<30} {avg_pf:>10.3f} ms  {avg_pf/avg_full*100:>10.1f}%")
    print(f"{'calculate_3phase_losses()':<30} {avg_loss:>10.3f} ms  {avg_loss/avg_full*100:>10.1f}%")
    print(f"{'_build_nodes_df()':<30} {avg_nodes_df:>10.3f} ms  {avg_nodes_df/avg_full*100:>10.1f}%")
    print(f"{'_build_lines_df()':<30} {avg_lines_df:>10.3f} ms  {avg_lines_df/avg_full*100:>10.1f}%")
    print(f"{'Total cal_pf(df=True)':<30} {avg_full:>10.3f} ms  {100:>10.1f}%")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("Running component timing analysis...")
    time_components()
    
    print("\n\nRunning cProfile analysis...")
    profile_cal_pf()

