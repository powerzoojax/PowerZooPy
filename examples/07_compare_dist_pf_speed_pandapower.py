"""Performance Comparison: PowerZoo FBS vs pandapower

This script compares the power flow calculation between:
1. PowerZoo: Forward-Backward Sweep (matrix-based)
2. pandapower: Newton-Raphson (industry standard)

Compares both speed and accuracy.
"""
import sys
import time
import warnings
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')

try:
    import pandapower as pp
    PANDAPOWER_AVAILABLE = True
except ImportError:
    PANDAPOWER_AVAILABLE = False
    print("Warning: pandapower not installed. Run: pip install pandapower")


def convert_case33bw_to_pandapower():
    """Convert Case33bw to pandapower network
    
    Returns:
        pandapower network object
    """
    from powerzoo.case.distribution.Case33bw import Case33bw
    
    case = Case33bw()
    case.init()
    
    # Get base values
    baseMVA = case.baseMVA  # 100.0
    baseKV = case.baseKV    # 12.66
    z_base = baseKV ** 2 / baseMVA  # Ohms
    
    # Create empty network
    net = pp.create_empty_network(sn_mva=baseMVA)
    
    # Add buses (0-indexed in pandapower)
    for idx, row in case.nodes.iterrows():
        bus_type = 'b'  # PQ bus
        if row['type'] == 3:
            bus_type = 'b'  # Will add ext_grid
        pp.create_bus(net, vn_kv=baseKV, name=f"Bus {int(row['id'])}")
    
    # Add external grid at bus 0 (slack bus)
    pp.create_ext_grid(net, bus=0, vm_pu=1.0, name="Grid Connection")
    
    # Add loads (skip bus 0 which has no load)
    for idx, row in case.loads.iterrows():
        bus_id = int(row['bus_id']) - 1  # Convert to 0-indexed
        p_mw = row['Pd']
        q_mvar = row['Qd']
        if p_mw > 0 or q_mvar > 0:
            pp.create_load(net, bus=bus_id, p_mw=p_mw, q_mvar=q_mvar, 
                          name=f"Load at Bus {bus_id + 1}")
    
    # Add lines (only in-service lines, status=1)
    lines_df = case.lines
    for idx, row in lines_df.iterrows():
        if row['status'] != 1:
            continue
            
        from_bus = int(row['from']) - 1  # Convert to 0-indexed
        to_bus = int(row['to']) - 1
        
        # Convert impedance from p.u. to Ohms
        r_ohm = row['r'] * z_base
        x_ohm = row['x'] * z_base
        
        # Use create_line_from_parameters with length_km=1
        # This way r_ohm_per_km = r_ohm, x_ohm_per_km = x_ohm
        pp.create_line_from_parameters(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=1.0,
            r_ohm_per_km=r_ohm,
            x_ohm_per_km=x_ohm,
            c_nf_per_km=0,  # No line charging
            max_i_ka=1.0,   # Arbitrary high value
            name=f"Line {int(row['from'])}-{int(row['to'])}"
        )
    
    return net


def benchmark_pandapower(net, n_runs=100, warmup=10):
    """Benchmark pandapower power flow"""
    
    # Warmup
    for _ in range(warmup):
        pp.runpp(net, algorithm='nr', init='flat')
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        pp.runpp(net, algorithm='nr', init='flat')
    end = time.perf_counter()
    
    avg_time = (end - start) / n_runs * 1000  # ms
    
    return {
        'avg_time_ms': avg_time,
        'throughput': n_runs / (end - start),
        'v_mag': net.res_bus.vm_pu.values.copy(),
        'v_min': net.res_bus.vm_pu.min(),
        'v_max': net.res_bus.vm_pu.max(),
        'p_loss_mw': net.res_line.pl_mw.sum(),
        'q_loss_mvar': net.res_line.ql_mvar.sum(),
    }


def benchmark_powerzoo(n_runs=100, warmup=10):
    """Benchmark PowerZoo FBS power flow"""
    from powerzoo.envs.grid import DistGridEnv
    
    env = DistGridEnv()
    
    # Warmup
    for _ in range(warmup):
        env.cal_pf(df=True)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        nodes_df, lines_df = env.cal_pf(df=True)
    end = time.perf_counter()
    
    avg_time = (end - start) / n_runs * 1000  # ms
    p_loss, q_loss = env.get_total_loss(lines_df)
    
    return {
        'avg_time_ms': avg_time,
        'throughput': n_runs / (end - start),
        'v_mag': nodes_df['v_mag'].values.copy(),
        'v_min': nodes_df['v_mag'].min(),
        'v_max': nodes_df['v_mag'].max(),
        'p_loss_mw': p_loss,
        'q_loss_mvar': q_loss,
    }


def main():
    print("=" * 90)
    print("Power Flow Performance Comparison")
    print("PowerZoo (FBS, Matrix) vs pandapower (Newton-Raphson)")
    print("=" * 90)
    
    if not PANDAPOWER_AVAILABLE:
        print("\nError: pandapower is required for this comparison.")
        print("Install with: pip install pandapower")
        return
    
    n_runs = 100
    warmup = 20
    
    # Convert case to pandapower
    print("\nConverting Case33bw to pandapower format...")
    net = convert_case33bw_to_pandapower()
    print(f"  Buses: {len(net.bus)}")
    print(f"  Lines: {len(net.line)}")
    print(f"  Loads: {len(net.load)}")
    
    # Benchmark pandapower
    print(f"\n{'='*90}")
    print(f"Benchmarking pandapower (Newton-Raphson)... ({n_runs} runs, {warmup} warmup)")
    print("=" * 90)
    pp_result = benchmark_pandapower(net, n_runs, warmup)
    
    print(f"\n  Average time: {pp_result['avg_time_ms']:.3f} ms")
    print(f"  Throughput:   {pp_result['throughput']:.1f} runs/second")
    print(f"  Min voltage:  {pp_result['v_min']:.4f} p.u.")
    print(f"  Max voltage:  {pp_result['v_max']:.4f} p.u.")
    print(f"  P loss:       {pp_result['p_loss_mw']:.4f} MW")
    print(f"  Q loss:       {pp_result['q_loss_mvar']:.4f} MVAr")
    
    # Benchmark PowerZoo
    print(f"\n{'='*90}")
    print(f"Benchmarking PowerZoo (FBS, Matrix)... ({n_runs} runs, {warmup} warmup)")
    print("=" * 90)
    pz_result = benchmark_powerzoo(n_runs, warmup)
    
    print(f"\n  Average time: {pz_result['avg_time_ms']:.3f} ms")
    print(f"  Throughput:   {pz_result['throughput']:.1f} runs/second")
    print(f"  Min voltage:  {pz_result['v_min']:.4f} p.u.")
    print(f"  Max voltage:  {pz_result['v_max']:.4f} p.u.")
    print(f"  P loss:       {pz_result['p_loss_mw']:.4f} MW")
    print(f"  Q loss:       {pz_result['q_loss_mvar']:.4f} MVAr")
    
    # Comparison
    print(f"\n{'='*90}")
    print("COMPARISON SUMMARY")
    print("=" * 90)
    
    speedup = pp_result['avg_time_ms'] / pz_result['avg_time_ms']
    v_diff = np.abs(pp_result['v_mag'] - pz_result['v_mag'])
    max_v_diff = v_diff.max()
    avg_v_diff = v_diff.mean()
    
    print(f"\n{'Metric':<25} {'pandapower (NR)':<20} {'PowerZoo (FBS)':<20} {'Comparison':<20}")
    print("-" * 85)
    print(f"{'Average Time':<25} {pp_result['avg_time_ms']:.3f} ms{'':<12} {pz_result['avg_time_ms']:.3f} ms{'':<12} {'PowerZoo ' + f'{speedup:.1f}x faster' if speedup > 1 else 'pandapower faster'}")
    print(f"{'Throughput':<25} {pp_result['throughput']:.1f} /s{'':<12} {pz_result['throughput']:.1f} /s{'':<12}")
    print(f"{'Min Voltage':<25} {pp_result['v_min']:.4f} p.u.{'':<8} {pz_result['v_min']:.4f} p.u.{'':<8} Δ={abs(pp_result['v_min']-pz_result['v_min']):.4f}")
    print(f"{'Max Voltage':<25} {pp_result['v_max']:.4f} p.u.{'':<8} {pz_result['v_max']:.4f} p.u.{'':<8} Δ={abs(pp_result['v_max']-pz_result['v_max']):.4f}")
    print(f"{'P Loss':<25} {pp_result['p_loss_mw']:.4f} MW{'':<10} {pz_result['p_loss_mw']:.4f} MW{'':<10} Δ={abs(pp_result['p_loss_mw']-pz_result['p_loss_mw']):.4f}")
    print(f"{'Q Loss':<25} {pp_result['q_loss_mvar']:.4f} MVAr{'':<7} {pz_result['q_loss_mvar']:.4f} MVAr{'':<7} Δ={abs(pp_result['q_loss_mvar']-pz_result['q_loss_mvar']):.4f}")
    
    # Voltage comparison by bus
    print(f"\n{'='*90}")
    print("Voltage Comparison by Bus")
    print("=" * 90)
    
    print(f"\n{'Bus':>5} {'pandapower':>12} {'PowerZoo':>12} {'Diff':>12} {'Error %':>12}")
    print("-" * 55)
    
    for i in range(len(pp_result['v_mag'])):
        pp_v = pp_result['v_mag'][i]
        pz_v = pz_result['v_mag'][i]
        diff = pz_v - pp_v
        error_pct = abs(diff) / pp_v * 100
        marker = " *" if error_pct > 1.0 else ""
        print(f"{i+1:>5} {pp_v:>12.4f} {pz_v:>12.4f} {diff:>12.4f} {error_pct:>11.2f}%{marker}")
    
    print(f"\n{'-'*55}")
    print(f"Average voltage error: {avg_v_diff / pp_result['v_mag'].mean() * 100:.3f}%")
    print(f"Max voltage error: {max_v_diff / pp_result['v_min'] * 100:.3f}%")
    
    # Assessment
    print(f"\n{'='*90}")
    print("ASSESSMENT")
    print("=" * 90)
    
    if speedup >= 2.0:
        print(f"\n✓ PowerZoo is {speedup:.1f}x faster than pandapower!")
    elif speedup >= 1.0:
        print(f"\n○ PowerZoo is {speedup:.1f}x faster than pandapower")
    else:
        print(f"\n△ pandapower is {1/speedup:.1f}x faster than PowerZoo")
    
    if max_v_diff < 0.01:
        print("✓ Excellent accuracy! Max voltage difference < 0.01 p.u.")
    elif max_v_diff < 0.02:
        print("○ Good accuracy. Max voltage difference < 0.02 p.u.")
    else:
        print(f"△ Noticeable difference. Max voltage diff = {max_v_diff:.4f} p.u.")
    
    print(f"\n{'='*90}")
    print("Summary:")
    print(f"  - PowerZoo FBS uses sparse matrix operations, optimized for radial networks")
    print(f"  - pandapower uses Newton-Raphson, a general-purpose iterative method")
    print(f"  - For radial distribution networks, FBS can be faster and sufficiently accurate")
    print("=" * 90)


if __name__ == "__main__":
    main()

