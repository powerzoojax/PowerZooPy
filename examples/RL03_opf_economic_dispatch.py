"""Example: OPF-based Economic Dispatch for Transmission Grid

This example demonstrates:
1. OPF-based unit commitment and economic dispatch
2. Integration with renewable resources (solar, wind)
3. Line security constraints via PTDF
4. Cost-optimal generation scheduling

Uses the free SciPy/HiGHS DC-OPF backend by default.
"""

import numpy as np
import matplotlib.pyplot as plt
from powerzoo.envs.grid import TransGridEnv
from powerzoo.envs.resource import SolarEnv, WindEnv
from powerzoo.case import load_case
from powerzoo.data import DataLoader, signals as S

print("="*80)
print("OPF-based Economic Dispatch Example")
print("="*80)

# ====== Setup Environment ======

# Create transmission grid environment
time_series = DataLoader().load_signals(
    [S.LOAD_ACTUAL_MW, S.SOLAR_AVAILABLE_MW, S.WIND_AVAILABLE_MW],
    source='gb',
    start_date='2024-01-01',
    end_date='2024-01-03',
    resample='30min',
)
env = TransGridEnv(
    case=load_case(5),
    solver_type='scipy',
    start_date='2024-01-01',
    end_date='2024-01-03',
    delta_t_minutes=30,
    time_series=time_series,
)

print(f"\nGrid Configuration:")
print(f"  Nodes: {len(env.case.nodes)}")
print(f"  Lines: {len(env.case.lines)}")
print(f"  Units: {len(env.case.units)}")
print(f"  Date range: 2024-01-01 to 2024-01-03")
print(f"  Time step: 30 minutes")

# Display unit information
print(f"\nUnit Information:")
print(f"  {'Unit':<6} {'Bus':<6} {'P_min':<8} {'P_max':<8} {'MC':<8}")
print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
for i, unit in env.case.units.iterrows():
    print(f"  {int(unit['id']):<6} {int(unit['bus_id']):<6} {unit['p_min']:<8.1f} {unit['p_max']:<8.1f} {unit['mc_c']:<8.1f}")

# Attach renewable resources
solar = SolarEnv(capacity_mw=50.0)
wind = WindEnv(capacity_mw=100.0)

solar.attach(env, bus_id=2)
wind.attach(env, bus_id=3)

print(f"\nRenewable Resources:")
print(f"  Solar: 50 MW at bus 2")
print(f"  Wind: 100 MW at bus 3")

# ====== Run Simulation ======

print(f"\n{'='*80}")
print("Running 48-step Simulation (Day 1)")
print('='*80)

# Reset environment
state, info = env.reset(day_id=0)

# Storage for results
results = {
    'time_steps': [],
    'total_load_mw': [],
    'total_generation_mw': [],
    'generation_cost': [],
    'solar_gen': [],
    'wind_gen': [],
    'unit_outputs': [[] for _ in range(len(env.case.units))],
    'line_violations': [],
    'lmp': [[] for _ in range(len(env.case.nodes))],  # Nodal LMP ($/MWh)
}

# Run simulation
n_steps = 48  # One day
for step in range(n_steps):
    # Step environment (OPF solves economic dispatch internally)
    state, reward, done, truncated, info = env.step({})
    
    # Extract results
    results['time_steps'].append(step)
    results['total_load_mw'].append(state['nodes']['node_net_load_mw'].sum())
    results['total_generation_mw'].append(info['total_generation_mw'])
    results['generation_cost'].append(state.get('opf_cost', 0))
    # Renewable current_p_mw is positive generation injected into the grid.
    results['solar_gen'].append(solar.current_p_mw)
    results['wind_gen'].append(wind.current_p_mw)
    
    # Unit outputs
    for i, p in enumerate(info['unit_power_mw']):
        results['unit_outputs'][i].append(p)
    
    # Line violations
    n_violations = len(state['safety_info']['unsafe_line_ids'])
    results['line_violations'].append(n_violations)
    
    # Nodal LMP (Locational Marginal Price)
    if 'lmp' in state:
        for i, price in enumerate(state['lmp']):
            results['lmp'][i].append(price)
    
    # Print progress every 6 steps (3 hours)
    if step % 6 == 0:
        hour = step * 0.5
        avg_lmp = np.mean(state.get('lmp', [0])) if 'lmp' in state else 0
        print(f"  Step {step:2d} ({hour:4.1f}h): "
              f"Load={state['nodes']['node_net_load_mw'].sum():6.1f} MW, "
              f"Gen={info['total_generation_mw']:6.1f} MW, "
              f"Cost=${state.get('opf_cost', 0):7.1f}, "
              f"LMP=${avg_lmp:5.1f}/MWh, "
              f"Solar={solar.current_p_mw:4.1f} MW, "
              f"Wind={wind.current_p_mw:5.1f} MW, "
              f"Violations={n_violations}")

# ====== Summary ======

print(f"\n{'='*80}")
print("Simulation Summary")
print('='*80)

print(f"\nEconomic Metrics:")
print(f"  Total Generation Cost: ${sum(results['generation_cost']):.2f}")
print(f"  Average Cost per Step: ${np.mean(results['generation_cost']):.2f}")
print(f"  Average Generation: {np.mean(results['total_generation_mw']):.1f} MW")
print(f"  Average Load: {np.mean(results['total_load_mw']):.1f} MW")

print(f"\nRenewable Energy:")
print(f"  Average Solar: {np.mean(results['solar_gen']):.1f} MW")
print(f"  Average Wind: {np.mean(results['wind_gen']):.1f} MW")
print(f"  Total Renewable: {np.mean(results['solar_gen']) + np.mean(results['wind_gen']):.1f} MW")
print(f"  Renewable Penetration: {(np.mean(results['solar_gen']) + np.mean(results['wind_gen'])) / np.mean(results['total_generation_mw']) * 100:.1f}%")

print(f"\nLocational Marginal Price (LMP):")
if results['lmp'][0]:  # Check if LMP data exists
    avg_lmps = [np.mean(node_lmp) for node_lmp in results['lmp']]
    for i, avg_lmp in enumerate(avg_lmps):
        print(f"  Node {i+1}: ${avg_lmp:.2f}/MWh")
    print(f"  System Average: ${np.mean(avg_lmps):.2f}/MWh")

print(f"\nSafety:")
print(f"  Total Line Violations: {sum(results['line_violations'])}")
print(f"  Safe Steps: {sum([1 for v in results['line_violations'] if v == 0])}/{n_steps}")

# ====== Visualization ======

print(f"\n{'='*80}")
print("Generating Visualizations")
print('='*80)

fig, axes = plt.subplots(4, 1, figsize=(14, 13))

# Plot 1: Load and Generation
ax = axes[0]
time_hours = np.array(results['time_steps']) * 0.5
ax.plot(time_hours, results['total_load_mw'], label='Net Load', linewidth=2, color='red')
ax.plot(time_hours, results['total_generation_mw'], label='Total Generation', linewidth=2, color='blue')
ax.plot(time_hours, results['solar_gen'], label='Solar', linewidth=1.5, color='orange', linestyle='--')
ax.plot(time_hours, results['wind_gen'], label='Wind', linewidth=1.5, color='green', linestyle='--')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Power (MW)')
ax.set_title('Load, Generation, and Renewables')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Unit Dispatch
ax = axes[1]
bottom = np.zeros(len(time_hours))
colors = plt.cm.tab10(np.linspace(0, 1, len(env.case.units)))
for i, unit in enumerate(env.case.units.itertuples()):
    ax.fill_between(time_hours, bottom, bottom + results['unit_outputs'][i],
                     label=f'Unit {int(unit.id)} (MC=${unit.mc_c:.0f})',
                     alpha=0.7, color=colors[i])
    bottom += results['unit_outputs'][i]
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Power (MW)')
ax.set_title('Unit Dispatch Stack (Sorted by Merit Order)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Plot 3: Nodal Locational Marginal Price (LMP)
ax = axes[2]
if results['lmp'][0]:  # Check if LMP data exists
    lmp_colors = plt.cm.viridis(np.linspace(0, 1, len(results['lmp'])))
    for i, node_lmp in enumerate(results['lmp']):
        ax.plot(time_hours, node_lmp, label=f'Node {i+1}', 
                linewidth=1.5, color=lmp_colors[i], alpha=0.8)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('LMP ($/MWh)')
    ax.set_title('Locational Marginal Price (LMP) by Node')
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No LMP data available', 
            ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('LMP ($/MWh)')
    ax.set_title('Locational Marginal Price (LMP) by Node')

# Plot 4: Generation Cost
ax = axes[3]
ax.plot(time_hours, results['generation_cost'], linewidth=2, color='purple')
ax.fill_between(time_hours, results['generation_cost'], alpha=0.3, color='purple')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Cost ($/period)')
ax.set_title('Generation Cost')
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
import os
output_dir = 'examples/x_RL03_output'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'opf_economic_dispatch.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")
plt.show()

print(f"\n{'='*80}")
print("Simulation Complete")
print('='*80)

print(f"\nKey Insights:")
print(f"  1. OPF automatically dispatches units in merit order (lowest MC first)")
print(f"  2. Renewable energy reduces net load and generation cost")
print(f"  3. Line security constraints ensure safe operation")
print(f"  4. Nodal LMP reflects marginal generation cost + transmission congestion")
print(f"  5. Total cost = ${sum(results['generation_cost']):.2f} over 24 hours")
