"""
Continuous simulation example for transmission grid with renewable resources

This example demonstrates:
1. Creating a TransGridEnv with time series data loading
2. Attaching solar and wind resources to grid nodes
3. Running multi-step simulation with data-driven load and renewable profiles
4. Visualizing results: load, renewables, node injection, and line flow
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from powerzoo.envs.grid.trans import TransGridEnv
from powerzoo.envs.resource import SolarEnv, WindEnv
from powerzoo.case import load_case
from powerzoo.data import DataLoader, signals as S

# ============== 1. Setup Environment ==============
print("=" * 80)
print("Setting up Transmission Grid Environment with Renewable Resources")
print("=" * 80)

# Create environment with time series data
# Note: Data will be loaded from 2024-01-01 to 2024-01-31
case = load_case(5)  # IEEE 5-bus system
time_series = DataLoader().load_signals(
    [S.LOAD_ACTUAL_MW, S.SOLAR_AVAILABLE_MW, S.WIND_AVAILABLE_MW],
    source='gb',
    start_date='2024-01-01',
    end_date='2024-01-31',
    resample='30min',
)
env = TransGridEnv(
    case=case,
    solver_type='scipy',
    delta_t_minutes=30,
    start_date='2024-01-01',
    end_date='2024-01-31',
    time_series=time_series,
)

print(f"\nEnvironment configured:")
print(f"  Case: {case.__class__.__name__} (5-bus system)")
print(f"  Time step: {env.delta_t_minutes} minutes")
print(f"  Steps per day: {env.steps_per_day}")
print(f"  Data range: {env.start_date.date()} to {env.end_date.date()}")

# ============== 2. Attach Renewable Resources ==============
print("\n" + "=" * 80)
print("Attaching Renewable Resources")
print("=" * 80)

# Create and attach solar resource to bus 2
# Uses default 'Solar' column from grid's data_loader
solar = SolarEnv(
    parent=env,
    bus_id=2,
    capacity_mw=200.0,  # 50 MW solar farm
    delta_t_minutes=30
    # profile_column='Solar'  # Optional: specify custom column name
    # custom_data_loader=None  # Optional: provide custom DataLoader
)
print(f"\nAttached Solar: {solar.capacity_mw} MW at bus {solar.bus_id}")
print(f"  Using column: {solar.profile_column or solar._get_default_column()}")

# Create and attach wind resource to bus 3
# Uses default 'Wind' column from grid's data_loader
wind = WindEnv(
    parent=env,
    bus_id=3,
    capacity_mw=500.0,  # 100 MW wind farm
    delta_t_minutes=30
    # profile_column='Wind Offshore'  # Optional: e.g., only offshore wind
)
print(f"Attached Wind: {wind.capacity_mw} MW at bus {wind.bus_id}")
print(f"  Using column: {wind.profile_column or wind._get_default_column()}")

# ============== 3. Run Simulation ==============
print("\n" + "=" * 80)
print("Running Multi-Step Simulation")
print("=" * 80)

# Reset for day_id=0 (first day: 2024-01-01)
day_id = 0
state, info = env.reset(day_id=day_id)
print(f"\nReset to day_id={day_id} ({env._get_datetime_from_day_and_step(day_id, 0).date()})")

# Run for 24 time steps (12 hours with 30-min intervals)
n_steps = 48
results = {
    'time': [],
    'total_load_mw': [],
    'solar_output': [],
    'wind_output': [],
    'node_inj_mw': [],
    'line_flow_mw': [],
    'is_safe': []
}

print(f"\nSimulating {n_steps} steps (12 hours)...\n")
print(f"{'Step':<6} {'Time':<12} {'Load(MW)':<12} {'Solar(MW)':<12} {'Wind(MW)':<12} {'Safe':<6}")
print("-" * 60)

for step in range(n_steps):
    # Step environment (renewable resources will automatically update)
    state, reward, done, truncated, info = env.step({})
    
    # Get current time
    current_time = env._get_datetime_from_day_and_step(day_id, step)
    
    # Get load data
    load_data = env._get_current_load_data()
    total_load_mw = load_data.get(S.LOAD_ACTUAL_MW, 0.0)
    
    # Get renewable outputs (positive = generation injected into the grid)
    solar_output = solar.current_p_mw
    wind_output = wind.current_p_mw
    
    # Store results
    results['time'].append(current_time)
    results['total_load_mw'].append(total_load_mw)
    results['solar_output'].append(solar_output)
    results['wind_output'].append(wind_output)
    results['node_inj_mw'].append(state['nodes']['node_inj_mw'].values.copy())
    results['line_flow_mw'].append(state['lines']['line_flow_mw'].values.copy())
    results['is_safe'].append(state['is_safe'])
    
    # Print progress
    print(f"{step:<6} {current_time.strftime('%H:%M'):<12} {total_load_mw:<12.1f} "
          f"{solar_output:<12.1f} {wind_output:<12.1f} {'OK' if state['is_safe'] else 'WARN':<6}")

# ============== 4. Display Results ==============
print("\n" + "=" * 80)
print("Simulation Summary")
print("=" * 80)

avg_load = np.mean(results['total_load_mw'])
avg_solar = np.mean(results['solar_output'])
avg_wind = np.mean(results['wind_output'])
total_renewable = avg_solar + avg_wind
renewable_penetration = (total_renewable / avg_load * 100) if avg_load > 0 else 0

print(f"\nAverage Load: {avg_load:.1f} MW")
print(f"Average Solar: {avg_solar:.1f} MW")
print(f"Average Wind: {avg_wind:.1f} MW")
print(f"Total Renewable: {total_renewable:.1f} MW")
print(f"Renewable Penetration: {renewable_penetration:.1f}%")
print(f"Safety Status: {sum(results['is_safe'])}/{n_steps} steps safe")

# ============== 5. Visualization ==============
print("\n" + "=" * 80)
print("Generating Visualizations")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Transmission Grid Simulation Results - Day {day_id} ({results["time"][0].date()})', 
             fontsize=14, fontweight='bold')

# Convert time to hours for x-axis
time_hours = [(t.hour + t.minute / 60) for t in results['time']]

# Plot 1: Load and Renewable Generation
ax1 = axes[0, 0]
ax1.plot(time_hours, results['total_load_mw'], 'k-', linewidth=2, label='Total Load')
ax1.plot(time_hours, results['solar_output'], 'gold', linewidth=2, label='Solar')
ax1.plot(time_hours, results['wind_output'], 'skyblue', linewidth=2, label='Wind')
ax1.fill_between(time_hours, 0, results['solar_output'], alpha=0.3, color='gold')
ax1.fill_between(time_hours, 0, results['wind_output'], alpha=0.3, color='skyblue')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Power (MW)')
ax1.set_title('System Load and Renewable Generation')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Renewable Penetration
ax2 = axes[0, 1]
renewable_total = np.array(results['solar_output']) + np.array(results['wind_output'])
load_array = np.array(results['total_load_mw'], dtype=float)
penetration = np.divide(
    renewable_total,
    load_array,
    out=np.zeros_like(renewable_total, dtype=float),
    where=load_array > 0,
) * 100
ax2.plot(time_hours, penetration, 'green', linewidth=2)
ax2.fill_between(time_hours, 0, penetration, alpha=0.3, color='green')
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Penetration (%)')
ax2.set_title('Renewable Penetration')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, max(float(np.nanmax(penetration)) * 1.2, 1.0)])

# Plot 3: Node Power Injection
ax3 = axes[1, 0]
node_inj_array = np.array(results['node_inj_mw'])  # shape: (n_steps, n_nodes)
for i in range(node_inj_array.shape[1]):
    ax3.plot(time_hours, node_inj_array[:, i], linewidth=1.5, label=f'Node {i+1}')
ax3.set_xlabel('Time (hours)')
ax3.set_ylabel('Power Injection (MW)')
ax3.set_title('Node Power Injection')
ax3.legend(loc='upper left', ncol=2)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# Plot 4: Line Power Flow
ax4 = axes[1, 1]
line_flow_array = np.array(results['line_flow_mw'])  # shape: (n_steps, n_lines)
for i in range(line_flow_array.shape[1]):
    ax4.plot(time_hours, line_flow_array[:, i], linewidth=1.5, label=f'Line {i+1}')
ax4.set_xlabel('Time (hours)')
ax4.set_ylabel('Line Flow (MW)')
ax4.set_title('Line Power Flow')
ax4.legend(loc='upper left', ncol=2)
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

plt.tight_layout()

# Save figure
output_dir = os.path.join(os.path.dirname(__file__), 'x_RL01_output')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'continuous_simulation_results.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: {output_file}")

plt.show()

print("\n" + "=" * 80)
print("Simulation Complete")
print("=" * 80)
