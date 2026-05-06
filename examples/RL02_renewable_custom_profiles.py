"""
Advanced renewable resource configuration examples

This example demonstrates:
1. Using default Solar/Wind columns
2. Using custom column names (e.g., 'Wind Offshore' only)
3. Comparing different renewable configurations
4. Multi-day simulation to show daily variations
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from powerzoo.envs.grid.trans import TransGridEnv
from powerzoo.envs.resource import SolarEnv, WindEnv
from powerzoo.case import load_case

# ============== Setup ==============
print("=" * 80)
print("Advanced Renewable Resource Configuration Example")
print("=" * 80)

# Create environment
case = load_case(5)
env = TransGridEnv(
    case=case,
    delta_t_minutes=30,
    start_date='2024-01-01',
    end_date='2024-01-07',  # One week
    load_columns=['ActualDemand', 'Solar', 'Wind Offshore', 'Wind Onshore']
    # Note: 'Wind' will be automatically computed as Wind Offshore + Wind Onshore
)

print(f"\nEnvironment: {case.__class__.__name__}")
print(f"  Time step: {env.delta_t_minutes} minutes")
print(f"  Date range: {env.start_date.date()} to {env.end_date.date()}")

# ============== Configure Different Renewable Resources ==============
print("\n" + "=" * 80)
print("Configuring Renewable Resources")
print("=" * 80)

# Configuration 1: Default Solar (uses 'Solar' column)
solar = SolarEnv(
    parent=env,
    bus_id=2,
    capacity_mw=50.0,
    delta_t_minutes=30
)
print(f"\n1. Default Solar at bus {solar.bus_id}")
print(f"   Column: {solar.profile_column or solar._get_default_column()}")
print(f"   Capacity: {solar.capacity_mw} MW")

# Configuration 2: Default Wind (uses 'Wind' = Wind Offshore + Wind Onshore)
wind_total = WindEnv(
    parent=env,
    bus_id=3,
    capacity_mw=100.0,
    delta_t_minutes=30
)
print(f"\n2. Total Wind at bus {wind_total.bus_id}")
print(f"   Column: {wind_total.profile_column or wind_total._get_default_column()}")
print(f"   Capacity: {wind_total.capacity_mw} MW")

# Configuration 3: Custom column - Wind Offshore only
wind_offshore = WindEnv(
    parent=env,
    bus_id=4,
    capacity_mw=60.0,
    profile_column='Wind Offshore',  # Custom column
    delta_t_minutes=30
)
print(f"\n3. Wind Offshore (custom) at bus {wind_offshore.bus_id}")
print(f"   Column: {wind_offshore.profile_column}")
print(f"   Capacity: {wind_offshore.capacity_mw} MW")

# Configuration 4: Custom column - Wind Onshore only
wind_onshore = WindEnv(
    parent=env,
    bus_id=5,
    capacity_mw=40.0,
    profile_column='Wind Onshore',  # Custom column
    delta_t_minutes=30
)
print(f"\n4. Wind Onshore (custom) at bus {wind_onshore.bus_id}")
print(f"   Column: {wind_onshore.profile_column}")
print(f"   Capacity: {wind_onshore.capacity_mw} MW")

# ============== Multi-Day Simulation ==============
print("\n" + "=" * 80)
print("Running Multi-Day Simulation")
print("=" * 80)

n_days = 3  # Simulate 3 days
steps_per_day = env.steps_per_day
results = {
    'day': [],
    'time_step': [],
    'hour': [],
    'load': [],
    'solar': [],
    'wind_total': [],
    'wind_offshore': [],
    'wind_onshore': [],
    'total_renewable': []
}

for day_id in range(n_days):
    print(f"\n{'='*60}")
    print(f"Day {day_id}: {env._get_datetime_from_day_and_step(day_id, 0).date()}")
    print(f"{'='*60}")
    
    # Reset for new day
    env.reset(day_id=day_id)
    
    # Run through the day
    for step in range(steps_per_day):
        # Step environment
        state, reward, done, truncated, info = env.step({})
        
        # Get current time
        current_datetime = env._get_datetime_from_day_and_step(day_id, step)
        hour = current_datetime.hour + current_datetime.minute / 60
        
        # Get data
        load_data = env._get_current_load_data()
        total_load_mw = load_data.get('ActualDemand', 0.0)
        
        # Store results
        results['day'].append(day_id)
        results['time_step'].append(step)
        results['hour'].append(hour)
        results['load'].append(total_load_mw)
        results['solar'].append(-solar.current_p_mw)
        results['wind_total'].append(-wind_total.current_p_mw)
        results['wind_offshore'].append(-wind_offshore.current_p_mw)
        results['wind_onshore'].append(-wind_onshore.current_p_mw)
        results['total_renewable'].append(-solar.current_p_mw - wind_total.current_p_mw - 
                                         wind_offshore.current_p_mw - wind_onshore.current_p_mw)
    
    # Print daily summary
    day_mask = np.array(results['day']) == day_id
    print(f"\nDaily Summary:")
    print(f"  Avg Load: {np.mean(np.array(results['load'])[day_mask]):.1f} MW")
    print(f"  Avg Solar: {np.mean(np.array(results['solar'])[day_mask]):.1f} MW")
    print(f"  Avg Wind Total: {np.mean(np.array(results['wind_total'])[day_mask]):.1f} MW")
    print(f"  Avg Wind Offshore: {np.mean(np.array(results['wind_offshore'])[day_mask]):.1f} MW")
    print(f"  Avg Wind Onshore: {np.mean(np.array(results['wind_onshore'])[day_mask]):.1f} MW")
    print(f"  Total Renewable: {np.mean(np.array(results['total_renewable'])[day_mask]):.1f} MW")

# ============== Visualization ==============
print("\n" + "=" * 80)
print("Generating Visualizations")
print("=" * 80)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Multi-Day Renewable Resource Simulation', fontsize=14, fontweight='bold')

# Create continuous time axis (hours from start)
time_hours = np.array([d * 24 + h for d, h in zip(results['day'], results['hour'])])

# Plot 1: Load and Total Renewables
ax1 = axes[0]
ax1.plot(time_hours, results['load'], 'k-', linewidth=2, label='Load', alpha=0.7)
ax1.plot(time_hours, results['total_renewable'], 'g-', linewidth=2, label='Total Renewable')
ax1.fill_between(time_hours, 0, results['total_renewable'], alpha=0.3, color='green')
ax1.set_ylabel('Power (MW)')
ax1.set_title('System Load and Total Renewable Generation')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, n_days * 24])

# Add day separators
for day in range(1, n_days):
    ax1.axvline(x=day*24, color='gray', linestyle='--', alpha=0.5)

# Plot 2: Renewable Breakdown
ax2 = axes[1]
ax2.plot(time_hours, results['solar'], 'gold', linewidth=2, label='Solar (50 MW)')
ax2.plot(time_hours, results['wind_total'], 'skyblue', linewidth=2, label='Wind Total (100 MW)')
ax2.plot(time_hours, results['wind_offshore'], 'navy', linewidth=1.5, label='Wind Offshore (60 MW)')
ax2.plot(time_hours, results['wind_onshore'], 'teal', linewidth=1.5, label='Wind Onshore (40 MW)')
ax2.set_ylabel('Power (MW)')
ax2.set_title('Renewable Generation by Type')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, n_days * 24])

for day in range(1, n_days):
    ax2.axvline(x=day*24, color='gray', linestyle='--', alpha=0.5)

# Plot 3: Renewable Penetration
ax3 = axes[2]
penetration = np.array(results['total_renewable']) / np.array(results['load']) * 100
ax3.plot(time_hours, penetration, 'green', linewidth=2)
ax3.fill_between(time_hours, 0, penetration, alpha=0.3, color='green')
ax3.set_xlabel('Time (hours from start)')
ax3.set_ylabel('Penetration (%)')
ax3.set_title('Renewable Energy Penetration')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, n_days * 24])

for day in range(1, n_days):
    ax3.axvline(x=day*24, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()

# Save figure
output_dir = os.path.join(os.path.dirname(__file__), 'x_RL02_output')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'multi_day_renewable_simulation.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: {output_file}")

plt.show()

print("\n" + "=" * 80)
print("Simulation Complete")
print("=" * 80)
print("\nKey Takeaways:")
print("1. Default columns ('Solar', 'Wind') work out-of-the-box")
print("2. Custom columns ('Wind Offshore', 'Wind Onshore') allow fine-grained control")
print("3. Daily profiles vary naturally from the time series data")
print("4. Multiple renewable resources can coexist on different buses")

