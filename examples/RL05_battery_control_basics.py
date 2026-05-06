"""Example: Battery Control Basics

This example demonstrates:
1. Creating a scenario with battery storage
2. Inspecting observation and action spaces
3. Using FlattenWrapper for RL library compatibility
4. Manual battery control
5. Visualizing battery SOC and grid performance

No external RL library required for this basic example.
"""

import numpy as np
import matplotlib.pyplot as plt
from powerzoo.envs.power_env import PowerEnv
from powerzoo.wrappers.flatten import FlattenWrapper

print("="*80)
print("Battery Control Example")
print("="*80)

# ====== Step 1: Build PowerEnv with Battery ======

print("\n" + "="*80)
print("Step 1: Build PowerEnv with Battery")
print("="*80)

battery_config = {
    'name': 'IEEE5Bus-Battery',
    'description': 'Economic dispatch with battery storage control',
    'grid': {
        'type': 'transmission',
        'case': 'Case5',
        'start_date': '2024-01-01',
        'end_date': '2024-01-03',  # 2 days
        'delta_t_minutes': 30,
        'max_load_ratio': 0.9,
    },
    'resources': [
        {'type': 'solar', 'capacity_mw': 50, 'bus_id': 2},
        {'type': 'wind', 'capacity_mw': 100, 'bus_id': 3},
        {'type': 'battery', 'name': 'bat0', 'capacity_mw': 50, 'power_mw': 20, 'bus_id': 4,
         'efficiency': 0.95, 'soc_min': 0.1, 'soc_max': 0.9, 'initial_soc': 0.5}
    ],
    'reward': {
        'type': 'economic_dispatch',
        'cost_weight': 0.01,
    },
    'episode': {
        'max_steps': 96,  # 2 days * 48 steps/day (30min intervals)
    },
}

# Create base environment (Dict spaces)
env_base = PowerEnv(battery_config)
print("✓ PowerEnv ready: IEEE5Bus-Battery")

# ====== Step 2: Inspect Default Spaces ======

print("\n" + "="*80)
print("Step 2: Inspect Default Observation and Action Spaces")
print("="*80)
print("\n[Base Environment - Dict Spaces]")
print(f"Observation Space: {env_base.observation_space}")
print(f"Action Space: {env_base.action_space}")

# Show observation space structure
print("\nObservation Space Components:")
for key, space in env_base.observation_space.spaces.items():
    print(f"  - {key}: {space}")

# Show what can be controlled
print("\nControllable Actions:")
print("  - Battery: charge/discharge power in range [-20, 20] MW")
print("  - Grid: unit commitment (not controlled in this example)")

# ====== Step 3: Apply Flattening for RL Compatibility ======

print("\n" + "="*80)
print("Step 3: Apply FlattenWrapper for RL Libraries")
print("="*80)

# Create flattened environment with resource-specific control
env = FlattenWrapper(
    env_base,
    resource_names=['bat0'],  # Control battery by name
    obs_keys=['grid', 'resources', 'time']  # Include all observations
)


print(f"\n[Flattened Environment - Box Spaces]")
print(f"Observation Space: {env.observation_space}")
print(f"Action Space: {env.action_space}")
print(f"\n✓ Environment is now compatible with Stable-Baselines3, RLlib, CleanRL!")

# ====== Step 4: Test Manual Battery Control ======

print("\n" + "="*80)
print("Step 4: Test Manual Battery Control")
print("="*80)

obs, info = env.reset()
print(f"\nInitial observation shape: {obs.shape}")

# Manually control battery for a few steps
print("\nManual control examples:")
actions = [
    ("Idle (0 MW)", 0.0),
    ("Charge 10 MW", -10.0),
    ("Discharge 15 MW", 15.0),
    ("Charge 5 MW", -5.0),
]

for action_name, action_value in actions:
    action = np.array([action_value], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    battery_status = env_base.get_resource_status()['bat0']  # Use custom name
    soc = float(np.atleast_1d(battery_status['soc'])[0])
    power = float(np.atleast_1d(battery_status['current_p_mw'])[0])
    print(f"  {action_name:20s}: SOC={soc*100:5.1f}%, "
          f"Power={-power:6.1f} MW, Reward={reward:7.2f}")

# ====== Step 5: Run Full Simulation ======

print("\n" + "="*80)
print("Step 5: Run Simulation with Simple Policy")
print("="*80)

# Reset environment
obs, info = env.reset()

results = {
    'rewards': [],
    'soc': [],
    'battery_power_mw': [],
    'generation_cost': [],
    'total_load_mw': [],
}

print("\nRunning simulation for 96 steps (2 days)...")

# Simple policy: oscillate to demonstrate control
# Charge/discharge in cycles to show SOC variation
for step in range(96):
    battery_status = env_base.get_resource_status()['bat0']  # Use custom name
    soc = float(np.atleast_1d(battery_status['soc'])[0])
    
    # Improved policy: actively control battery
    # - Charge during low-load periods (assumed: steps 0-11, 48-59)
    # - Discharge during high-load periods (assumed: steps 12-23, 36-47)
    # - Or use simple cycling based on SOC
    hour_of_day = (step % 48) // 2  # 0-23 (hour of day in 30-min steps)
    
    if soc < 0.4:
        # Low SOC: must charge
        action_value = -15.0
    elif soc > 0.6:
        # High SOC: must discharge
        action_value = 10.0
    elif hour_of_day < 6 or hour_of_day > 22:
        # Night hours: charge (prepare for day)
        action_value = -10.0
    elif 9 <= hour_of_day <= 17:
        # Peak hours: discharge (support grid)
        action_value = 8.0
    else:
        # Transition periods: idle
        action_value = 0.0
    
    action = np.array([action_value], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    battery_status = env_base.get_resource_status()['bat0']  # Use custom name
    soc = float(np.atleast_1d(battery_status['soc'])[0])
    power = float(np.atleast_1d(battery_status['current_p_mw'])[0])
    
    results['rewards'].append(reward)
    results['soc'].append(soc)
    results['battery_power_mw'].append(-power)  # Negative to show charge/discharge
    results['generation_cost'].append(info.get('opf_cost', 0))
    
    if (step + 1) % 24 == 0:
        half_day = (step + 1) // 24
        avg_reward = np.mean(results['rewards'][-24:])
        avg_soc = np.mean(results['soc'][-24:])
        print(f"  Half-day {half_day}: Avg Reward={avg_reward:7.2f}, Avg SOC={avg_soc:.2%}")
    
    if terminated or truncated:
        break

print(f"\n✓ Simulation complete!")
print(f"  Total reward: {sum(results['rewards']):.2f}")
print(f"  Average SOC: {np.mean(results['soc']):.2%}")
print(f"  Total generation cost: ${sum(results['generation_cost']):.2f}")

# ====== Step 6: Visualization ======

print("\n" + "="*80)
print("Step 6: Generate Visualization")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Rewards
ax = axes[0, 0]
ax.plot(results['rewards'], linewidth=1, color='blue', alpha=0.7)
ax.set_xlabel('Time Step')
ax.set_ylabel('Reward')
ax.set_title('Reward over Time')
ax.grid(True, alpha=0.3)

# Plot 2: Battery SOC
ax = axes[0, 1]
ax.plot(np.array(results['soc']) * 100, linewidth=2, color='green')
ax.axhline(y=10, color='r', linestyle='--', label='SOC Min (10%)', linewidth=1)
ax.axhline(y=90, color='r', linestyle='--', label='SOC Max (90%)', linewidth=1)
ax.axhline(y=40, color='orange', linestyle=':', label='Low Threshold (40%)', linewidth=1)
ax.axhline(y=60, color='orange', linestyle=':', label='High Threshold (60%)', linewidth=1)
ax.set_xlabel('Time Step')
ax.set_ylabel('State of Charge (%)')
ax.set_title('Battery State of Charge')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: Battery Power
ax = axes[1, 0]
ax.plot(results['battery_power_mw'], linewidth=1.5, color='orange')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.fill_between(range(len(results['battery_power_mw'])), 
                 results['battery_power_mw'], 0,
                 where=np.array(results['battery_power_mw']) > 0,
                 alpha=0.3, color='red', label='Discharging')
ax.fill_between(range(len(results['battery_power_mw'])), 
                 results['battery_power_mw'], 0,
                 where=np.array(results['battery_power_mw']) < 0,
                 alpha=0.3, color='blue', label='Charging')
ax.set_xlabel('Time Step')
ax.set_ylabel('Power (MW)')
ax.set_title('Battery Charge/Discharge')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Generation Cost
ax = axes[1, 1]
ax.plot(results['generation_cost'], linewidth=1.5, color='purple')
ax.fill_between(range(len(results['generation_cost'])), 
                 results['generation_cost'], alpha=0.3, color='purple')
ax.set_xlabel('Time Step')
ax.set_ylabel('Cost ($)')
ax.set_title('Generation Cost')
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
import os
output_dir = 'examples/x_RL05_output'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'battery_control_basics.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_path}")

# ====== Summary ======

print("\n" + "="*80)
print("Summary")
print("="*80)

print(f"\nKey Takeaways:")
print(f"  1. Dict spaces can be inspected with env.observation_space")
print(f"  2. FlattenWrapper makes environments compatible with SB3/RLlib/CleanRL")
print(f"  3. Battery control action: continuous in [-20, 20] MW")
print(f"  4. SOC dynamics automatically handled by BatteryEnv")
print(f"  5. Simple threshold policy maintains SOC within bounds")

print(f"\nNext Steps:")
print(f"  - Try RL05_battery_control_sb3.py for RL training with Stable-Baselines3")
print(f"  - Tune reward function for better economics")
print(f"  - Experiment with different control policies")
print(f"  - Scale to larger grids with multiple batteries")

print("\n" + "="*80)
print("Example Complete!")
print("="*80)

