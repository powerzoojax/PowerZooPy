"""Example: Dual-Battery Coordinated Control (Rule-Based)

This example demonstrates:
1. Creating a scenario with 2 batteries
2. Using a single agent to control 2 batteries
3. Rule-based coordinated control strategy
4. Visualizing dual-battery operation status

Scenario Setup:
- bat0: Peak-valley arbitrage strategy (charge at low price, discharge at high price)
- bat1: Load balancing strategy (peak shaving and valley filling)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from powerzoo.envs.power_env import PowerEnv
from powerzoo.wrappers.flatten import FlattenWrapper

print("=" * 80)
print("Dual-Battery Coordinated Control Example (Rule-Based)")
print("=" * 80)

# ====== Step 1: Register Scenario with Dual Batteries ======

print("\n" + "=" * 80)
print("Step 1: Register Scenario with Dual Batteries")
print("=" * 80)

dual_battery_config = {
    'name': 'IEEE5Bus-DualBattery',
    'description': 'Dual-battery coordinated control: one for arbitrage, one for balancing',
    'grid': {
        'type': 'transmission',
        'case': 'Case5',
        'start_date': '2024-01-01',
        'end_date': '2024-01-08',  # 7 days
        'delta_t_minutes': 60,  # 1-hour intervals
        'max_load_ratio': 0.9,
    },
    'resources': [
        {'type': 'solar', 'capacity_mw': 50, 'bus_id': 2},
        {'type': 'wind', 'capacity_mw': 100, 'bus_id': 3},
        # Battery 0: For peak-valley arbitrage
        {'type': 'battery', 'name': 'bat0', 'capacity_mw': 50, 'power_mw': 20, 'bus_id': 4,
         'efficiency': 0.95, 'soc_min': 0.1, 'soc_max': 0.9, 'initial_soc': 0.5},
        # Battery 1: For load balancing
        {'type': 'battery', 'name': 'bat1', 'capacity_mw': 40, 'power_mw': 15, 'bus_id': 5,
         'efficiency': 0.95, 'soc_min': 0.1, 'soc_max': 0.9, 'initial_soc': 0.5},
    ],
    'reward': {
        'type': 'battery_arbitrage',
        'peak_hours': [10, 11, 12, 13, 14, 15, 16, 17, 18],
        'off_peak_hours': [0, 1, 2, 3, 4, 5],
        'arbitrage_weight': 1.0,
        'soc_penalty_weight': 0.05,
    },
    'episode': {
        'max_steps': 168,  # 7 days * 24 hours
    },
}

print("Using PowerEnv with dual_battery_config")

# ====== Step 2: Inspect Space Dimensions ======

print("\n" + "=" * 80)
print("Step 2: Inspect Observation and Action Spaces")
print("=" * 80)

# Create base environment
env_base = PowerEnv(dual_battery_config)
print("\n[Base Environment - Dict Space]")
print(f"Observation Space: {env_base.observation_space}")
print(f"Action Space: {env_base.action_space}")

# Show controllable resources
print("\nControllable Resources:")
print(f"  Resources in environment: {list(env_base.resources.keys())}")
print(f"  - bat0 (Battery 0): Power range [-20, 20] MW, Capacity 50 MWh")
print(f"  - bat1 (Battery 1): Power range [-15, 15] MW, Capacity 40 MWh")

# ====== Step 3: Apply FlattenWrapper for Dual-Battery Control ======

print("\n" + "=" * 80)
print("Step 3: Apply FlattenWrapper for Dual-Battery Control")
print("=" * 80)

# Create flattened environment, controlling two batteries
env = FlattenWrapper(
    env_base,
    resource_names=['bat0', 'bat1'],  # Control two batteries
    obs_keys=['grid', 'resources', 'time']
)

# Important: Update env_base to point to wrapped environment
env_base = env.env

print(f"\n[Flattened Environment - Box Space]")
print(f"Observation Space: {env.observation_space}")
print(f"  Shape: {env.observation_space.shape}")
print(f"Action Space: {env.action_space}")
print(f"  Shape: {env.action_space.shape}")
print(f"  bat0 bounds: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}] MW")
print(f"  bat1 bounds: [{env.action_space.low[1]:.1f}, {env.action_space.high[1]:.1f}] MW")
print(f"\nAction space dimension is 2, successfully supports dual-battery control!")

# ====== Step 4: Test Manual Control ======

print("\n" + "=" * 80)
print("Step 4: Test Manual Dual-Battery Control")
print("=" * 80)

obs, info = env.reset()
print(f"\nInitial observation shape: {obs.shape}")

# Manual control test
print("\nManual control examples:")
test_actions = [
    ("Both batteries idle", [0.0, 0.0]),
    ("bat0 charge 10MW, bat1 charge 5MW", [-10.0, -5.0]),
    ("bat0 discharge 15MW, bat1 discharge 10MW", [15.0, 10.0]),
    ("bat0 charge 5MW, bat1 discharge 8MW", [-5.0, 8.0]),
]

for action_name, action_values in test_actions:
    action = np.array(action_values, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    bat0_status = env_base.get_resource_status()['bat0']
    bat1_status = env_base.get_resource_status()['bat1']
    
    soc0 = float(np.atleast_1d(bat0_status['soc'])[0])
    soc1 = float(np.atleast_1d(bat1_status['soc'])[0])
    power0 = float(np.atleast_1d(bat0_status['current_p_mw'])[0])
    power1 = float(np.atleast_1d(bat1_status['current_p_mw'])[0])
    
    print(f"  {action_name:30s}: "
          f"bat0[SOC={soc0*100:5.1f}%, P={power0:6.1f}MW], "
          f"bat1[SOC={soc1*100:5.1f}%, P={power1:6.1f}MW], "
          f"Reward={reward:7.2f}")

# ====== Step 5: Run Rule-Based Coordinated Control ======

print("\n" + "=" * 80)
print("Step 5: Run Rule-Based Coordinated Control Strategy")
print("=" * 80)

# Reset environment
obs, info = env.reset()

results = {
    'rewards': [],
    'soc0': [],
    'soc1': [],
    'power0': [],
    'power1': [],
    'generation_cost': [],
}

print("\nRunning 168-step simulation (7 days)...")
print("\nStrategy description:")
print("  bat0: Peak-valley arbitrage - charge at dawn, discharge in afternoon")
print("  bat1: Load balancing - respond to demand, peak shaving and valley filling\n")

for step in range(168):
    bat0_status = env_base.get_resource_status()['bat0']
    bat1_status = env_base.get_resource_status()['bat1']
    
    soc0 = float(np.atleast_1d(bat0_status['soc'])[0])
    soc1 = float(np.atleast_1d(bat1_status['soc'])[0])
    
    hour_of_day = step % 24
    
    # === bat0 strategy: Peak-valley arbitrage ===
    if soc0 < 0.2:
        # SOC too low, must charge
        action0 = -15.0
    elif soc0 > 0.8:
        # SOC too high, must discharge
        action0 = 12.0
    elif hour_of_day in [0, 1, 2, 3, 4, 5]:
        # Dawn (0-5): Low-price charging
        action0 = -15.0
    elif hour_of_day in [14, 15, 16, 17, 18, 19]:
        # Afternoon peak (14-19): High-price discharging
        action0 = 18.0
    else:
        # Other periods: Standby
        action0 = 0.0
    
    # === bat1 strategy: Load balancing ===
    if soc1 < 0.2:
        # SOC too low, must charge
        action1 = -10.0
    elif soc1 > 0.8:
        # SOC too high, must discharge
        action1 = 8.0
    elif hour_of_day in [10, 11, 12, 13]:
        # Noon peak: Moderate discharge to support grid
        action1 = 10.0 if soc1 > 0.4 else 5.0
    elif hour_of_day in [6, 7, 8]:
        # Morning peak: Discharge support
        action1 = 8.0 if soc1 > 0.4 else 0.0
    elif hour_of_day in [20, 21, 22, 23]:
        # Evening low load: Charge preparation
        action1 = -8.0 if soc1 < 0.6 else 0.0
    else:
        # Other periods: Small balancing
        if soc1 < 0.4:
            action1 = -5.0
        elif soc1 > 0.6:
            action1 = 5.0
        else:
            action1 = 0.0
    
    # Execute action
    action = np.array([action0, action1], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Get post-execution status
    bat0_status = env_base.get_resource_status()['bat0']
    bat1_status = env_base.get_resource_status()['bat1']
    
    soc0 = float(np.atleast_1d(bat0_status['soc'])[0])
    soc1 = float(np.atleast_1d(bat1_status['soc'])[0])
    power0 = float(np.atleast_1d(bat0_status['current_p_mw'])[0])
    power1 = float(np.atleast_1d(bat1_status['current_p_mw'])[0])
    
    results['rewards'].append(reward)
    results['soc0'].append(soc0)
    results['soc1'].append(soc1)
    results['power0'].append(power0)
    results['power1'].append(power1)
    results['generation_cost'].append(info.get('opf_cost', 0))
    
    if (step + 1) % 24 == 0:
        day = (step + 1) // 24
        avg_reward = np.mean(results['rewards'][-24:])
        avg_soc0 = np.mean(results['soc0'][-24:])
        avg_soc1 = np.mean(results['soc1'][-24:])
        print(f"  Day {day}: "
              f"Avg Reward={avg_reward:7.2f}, "
              f"bat0_SOC={avg_soc0:.2%}, "
              f"bat1_SOC={avg_soc1:.2%}")
    
    if terminated or truncated:
        break

print(f"\nSimulation complete!")
print(f"  Total reward: {sum(results['rewards']):.2f}")
print(f"  bat0 avg SOC: {np.mean(results['soc0']):.2%}")
print(f"  bat1 avg SOC: {np.mean(results['soc1']):.2%}")
print(f"  bat0 SOC range: [{min(results['soc0'])*100:.1f}%, {max(results['soc0'])*100:.1f}%]")
print(f"  bat1 SOC range: [{min(results['soc1'])*100:.1f}%, {max(results['soc1'])*100:.1f}%]")
print(f"  bat0 power std: {np.std(results['power0']):.2f} MW")
print(f"  bat1 power std: {np.std(results['power1']):.2f} MW")
print(f"  Total generation cost: ${sum(results['generation_cost']):.2f}")

# ====== Step 6: Visualization ======

print("\n" + "=" * 80)
print("Step 6: Generate Visualization")
print("=" * 80)

fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# Plot 1: Reward curve
ax = axes[0, 0]
ax.plot(results['rewards'], linewidth=1, color='blue', alpha=0.7)
ax.set_xlabel('Time Step')
ax.set_ylabel('Reward')
ax.set_title('Reward Curve')
ax.grid(True, alpha=0.3)

# Plot 2: Dual-battery SOC comparison
ax = axes[0, 1]
ax.plot(np.array(results['soc0']) * 100, linewidth=2, color='green', label='bat0 (Arbitrage)')
ax.plot(np.array(results['soc1']) * 100, linewidth=2, color='purple', label='bat1 (Balancing)')
ax.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='SOC Lower Limit')
ax.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='SOC Upper Limit')
ax.set_xlabel('Time Step')
ax.set_ylabel('State of Charge (%)')
ax.set_title('Dual-Battery SOC Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: bat0 power curve
ax = axes[1, 0]
ax.plot(results['power0'], linewidth=1.5, color='green')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.fill_between(range(len(results['power0'])),
                results['power0'], 0,
                where=np.array(results['power0']) > 0,
                alpha=0.3, color='red', label='Discharging')
ax.fill_between(range(len(results['power0'])),
                results['power0'], 0,
                where=np.array(results['power0']) < 0,
                alpha=0.3, color='blue', label='Charging')
ax.set_xlabel('Time Step')
ax.set_ylabel('Power (MW)')
ax.set_title('bat0 Power (Peak-Valley Arbitrage)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: bat1 power curve
ax = axes[1, 1]
ax.plot(results['power1'], linewidth=1.5, color='purple')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.fill_between(range(len(results['power1'])),
                results['power1'], 0,
                where=np.array(results['power1']) > 0,
                alpha=0.3, color='red', label='Discharging')
ax.fill_between(range(len(results['power1'])),
                results['power1'], 0,
                where=np.array(results['power1']) < 0,
                alpha=0.3, color='blue', label='Charging')
ax.set_xlabel('Time Step')
ax.set_ylabel('Power (MW)')
ax.set_title('bat1 Power (Load Balancing)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Dual-battery power stacking
ax = axes[2, 0]
total_power = np.array(results['power0']) + np.array(results['power1'])
ax.plot(results['power0'], linewidth=1, alpha=0.6, color='green', label='bat0')
ax.plot(results['power1'], linewidth=1, alpha=0.6, color='purple', label='bat1')
ax.plot(total_power, linewidth=2, color='red', label='Total Power', linestyle='--')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Time Step')
ax.set_ylabel('Power (MW)')
ax.set_title('Dual-Battery Power Stacking')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Generation cost
ax = axes[2, 1]
ax.plot(results['generation_cost'], linewidth=1.5, color='orange')
ax.fill_between(range(len(results['generation_cost'])),
                results['generation_cost'], alpha=0.3, color='orange')
ax.set_xlabel('Time Step')
ax.set_ylabel('Cost ($)')
ax.set_title('Generation Cost')
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_dir = 'examples/x_RL07_output'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'battery_dual_control_basics.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

# ====== Summary ======

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

print(f"\nKey Takeaways:")
print(f"  1. FlattenWrapper supports controlling multiple resources (resource_names=['bat0', 'bat1'])")
print(f"  2. Action space automatically expands to 2D: [bat0_power, bat1_power]")
print(f"  3. A single agent can control multiple batteries simultaneously")
print(f"  4. Different batteries can execute different control strategies")
print(f"  5. Coordinated control is more flexible than single-battery control")

print(f"\nNext Steps:")
print(f"  - Run RL07_battery_dual_control_sb3.py to train RL agent")
print(f"  - Compare performance of rule-based vs RL strategies")
print(f"  - Extend to more batteries or other resource types")

print("\n" + "=" * 80)
print("Example Complete!")
print("=" * 80)
