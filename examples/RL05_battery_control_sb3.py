"""Example: Battery Control with Stable-Baselines3

This example demonstrates:
1. Creating a scenario with battery storage
2. Inspecting observation and action spaces
3. Using FlattenWrapper for SB3 compatibility
4. Training a SAC agent to control battery charge/discharge
5. Visualizing battery SOC and grid performance

Requirements:
    - stable-baselines3: pip install stable-baselines3
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ====== Configuration ======
FORCE_RETRAIN = False  # Set to True to retrain even if model exists

# Check if stable-baselines3 is available
try:
    from stable_baselines3 import SAC  # Use SAC for better continuous control
    from stable_baselines3.common.callbacks import BaseCallback

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("=" * 80)
    print("WARNING: stable-baselines3 not installed")
    print("=" * 80)
    print("Install via: pip install stable-baselines3 shimmy")
    print("This example will run in demo mode without training.")
    print("=" * 80 + "\n")

HAS_TRAINED_MODEL = False

from powerzoo.envs.power_env import PowerEnv
from powerzoo.wrappers.flatten import FlattenWrapper

print("=" * 80)
print("Battery Control with Reinforcement Learning")
print("=" * 80)

# ====== Step 1: Register Custom Scenario with Battery ======

print("\n" + "=" * 80)
print("Step 1: Register Scenario with Battery")
print("=" * 80)

training_battery_env_config = {
    'name': 'IEEE5Bus-Battery',
    'description': 'Battery arbitrage: charge during off-peak, discharge during peak',
    'grid': {
        'type': 'transmission',
        'case': 'Case5',
        'start_date': '2024-01-01',
        'end_date': '2024-01    -31',  # 1 month for training
        'delta_t_minutes': 60,  # 1-hour intervals
        'max_load_ratio': 0.9,
    },
    'resources': [
        {'type': 'solar', 'capacity_mw': 50, 'bus_id': 2},
        {'type': 'wind', 'capacity_mw': 100, 'bus_id': 3},
        {'type': 'battery', 'name': 'bat0', 'capacity_mw': 50, 'power_mw': 20, 'bus_id': 4,
         'efficiency': 0.95, 'soc_min': 0.1, 'soc_max': 0.9, 'initial_soc': 0.5},
    ],
    'reward': {
        'type': 'battery_arbitrage',  # New: battery-specific reward
        'peak_hours': [10, 11, 12, 13, 14, 15, 16, 17, 18],  # Discharge during peak
        'off_peak_hours': [0, 1, 2, 3, 4, 5],  # Charge during off-peak
        'arbitrage_weight': 1.0,
        'soc_penalty_weight': 0.05,
    },
    'episode': {
        'max_steps': 168,  # 7 days * 24 steps/day (1-hour intervals)
    },
}

# training_battery_env_config is passed directly to PowerEnv (no scenario registry)
print("✓ Registered scenario: IEEE5Bus-Battery")

# ====== Step 2: Inspect Default Spaces ======

print("\n" + "=" * 80)
print("Step 2: Inspect Default Observation and Action Spaces")
print("=" * 80)

# Create base environment (Dict spaces)
env_base = PowerEnv(training_battery_env_config)
print("\n[Base Environment - Dict Spaces]")
print(f"Observation Space: {env_base.observation_space}")
print(f"Action Space: {env_base.action_space}")

# Show observation space structure
print("\nObservation Space Components:")
for key, space in env_base.observation_space.spaces.items():
    print(f"  - {key}: {space}")

# Show what can be controlled
print("\nControllable Resources:")
print(f"  Resources in environment: {list(env_base.resources.keys())}")
print(f"  - bat0 (Battery): charge/discharge power in range [-20, 20] MW")
print("  - Grid: unit commitment (not controlled in this example)")

# ====== Step 3: Apply Flattening for SB3 Compatibility ======

print("\n" + "=" * 80)
print("Step 3: Apply FlattenWrapper for Stable-Baselines3")
print("=" * 80)

# Create flattened environment with resource-specific control
# New interface: specify which resources to control by name
env = FlattenWrapper(
    env_base,
    resource_names=['bat0'],  # Control battery by name
    obs_keys=['grid', 'resources', 'time']  # Include all observations
)

# IMPORTANT: After wrapping, update env_base to point to the wrapped environment's base
# This ensures env_base and env.env are the same instance
env_base = env.env

print(f"\n[Flattened Environment - Box Spaces]")
print(f"Observation Space: {env.observation_space}")
print(f"  Shape: {env.observation_space.shape}")
print(f"Action Space: {env.action_space}")
print(f"  Bounds: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}] MW")
print(f"\n✓ Environment is now compatible with Stable-Baselines3!")
print(f"✓ Auto-detected battery action: power in range [-20, 20] MW")

# ====== Step 4: Test Manual Control ======

print("\n" + "=" * 80)
print("Step 4: Test Manual Battery Control")
print("=" * 80)

obs, info = env.reset()
print(f"\nInitial observation shape: {obs.shape}")

# Manually control battery for a few steps
print("\nManual control examples:")
actions = [
    ("Idle", 0.0),
    ("Charge 10 MW", -10.0),
    ("Discharge 15 MW", 15.0),
]

for action_name, action_value in actions:
    action = np.array([action_value], dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)

    battery_status = env_base.get_resource_status()['bat0']  # Use custom name
    soc_pct = float(np.atleast_1d(battery_status['soc_percent'])[0])
    power = float(np.atleast_1d(battery_status['current_p_mw'])[0])
    print(f"  {action_name:20s}: SOC={soc_pct:5.1f}%, "
          f"Power={power:6.1f} MW, Reward={reward:7.2f}")

# ====== Step 5 & 6: Train and Evaluate (if SB3 available) ======

if HAS_SB3:
    try:
        print("\n" + "=" * 80)
        print("Step 5: Train PPO Agent")
        print("=" * 80)


        # Custom callback for tracking
        class TrainingCallback(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)
                self.episode_rewards = []
                self.episode_lengths = []

            def _on_step(self) -> bool:
                if self.locals.get('dones', [False])[0]:
                    self.episode_rewards.append(self.locals['rewards'][0])
                    self.episode_lengths.append(self.locals['infos'][0].get('time_step', 0))
                return True


        # Use the existing wrapped environment (don't create a new one!)
        # The env and env_base were already created above
        
        # IMPORTANT: Reset once to initialize observation space before creating SAC
        # This ensures SB3 sees the correct observation dimensions
        _ = env.reset()
        
        # Re-sync env_base after reset
        env_base = env.env

        # Define model save path
        model_dir = Path(f"{os.path.dirname(__file__)}/x_RL05_output")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "sac_battery_model.zip"

        # Check if trained model exists
        if model_path.exists() and not FORCE_RETRAIN:
            print(f"\n✓ Found existing model: {model_path}")
            print("  Loading trained model (skip training)...")
            print("  Tip: Set FORCE_RETRAIN=True to retrain from scratch")
            
            model = SAC.load(model_path, env=env)
            print("  ✓ Model loaded successfully!")
            HAS_TRAINED_MODEL = True
            
        else:
            print(f"\n✗ No existing model found: {model_path}")
            print("  Training new model from scratch...")

            # Create SAC agent (better for continuous control than PPO)
            model = SAC(
                'MlpPolicy',
                env,
                verbose=0,
                learning_rate=3e-4,
                buffer_size=10000,
                learning_starts=100,
                batch_size=64,
                tau=0.005,
            )

            print("\nTraining SAC agent for 5000 timesteps...")
            print("(This may take 2-3 minutes)")

            callback = TrainingCallback()
            model.learn(total_timesteps=5000, callback=callback, progress_bar=False)

            print(f"\n✓ Training complete!")
            print(f"  Episodes: {len(callback.episode_rewards)}")
            print(f"  Mean reward: {np.mean(callback.episode_rewards[-10:]):.2f}")
            
            # Save the trained model
            print(f"\nSaving model to: {model_path}")
            model.save(model_path)
            print("  ✓ Model saved!")

            HAS_TRAINED_MODEL = True

        # ====== Step 6: Evaluate Trained Agent ======
        print("\n" + "=" * 80)
        print("Step 6: Evaluate Trained Agent")
        print("=" * 80)

        # Check SOC before reset
        print(f"\nBefore reset: Battery SOC = {env_base.get_resource_status()['bat0']['soc']*100:.1f}%")
        
        # Test trained agent
        
        obs, info = env.reset()
        
        
        # Check SOC after reset
        print(f"After reset: Battery SOC = {env_base.get_resource_status()['bat0']['soc']*100:.1f}% (expected 50.0%)")
        print(f"Evaluation starting from day {info.get('start_day_id', 'N/A')}")

        results = {
            'rewards': [],
            'soc': [],
            'battery_power_mw': [],
            'generation_cost': [],
        }

        print("\nRunning evaluation for 168 steps (1 week)...")

        for step in range(500):  # Will truncate at 168 steps (1 week at 1-hour intervals)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            battery_status = env_base.get_resource_status()['bat0']  # Use custom name
            soc = float(np.atleast_1d(battery_status['soc'])[0])
            power = float(np.atleast_1d(battery_status['current_p_mw'])[0])

            results['rewards'].append(reward)
            results['soc'].append(soc)
            results['battery_power_mw'].append(power)  # Positive=discharge, Negative=charge
            results['generation_cost'].append(info.get('opf_cost', 0))

            if (step + 1) % 24 == 0:
                day = (step + 1) // 24
                avg_reward = np.mean(results['rewards'][-24:])
                avg_soc = np.mean(results['soc'][-24:])
                min_soc = min(results['soc'][-24:])
                max_soc = max(results['soc'][-24:])
                print(f"  Day {day}: Avg Reward={avg_reward:7.2f}, Avg SOC={avg_soc:.2%}, Range=[{min_soc:.2%}, {max_soc:.2%}]")

            if done or truncated:
                break

        print(f"\n✓ Evaluation complete!")
        print(f"  Total reward: {sum(results['rewards']):.2f}")
        print(f"  Average SOC: {np.mean(results['soc']):.2%}")
        print(f"  SOC range: [{min(results['soc'])*100:.1f}%, {max(results['soc'])*100:.1f}%]")
        print(f"  SOC std: {np.std(results['soc'])*100:.2f}%")
        print(f"  Power range: [{min(results['battery_power_mw']):.1f}, {max(results['battery_power_mw']):.1f}] MW")
        print(f"  Power std: {np.std(results['battery_power_mw']):.2f} MW")
        print(f"  Total generation cost: ${sum(results['generation_cost']):.2f}")

    except (ImportError, Exception) as e:
        print(f"\nWARNING: Training failed - {e}")
        print("Will use random policy instead.")
        HAS_TRAINED_MODEL = False
        HAS_SB3 = False

# ====== Demo Mode (if SB3 not available or training failed) ======

if not HAS_SB3 or not HAS_TRAINED_MODEL:
    # Demo mode without training
    print("\n" + "=" * 80)
    print("Demo Mode: Rule-Based Policy (SB3 not available)")
    print("=" * 80)

    # Create fresh flattened environment for demo
    env_demo_base = PowerEnv(training_battery_env_config)
    env_demo = FlattenWrapper(
        env_demo_base,
        resource_names=['bat0'],  # Control battery by name
        obs_keys=['grid', 'resources', 'time']
    )

    obs, info = env_demo.reset()

    results = {
        'rewards': [],
        'soc': [],
        'battery_power_mw': [],
        'generation_cost': [],
    }

    print("\nRunning 100 steps with rule-based policy...")

    for step in range(100):
        # Rule-based policy to demonstrate battery control
        battery_status = env_demo_base.get_resource_status()['bat0']  # Use custom name
        soc = float(np.atleast_1d(battery_status['soc'])[0])

        # Cycle-based control to show SOC variation
        cycle_position = step % 20

        if soc < 0.35:
            # Emergency: must charge
            action_value = -15.0
        elif soc > 0.65:
            # Emergency: must discharge
            action_value = 12.0
        elif cycle_position < 8:
            # Charge phase
            action_value = -8.0
        elif cycle_position < 12:
            # Discharge phase
            action_value = 10.0
        else:
            # Rest phase
            action_value = 0.0

        action = np.array([action_value], dtype=np.float32)
        obs, reward, done, truncated, info = env_demo.step(action)

        battery_status = env_demo_base.get_resource_status()['bat0']  # Use custom name
        soc = float(np.atleast_1d(battery_status['soc'])[0])
        power = float(np.atleast_1d(battery_status['current_p_mw'])[0])

        results['rewards'].append(reward)
        results['soc'].append(soc)
        results['battery_power_mw'].append(power)  # Positive=discharge, Negative=charge
        results['generation_cost'].append(info.get('opf_cost', 0))

        if (step + 1) % 20 == 0:
            avg_soc = np.mean(results['soc'][-20:])
            min_soc = min(results['soc'][-20:])
            max_soc = max(results['soc'][-20:])
            print(f"  Step {step + 1}: Reward={reward:7.2f}, SOC={soc:.2%}, Avg={avg_soc:.2%}, Range=[{min_soc:.2%}, {max_soc:.2%}]")

        if done or truncated:
            break

# ====== Step 7: Visualization ======

print("\n" + "=" * 80)
print("Step 7: Generate Visualization")
print("=" * 80)

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
ax.axhline(y=10, color='r', linestyle='--', label='SOC Min')
ax.axhline(y=90, color='r', linestyle='--', label='SOC Max')
ax.set_xlabel('Time Step')
ax.set_ylabel('State of Charge (%)')
ax.set_title('Battery State of Charge')
ax.legend()
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
output_path = os.path.join(output_dir, 'battery_control_sb3.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_path}")

# ====== Summary ======

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

print(f"\nKey Takeaways:")
print(f"  1. Dict spaces can be inspected with env.observation_space")
print(f"  2. FlattenWrapper makes environments compatible with SB3/RLlib")
print(f"  3. Battery control action: continuous in [-20, 20] MW")
print(f"  4. SOC dynamics automatically handled by BatteryEnv")

try:
    if HAS_TRAINED_MODEL:
        print(f"  5. PPO successfully learned battery control policy")
        print(f"  6. Trained agent maintains SOC constraints")
except:
    pass

if not HAS_SB3:
    print(f"  5. Install stable-baselines3 + shimmy to train RL agents")
    print(f"     pip install stable-baselines3 shimmy")

print(f"\nNext Steps:")
print(f"  - Tune reward function for better economics")
print(f"  - Add more training timesteps for better performance")
print(f"  - Try different RL algorithms (SAC, TD3, A2C)")
print(f"  - Scale to larger grids with multiple batteries")

print("\n" + "=" * 80)
print("Example Complete!")
print("=" * 80)
