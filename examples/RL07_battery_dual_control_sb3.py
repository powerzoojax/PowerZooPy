"""Example: Dual-Battery Coordinated Control (Stable-Baselines3 Training)

This example demonstrates:
1. Using a single agent to control 2 batteries
2. Training coordinated control strategy with SAC algorithm
3. Visualizing dual-battery operation after training

Requirements:
    - stable-baselines3: pip install stable-baselines3
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ====== Configuration ======
FORCE_RETRAIN = False  # Set to True to force retraining

# Check if stable-baselines3 is available
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("=" * 80)
    print("WARNING: stable-baselines3 not installed")
    print("=" * 80)
    print("Install: pip install stable-baselines3 shimmy")
    print("Will use rule-based strategy for demonstration")
    print("=" * 80 + "\n")

from powerzoo.envs.power_env import PowerEnv
from powerzoo.wrappers.flatten import FlattenWrapper

print("=" * 80)
print("Dual-Battery Coordinated Control - SAC Training")
print("=" * 80)

# ====== Step 1: Register Training Scenario ======

print("\nRegistering training scenario...")

training_config = {
    'name': 'IEEE5Bus-DualBattery-Train',
    'description': 'Dual-battery coordinated control training scenario',
    'grid': {
        'type': 'transmission',
        'case': 'Case5',
        'start_date': '2024-01-01',
        'end_date': '2024-01-31',  # 1 month for training
        'delta_t_minutes': 60,
        'max_load_ratio': 0.9,
    },
    'resources': [
        {'type': 'solar', 'capacity_mw': 50, 'bus_id': 2},
        {'type': 'wind', 'capacity_mw': 100, 'bus_id': 3},
        {'type': 'battery', 'name': 'bat0', 'capacity_mw': 50, 'power_mw': 20, 'bus_id': 4,
         'efficiency': 0.95, 'soc_min': 0.1, 'soc_max': 0.9, 'initial_soc': 0.5},
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
        'max_steps': 168,  # 7 days
    },
}

# Create environment
env_base = PowerEnv(training_config)
env = FlattenWrapper(
    env_base,
    resource_names=['bat0', 'bat1'],  # Control 2 batteries
    obs_keys=['grid', 'resources', 'time']
)
env_base = env.env

print(f"Environment created successfully")
print(f"  Observation space: {env.observation_space.shape}")
print(f"  Action space: {env.action_space.shape} (controlling 2 batteries)")

# ====== Step 2: Train or Load Model ======

HAS_TRAINED_MODEL = False

if HAS_SB3:
    print("\n" + "=" * 80)
    print("Train/Load SAC Model")
    print("=" * 80)
    
    # Reset environment to initialize observation space
    _ = env.reset()
    env_base = env.env
    
    # Define model save path
    model_dir = Path(f"{os.path.dirname(__file__)}/x_RL07_output")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "sac_dual_battery_model.zip"
    
    # Check if trained model exists
    if model_path.exists() and not FORCE_RETRAIN:
        print(f"\nFound existing model: {model_path}")
        print("  Loading trained model...")
        print("  Tip: Set FORCE_RETRAIN=True to retrain")
        
        model = SAC.load(model_path, env=env)
        print("  Model loaded successfully!")
        HAS_TRAINED_MODEL = True
        
    else:
        print(f"\nStarting new model training...")
        
        # Training callback
        class TrainingCallback(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)
                self.episode_rewards = []
                self.episode_lengths = []
            
            def _on_step(self) -> bool:
                if self.locals.get('dones', [False])[0]:
                    self.episode_rewards.append(self.locals['rewards'][0])
                return True
        
        # Create SAC model
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
        
        print("\nTraining SAC agent (5000 steps)...")
        print("(Estimated time: 2-3 minutes)")
        
        callback = TrainingCallback()
        model.learn(total_timesteps=5000, callback=callback, progress_bar=False)
        
        print(f"\nTraining complete!")
        print(f"  Training episodes: {len(callback.episode_rewards)}")
        print(f"  Average reward (last 10 episodes): {np.mean(callback.episode_rewards[-10:]):.2f}")
        
        # Save model
        print(f"\nSaving model to: {model_path}")
        model.save(model_path)
        print("  Model saved successfully!")
        
        HAS_TRAINED_MODEL = True

# ====== Step 3: Evaluate Model ======

print("\n" + "=" * 80)
print("Evaluate Model Performance")
print("=" * 80)

if not HAS_SB3 or not HAS_TRAINED_MODEL:
    # Use rule-based strategy
    print("\nUsing rule-based strategy for demonstration...")
    
    obs, info = env.reset()
    
    results = {
        'rewards': [],
        'soc0': [],
        'soc1': [],
        'power0': [],
        'power1': [],
        'generation_cost': [],
    }
    
    for step in range(168):
        bat0_status = env_base.get_resource_status()['bat0']
        bat1_status = env_base.get_resource_status()['bat1']
        
        soc0 = float(np.atleast_1d(bat0_status['soc'])[0])
        soc1 = float(np.atleast_1d(bat1_status['soc'])[0])
        
        hour_of_day = step % 24
        
        # bat0 strategy: Peak-valley arbitrage
        if soc0 < 0.2:
            action0 = -15.0
        elif soc0 > 0.8:
            action0 = 12.0
        elif hour_of_day in [0, 1, 2, 3, 4, 5]:
            action0 = -15.0
        elif hour_of_day in [14, 15, 16, 17, 18, 19]:
            action0 = 18.0
        else:
            action0 = 0.0
        
        # bat1 strategy: Load balancing
        if soc1 < 0.2:
            action1 = -10.0
        elif soc1 > 0.8:
            action1 = 8.0
        elif hour_of_day in [10, 11, 12, 13]:
            action1 = 10.0 if soc1 > 0.4 else 5.0
        elif hour_of_day in [6, 7, 8]:
            action1 = 8.0 if soc1 > 0.4 else 0.0
        else:
            action1 = 0.0
        
        action = np.array([action0, action1], dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
        
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
        
        if done or truncated:
            break

else:
    # Use trained model
    print(f"\nRunning evaluation (168 steps, 7 days)...")
    
    obs, info = env.reset()
    print(f"Initial state: bat0 SOC={env_base.get_resource_status()['bat0']['soc']*100:.1f}%, "
          f"bat1 SOC={env_base.get_resource_status()['bat1']['soc']*100:.1f}%")
    
    results = {
        'rewards': [],
        'soc0': [],
        'soc1': [],
        'power0': [],
        'power1': [],
        'generation_cost': [],
    }
    
    for step in range(500):  # Max 500 steps, will truncate at 168
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
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
            print(f"  Day {day}: Reward={avg_reward:7.2f}, "
                  f"bat0_SOC={avg_soc0:.2%}, bat1_SOC={avg_soc1:.2%}")
        
        if done or truncated:
            break

print(f"\nEvaluation complete!")
print(f"  Total reward: {sum(results['rewards']):.2f}")
print(f"  bat0 avg SOC: {np.mean(results['soc0']):.2%}")
print(f"  bat1 avg SOC: {np.mean(results['soc1']):.2%}")
print(f"  bat0 SOC range: [{min(results['soc0'])*100:.1f}%, {max(results['soc0'])*100:.1f}%]")
print(f"  bat1 SOC range: [{min(results['soc1'])*100:.1f}%, {max(results['soc1'])*100:.1f}%]")
print(f"  Total generation cost: ${sum(results['generation_cost']):.2f}")

# ====== Step 4: Visualization ======

print("\n" + "=" * 80)
print("Generate Visualization")
print("=" * 80)

fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# Plot 1: Reward curve
ax = axes[0, 0]
ax.plot(results['rewards'], linewidth=1, color='blue', alpha=0.7)
ax.set_xlabel('Time Step')
ax.set_ylabel('Reward')
ax.set_title('Reward Curve (SAC Training)' if HAS_TRAINED_MODEL else 'Reward Curve (Rule-Based)')
ax.grid(True, alpha=0.3)

# Plot 2: Dual-battery SOC comparison
ax = axes[0, 1]
ax.plot(np.array(results['soc0']) * 100, linewidth=2, color='green', label='bat0')
ax.plot(np.array(results['soc1']) * 100, linewidth=2, color='purple', label='bat1')
ax.axhline(y=10, color='r', linestyle='--', alpha=0.5)
ax.axhline(y=90, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('Time Step')
ax.set_ylabel('State of Charge (%)')
ax.set_title('Dual-Battery SOC Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: bat0 power
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
ax.set_title('bat0 Power')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: bat1 power
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
ax.set_title('bat1 Power')
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
output_path = os.path.join(output_dir, 'battery_dual_control_sb3.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

# ====== Summary ======

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

print(f"\nKey Takeaways:")
print(f"  1. Single SAC agent successfully controls 2 batteries")
print(f"  2. Action space dimension is 2: [bat0_power, bat1_power]")
print(f"  3. Agent automatically learns coordinated control strategy")
if HAS_TRAINED_MODEL:
    print(f"  4. Trained strategy maintains SOC within reasonable range")
    print(f"  5. Agent learned peak-valley arbitrage and load balancing")

print(f"\nNext Steps:")
print(f"  - Increase training steps to improve performance")
print(f"  - Tune reward function for different objectives")
print(f"  - Try other algorithms (TD3, PPO)")
print(f"  - Extend to more batteries or other resources")

print("\n" + "=" * 80)
print("Example Complete!")
print("=" * 80)
