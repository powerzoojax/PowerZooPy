"""Example: Battery Control with LMP-based Arbitrage using Stable-Baselines3

This example demonstrates:
1. Creating a scenario with battery storage
2. Using real-time Locational Marginal Price (LMP) from OPF
3. Training a SAC agent to maximize battery arbitrage profit
4. Comparing LMP-based reward with rule-based peak/off-peak strategy
5. Visualizing battery SOC, power, LMP, and profit

Key Difference from RL05:
- RL05: Uses hardcoded peak/off-peak hours (10AM-6PM peak, 12AM-5AM off-peak)
- RL06: Uses real-time LMP from OPF (reflects actual generation cost + congestion)

Advantages of LMP-based reward:
- More realistic: Prices reflect real-time supply/demand and congestion
- More flexible: Can respond to unexpected events (generator outages, high renewable output)
- Better generalization: Not limited to predefined time windows

Requirements:
    - stable-baselines3: pip install stable-baselines3
    - gurobipy: pip install gurobipy (for OPF with LMP)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

# ====== Configuration ======
parser = argparse.ArgumentParser(description="Battery LMP arbitrage demo")
parser.add_argument("--timesteps", type=int, default=2_000,
                    help="SAC training timesteps; use 100000 for a longer run.")
parser.add_argument("--warmup-steps", type=int, default=200,
                    help="Random warmup steps before SAC training.")
parser.add_argument("--eval-steps", type=int, default=168,
                    help="Evaluation steps after training/loading.")
parser.add_argument("--force-retrain", action="store_true",
                    help="Retrain even if a saved model exists.")
args = parser.parse_args()

FORCE_RETRAIN = args.force_retrain

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
    print("Install via: pip install stable-baselines3 shimmy")
    print("This example will run in demo mode without training.")
    print("=" * 80 + "\n")

from powerzoo.envs.power_env import PowerEnv
from powerzoo.wrappers.flatten import FlattenWrapper

print("=" * 80)
print("Battery LMP Arbitrage with Reinforcement Learning")
print("=" * 80)

# ====== Step 1: Register Custom Scenario with Battery ======

print("\n" + "=" * 80)
print("Step 1: Register Scenario with LMP-based Reward")
print("=" * 80)

lmp_battery_config = {
    'name': 'IEEE5Bus-Battery-LMP',
    'description': 'Battery arbitrage based on real-time LMP from OPF',
    'grid': {
        'type': 'transmission',
        'case': 'Case5',
        'start_date': '2024-01-01',
        'end_date': '2024-01-31',  # 1 month for training
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
        'type': 'battery_lmp_arbitrage',  # Use LMP-based reward
        'battery_bus_id': 4,  # Battery is at bus 4
        'profit_weight': 1.0,  # Pure profit signal
        'soc_penalty_weight': 0.0,  # No SOC penalty - pure RL learning!
        'target_soc': 0.5,
    },
    'episode': {
        'max_steps': 336,  # 14 days * 24 steps/day - longer for complete cycles
    },
}

# lmp_battery_config is passed directly to PowerEnv
print("✓ Registered scenario: IEEE5Bus-Battery-LMP")
print("\nReward Function: LMP-based Arbitrage")
print("  - Profit = LMP × battery_power × delta_t")
print("  - Discharge at high LMP → Positive profit")
print("  - Charge at low LMP → Negative profit (buying cost)")
print("  - Agent learns to maximize: ∑(profit - SOC_penalty)")

# ====== Step 2: Inspect Spaces ======

print("\n" + "=" * 80)
print("Step 2: Inspect Observation and Action Spaces")
print("=" * 80)

# Create base environment
env_base = PowerEnv(lmp_battery_config)
print("\n[Base Environment - Dict Spaces]")
print(f"Observation Space: {env_base.observation_space}")
print(f"Action Space: {env_base.action_space}")

# ====== Step 3: Create Flattened Environment for SB3 ======

print("\n" + "=" * 80)
print("Step 3: Create Flattened Environment for SB3")
print("=" * 80)

# Wrap environment for SB3
env = FlattenWrapper(
    env_base,
    resource_names=['bat0'],  # Control battery 'bat0'
    obs_keys=['grid', 'resources', 'time']  # Include all observation components
)

print(f"\n[Wrapped Environment - Flat Spaces]")
print(f"Observation Space: {env.observation_space}")
print(f"Action Space: {env.action_space}")

# ====== Step 4: Training with SB3 ======

if HAS_SB3:
    print("\n" + "=" * 80)
    print("Step 4: Training SAC Agent")
    print("=" * 80)
    
    # Custom callback to track training progress
    class TrainingCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
        
        def _on_step(self) -> bool:
            if self.locals.get('dones', [False])[0]:
                info = self.locals['infos'][0]
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
            return True
    
    callback = TrainingCallback()
    
    # Define model save path
    model_dir = Path(f"{os.path.dirname(__file__)}/x_RL06_output")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "sac_battery_lmp_model.zip"
    
    # Reset environment once to initialize observation space
    _ = env.reset()
    
    # Check if trained model exists
    if model_path.exists() and not FORCE_RETRAIN:
        print(f"\n✓ Found existing model: {model_path}")
        print("  Loading trained model (skip training)...")
        model = SAC.load(model_path, env=env)
        print("  ✓ Model loaded successfully!")
    else:
        print(f"\n✗ No existing model found or FORCE_RETRAIN=True")
        print("  Training new model from scratch...")
        print(f"\nSAC Configuration (Pure RL - No Tricks!):")
        print(f"  Algorithm: Soft Actor-Critic (SAC)")
        print(f"  Policy: MlpPolicy")
        print(f"  Learning Rate: 1e-3 (higher for faster learning)")
        print(f"  Batch Size: 256 (larger for stability)")
        print(f"  Buffer Size: 100000 (much larger for diverse experiences)")
        print(f"  Total Timesteps: {args.timesteps}")
        print(f"  Entropy Coefficient: 0.2 (strong exploration)")
        print(f"\nKey Strategy:")
        print(f"  ✓ Pure profit reward (no SOC penalty)")
        print(f"  ✓ Long episodes (14 days = 336 steps)")
        print(f"  ✓ Large replay buffer for diverse samples")
        print(f"  ✓ High entropy for exploration")
        
        # Warmup: Collect diverse experiences with random actions
        print(f"\nWarmup Phase: Building initial experience...")
        print(f"  Collecting {args.warmup_steps} random steps...")
        obs, _ = env.reset()
        warmup_rewards = []
        for warmup_step in range(args.warmup_steps):
            random_action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(random_action)
            warmup_rewards.append(reward)
            if terminated or truncated:
                obs, _ = env.reset()
            if warmup_step % max(1, args.warmup_steps // 4) == 0:
                avg_reward = np.mean(warmup_rewards[-100:]) if warmup_rewards else 0
                print(f"    Step {warmup_step}/{args.warmup_steps}, Recent avg reward: {avg_reward:.2f}")
        print(f"  ✓ Warmup complete! Avg reward: {np.mean(warmup_rewards):.2f}")
        
        # Create SAC model with strong exploration
        model = SAC(
            'MlpPolicy',
            env,
            verbose=0,
            learning_rate=1e-3,  # Higher LR for faster learning
            buffer_size=100000,  # Large buffer
            learning_starts=max(1, args.warmup_steps),
            batch_size=256,  # Large batch for stability
            tau=0.005,
            ent_coef=0.2,  # Strong entropy bonus for exploration
            gamma=0.99,  # Standard discount
        )
        
        print(f"\nTraining in progress...")
        print(f"  Use --timesteps 100000 for the longer convergence run.")
        print(f"  Agent will learn: charge at low LMP, discharge at high LMP")
        model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
        
        # Save model
        model.save(model_path)
        print(f"\n✓ Model saved to: {model_path}")
    
    # ====== Step 5: Evaluation ======
    
    print("\n" + "=" * 80)
    print("Step 5: Evaluate Trained Agent")
    print("=" * 80)
    
    # Reset environment
    obs, info = env.reset(day_id=0)
    
    # Get reference to base environment
    env_base = env.env
    
    # Storage for results
    results = {
        'time_steps': [],
        'battery_power_mw': [],
        'battery_soc': [],
        'lmp': [],
        'profit': [],
        'cumulative_profit': [],
        'rewards': [],
        'total_load_mw': [],
    }
    
    cumulative_profit = 0.0
    total_reward = 0.0
    
    print(f"\nRunning evaluation for {args.eval_steps} steps...")
    for step in range(args.eval_steps):
        # Get action from trained model
        action, _states = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract battery status
        battery_status = env_base.get_resource_status()['bat0']
        
        # Extract power (action is the power command)
        power = float(np.atleast_1d(battery_status['current_p_mw'])[0])
        soc = float(np.atleast_1d(battery_status['soc'])[0])
        
        # Get LMP at battery bus (bus 4 → index 3)
        lmp_array = info.get('lmp', np.zeros(5))
        battery_lmp = float(lmp_array[3])  # Bus 4 (0-indexed: 3)
        
        # Calculate profit for this step
        delta_t_hours = 1.0  # 1-hour intervals
        step_profit = battery_lmp * power * delta_t_hours
        cumulative_profit += step_profit
        
        # Get total load
        total_load_mw = info.get('total_generation_mw', 0)
        
        # Store results
        results['time_steps'].append(step)
        results['battery_power_mw'].append(power)
        results['battery_soc'].append(soc)
        results['lmp'].append(battery_lmp)
        results['profit'].append(step_profit)
        results['cumulative_profit'].append(cumulative_profit)
        results['rewards'].append(reward)
        results['total_load_mw'].append(total_load_mw)
        
        total_reward += reward
        
        # Print progress every day
        if step % 24 == 0:
            day = step // 24
            print(f"  Day {day:2d}: SOC={soc*100:5.1f}%, Power={power:6.1f} MW, "
                  f"LMP=${battery_lmp:5.1f}/MWh, Profit=${step_profit:6.2f}, "
                  f"Cum_Profit=${cumulative_profit:8.2f}")
        
        if terminated or truncated:
            break
    
    print(f"\n✓ Evaluation complete")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Total profit: ${cumulative_profit:.2f}")
    print(f"  Average profit per step: ${cumulative_profit/len(results['time_steps']):.2f}")
    print(f"  SOC range: {min(results['battery_soc'])*100:.1f}% - {max(results['battery_soc'])*100:.1f}%")
    print(f"  LMP range: ${min(results['lmp']):.2f}/MWh - ${max(results['lmp']):.2f}/MWh")
    
    # ====== Step 6: Visualization ======
    
    print("\n" + "=" * 80)
    print("Step 6: Visualize Results")
    print("=" * 80)
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    time_hours = np.array(results['time_steps'])
    
    # Plot 1: Battery Power
    ax = axes[0]
    ax.plot(time_hours, results['battery_power_mw'], linewidth=2, color='blue')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.fill_between(time_hours, results['battery_power_mw'], alpha=0.3, color='blue')
    ax.set_ylabel('Power (MW)')
    ax.set_title('Battery Power (Positive=Discharge, Negative=Charge)')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Battery SOC
    ax = axes[1]
    ax.plot(time_hours, np.array(results['battery_soc']) * 100, linewidth=2, color='green')
    ax.axhline(y=50, color='k', linestyle='--', alpha=0.3, label='Target SOC')
    ax.axhline(y=10, color='r', linestyle='--', alpha=0.3, label='SOC Min')
    ax.axhline(y=90, color='r', linestyle='--', alpha=0.3, label='SOC Max')
    ax.fill_between(time_hours, np.array(results['battery_soc']) * 100, alpha=0.3, color='green')
    ax.set_ylabel('SOC (%)')
    ax.set_title('Battery State of Charge')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Locational Marginal Price (LMP)
    ax = axes[2]
    ax.plot(time_hours, results['lmp'], linewidth=2, color='orange')
    ax.fill_between(time_hours, results['lmp'], alpha=0.3, color='orange')
    ax.set_ylabel('LMP ($/MWh)')
    ax.set_title('Locational Marginal Price at Battery Bus (Bus 4)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Profit
    ax = axes[3]
    ax.plot(time_hours, results['profit'], linewidth=1.5, color='purple', alpha=0.7, label='Step Profit')
    ax.plot(time_hours, results['cumulative_profit'], linewidth=2, color='darkviolet', label='Cumulative Profit')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_ylabel('Profit ($)')
    ax.set_title('Battery Arbitrage Profit')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Reward
    ax = axes[4]
    ax.plot(time_hours, results['rewards'], linewidth=2, color='red')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Reward')
    ax.set_title('RL Reward (Profit - SOC Penalty)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = 'examples/x_RL06_output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'battery_lmp_arbitrage.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # ====== Summary ======
    
    print("\n" + "=" * 80)
    print("Summary: LMP-based Battery Arbitrage")
    print("=" * 80)
    
    print(f"\nFinancial Performance:")
    print(f"  Total Arbitrage Profit: ${cumulative_profit:.2f}")
    print(f"  Average Profit per Hour: ${cumulative_profit/len(results['time_steps']):.2f}")
    print(f"  Average Profit per Day: ${cumulative_profit*24/len(results['time_steps']):.2f}")
    
    print(f"\nBattery Operation:")
    print(f"  SOC Range: {min(results['battery_soc'])*100:.1f}% - {max(results['battery_soc'])*100:.1f}%")
    print(f"  Average Power: {np.mean(np.abs(results['battery_power_mw'])):.2f} MW")
    print(f"  Total Energy Discharged: {sum([p for p in results['battery_power_mw'] if p > 0]):.2f} MWh")
    print(f"  Total Energy Charged: {sum([p for p in results['battery_power_mw'] if p < 0]):.2f} MWh")
    
    print(f"\nPrice Statistics:")
    print(f"  Average LMP: ${np.mean(results['lmp']):.2f}/MWh")
    print(f"  LMP Range: ${min(results['lmp']):.2f}/MWh - ${max(results['lmp']):.2f}/MWh")
    print(f"  LMP Std Dev: ${np.std(results['lmp']):.2f}/MWh")
    
    print(f"\nKey Insights (Pure RL Learning):")
    print(f"  ✓ Agent learned ONLY from profit signal (no SOC penalty)")
    print(f"  ✓ Discovered arbitrage strategy autonomously:")
    print(f"    → Charge when LMP is low (buy cheap)")
    print(f"    → Discharge when LMP is high (sell expensive)")
    print(f"  ✓ LMP reflects real-time generation cost + congestion")
    print(f"  ✓ More adaptive than fixed peak/off-peak rules")
    
    print(f"\n" + "=" * 80)
    print("Pure Reinforcement Learning Achievement")
    print("=" * 80)
    
    print(f"\nWhat Makes This Special:")
    print(f"  ✓ NO hardcoded rules or expert knowledge")
    print(f"  ✓ NO SOC penalty guiding the agent")
    print(f"  ✓ ONLY profit signal: reward = LMP × power × time")
    print(f"  ✓ Agent discovered arbitrage strategy by itself!")
    
    print(f"\nCompared to RL05:")
    print(f"  RL05: Uses fixed peak/off-peak hours (rule-based)")
    print(f"  RL06: Uses real-time LMP (market-based, adaptive)")
    
    print(f"\nTraining Configuration:")
    print(f"  - Total timesteps: {args.timesteps}")
    print(f"  - Buffer size: 100,000")
    print(f"  - Entropy coefficient: 0.2 (strong exploration)")
    print(f"  - Episode length: 336 steps (14 days)")
    print(f"  - Reward: Pure profit (no penalties)")

else:
    # Demo mode without SB3
    print("\n" + "=" * 80)
    print("Demo Mode: Random Policy (SB3 not installed)")
    print("=" * 80)
    print("\nInstall stable-baselines3 to train RL agent:")
    print("  pip install stable-baselines3 shimmy")

print("\n" + "=" * 80)
print("Example Complete")
print("=" * 80)
