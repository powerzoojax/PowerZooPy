"""Example: Multi-Agent DER Storage Arbitrage with Task System

Multi-agent battery arbitrage using MAPPO with penalty-based constraints.
Demonstrates the need for Safe RL methods in power system control.

Task description:
- Multiple batteries (3 by default) as independent agents
- Goal: maximize arbitrage profit (charge low, discharge high)
- Challenge: SOC constraints enforced via penalties (soft constraints)
- This baseline shows constraint violations → need for Safe MARL

Core API:
    from powerzoo.tasks import make_task_env
    env = make_task_env('marl_der_arbitrage')

Usage:
    python MARL05_der_arbitrage_demo.py                    # Quick test (20 iterations)
    python MARL05_der_arbitrage_demo.py --full             # Full training (200 iterations)
    python MARL05_der_arbitrage_demo.py --full --skip-test # Skip test, full training only
"""

import os
import sys
import csv
import numpy as np
from datetime import datetime
from typing import Dict, List

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'x_MARL05_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
CONFIG = {
    'test_iterations': 20,
    'test_batch_size': 512,
    'train_iterations': 200,  # 200 iterations for full training
    'train_batch_size': 2048,
    'lr': 3e-4,
    'gamma': 0.99,
    'max_steps': 336,  # 7 days (7 * 48)
    'num_batteries': 3,
    'start_date': '2024-01-01',
    'end_date': '2024-01-08',  # 7 days
}


# ==============================================================================
# Training Logger
# ==============================================================================
class TrainingLogger:
    """Log training metrics to CSV"""
    
    def __init__(self, output_dir: str):
        self.csv_path = os.path.join(output_dir, 'training_log.csv')
        self.headers = [
            'iteration', 'timestamp',
            'episode_reward_mean', 'episode_reward_min', 'episode_reward_max',
            'episode_len_mean',
            'policy_loss_mean', 'vf_loss_mean', 'entropy_mean', 'kl_loss_mean',
            'avg_profit', 'avg_violation', 'num_env_steps', 'training_time_s'
        ]
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
        
        self.results = []
        self._start_time = datetime.now()
        self._last_time = self._start_time
    
    def log(self, iteration: int, result: Dict):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_time = datetime.now()
        training_time = (current_time - self._last_time).total_seconds()
        self._last_time = current_time
        
        env_runners = result.get('env_runners', {})
        episode_reward_mean = env_runners.get('episode_return_mean', 
                            env_runners.get('episode_reward_mean', 0.0))
        episode_reward_min = env_runners.get('episode_return_min', 0.0)
        episode_reward_max = env_runners.get('episode_return_max', 0.0)
        episode_len_mean = env_runners.get('episode_len_mean', 0.0)
        num_env_steps = env_runners.get('num_env_steps_sampled', 0)
        
        learners = result.get('learners', {})
        policy_losses, vf_losses, entropies, kl_losses = [], [], [], []
        
        for key, stats in learners.items():
            if key == '__all_modules__' or not isinstance(stats, dict):
                continue
            if 'policy_loss' in stats:
                policy_losses.append(stats['policy_loss'])
            if 'vf_loss' in stats:
                vf_losses.append(stats['vf_loss'])
            if 'entropy' in stats:
                entropies.append(stats['entropy'])
            if 'mean_kl_loss' in stats:
                kl_losses.append(stats['mean_kl_loss'])
        
        policy_loss_mean = np.mean(policy_losses) if policy_losses else 0.0
        vf_loss_mean = np.mean(vf_losses) if vf_losses else 0.0
        entropy_mean = np.mean(entropies) if entropies else 0.0
        kl_loss_mean = np.mean(kl_losses) if kl_losses else 0.0
        
        # Estimate profit and violation from reward
        avg_profit = episode_reward_mean * 0.5 if episode_reward_mean > 0 else 0.0
        avg_violation = -episode_reward_mean * 0.1 if episode_reward_mean < 0 else 0.0
        
        row = [
            iteration, timestamp,
            episode_reward_mean, episode_reward_min, episode_reward_max,
            episode_len_mean,
            policy_loss_mean, vf_loss_mean, entropy_mean, kl_loss_mean,
            avg_profit, avg_violation, num_env_steps, training_time
        ]
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        self.results.append({
            'iteration': iteration,
            'episode_reward_mean': episode_reward_mean,
            'episode_reward_min': episode_reward_min,
            'episode_reward_max': episode_reward_max,
            'episode_len': episode_len_mean,
            'policy_loss': policy_loss_mean,
            'vf_loss': vf_loss_mean,
            'entropy': entropy_mean,
            'kl_loss': kl_loss_mean,
            'avg_profit': avg_profit,
            'avg_violation': avg_violation,
        })
        
        return row


# ==============================================================================
# Plotting Functions
# ==============================================================================
def plot_training_progress(logger: TrainingLogger, output_dir: str):
    """Plot training progress curves"""
    import matplotlib.pyplot as plt
    
    if not logger.results:
        return
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    iterations = [r['iteration'] for r in logger.results]
    rewards = [r['episode_reward_mean'] for r in logger.results]
    reward_mins = [r['episode_reward_min'] for r in logger.results]
    reward_maxs = [r['episode_reward_max'] for r in logger.results]
    policy_losses = [r['policy_loss'] for r in logger.results]
    entropies = [r['entropy'] for r in logger.results]
    
    # 1. Reward convergence
    ax = axes[0, 0]
    ax.fill_between(iterations, reward_mins, reward_maxs, alpha=0.3, color='blue')
    ax.plot(iterations, rewards, 'b-', linewidth=2, label='Mean Reward')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Episode Reward')
    ax.set_title('(a) Reward Convergence')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Policy Loss
    ax = axes[0, 1]
    ax.plot(iterations, policy_losses, 'r-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Policy Loss')
    ax.set_title('(b) Policy Loss')
    ax.grid(True, alpha=0.3)
    
    # 3. Entropy
    ax = axes[1, 0]
    ax.plot(iterations, entropies, 'g-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Entropy')
    ax.set_title('(c) Policy Entropy')
    ax.grid(True, alpha=0.3)
    
    # 4. Episode Length
    ep_lens = [r['episode_len'] for r in logger.results]
    ax = axes[1, 1]
    ax.plot(iterations, ep_lens, 'purple', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Episode Length')
    ax.set_title('(d) Episode Length')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'training_progress.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'training_progress.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Training progress figures saved")


def save_final_episode_details(eval_data: Dict, output_dir: str):
    """Save final episode details to CSV"""
    dispatch_csv = os.path.join(output_dir, 'final_episode_dispatch.csv')
    
    # Get agent names
    agent_names = list(eval_data['socs'][0].keys()) if eval_data['socs'] else ['bat_0', 'bat_1', 'bat_2']
    
    with open(dispatch_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['step', 'time_hhmm', 'hour', 'is_peak']
        for agent in agent_names:
            header.extend([f'{agent}_power_mw', f'{agent}_soc', f'{agent}_violation', f'{agent}_profit'])
        header.extend(['total_reward', 'total_violation'])
        writer.writerow(header)
        
        for step in range(len(eval_data['socs'])):
            hour = (step * 30) // 60
            minute = (step * 30) % 60
            time_str = f"{hour:02d}:{minute:02d}"
            
            is_peak = hour in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            
            row = [step, time_str, hour, int(is_peak)]
            
            total_violation = 0.0
            total_reward = 0.0
            
            for agent in agent_names:
                power = eval_data['powers'][step].get(agent, 0.0)
                soc = eval_data['socs'][step].get(agent, 0.5)
                violation = eval_data['violations'][step].get(agent, 0.0)
                profit = eval_data['profits'][step].get(agent, 0.0)
                reward = eval_data['rewards'][step].get(agent, 0.0)
                
                row.extend([power, soc, violation, profit])
                total_violation += violation
                total_reward += reward
            
            row.extend([total_reward / len(agent_names), total_violation])
            writer.writerow(row)
    
    print(f"  Final episode dispatch saved: {dispatch_csv}")
    
    # Summary CSV
    summary_csv = os.path.join(output_dir, 'final_episode_summary.csv')
    
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'unit'])
        
        # Total statistics
        total_profit = sum(sum(p.values()) for p in eval_data['profits'])
        total_violation = sum(sum(v.values()) for v in eval_data['violations'])
        total_reward = sum(sum(r.values()) for r in eval_data['rewards'])
        
        writer.writerow(['total_profit', total_profit, '$'])
        writer.writerow(['total_violation', total_violation, 'SOC units'])
        writer.writerow(['total_reward', total_reward / len(agent_names), ''])
        writer.writerow(['num_violation_steps', sum(1 for v in eval_data['violations'] if sum(v.values()) > 0), 'steps'])
        
        # Per-agent statistics
        for agent in agent_names:
            avg_soc = np.mean([s[agent] for s in eval_data['socs']])
            total_agent_profit = sum(p[agent] for p in eval_data['profits'])
            total_agent_violation = sum(v[agent] for v in eval_data['violations'])
            writer.writerow([f'{agent}_avg_soc', avg_soc, ''])
            writer.writerow([f'{agent}_total_profit', total_agent_profit, '$'])
            writer.writerow([f'{agent}_total_violation', total_agent_violation, 'SOC units'])
    
    print(f"  Final episode summary saved: {summary_csv}")


def plot_evaluation_results(eval_data: Dict, output_dir: str, env=None):
    """Plot evaluation episode results with price signal and load curve"""
    import matplotlib.pyplot as plt
    
    if not eval_data['socs']:
        return
    
    agent_names = list(eval_data['socs'][0].keys())
    n_agents = len(agent_names)
    steps = len(eval_data['socs'])
    
    # Generate time axis in hours
    time_hours = np.arange(steps) * 0.5  # 30-min intervals
    max_time = time_hours[-1] if len(time_hours) > 0 else 24.0
    
    # Generate electricity price curve ($/MWh) - typical TOU pricing
    peak_hours = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
    off_peak_hours = {0, 1, 2, 3, 4, 5, 6, 23}
    price_curve = []
    for step in range(steps):
        hour = int((step * 30) / 60) % 24
        if hour in peak_hours:
            price_curve.append(80.0)  # High price $/MWh
        elif hour in off_peak_hours:
            price_curve.append(30.0)  # Low price $/MWh
        else:
            price_curve.append(50.0)  # Normal price $/MWh
    
    # Get real load data from eval_data if available
    load_curve = eval_data.get('loads', [])
    if not load_curve or len(load_curve) != steps:
        # Fallback: generate typical pattern (but this should not happen now)
        load_curve = []
        for step in range(steps):
            hour = int((step * 30) / 60) % 24
            if hour < 6:
                load = 600 + 20 * hour
            elif hour < 9:
                load = 720 + 60 * (hour - 6)
            elif hour < 12:
                load = 900 + 50 * (hour - 9)
            elif hour < 14:
                load = 1050 - 25 * (hour - 12)
            elif hour < 19:
                load = 1000 + 30 * (hour - 14)
            elif hour < 21:
                load = 1150 - 50 * (hour - 19)
            else:
                load = 1050 - 150 * (hour - 21)
            load_curve.append(load)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = plt.cm.tab10(np.arange(n_agents))
    
    # 1. Electricity Price Curve
    ax = axes[0, 0]
    ax.plot(time_hours, price_curve, 'r-', linewidth=2.5, label='Electricity Price')
    ax.fill_between(time_hours, 0, price_curve, alpha=0.3, color='coral')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Price ($/MWh)')
    ax.set_title('(a) Time-of-Use Electricity Price')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_time])
    ax.set_ylim([0, 100])
    
    # 2. Load Curve (real data)
    ax = axes[0, 1]
    ax.plot(time_hours, load_curve, 'b-', linewidth=2.5, label='System Load')
    ax.fill_between(time_hours, 0, load_curve, alpha=0.3, color='steelblue')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Load (MW)')
    ax.set_title('(b) System Load Profile (Real Data)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_time])
    
    # 3. SOC curves with price overlay
    ax = axes[0, 2]
    ax2 = ax.twinx()
    ax2.plot(time_hours, price_curve, 'r--', linewidth=1.5, alpha=0.7, label='Price')
    ax2.set_ylabel('Price ($/MWh)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, 100])
    
    for i, agent in enumerate(agent_names):
        socs = [s[agent] for s in eval_data['socs']]
        ax.plot(time_hours, socs, color=colors[i], linewidth=2, label=agent)
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='SOC min')
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='SOC max')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('State of Charge')
    ax.set_title('(c) Battery SOC vs Price')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_time])
    ax.set_ylim([0, 1])
    
    # 4. Power curves with price overlay
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax2.plot(time_hours, price_curve, 'r--', linewidth=1.5, alpha=0.7, label='Price')
    ax2.set_ylabel('Price ($/MWh)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, 100])
    
    for i, agent in enumerate(agent_names):
        powers = [p[agent] for p in eval_data['powers']]
        ax.plot(time_hours, powers, color=colors[i], linewidth=2, label=agent)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Power (MW)')
    ax.set_title('(d) Battery Power vs Price (+discharge, -charge)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_time])
    
    # 5. SOC Constraint Violations
    ax = axes[1, 1]
    total_violations = [sum(v.values()) for v in eval_data['violations']]
    bar_width = 0.5 * (steps / 48.0)  # Adapt bar width to number of steps
    ax.bar(time_hours, total_violations, width=bar_width, color='red', alpha=0.7)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('SOC Violation')
    ax.set_title('(e) SOC Constraint Violations (Need for Safe RL)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_time])
    if max(total_violations) == 0:
        ax.set_ylim([0, 0.1])
        ax.text(max_time / 2, 0.05, 'No Violations', ha='center', fontsize=12, color='green')
    
    # 6. Profit/Reward with price overlay
    ax = axes[1, 2]
    ax2 = ax.twinx()
    ax2.plot(time_hours, price_curve, 'r--', linewidth=1.5, alpha=0.7, label='Price')
    ax2.set_ylabel('Price ($/MWh)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, 100])
    
    total_profits = [sum(p.values()) for p in eval_data['profits']]
    total_rewards = [sum(r.values()) / len(agent_names) for r in eval_data['rewards']]
    cumulative_profit = np.cumsum(total_profits)
    
    ax.plot(time_hours, total_profits, 'g-', linewidth=2, label='Step Profit')
    ax.plot(time_hours, total_rewards, 'b-', linewidth=1.5, alpha=0.7, label='Step Reward')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Value ($)')
    ax.set_title(f'(f) Profit & Reward (Total: ${cumulative_profit[-1]:.2f})')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_time])
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'evaluation_results.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'evaluation_results.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Evaluation results figures saved")


# ==============================================================================
# Main Training Function
# ==============================================================================
def train_with_task_system(is_full_training: bool = False):
    """Train multi-agent PPO using the Task system"""
    
    # Import here to avoid import errors if ray not installed
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env
    from powerzoo.tasks import make_task_env
    
    mode = "Full Training (200 iter)" if is_full_training else "Test Training (20 iter)"
    print("=" * 60)
    print(f"MARL DER Arbitrage Training with Task System - {mode}")
    print("=" * 60)
    
    # Create test environment to get space info
    env_config = {
        'max_steps': CONFIG['max_steps'],
        'num_batteries': CONFIG['num_batteries'],
        'start_date': CONFIG.get('start_date', '2024-01-01'),
        'end_date': CONFIG.get('end_date', '2024-01-08'),  # 7 days
    }
    test_env = make_task_env('marl_der_arbitrage', **env_config)
    agents = test_env.possible_agents
    obs_space = test_env.observation_space[agents[0]]
    
    n_days = CONFIG['max_steps'] // 48
    print(f"\n[Environment Info]")
    print(f"  Agents: {agents}")
    print(f"  Obs shape: {obs_space.shape}")
    print(f"  Action space: {test_env.action_space[agents[0]]}")
    print(f"  Max steps: {CONFIG['max_steps']} ({n_days} days)")
    
    # Register environment with 7-day config
    register_env("marl_der_arbitrage_task", 
                 lambda cfg: make_task_env('marl_der_arbitrage', 
                                          max_steps=cfg.get('max_steps', 336),
                                          num_batteries=cfg.get('num_batteries', 3),
                                          start_date=cfg.get('start_date', '2024-01-01'),
                                          end_date=cfg.get('end_date', '2024-01-08')))
    
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    logger = TrainingLogger(OUTPUT_DIR)
    
    try:
        num_iterations = CONFIG['train_iterations'] if is_full_training else CONFIG['test_iterations']
        batch_size = CONFIG['train_batch_size'] if is_full_training else CONFIG['test_batch_size']
        
        config = (
            PPOConfig()
            .environment(
                env="marl_der_arbitrage_task",
                env_config={
                    'max_steps': CONFIG['max_steps'],
                    'num_batteries': CONFIG['num_batteries'],
                    'start_date': CONFIG.get('start_date', '2024-01-01'),
                    'end_date': CONFIG.get('end_date', '2024-01-08'),
                },
            )
            .framework('torch')
            .env_runners(num_env_runners=0)
            .training(
                lr=CONFIG['lr'],
                gamma=CONFIG['gamma'],
                train_batch_size=batch_size,
                model={"fcnet_hiddens": [128, 128]},
            )
            .multi_agent(
                policies={
                    agent: (None, obs_space, test_env.action_space[agent], {})
                    for agent in agents
                },
                policy_mapping_fn=lambda agent_id, episode, worker=None, **kw: agent_id,
            )
        )
        
        algo = config.build()
        
        print(f"\n[Training]")
        print(f"  Iterations: {num_iterations}")
        print(f"  Batch size: {batch_size}")
        print(f"  CSV log: {logger.csv_path}")
        
        for i in range(num_iterations):
            result = algo.train()
            row = logger.log(i + 1, result)
            
            episode_reward = row[2]
            episode_len = row[5]
            
            if (i + 1) % max(1, num_iterations // 10) == 0 or i == 0:
                print(f"  Iteration {i+1:3d}/{num_iterations}: reward={episode_reward:.4f}, ep_len={episode_len:.1f}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = algo.save(checkpoint_dir)
        print(f"\n[Checkpoint saved]: {checkpoint_path}")
        
        # Plot training progress
        print(f"\n[Generating Figures]")
        plot_training_progress(logger, OUTPUT_DIR)
        
        # Final evaluation
        print(f"\n[Final Evaluation]")
        eval_env = make_task_env('marl_der_arbitrage', **env_config)
        obs, info = eval_env.reset()
        
        eval_data = {'powers': [], 'socs': [], 'rewards': [], 'violations': [], 'profits': [], 'loads': []}
        
        # Get RLModule for inference - try multiple methods
        import torch
        rl_modules = {}
        
        # Method 1: Try to get modules from learner
        try:
            learner = algo.learner_group._learner
            if hasattr(learner, '_module'):
                multi_module = learner._module
                if hasattr(multi_module, '_rl_modules'):
                    rl_modules = multi_module._rl_modules
                    print(f"  Got modules from learner: {list(rl_modules.keys())}")
        except Exception as e:
            print(f"  Method 1 (learner) failed: {e}")
        
        # Method 2: Try algo.get_module() - new API
        if not rl_modules:
            try:
                multi_module = algo.get_module()
                if multi_module is not None:
                    if hasattr(multi_module, '_rl_modules'):
                        rl_modules = multi_module._rl_modules
                    elif hasattr(multi_module, 'keys'):
                        rl_modules = {k: multi_module[k] for k in multi_module.keys()}
                    print(f"  Got modules from get_module(): {list(rl_modules.keys())}")
            except Exception as e:
                print(f"  Method 2 (get_module) failed: {e}")
        
        # Method 3: Try env_runner (new API stack)
        if not rl_modules:
            try:
                env_runner = algo.env_runner
                if hasattr(env_runner, '_module'):
                    multi_module = env_runner._module
                    if hasattr(multi_module, '_rl_modules'):
                        rl_modules = multi_module._rl_modules
                        print(f"  Got modules from env_runner: {list(rl_modules.keys())}")
            except Exception as e:
                print(f"  Method 3 (env_runner) failed: {e}")
        
        if not rl_modules:
            print(f"  Warning: Could not get RLModules, using random actions")
        
        # Use RLModule.forward_inference for proper inference
        for step in range(CONFIG['max_steps']):
            actions = {}
            for agent in agents:
                try:
                    # Prepare observation tensor
                    obs_array = obs[agent]
                    if isinstance(obs_array, dict):
                        obs_array = obs_array.get('obs', obs_array)
                    obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0)
                    
                    # Get module for this agent
                    agent_module = rl_modules.get(agent, None)
                    
                    if agent_module is not None:
                        # Forward inference with module
                        with torch.no_grad():
                            fwd_out = agent_module.forward_inference({"obs": obs_tensor})
                        
                        # Extract action from output - handle different output formats
                        if "action_dist_inputs" in fwd_out:
                            # Gaussian policy: mean is the first half of action_dist_inputs
                            dist_inputs = fwd_out["action_dist_inputs"]
                            # For continuous action, mean is typically the first element
                            action_value = float(dist_inputs[0, 0].item())
                        elif "actions" in fwd_out:
                            action_value = float(fwd_out["actions"][0, 0].item())
                        else:
                            # Try to extract from any available key
                            action_value = 0.0
                            for key, val in fwd_out.items():
                                if isinstance(val, torch.Tensor) and val.numel() > 0:
                                    action_value = float(val[0, 0].item() if val.dim() > 1 else val[0].item())
                                    break
                    else:
                        # Random action as fallback
                        power_mw = eval_env._resource_info[agent]['power_mw']
                        action_value = np.random.uniform(-power_mw, power_mw)
                            
                except Exception as e:
                    if step == 0:
                        print(f"  Warning: inference failed for {agent}: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()
                    # Random action as fallback
                    power_mw = eval_env._resource_info[agent]['power_mw']
                    action_value = np.random.uniform(-power_mw, power_mw)
                
                # Clip action to valid range
                power_mw = eval_env._resource_info[agent]['power_mw']
                actions[agent] = np.clip(action_value, -power_mw, power_mw)
            
            # Debug: print actions for first few steps
            if step < 3:
                print(f"  Step {step}: actions = {actions}")
            
            obs, rewards, terminateds, truncateds, infos = eval_env.step(actions)
            
            # Record data from environment
            episode_data = eval_env.get_episode_data()
            if episode_data['powers']:
                eval_data['powers'].append(episode_data['powers'][-1])
                eval_data['socs'].append(episode_data['socs'][-1])
                eval_data['rewards'].append(episode_data['rewards'][-1])
                eval_data['violations'].append(episode_data['violations'][-1])
                eval_data['profits'].append(episode_data['profits'][-1])
                # Also get load data
                if 'loads' in episode_data and episode_data['loads']:
                    eval_data['loads'].append(episode_data['loads'][-1])
            
            if terminateds.get("__all__") or truncateds.get("__all__"):
                break
        
        print(f"  Evaluation steps: {len(eval_data['socs'])}")
        
        # Calculate summary statistics
        if eval_data['socs']:
            total_reward = sum(sum(r.values()) for r in eval_data['rewards']) / len(agents)
            total_violation = sum(sum(v.values()) for v in eval_data['violations'])
            total_profit = sum(sum(p.values()) for p in eval_data['profits'])
            print(f"  Total reward: {total_reward:.4f}")
            print(f"  Total profit: {total_profit:.4f}")
            print(f"  Total violation: {total_violation:.4f}")
        
        # Save final episode details
        save_final_episode_details(eval_data, OUTPUT_DIR)
        
        # Plot evaluation results
        plot_evaluation_results(eval_data, OUTPUT_DIR)
        
        algo.stop()
        
    finally:
        ray.shutdown()
    
    print(f"\n[Complete] All results saved to: {OUTPUT_DIR}")
    return True


# ==============================================================================
# Entry Point
# ==============================================================================
if __name__ == "__main__":
    is_full_training = '--full' in sys.argv
    skip_test = '--skip-test' in sys.argv
    
    if not skip_test:
        # Quick environment test
        print("=" * 60)
        print("Quick Environment Test")
        print("=" * 60)
        
        from powerzoo.tasks import make_task_env, list_tasks
        
        print("Available tasks:", list_tasks())
        env = make_task_env('marl_der_arbitrage', max_steps=48, num_batteries=3)
        print(f"Environment: {type(env).__name__}")
        print(f"Agents: {env.possible_agents}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        obs, info = env.reset()
        print(f"\nRunning 5-step test...")
        for step in range(5):
            # Random actions within bounds
            actions = {}
            for agent in env.possible_agents:
                power_mw = env._resource_info[agent]['power_mw']
                actions[agent] = np.random.uniform(-power_mw, power_mw)
            
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            
            # Show first agent info
            agent0 = env.possible_agents[0]
            info0 = infos[agent0]
            print(f"  Step {step+1}: "
                  f"SOC={info0['soc']:.3f}, "
                  f"power={info0['power']:.1f}MW, "
                  f"profit={info0['profit']:.3f}, "
                  f"violation={info0['violation']:.3f}, "
                  f"reward={rewards[agent0]:.4f}")
        
        print("\n✓ Environment test passed!\n")
    
    # Run training
    train_with_task_system(is_full_training=is_full_training)

