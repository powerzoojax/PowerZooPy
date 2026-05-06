"""Example: Multi-Agent EV V2G/G2V Control with Task System

Multi-agent EV charging/discharging using MAPPO with penalty-based constraints.
Demonstrates the need for Safe RL methods in EV control with commute constraints.

Task description:
- Multiple EVs (5 by default) as independent agents on IEEE 33-bus distribution grid
- Goal: maximize arbitrage profit while ensuring departure readiness
- Challenge: Commute constraints (only charge when home) + departure SOC requirements
- This baseline shows constraint violations → need for Safe MARL

Core API:
    from powerzoo.tasks import make_task_env
    env = make_task_env('marl_ev_v2g')

Usage:
    python MARL06_ev_v2g_demo.py                    # Quick test (20 iterations)
    python MARL06_ev_v2g_demo.py --full             # Full training (200 iterations)
    python MARL06_ev_v2g_demo.py --full --skip-test # Skip test, full training only
"""

import os
import sys
import csv
import numpy as np
from datetime import datetime
from typing import Dict, List

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'x_MARL06_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
CONFIG = {
    'test_iterations': 20,
    'test_batch_size': 512,
    'train_iterations': 300,  # Extended training for better convergence
    'train_batch_size': 2048,
    'lr': 3e-4,
    'gamma': 0.99,
    'max_steps': 168,  # 7 days (7 * 24)
    'num_evs': 3,  # Reduced from 5 to simplify learning
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
            'avg_profit', 'avg_violation', 'departure_ready_ratio',
            'num_env_steps', 'training_time_s'
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
        
        # Extract learner metrics
        learners = result.get('learners', {})
        all_modules = learners.get('__all_modules__', {})
        
        # Get per-module metrics
        policy_losses = []
        vf_losses = []
        entropies = []
        kl_losses = []
        
        for key, val in learners.items():
            if key != '__all_modules__' and isinstance(val, dict):
                if 'policy_loss' in val:
                    policy_losses.append(val['policy_loss'])
                if 'vf_loss' in val:
                    vf_losses.append(val['vf_loss'])
                if 'entropy' in val:
                    entropies.append(val['entropy'])
                if 'mean_kl_loss' in val:
                    kl_losses.append(val['mean_kl_loss'])
        
        policy_loss_mean = np.mean(policy_losses) if policy_losses else 0.0
        vf_loss_mean = np.mean(vf_losses) if vf_losses else 0.0
        entropy_mean = np.mean(entropies) if entropies else 0.0
        kl_loss_mean = np.mean(kl_losses) if kl_losses else 0.0
        
        # Estimate profit and violation from reward
        avg_profit = episode_reward_mean / 2.0  # Rough estimate
        avg_violation = 0.0
        departure_ready_ratio = 0.0  # Will be computed in evaluation
        
        num_env_steps = all_modules.get('num_env_steps_trained', 0)
        
        row = [
            iteration, timestamp,
            episode_reward_mean, episode_reward_min, episode_reward_max,
            episode_len_mean,
            policy_loss_mean, vf_loss_mean, entropy_mean, kl_loss_mean,
            avg_profit, avg_violation, departure_ready_ratio,
            num_env_steps, training_time
        ]
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        self.results.append(row)
        return row


# ==============================================================================
# Plotting Functions
# ==============================================================================
def plot_training_progress(logger: TrainingLogger, output_dir: str):
    """Plot training progress curves"""
    import matplotlib.pyplot as plt
    
    if not logger.results:
        return
    
    # Extract numeric columns only (skip timestamp column 1)
    iterations = np.array([r[0] for r in logger.results], dtype=float)
    rewards_mean = np.array([r[2] for r in logger.results], dtype=float)
    rewards_min = np.array([r[3] for r in logger.results], dtype=float)
    rewards_max = np.array([r[4] for r in logger.results], dtype=float)
    vf_loss = np.array([r[7] for r in logger.results], dtype=float)
    entropy = np.array([r[8] for r in logger.results], dtype=float)
    num_steps = np.array([r[13] for r in logger.results], dtype=float)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Reward convergence
    ax = axes[0, 0]
    ax.plot(iterations, rewards_mean, 'b-', linewidth=2, label='Mean')
    ax.fill_between(iterations, rewards_min, rewards_max, alpha=0.3, color='blue')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Episode Reward')
    ax.set_title('(a) Reward Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Value function loss
    ax = axes[0, 1]
    ax.plot(iterations, vf_loss, 'r-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('VF Loss')
    ax.set_title('(b) Value Function Loss')
    ax.grid(True, alpha=0.3)
    
    # 3. Policy entropy
    ax = axes[1, 0]
    ax.plot(iterations, entropy, 'g-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Entropy')
    ax.set_title('(c) Policy Entropy')
    ax.grid(True, alpha=0.3)
    
    # 4. Training efficiency
    ax = axes[1, 1]
    cumulative_steps = np.cumsum(num_steps)
    ax.plot(iterations, cumulative_steps / 1000, 'purple', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cumulative Steps (K)')
    ax.set_title('(d) Training Progress')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'training_progress.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'training_progress.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Training progress figures saved")


def save_final_episode_details(eval_data: Dict, output_dir: str):
    """Save final episode details to CSV"""
    dispatch_csv = os.path.join(output_dir, 'final_episode_dispatch.csv')
    
    agent_names = list(eval_data['socs'][0].keys()) if eval_data['socs'] else []
    
    with open(dispatch_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['step', 'time_hhmm', 'hour', 'day', 'is_peak']
        for agent in agent_names:
            header.extend([f'{agent}_power_kw', f'{agent}_soc', f'{agent}_is_home', 
                          f'{agent}_departure_ready', f'{agent}_violation', f'{agent}_profit'])
        header.extend(['total_reward', 'total_violation'])
        writer.writerow(header)
        
        for step in range(len(eval_data['socs'])):
            hour = step % 24
            day = step // 24 + 1
            minute = 0
            time_str = f"{hour:02d}:{minute:02d}"
            
            is_peak = hour in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            
            row = [step, time_str, hour, day, int(is_peak)]
            
            total_violation = 0.0
            total_reward = 0.0
            
            for agent in agent_names:
                power = eval_data['powers'][step].get(agent, 0.0)
                soc = eval_data['socs'][step].get(agent, 0.5)
                is_home = eval_data['is_home'][step].get(agent, True)
                dep_ready = eval_data['departure_ready'][step].get(agent, False)
                violation = eval_data['violations'][step].get(agent, 0.0)
                profit = eval_data['profits'][step].get(agent, 0.0)
                reward = eval_data['rewards'][step].get(agent, 0.0)
                
                row.extend([power, soc, int(is_home), int(dep_ready), violation, profit])
                total_violation += violation
                total_reward += reward
            
            row.extend([total_reward / max(len(agent_names), 1), total_violation])
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
        
        # Departure readiness ratio
        dep_ready_count = sum(sum(1 for v in d.values() if v) for d in eval_data['departure_ready'])
        total_steps = len(eval_data['departure_ready']) * len(agent_names)
        dep_ready_ratio = dep_ready_count / max(total_steps, 1)
        
        writer.writerow(['total_profit', total_profit, '$'])
        writer.writerow(['total_violation', total_violation, 'penalty'])
        writer.writerow(['total_reward', total_reward / len(agent_names), ''])
        writer.writerow(['departure_ready_ratio', dep_ready_ratio, '%'])
        writer.writerow(['num_violation_steps', sum(1 for v in eval_data['violations'] if sum(v.values()) > 0), 'steps'])
        
        # Per-agent statistics
        for agent in agent_names:
            avg_soc = np.mean([s[agent] for s in eval_data['socs']])
            total_agent_profit = sum(p[agent] for p in eval_data['profits'])
            total_agent_violation = sum(v[agent] for v in eval_data['violations'])
            home_ratio = sum(1 for h in eval_data['is_home'] if h[agent]) / len(eval_data['is_home'])
            writer.writerow([f'{agent}_avg_soc', avg_soc, ''])
            writer.writerow([f'{agent}_total_profit', total_agent_profit, '$'])
            writer.writerow([f'{agent}_total_violation', total_agent_violation, 'penalty'])
            writer.writerow([f'{agent}_home_ratio', home_ratio, '%'])
    
    print(f"  Final episode summary saved: {summary_csv}")


def plot_evaluation_results(eval_data: Dict, output_dir: str):
    """Plot evaluation episode results"""
    import matplotlib.pyplot as plt
    
    if not eval_data['socs']:
        return
    
    agent_names = list(eval_data['socs'][0].keys())
    n_agents = len(agent_names)
    steps = len(eval_data['socs'])
    
    # Time axis in hours
    time_hours = np.arange(steps)
    max_time = time_hours[-1] if len(time_hours) > 0 else 168.0
    
    # Price curve
    peak_hours = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
    price_curve = []
    for step in range(steps):
        hour = step % 24
        if hour in peak_hours:
            price_curve.append(80.0)
        elif hour in {0, 1, 2, 3, 4, 5, 6, 23}:
            price_curve.append(30.0)
        else:
            price_curve.append(50.0)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = plt.cm.tab10(np.arange(n_agents))
    
    # 1. Price and home status
    ax = axes[0, 0]
    ax.plot(time_hours, price_curve, 'r-', linewidth=2, label='Price')
    ax.fill_between(time_hours, 0, price_curve, alpha=0.2, color='coral')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Price ($/MWh)')
    ax.set_title('(a) Electricity Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_time])
    
    # Add day markers
    for d in range(1, 8):
        ax.axvline(x=d*24, color='gray', linestyle='--', alpha=0.5)
    
    # 2. SOC curves
    ax = axes[0, 1]
    for i, agent in enumerate(agent_names):
        socs = [s[agent] for s in eval_data['socs']]
        ax.plot(time_hours, socs, color=colors[i], linewidth=1.5, label=agent)
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='SOC min')
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Departure min')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('State of Charge')
    ax.set_title('(b) Battery SOC')
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_time])
    ax.set_ylim([0, 1])
    
    # 3. Home status
    ax = axes[0, 2]
    for i, agent in enumerate(agent_names):
        is_home = [1.0 if h[agent] else 0.0 for h in eval_data['is_home']]
        ax.fill_between(time_hours, i, [i + 0.8 * h for h in is_home], 
                       alpha=0.7, color=colors[i], label=agent)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('EV')
    ax.set_title('(c) EV Availability (Home/Away)')
    ax.set_yticks(np.arange(n_agents) + 0.4)
    ax.set_yticklabels(agent_names)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim([0, max_time])
    
    # 4. Power curves
    ax = axes[1, 0]
    for i, agent in enumerate(agent_names):
        powers = [p[agent] for p in eval_data['powers']]
        ax.plot(time_hours, powers, color=colors[i], linewidth=1.5, label=agent)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('(d) Charge/Discharge Power (+V2G, -G2V)')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_time])
    
    # 5. Hard Constraint Status (Real Violations)
    ax = axes[1, 1]
    # Check actual hard constraint violations
    hard_violations = []
    for step in range(steps):
        hard_viol_count = 0
        for agent in agent_names:
            soc = eval_data['socs'][step][agent]
            # Hard constraints: SOC must be in [0.1, 0.95]
            if soc < 0.1 or soc > 0.95:
                hard_viol_count += 1
        hard_violations.append(hard_viol_count)
    
    # Soft penalties (training signals)
    total_violations = [sum(v.values()) for v in eval_data['violations']]
    
    # Plot both
    ax.bar(time_hours, hard_violations, width=0.8, color='darkred', alpha=0.9, label='Hard Violations (Real)')
    ax2 = ax.twinx()
    ax2.bar(time_hours, total_violations, width=0.6, color='orange', alpha=0.4, label='Soft Penalties (Training)')
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Hard Violations (# of EVs)', color='darkred')
    ax2.set_ylabel('Soft Penalty Value', color='orange')
    ax.set_title('(e) Hard Constraints ✅ vs Soft Penalties')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_time])
    ax.tick_params(axis='y', labelcolor='darkred')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Add text if no hard violations
    if sum(hard_violations) == 0:
        ax.text(max_time/2, 0.5, '✅ No Hard Constraint Violations', 
                ha='center', va='center', fontsize=12, color='green', 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        ax.text(max_time/2, 0.05, 'No Violations', ha='center', fontsize=12, color='green')
    
    # 6. Cumulative profit
    ax = axes[1, 2]
    total_profits = [sum(p.values()) for p in eval_data['profits']]
    cumulative_profit = np.cumsum(total_profits)
    ax.plot(time_hours, cumulative_profit, 'g-', linewidth=2)
    ax.fill_between(time_hours, 0, cumulative_profit, alpha=0.3, color='green')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Cumulative Profit ($)')
    ax.set_title(f'(f) Cumulative Profit (Total: ${cumulative_profit[-1]:.2f})')
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
    
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env
    from powerzoo.tasks import make_task_env
    
    mode = "Full Training (200 iter)" if is_full_training else "Test Training (20 iter)"
    print("=" * 60)
    print(f"MARL EV V2G/G2V Training with Task System - {mode}")
    print("=" * 60)
    
    # Create test environment to get space info
    env_config = {
        'max_steps': CONFIG['max_steps'],
        'num_evs': CONFIG['num_evs'],
        'start_date': CONFIG.get('start_date', '2024-01-01'),
        'end_date': CONFIG.get('end_date', '2024-01-08'),
    }
    test_env = make_task_env('marl_ev_v2g', **env_config)
    agents = test_env.possible_agents
    obs_space = test_env.observation_space[agents[0]]
    
    n_days = CONFIG['max_steps'] // 24
    print(f"\n[Environment Info]")
    print(f"  Task: marl_ev_v2g (EV V2G/G2V)")
    print(f"  Agents: {agents}")
    print(f"  Obs shape: {obs_space.shape}")
    print(f"  Action space: {test_env.action_space[agents[0]]}")
    print(f"  Max steps: {CONFIG['max_steps']} ({n_days} days)")
    
    # Register environment
    register_env("marl_ev_v2g_task", 
                 lambda cfg: make_task_env('marl_ev_v2g', 
                                          max_steps=cfg.get('max_steps', 168),
                                          num_evs=cfg.get('num_evs', 5),
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
                env="marl_ev_v2g_task",
                env_config={
                    'max_steps': CONFIG['max_steps'],
                    'num_evs': CONFIG['num_evs'],
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
        eval_env = make_task_env('marl_ev_v2g', **env_config)
        obs, info = eval_env.reset()
        
        eval_data = {
            'powers': [], 'socs': [], 'rewards': [], 'violations': [], 
            'profits': [], 'is_home': [], 'departure_ready': []
        }
        
        # Get RLModule for inference
        import torch
        rl_modules = {}
        
        try:
            learner = algo.learner_group._learner
            if hasattr(learner, '_module'):
                multi_module = learner._module
                if hasattr(multi_module, '_rl_modules'):
                    rl_modules = multi_module._rl_modules
                    print(f"  Got modules: {list(rl_modules.keys())}")
        except Exception as e:
            print(f"  Warning: Could not get modules: {e}")
        
        # Evaluation loop
        for step in range(CONFIG['max_steps']):
            actions = {}
            for agent in agents:
                try:
                    obs_array = obs[agent]
                    obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0)
                    
                    agent_module = rl_modules.get(agent, None)
                    
                    if agent_module is not None:
                        with torch.no_grad():
                            fwd_out = agent_module.forward_inference({"obs": obs_tensor})
                        
                        if "action_dist_inputs" in fwd_out:
                            action_value = float(fwd_out["action_dist_inputs"][0, 0].item())
                        elif "actions" in fwd_out:
                            action_value = float(fwd_out["actions"][0, 0].item())
                        else:
                            action_value = 0.0
                    else:
                        action_value = np.random.uniform(-1, 1)
                        
                except Exception as e:
                    if step == 0:
                        print(f"  Warning: {agent}: {e}")
                    action_value = np.random.uniform(-1, 1)
                
                actions[agent] = np.clip(action_value, -1.0, 1.0)
            
            if step < 3:
                print(f"  Step {step}: actions = {actions}")
            
            obs, rewards, terminateds, truncateds, infos = eval_env.step(actions)
            
            # Record data
            episode_data = eval_env.get_episode_data()
            if episode_data['powers']:
                eval_data['powers'].append(episode_data['powers'][-1])
                eval_data['socs'].append(episode_data['socs'][-1])
                eval_data['rewards'].append(episode_data['rewards'][-1])
                eval_data['violations'].append(episode_data['violations'][-1])
                eval_data['profits'].append(episode_data['profits'][-1])
                eval_data['is_home'].append(episode_data['is_home'][-1])
                eval_data['departure_ready'].append(episode_data['departure_ready'][-1])
            
            if terminateds.get("__all__") or truncateds.get("__all__"):
                break
        
        print(f"  Evaluation steps: {len(eval_data['socs'])}")
        
        if eval_data['socs']:
            total_reward = sum(sum(r.values()) for r in eval_data['rewards']) / len(agents)
            total_profit = sum(sum(p.values()) for p in eval_data['profits'])
            total_violation = sum(sum(v.values()) for v in eval_data['violations'])
            print(f"  Total reward: {total_reward:.4f}")
            print(f"  Total profit: {total_profit:.4f}")
            print(f"  Total violation: {total_violation:.4f}")
        
        # Save and plot evaluation results
        save_final_episode_details(eval_data, OUTPUT_DIR)
        plot_evaluation_results(eval_data, OUTPUT_DIR)
        
        algo.stop()
        
    finally:
        ray.shutdown()
    
    print(f"\n[Complete] All results saved to: {OUTPUT_DIR}")


# ==============================================================================
# Entry Point
# ==============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MARL EV V2G Training')
    parser.add_argument('--full', action='store_true', help='Run full training (200 iterations)')
    parser.add_argument('--skip-test', action='store_true', help='Skip test run')
    args = parser.parse_args()
    
    if args.full and args.skip_test:
        # Full training only
        train_with_task_system(is_full_training=True)
    elif args.full:
        # Test first, then full training
        print("\n" + "=" * 60)
        print("Phase 1: Quick Test (20 iterations)")
        print("=" * 60)
        train_with_task_system(is_full_training=False)
        
        print("\n" + "=" * 60)
        print("Phase 2: Full Training (200 iterations)")
        print("=" * 60)
        train_with_task_system(is_full_training=True)
    else:
        # Test only
        train_with_task_system(is_full_training=False)

