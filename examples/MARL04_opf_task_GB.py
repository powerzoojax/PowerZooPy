"""Example: Multi-Agent OPF with Task System for GB Network (GPU Enabled)

Case29GB: 29 buses, 66 generators, 99 transmission lines
Compared to Case5 (5 generators), this is a significantly larger problem.

Training parameters are adjusted for the larger scale:
- Larger neural network (256x256 vs 128x128)
- Larger batch size for better gradient estimation
- Slightly lower learning rate for stability with more agents
- More training iterations for convergence
- GPU acceleration for faster training

Usage:
    python MARL04_opf_task_GB.py                    # Quick test (10 iterations)
    python MARL04_opf_task_GB.py --full             # Full training (500 iterations)
    python MARL04_opf_task_GB.py --full --skip-test # Skip test, full training only
"""

import os
import sys
import csv
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'x_MARL04_GB_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check GPU availability
GPU_AVAILABLE = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if GPU_AVAILABLE else 0
GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "N/A"

# Configuration - Adjusted for Case29GB (66 generators) - FAST VERSION
CONFIG = {
    # Test mode (quick check)
    'test_iterations': 5,
    'test_batch_size': 480,
    
    # Full training mode - accelerated
    'train_iterations': 200,      # Total iterations
    'train_batch_size': 960,     # Optimized batch size
    
    # Learning parameters
    'lr': 3e-4,                   # Learning rate
    'gamma': 0.99,
    'max_steps': 48,              # Half day for faster episodes
    
    # Network architecture - smaller for speed
    'fcnet_hiddens': [128, 128],  # Reduced for faster training
    
    # GB specific
    'case': 'Case552GB',
    'n_units': 2385,                # For reference
    
    # GPU settings - more parallelism
    'num_gpus': 1 if GPU_AVAILABLE else 0,
    'num_env_runners': 8 if GPU_AVAILABLE else 0,  # More workers
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
            'avg_cost_per_mwh', 'num_env_steps', 'training_time_s'
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
        
        avg_cost_per_mwh = -episode_reward_mean * 100 / 5 if episode_reward_mean != 0 else 0.0
        
        row = [
            iteration, timestamp,
            episode_reward_mean, episode_reward_min, episode_reward_max,
            episode_len_mean,
            policy_loss_mean, vf_loss_mean, entropy_mean, kl_loss_mean,
            avg_cost_per_mwh, num_env_steps, training_time
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
            'avg_cost_per_mwh': avg_cost_per_mwh,
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
    costs = [r['avg_cost_per_mwh'] for r in logger.results]
    
    # 1. Reward convergence
    ax = axes[0, 0]
    ax.fill_between(iterations, reward_mins, reward_maxs, alpha=0.3, color='blue')
    ax.plot(iterations, rewards, 'b-', linewidth=2, label='Mean Reward')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Episode Reward')
    ax.set_title('(a) Reward Convergence - GB Network (66 units)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Cost per MWh
    ax = axes[0, 1]
    ax.plot(iterations, costs, 'g-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost per MWh ($)')
    ax.set_title('(b) Generation Cost')
    ax.grid(True, alpha=0.3)
    
    # 3. Policy Loss
    ax = axes[1, 0]
    ax.plot(iterations, policy_losses, 'r-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Policy Loss')
    ax.set_title('(c) Policy Loss')
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
    
    with open(dispatch_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        n_units = len(eval_data['powers'][0]) if eval_data['powers'] else CONFIG['n_units']
        header = ['step', 'time_hhmm', 'total_load_mw', 'total_gen_mw', 'total_cost', 'cost_per_mwh']
        header += [f'unit_{i}_power_mw' for i in range(n_units)]
        writer.writerow(header)
        
        for step in range(len(eval_data['powers'])):
            hour = (step * 30) // 60
            minute = (step * 30) % 60
            time_str = f"{hour:02d}:{minute:02d}"
            
            unit_power_mws = eval_data['powers'][step]
            total_gen = np.sum(unit_power_mws)
            total_load_mw = eval_data['loads'][step]
            total_cost = eval_data['costs'][step]
            cost_per_mwh = total_cost / max(total_gen, 1.0)
            
            row = [step, time_str, total_load_mw, total_gen, total_cost, cost_per_mwh]
            row.extend(unit_power_mws)
            writer.writerow(row)
    
    print(f"  Final episode dispatch saved: {dispatch_csv}")
    
    # Summary CSV
    summary_csv = os.path.join(output_dir, 'final_episode_summary.csv')
    powers = np.array(eval_data['powers'])
    loads = np.array(eval_data['loads'])
    costs = np.array(eval_data['costs'])
    
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'unit'])
        writer.writerow(['total_generation_mw', np.sum(powers), 'MWh'])
        writer.writerow(['total_load_mw', np.sum(loads), 'MWh'])
        writer.writerow(['total_cost', np.sum(costs), '$'])
        writer.writerow(['avg_cost_per_mwh', np.mean(costs / np.maximum(np.sum(powers, axis=1), 1)), '$/MWh'])
        writer.writerow(['num_units', powers.shape[1], 'units'])
        for i in range(min(powers.shape[1], 10)):  # Only show first 10 units in summary
            writer.writerow([f'unit_{i}_avg_power', np.mean(powers[:, i]), 'MW'])
        if powers.shape[1] > 10:
            writer.writerow(['...', f'{powers.shape[1] - 10} more units', ''])
    
    print(f"  Final episode summary saved: {summary_csv}")


# ==============================================================================
# Main Training Function
# ==============================================================================
def train_with_task_system(is_full_training: bool = False):
    """Train multi-agent PPO using the Task system for GB network"""
    
    # Import here to avoid import errors if ray not installed
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env
    from powerzoo.tasks import make_task_env
    
    mode = "Full Training (500 iter)" if is_full_training else "Test Training (10 iter)"
    print("=" * 70)
    print(f"MARL OPF Training with Task System - GB Network - {mode}")
    print("=" * 70)
    
    # GPU info
    print(f"\n[GPU Status]")
    print(f"  CUDA Available: {GPU_AVAILABLE}")
    if GPU_AVAILABLE:
        print(f"  GPU Count: {GPU_COUNT}")
        print(f"  GPU Name: {GPU_NAME}")
        print(f"  Using GPU: Yes (num_gpus={CONFIG['num_gpus']})")
    else:
        print(f"  Using GPU: No (will run on CPU)")
    
    # Create test environment to get space info
    test_env = make_task_env(
        'marl_opf',
        case=CONFIG['case'],
        max_steps=CONFIG['max_steps'],
        scenario={'grid': {'solver_type': 'scipy'}},
    )
    agents = test_env.possible_agents
    obs_space = test_env.observation_space[agents[0]]
    
    print(f"\n[Environment Info - GB Network]")
    print(f"  Case: {CONFIG['case']}")
    print(f"  Number of agents (generators): {len(agents)}")
    print(f"  Obs shape per agent: {obs_space.shape}")
    print(f"  Max steps: {CONFIG['max_steps']}")
    print(f"  Network architecture: {CONFIG['fcnet_hiddens']}")
    print(f"  Num env runners: {CONFIG['num_env_runners']}")
    
    # Register environment
    def make_env(cfg):
        cfg = dict(cfg)
        cfg.setdefault('scenario', {'grid': {'solver_type': 'scipy'}})
        return make_task_env('marl_opf', case=CONFIG['case'], **cfg)
    
    register_env("marl_opf_gb", make_env)
    
    ray.init(ignore_reinit_error=True, log_to_driver=False, include_dashboard=False)
    
    logger = TrainingLogger(OUTPUT_DIR)
    
    try:
        num_iterations = CONFIG['train_iterations'] if is_full_training else CONFIG['test_iterations']
        batch_size = CONFIG['train_batch_size'] if is_full_training else CONFIG['test_batch_size']
        
        # Configure for GPU training (Ray 2.x/3.x New API Stack)
        config = (
            PPOConfig()
            .environment(
                env="marl_opf_gb",
                env_config={'max_steps': CONFIG['max_steps']},
            )
            .framework('torch')
            .env_runners(num_env_runners=CONFIG['num_env_runners'])
            .learners(
                num_learners=1 if GPU_AVAILABLE else 0,
                num_gpus_per_learner=CONFIG['num_gpus'],
            )
            .training(
                lr=CONFIG['lr'],
                gamma=CONFIG['gamma'],
                train_batch_size=batch_size,
            )
            .rl_module(
                model_config={
                    "fcnet_hiddens": CONFIG['fcnet_hiddens'],
                    "fcnet_activation": "tanh",
                }
            )
            .multi_agent(
                policies={
                    agent: (None, obs_space, test_env.action_space[agent], {})
                    for agent in agents
                },
                policy_mapping_fn=lambda agent_id, episode, worker=None, **kw: agent_id,
            )
        )
        
        algo = config.build_algo()
        
        print(f"\n[Training Configuration]")
        print(f"  Iterations: {num_iterations}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {CONFIG['lr']}")
        print(f"  Num GPUs: {CONFIG['num_gpus']}")
        print(f"  Num env runners: {CONFIG['num_env_runners']}")
        print(f"  CSV log: {logger.csv_path}")
        
        print(f"\n[Training Progress]")
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
        eval_env = make_task_env(
            'marl_opf',
            case=CONFIG['case'],
            max_steps=CONFIG['max_steps'],
            scenario={'grid': {'solver_type': 'scipy'}},
        )
        obs, info = eval_env.reset()
        
        eval_data = {'powers': [], 'costs': [], 'loads': []}
        total_reward = 0
        
        try:
            rl_module = algo.get_module()
        except:
            rl_module = None
        
        # Determine device for inference
        device = torch.device("cuda" if GPU_AVAILABLE and CONFIG['num_gpus'] > 0 else "cpu")
        
        for step in range(CONFIG['max_steps']):
            actions = {}
            if rl_module is not None:
                with torch.no_grad():
                    for agent in agents:
                        try:
                            obs_tensor = torch.tensor(obs[agent], dtype=torch.float32).unsqueeze(0).to(device)
                            module = rl_module[agent] if hasattr(rl_module, '__getitem__') else rl_module
                            fwd_out = module.forward_inference({"obs": obs_tensor})
                            if "action_dist_inputs" in fwd_out:
                                action = fwd_out["action_dist_inputs"][:, :1].cpu().numpy().flatten()[0]
                            elif "actions" in fwd_out:
                                action = fwd_out["actions"].cpu().numpy().flatten()[0]
                            else:
                                action = 0.5
                        except:
                            action = 0.5
                        actions[agent] = np.clip(action, 0.0, 1.0)
            else:
                actions = {agent: 0.5 for agent in agents}
            
            obs, rewards, terminateds, truncateds, infos = eval_env.step(actions)
            total_reward += rewards[agents[0]]
            
            # Record data
            agent_info = infos[agents[0]]
            eval_data['powers'].append(agent_info.get('unit_power_mw', np.zeros(CONFIG['n_units'])))
            eval_data['costs'].append(agent_info.get('total_cost', 0))
            eval_data['loads'].append(agent_info.get('total_load_mw', 0))
            
            if terminateds.get("__all__") or truncateds.get("__all__"):
                break
        
        print(f"  Evaluation steps: {len(eval_data['powers'])}")
        print(f"  Total reward: {total_reward:.4f}")
        
        # Save final episode details
        save_final_episode_details(eval_data, OUTPUT_DIR)
        
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
    run_training = '--train' in sys.argv or is_full_training
    skip_test = '--skip-test' in sys.argv
    
    if not skip_test:
        # Quick environment test
        print("=" * 70)
        print("Quick Environment Test - GB Network")
        print("=" * 70)
        
        from powerzoo.tasks import make_task_env, list_tasks
        
        print("Available tasks:", list_tasks())
        env = make_task_env(
            'marl_opf',
            case=CONFIG['case'],
            max_steps=48,
            scenario={'grid': {'solver_type': 'scipy'}},
        )
        print(f"Environment: {type(env).__name__}")
        print(f"Case: {CONFIG['case']}")
        print(f"Number of agents: {len(env.possible_agents)}")
        print(f"Agents (first 5): {env.possible_agents[:5]}...")
        
        obs, info = env.reset()
        for step in range(5):
            actions = {agent: np.array([0.5]) for agent in env.possible_agents}
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            print(f"  Step {step+1}: reward={rewards[env.possible_agents[0]]:.4f}")
        
        print("\n✓ Environment test passed!\n")
    
    # The GB case has thousands of unit agents; require an explicit training flag.
    if run_training:
        train_with_task_system(is_full_training=is_full_training)
    else:
        print("Skipping GB RLlib training by default. Use --train for the short run or --full for full training.")
