"""Example: Multi-Agent OPF with Task System

Shows how to use the PowerZoo Task system to simplify MARL training code.
Compared to the 1000+ lines in MARL02, this demo achieves the same goal in ~200 lines.

Core API:
    from powerzoo.tasks import make_task_env
    env = make_task_env('marl_opf')

Usage:
    python MARL04_opf_task_demo.py                    # Quick test (20 iterations)
    python MARL04_opf_task_demo.py --full             # Full training (200 iterations)
    python MARL04_opf_task_demo.py --full --skip-test # Skip test, full training only
"""

import os
import sys
import csv
import numpy as np
from datetime import datetime
from typing import Dict, List

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'x_MARL04_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
CONFIG = {
    'test_iterations': 20,
    'test_batch_size': 512,
    'train_iterations': 200,
    'train_batch_size': 1920,
    'lr': 3e-4,
    'gamma': 0.99,
    'max_steps': 48,  # 1 day
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
    ax.set_title('(a) Reward Convergence')
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
        n_units = len(eval_data['powers'][0]) if eval_data['powers'] else 5
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
        for i in range(powers.shape[1]):
            writer.writerow([f'unit_{i}_avg_power', np.mean(powers[:, i]), 'MW'])
    
    print(f"  Final episode summary saved: {summary_csv}")


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
    print(f"MARL OPF Training with Task System - {mode}")
    print("=" * 60)
    
    # Create test environment to get space info
    test_env = make_task_env('marl_opf', max_steps=CONFIG['max_steps'])
    agents = test_env.possible_agents
    obs_space = test_env.observation_space[agents[0]]
    
    print(f"\n[Environment Info]")
    print(f"  Agents: {agents}")
    print(f"  Obs shape: {obs_space.shape}")
    print(f"  Max steps: {CONFIG['max_steps']}")
    
    # Register environment
    register_env("marl_opf_task", lambda cfg: make_task_env('marl_opf', **cfg))
    
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    logger = TrainingLogger(OUTPUT_DIR)
    
    try:
        num_iterations = CONFIG['train_iterations'] if is_full_training else CONFIG['test_iterations']
        batch_size = CONFIG['train_batch_size'] if is_full_training else CONFIG['test_batch_size']
        
        config = (
            PPOConfig()
            .environment(
                env="marl_opf_task",
                env_config={'max_steps': CONFIG['max_steps']},
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
        eval_env = make_task_env('marl_opf', max_steps=CONFIG['max_steps'])
        obs, info = eval_env.reset()
        
        eval_data = {'powers': [], 'costs': [], 'loads': []}
        total_reward = 0
        
        import torch
        try:
            rl_module = algo.get_module()
        except:
            rl_module = None
        
        for step in range(CONFIG['max_steps']):
            actions = {}
            if rl_module is not None:
                with torch.no_grad():
                    for agent in agents:
                        try:
                            obs_tensor = torch.tensor(obs[agent], dtype=torch.float32).unsqueeze(0)
                            module = rl_module[agent] if hasattr(rl_module, '__getitem__') else rl_module
                            fwd_out = module.forward_inference({"obs": obs_tensor})
                            if "action_dist_inputs" in fwd_out:
                                action = fwd_out["action_dist_inputs"][:, :1].numpy().flatten()[0]
                            elif "actions" in fwd_out:
                                action = fwd_out["actions"].numpy().flatten()[0]
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
            eval_data['powers'].append(agent_info.get('unit_power_mw', np.zeros(5)))
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
    skip_test = '--skip-test' in sys.argv
    
    if not skip_test:
        # Quick environment test
        print("=" * 60)
        print("Quick Environment Test")
        print("=" * 60)
        
        from powerzoo.tasks import make_task_env, list_tasks
        
        print("Available tasks:", list_tasks())
        env = make_task_env('marl_opf', max_steps=48)
        print(f"Environment: {type(env).__name__}")
        print(f"Agents: {env.possible_agents}")
        
        obs, info = env.reset()
        for step in range(5):
            actions = {agent: np.array([0.5]) for agent in env.possible_agents}
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            print(f"  Step {step+1}: reward={rewards[env.possible_agents[0]]:.4f}")
        
        print("\n✓ Environment test passed!\n")
    
    # Run training
    train_with_task_system(is_full_training=is_full_training)
