"""Example: Multi-Agent OPF Control with RLlib PPO - 7 Days Training

This example extends MARL02 with:
1. Extended training period: 7 days (336 steps per episode)
2. More training iterations: 200
3. Longer episode for better policy learning

Inherits all core functionality from MARL02:
- Score-based power allocation
- Power balance guarantee
- Cost per MWh reward

Usage:
    python MARL03_opf_rllib_7days.py
    python MARL03_opf_rllib_7days.py --full  # Full training (200 iterations)
"""

import os
import sys

# Add parent directory to path to import from MARL02
sys.path.insert(0, os.path.dirname(__file__))

# Import all necessary components from MARL02
from MARL02_opf_rllib import (
    OPFMultiAgentEnv,
    TrainingLogger,
    save_final_episode_details,
    plot_report_figures,
    quick_test as marl02_quick_test,
)

import numpy as np
import csv
from datetime import datetime
from typing import Dict

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# ==============================================================================
# Output directory
# ==============================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'x_MARL03_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# Configuration for 7-day training
# ==============================================================================
CONFIG = {
    # Training config
    'test_iterations': 10,
    'test_batch_size': 512,
    'train_iterations': 200,  # More iterations for 7-day episodes
    'train_batch_size': 4096,  # Larger batch for longer episodes
    
    # Shared config
    'lr': 3e-4,
    'gamma': 0.99,
    'lambda_': 0.95,
    'clip_param': 0.2,
    'num_workers': 0,
    'framework': 'torch',
    
    # Episode config - 7 days
    'max_steps': 336,  # 7 days * 48 steps/day
    'days': 7,
}


# ==============================================================================
# Extended Training Logger with more metrics
# ==============================================================================
class ExtendedTrainingLogger(TrainingLogger):
    """Extended logger with additional metrics for 7-day training"""
    
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        # Override CSV path
        self.csv_path = os.path.join(output_dir, 'training_log.csv')
        
        # Extended headers
        self.headers = [
            'iteration', 'timestamp',
            'episode_reward_mean', 'episode_reward_min', 'episode_reward_max',
            'episode_len_mean',
            'policy_loss_mean', 'vf_loss_mean', 'entropy_mean', 'kl_loss_mean',
            'total_gen_cost', 'avg_cost_per_mwh',
            'num_env_steps_sampled', 'training_time_s',
            'cumulative_time_s', 'episodes_total'
        ]
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
        
        self._cumulative_time = 0.0
        self._episodes_total = 0
    
    def log(self, iteration: int, result: Dict):
        """Log with extended metrics"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_time = datetime.now()
        training_time = (current_time - self._last_time).total_seconds()
        self._last_time = current_time
        self._cumulative_time += training_time
        
        # Extract from env_runners
        env_runners = result.get('env_runners', {})
        
        episode_reward_mean = env_runners.get('episode_return_mean', 
                            env_runners.get('episode_reward_mean', 0.0))
        episode_reward_min = env_runners.get('episode_return_min',
                            env_runners.get('episode_reward_min', 0.0))
        episode_reward_max = env_runners.get('episode_return_max',
                            env_runners.get('episode_reward_max', 0.0))
        episode_len_mean = env_runners.get('episode_len_mean', 0.0)
        num_env_steps = env_runners.get('num_env_steps_sampled', 0)
        num_episodes = env_runners.get('num_episodes', 0)
        self._episodes_total += num_episodes
        
        # Extract learner stats
        learners = result.get('learners', {})
        
        policy_losses = []
        vf_losses = []
        entropies = []
        kl_losses = []
        
        for key, stats in learners.items():
            if key == '__all_modules__':
                continue
            if isinstance(stats, dict):
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
        
        n_agents = 5
        avg_cost_per_mwh = -episode_reward_mean * 100 / n_agents if episode_reward_mean != 0 else 0.0
        total_gen_cost = avg_cost_per_mwh * CONFIG['max_steps']
        
        row = [
            iteration, timestamp,
            episode_reward_mean, episode_reward_min, episode_reward_max,
            episode_len_mean,
            policy_loss_mean, vf_loss_mean, entropy_mean, kl_loss_mean,
            total_gen_cost, avg_cost_per_mwh,
            num_env_steps, training_time,
            self._cumulative_time, self._episodes_total
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
# Extended Plotting for Training Progress
# ==============================================================================
def plot_training_progress(logger, output_dir: str):
    """Plot comprehensive training progress curves"""
    import matplotlib.pyplot as plt
    
    if not logger.results:
        return
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    iterations = [r['iteration'] for r in logger.results]
    rewards = [r['episode_reward_mean'] for r in logger.results]
    reward_mins = [r['episode_reward_min'] for r in logger.results]
    reward_maxs = [r['episode_reward_max'] for r in logger.results]
    policy_losses = [r['policy_loss'] for r in logger.results]
    vf_losses = [r['vf_loss'] for r in logger.results]
    entropies = [r['entropy'] for r in logger.results]
    kl_losses = [r.get('kl_loss', 0) for r in logger.results]
    costs = [r['avg_cost_per_mwh'] for r in logger.results]
    
    # 1. Reward convergence with confidence band
    ax = axes[0, 0]
    ax.fill_between(iterations, reward_mins, reward_maxs, alpha=0.3, color='blue', label='Min-Max Range')
    ax.plot(iterations, rewards, 'b-', linewidth=2, label='Mean Reward')
    # Add smoothed trend
    if len(rewards) > 10:
        window = min(10, len(rewards) // 5)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(iterations[window-1:], smoothed, 'r--', linewidth=2, label='Smoothed Trend')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Episode Reward')
    ax.set_title('(a) Reward Convergence')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2. Cost per MWh
    ax = axes[0, 1]
    ax.plot(iterations, costs, 'g-', linewidth=2)
    ax.axhline(y=np.mean(costs[-10:]) if len(costs) >= 10 else np.mean(costs), 
               color='r', linestyle='--', label=f'Final Avg: {np.mean(costs[-10:]) if len(costs) >= 10 else np.mean(costs):.2f}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost per MWh ($)')
    ax.set_title('(b) Generation Cost Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Policy Loss
    ax = axes[1, 0]
    ax.plot(iterations, policy_losses, 'r-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Policy Loss')
    ax.set_title('(c) Policy Loss')
    ax.grid(True, alpha=0.3)
    
    # 4. Value Function Loss
    ax = axes[1, 1]
    ax.plot(iterations, vf_losses, 'purple', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('VF Loss')
    ax.set_title('(d) Value Function Loss')
    ax.grid(True, alpha=0.3)
    
    # 5. Entropy (exploration)
    ax = axes[2, 0]
    ax.plot(iterations, entropies, 'orange', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Entropy')
    ax.set_title('(e) Policy Entropy (Exploration)')
    ax.grid(True, alpha=0.3)
    
    # 6. KL Divergence
    ax = axes[2, 1]
    ax.plot(iterations, kl_losses, 'cyan', linewidth=2)
    ax.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='KL Target')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('KL Divergence')
    ax.set_title('(f) KL Divergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'training_progress.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'training_progress.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    print(f"  Training progress figures saved")


# ==============================================================================
# Quick Test
# ==============================================================================
def quick_test():
    """Quick test for 7-day environment"""
    print("=" * 60)
    print("Quick Test: 7-Day OPF Environment")
    print("=" * 60)
    
    
    env = OPFMultiAgentEnv({
        'scenario_name': 'IEEE5Bus-OPF-MARL-7D',
        'max_steps': CONFIG['max_steps']
    })
    
    print(f"\n[Environment Info]")
    print(f"  Agents: {env.possible_agents}")
    print(f"  Max steps: {env._max_steps} (= {CONFIG['days']} days)")
    print(f"  P_min: {env.p_min}")
    print(f"  P_max: {env.p_max}")
    
    obs, info = env.reset()
    print(f"\n[Test Run - 20 steps]")
    
    total_reward = 0
    for step in range(20):
        actions = {agent: env.action_space[agent].sample() for agent in env.possible_agents}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        reward = rewards[env.possible_agents[0]]
        total_reward += reward
        
        if step % 5 == 0:
            agent0_info = infos[env.possible_agents[0]]
            print(f"  Step {step+1}: load={agent0_info['total_load_mw']:.1f}MW, "
                  f"cost/MWh={agent0_info['cost_per_mwh']:.2f}, reward={reward:.4f}")
        
        if terminateds.get('__all__') or truncateds.get('__all__'):
            break
    
    print(f"\n  Total reward (20 steps): {total_reward:.4f}")
    print("\n✓ Quick test passed!")
    
    return True


# ==============================================================================
# Main Training Function
# ==============================================================================
def train_marl_opf_7days(is_full_training: bool = False):
    """Train multi-agent PPO for 7-day OPF control"""
    
    mode = "Full Training (200 iter)" if is_full_training else "Quick Test Training"
    print("=" * 60)
    print(f"Multi-Agent OPF - 7 Days - RLlib PPO - {mode}")
    print("=" * 60)
    
    
    test_env = OPFMultiAgentEnv({
        'scenario_name': 'IEEE5Bus-OPF-MARL-7D',
        'max_steps': CONFIG['max_steps']
    })
    agents = test_env.possible_agents
    obs_space = test_env.observation_space[agents[0]]
    
    print(f"\n[Environment Info]")
    print(f"  Agents: {agents}")
    print(f"  Observation space: {obs_space}")
    print(f"  Action space: {test_env.action_space[agents[0]]}")
    print(f"  Max steps per episode: {CONFIG['max_steps']} ({CONFIG['days']} days)")
    
    def env_creator(config):
        return OPFMultiAgentEnv(config)
    
    register_env("OPF-MARL-7D", env_creator)
    
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    logger = ExtendedTrainingLogger(OUTPUT_DIR)
    
    try:
        num_iterations = CONFIG['train_iterations'] if is_full_training else CONFIG['test_iterations']
        batch_size = CONFIG['train_batch_size'] if is_full_training else CONFIG['test_batch_size']
        
        config = (
            PPOConfig()
            .environment(
                env="OPF-MARL-7D",
                env_config={
                    'scenario_name': 'IEEE5Bus-OPF-MARL-7D',
                    'max_steps': CONFIG['max_steps']
                },
            )
            .framework(CONFIG['framework'])
            .env_runners(
                num_env_runners=CONFIG['num_workers'],
                rollout_fragment_length='auto',
            )
            .training(
                lr=CONFIG['lr'],
                gamma=CONFIG['gamma'],
                lambda_=CONFIG['lambda_'],
                clip_param=CONFIG['clip_param'],
                train_batch_size=batch_size,
                model={
                    "fcnet_hiddens": [256, 256],  # Larger network for longer episodes
                    "fcnet_activation": "relu",
                },
            )
            .multi_agent(
                policies={
                    agent: (None, obs_space, test_env.action_space[agent], {})
                    for agent in agents
                },
                policy_mapping_fn=lambda agent_id, episode, worker=None, **kwargs: agent_id,
            )
        )
        
        algo = config.build()
        
        print(f"\n[Training]")
        print(f"  Algorithm: PPO (Independent PPO)")
        print(f"  Iterations: {num_iterations}")
        print(f"  Train batch size: {batch_size}")
        print(f"  Network: [256, 256]")
        print(f"  CSV log: {logger.csv_path}")
        
        for i in range(num_iterations):
            result = algo.train()
            row = logger.log(i + 1, result)
            
            episode_reward = row[2]
            episode_len = row[5]
            policy_loss = row[6]
            
            if (i + 1) % max(1, num_iterations // 20) == 0 or i == 0:
                print(f"  Iteration {i+1:3d}/{num_iterations}: reward={episode_reward:.4f}, "
                      f"ep_len={episode_len:.1f}, policy_loss={policy_loss:.4f}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = algo.save(checkpoint_dir)
        print(f"\n[Checkpoint saved]: {checkpoint_path}")
        
        # Plot training progress
        print(f"\n[Generating Training Progress Figures]")
        plot_training_progress(logger, OUTPUT_DIR)
        
        # Final Evaluation (7 days)
        print(f"\n[Final Evaluation - 7 Days ({CONFIG['max_steps']} steps)]")
        eval_env = OPFMultiAgentEnv({
            'scenario_name': 'IEEE5Bus-OPF-MARL-7D',
            'max_steps': CONFIG['max_steps']
        })
        obs, info = eval_env.reset()
        
        import torch
        try:
            rl_module = algo.get_module()
        except Exception as e:
            print(f"  Warning: Could not get module: {e}")
            rl_module = None
        
        total_reward = 0
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
                                action_mean = fwd_out["action_dist_inputs"]
                                if isinstance(action_mean, torch.Tensor):
                                    action = action_mean[:, :1].numpy().flatten()[0]
                                else:
                                    action = 0.5
                            elif "actions" in fwd_out:
                                action = fwd_out["actions"].numpy().flatten()[0]
                            else:
                                action = 0.5
                        except Exception:
                            action = 0.5
                        
                        actions[agent] = np.clip(action, 0.0, 1.0)
            else:
                for agent in agents:
                    actions[agent] = 0.5
            
            obs, rewards, terminateds, truncateds, infos = eval_env.step(actions)
            total_reward += rewards[agents[0]]
            
            if terminateds.get("__all__") or truncateds.get("__all__"):
                break
        
        eval_data = eval_env.get_episode_data()
        
        avg_cost_per_mwh = np.mean([c / max(sum(p), 1.0) 
                                    for c, p in zip(eval_data['costs'], eval_data['powers'])])
        
        print(f"  Evaluation steps: {len(eval_data['powers'])}")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Avg cost per MWh: {avg_cost_per_mwh:.2f} $/MWh")
        
        # Save detailed results
        save_final_episode_details(eval_data, eval_env.case, OUTPUT_DIR)
        
        # Generate report figures
        print(f"\n[Generating Report Figures]")
        plot_report_figures(eval_data, logger, eval_env.case, OUTPUT_DIR)
        
        algo.stop()
        
    finally:
        ray.shutdown()
    
    print(f"\n[Complete] All results saved to: {OUTPUT_DIR}")
    return True


# ==============================================================================
# Entry point
# ==============================================================================
if __name__ == "__main__":
    is_full_training = '--full' in sys.argv or '-f' in sys.argv
    skip_test = '--skip-test' in sys.argv
    
    if not skip_test:
        if not quick_test():
            print("Quick test failed!")
            sys.exit(1)
        print("\n" + "=" * 60 + "\n")
    
    train_marl_opf_7days(is_full_training=is_full_training)

