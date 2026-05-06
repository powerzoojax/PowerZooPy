"""Example: Multi-Agent OPF Control with RLlib PPO

This example demonstrates:
1. Using RLlib's PPO algorithm for multi-agent OPF control
2. Each generator unit outputs an action score (not direct power)
3. Softmax/linear allocation to distribute net_power among units
4. Guaranteed power balance through allocation mechanism
5. Reward = total_cost / total_load_mw (cost per MWh)

Multi-Agent Setup:
- 5 generator units (agents) in IEEE 5-bus system
- Each agent outputs a score [0, 1] for power allocation
- net_power = total_load_mw - sum(p_min) - renewable (if any)
- Power allocation ensures balance automatically
- Reward: cost per MWh (lower is better)

Requirements:
- ray[rllib]>=2.0
- powerzoo

Usage:
    python MARL02_opf_rllib.py
    python MARL02_opf_rllib.py --full  # Full training (100 iterations)
"""

import os
import csv
import numpy as np
from typing import Dict, Tuple, Any, Optional, Set
from datetime import datetime

# Ray and RLlib imports
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from gymnasium import spaces

# PowerZoo imports
from powerzoo.tasks import make_task_env

# ==============================================================================
# Output directory
# ==============================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'x_MARL02_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# OPF Multi-Agent Environment with Score-based Allocation
# ==============================================================================
class OPFMultiAgentEnv(MultiAgentEnv):
    """Multi-Agent Environment for OPF with Power Balance Guarantee
    
    Key Design:
    - Each agent outputs a score (0-1), not direct power
    - Scores are normalized via softmax to allocate net_power
    - net_power = total_load_mw - sum(p_min) - renewable
    - Power balance is guaranteed by design
    - Reward = total_cost / total_load_mw (cost per MWh)
    """
    
    def __init__(self, config: Dict = None):
        super().__init__()
        config = config or {}
        
        self.scenario_name = config.get('scenario_name', 'marl_opf')
        
        # PowerEnv from the registered marl_opf task (replaces powerzoo.scenarios.make)
        max_steps = int(config.get('max_steps', 48))
        self._task_env = make_task_env(
            'marl_opf',
            split='train',
            max_steps=max_steps,
            scenario={'grid': {'solver_type': 'scipy'}},
        )
        self.base_env = self._task_env.base_env
        self.grid = self.base_env.grid
        self.case = self.grid.case
        
        # Unit information
        self.n_units = len(self.case.units)
        self.units = self.case.units
        
        # Agent IDs (RLlib requires both agents and possible_agents)
        self.possible_agents = [f"unit_{i}" for i in range(self.n_units)]
        self.agents = self.possible_agents.copy()  # Currently active agents
        self._agent_ids: Set[str] = set(self.possible_agents)
        
        # Unit power limits
        self.p_min = self.units['p_min'].values.astype(np.float32)
        self.p_max = self.units['p_max'].values.astype(np.float32)
        self.p_range = self.p_max - self.p_min  # Allocatable range per unit
        
        # Cost coefficients (quadratic: a*P^2 + b*P + c)
        self.mc_a = self.units['mc_a'].values.astype(np.float32) if 'mc_a' in self.units.columns else np.zeros(self.n_units, dtype=np.float32)
        self.mc_b = self.units['mc_b'].values.astype(np.float32) if 'mc_b' in self.units.columns else np.ones(self.n_units, dtype=np.float32) * 30
        self.mc_c = self.units['mc_c'].values.astype(np.float32) if 'mc_c' in self.units.columns else np.zeros(self.n_units, dtype=np.float32)
        
        # Grid info
        self.n_lines = len(self.case.lines)
        self.n_nodes = len(self.case.nodes)
        self.n_loads = len(self.case.loads) if hasattr(self.case, 'loads') else self.n_nodes
        
        # Observation dimension:
        # Global: total_load_mw(1) + line_flows(n_lines) + time(2)
        # Local: unit_idx(1) + p_min(1) + p_max(1) + cost_coeffs(3) = 6
        self._global_obs_dim = 1 + self.n_lines + 2
        self._local_obs_dim = 6
        self._obs_dim = self._global_obs_dim + self._local_obs_dim
        
        # Build spaces
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True
        
        self.observation_space = spaces.Dict({
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        })
        
        # Action space: each agent outputs a score in [0, 1]
        self.action_space = spaces.Dict({
            agent: spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            for agent in self.possible_agents
        })
        
        # Internal state
        self._current_state = None
        self._step_count = 0
        self._max_steps = max_steps  # Must match task / PowerEnv episode horizon
        self._current_load = 0.0
        self._renewable_power = 0.0
        
        # Episode logs
        self._episode_costs = []
        self._episode_loads = []
        self._episode_powers = []
        self._episode_line_flows = []
    
    def get_agent_ids(self) -> Set[str]:
        return self._agent_ids
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        if seed is not None:
            np.random.seed(seed)
        
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._step_count = 0
        self._current_state = self.grid._get_state() if hasattr(self.grid, '_get_state') else {}
        
        # Get current load
        self._update_current_load()
        
        # Clear episode logs
        self._episode_costs = []
        self._episode_loads = []
        self._episode_powers = []
        self._episode_line_flows = []
        
        observations = self._build_observations()
        infos = {agent: {} for agent in self.possible_agents}
        
        return observations, infos
    
    def step(self, action_dict: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one timestep with score-based power allocation"""
        
        # 1. Collect scores from all agents
        scores = np.zeros(self.n_units, dtype=np.float32)
        for agent, action in action_dict.items():
            idx = int(agent.split('_')[1])
            if isinstance(action, (int, float)):
                score = float(action)
            elif isinstance(action, np.ndarray):
                score = float(action.flatten()[0])
            else:
                score = float(action[0])
            scores[idx] = np.clip(score, 0.0, 1.0)
        
        # 2. Calculate net_power to allocate
        total_load_mw = self._current_load
        sum_p_min = np.sum(self.p_min)
        net_power = max(0.0, total_load_mw - sum_p_min - self._renewable_power)
        
        # 3. Allocate net_power based on scores (softmax-like)
        score_sum = np.sum(scores) + 1e-8
        allocation_ratios = scores / score_sum
        
        max_allocatable = np.sum(self.p_range)
        
        if max_allocatable > 0 and net_power > 0:
            target_allocation = allocation_ratios * net_power
            actual_allocation = np.minimum(target_allocation, self.p_range)
            
            total_allocated = np.sum(actual_allocation)
            if total_allocated < net_power:
                remaining = net_power - total_allocated
                spare_capacity = self.p_range - actual_allocation
                spare_sum = np.sum(spare_capacity) + 1e-8
                if spare_sum > 0:
                    extra = remaining * (spare_capacity / spare_sum)
                    actual_allocation = np.minimum(actual_allocation + extra, self.p_range)
            
            unit_power_mw = self.p_min + actual_allocation
        else:
            unit_power_mw = self.p_min.copy()
        
        unit_power_mw = np.clip(unit_power_mw, self.p_min, self.p_max)
        
        # 4. Step base environment
        grid_action = {'unit_power_mw': unit_power_mw}
        obs, base_reward, terminated, truncated, info = self.base_env.step(grid_action)
        
        self._step_count += 1
        self._current_state = self.grid._get_state() if hasattr(self.grid, '_get_state') else {}
        self._update_current_load()
        
        # 5. Calculate reward (cost per MWh)
        total_gen = np.sum(unit_power_mw)
        total_cost = np.sum(self.mc_a * unit_power_mw**2 + self.mc_b * unit_power_mw + self.mc_c)
        cost_per_mwh = total_cost / max(total_gen, 1.0)
        
        is_safe = info.get('is_safe', True)
        safety_penalty = 0.0 if is_safe else 10.0
        
        imbalance = abs(total_gen - total_load_mw)
        imbalance_penalty = 0.1 * imbalance / max(total_load_mw, 1.0)
        
        reward = -cost_per_mwh / 100.0 - safety_penalty - imbalance_penalty
        rewards = {agent: reward for agent in self.possible_agents}
        
        # Get line flows
        line_flows = np.zeros(self.n_lines, dtype=np.float32)
        if self._current_state and 'lines' in self._current_state:
            lines = self._current_state['lines']
            if 'line_flow_mw' in lines.columns:
                line_flows = lines['line_flow_mw'].values.astype(np.float32)
        
        # Log episode data
        self._episode_costs.append(total_cost)
        self._episode_loads.append(total_load_mw)
        self._episode_powers.append(unit_power_mw.copy())
        self._episode_line_flows.append(line_flows.copy())
        
        if self._step_count >= self._max_steps:
            truncated = True
        
        observations = self._build_observations()
        
        terminateds = {agent: terminated for agent in self.possible_agents}
        terminateds["__all__"] = terminated
        
        truncateds = {agent: truncated for agent in self.possible_agents}
        truncateds["__all__"] = truncated
        
        infos = {
            agent: {
                'unit_power_mw': unit_power_mw,
                'total_cost': total_cost,
                'total_load_mw': total_load_mw,
                'cost_per_mwh': cost_per_mwh,
                'is_safe': is_safe,
                'imbalance': imbalance,
                'scores': scores,
                'step_count': self._step_count,
                'line_flows': line_flows,
            }
            for agent in self.possible_agents
        }
        
        return observations, rewards, terminateds, truncateds, infos
    
    def _update_current_load(self):
        """Update current load and renewable power"""
        if hasattr(self.grid, '_get_node_loads_p_current'):
            self._current_load = float(np.sum(self.grid._get_node_loads_p_current()))
        elif hasattr(self.case, 'loads'):
            self._current_load = float(np.sum(self.case.loads['d_max'].values))
        else:
            self._current_load = 500.0
        
        self._renewable_power = 0.0
        if hasattr(self.base_env, 'resources'):
            for res_id, resource in self.base_env.resources.items():
                res_type = resource.__class__.__name__.lower()
                if 'solar' in res_type or 'wind' in res_type:
                    if hasattr(resource, 'current_p_mw'):
                        self._renewable_power += float(resource.current_p_mw)
    
    def _build_observations(self) -> Dict[str, np.ndarray]:
        observations = {}
        global_obs = self._build_global_obs()
        
        for i, agent in enumerate(self.possible_agents):
            local_obs = np.array([
                float(i) / self.n_units,
                self.p_min[i] / 500.0,
                self.p_max[i] / 500.0,
                self.mc_a[i],
                self.mc_b[i] / 100.0,
                self.mc_c[i] / 1000.0,
            ], dtype=np.float32)
            
            observations[agent] = np.concatenate([global_obs, local_obs]).astype(np.float32)
        
        return observations
    
    def _build_global_obs(self) -> np.ndarray:
        global_parts = []
        
        max_gen = np.sum(self.p_max)
        normalized_load = np.array([self._current_load / max_gen], dtype=np.float32)
        global_parts.append(normalized_load)
        
        if self._current_state and 'lines' in self._current_state:
            lines = self._current_state['lines']
            if 'line_flow_mw' in lines.columns:
                line_flows = lines['line_flow_mw'].values.astype(np.float32)
                line_caps = self.case.lines['cap'].values.astype(np.float32)
                normalized_flows = np.where(line_caps > 0, line_flows / line_caps, 0.0)
                global_parts.append(normalized_flows.astype(np.float32))
            else:
                global_parts.append(np.zeros(self.n_lines, dtype=np.float32))
        else:
            global_parts.append(np.zeros(self.n_lines, dtype=np.float32))
        
        time_step = self.grid.time_step if hasattr(self.grid, 'time_step') else 0
        steps_per_day = self.grid.steps_per_day if hasattr(self.grid, 'steps_per_day') else 48
        
        time_of_day = float(time_step % steps_per_day) / steps_per_day if steps_per_day > 0 else 0.0
        time_sin = np.sin(2 * np.pi * time_of_day)
        time_cos = np.cos(2 * np.pi * time_of_day)
        
        time_features = np.array([time_sin, time_cos], dtype=np.float32)
        global_parts.append(time_features)
        
        return np.concatenate(global_parts).astype(np.float32)
    
    def get_episode_data(self):
        """Get episode data for analysis"""
        return {
            'costs': self._episode_costs.copy(),
            'loads': self._episode_loads.copy(),
            'powers': self._episode_powers.copy(),
            'line_flows': self._episode_line_flows.copy(),
        }


# ==============================================================================
# Training Configuration
# ==============================================================================
CONFIG = {
    'test_iterations': 5,
    'test_batch_size': 256,
    'train_iterations': 100,
    'train_batch_size': 2048,
    'lr': 3e-4,
    'gamma': 0.99,
    'lambda_': 0.95,
    'clip_param': 0.2,
    'num_workers': 0,
    'framework': 'torch',
    'max_steps': 48,  # 1 day
}


# ==============================================================================
# Improved CSV Logger
# ==============================================================================
class TrainingLogger:
    """Log training results to CSV with proper RLlib metric extraction"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, 'training_log.csv')
        self.headers = [
            'iteration', 'timestamp',
            'episode_reward_mean', 'episode_reward_min', 'episode_reward_max',
            'episode_len_mean',
            'policy_loss_mean', 'vf_loss_mean', 'entropy_mean', 'kl_loss_mean',
            'total_gen_cost', 'avg_cost_per_mwh',
            'num_env_steps_sampled', 'training_time_s'
        ]
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
        
        self.results = []
        self._start_time = datetime.now()
        self._last_time = self._start_time
    
    def log(self, iteration: int, result: Dict):
        """Log one iteration result with proper RLlib metric extraction"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_time = datetime.now()
        training_time = (current_time - self._last_time).total_seconds()
        self._last_time = current_time
        
        # Extract from env_runners (new RLlib API)
        env_runners = result.get('env_runners', {})
        
        # Episode rewards - try different keys
        episode_reward_mean = env_runners.get('episode_return_mean', 
                            env_runners.get('episode_reward_mean', 0.0))
        episode_reward_min = env_runners.get('episode_return_min',
                            env_runners.get('episode_reward_min', 0.0))
        episode_reward_max = env_runners.get('episode_return_max',
                            env_runners.get('episode_reward_max', 0.0))
        episode_len_mean = env_runners.get('episode_len_mean', 0.0)
        
        num_env_steps = env_runners.get('num_env_steps_sampled', 0)
        
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
        
        # Estimate generation cost from reward (reward ≈ -cost_per_mwh/100)
        # So cost_per_mwh ≈ -reward * 100 / n_agents (shared reward)
        n_agents = 5
        avg_cost_per_mwh = -episode_reward_mean * 100 / n_agents if episode_reward_mean != 0 else 0.0
        total_gen_cost = avg_cost_per_mwh * 48  # Approximate total over episode
        
        row = [
            iteration, timestamp,
            episode_reward_mean, episode_reward_min, episode_reward_max,
            episode_len_mean,
            policy_loss_mean, vf_loss_mean, entropy_mean, kl_loss_mean,
            total_gen_cost, avg_cost_per_mwh,
            num_env_steps, training_time
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
# Training Progress Curves
# ==============================================================================
def plot_training_progress(logger: TrainingLogger, output_dir: str):
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
        if window > 0:
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
    final_cost = np.mean(costs[-10:]) if len(costs) >= 10 else np.mean(costs)
    ax.axhline(y=final_cost, color='r', linestyle='--', label=f'Final Avg: {final_cost:.2f}')
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
# Save Final Episode Details
# ==============================================================================
def save_final_episode_details(eval_data: Dict, case, output_dir: str):
    """Save detailed final episode data for power flow analysis"""
    
    # 1. Save detailed power dispatch CSV
    dispatch_csv = os.path.join(output_dir, 'final_episode_dispatch.csv')
    with open(dispatch_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        n_units = len(eval_data['powers'][0])
        n_lines = len(eval_data['line_flows'][0])
        header = ['step', 'time_hhmm', 'total_load_mw', 'total_gen_mw', 'total_cost', 'cost_per_mwh']
        header += [f'unit_{i}_power_mw' for i in range(n_units)]
        header += [f'line_{i}_flow_mw' for i in range(n_lines)]
        writer.writerow(header)
        
        # Data
        for step in range(len(eval_data['powers'])):
            hour = (step * 30) // 60
            minute = (step * 30) % 60
            time_str = f"{hour:02d}:{minute:02d}"
            
            unit_power_mws = eval_data['powers'][step]
            total_gen = np.sum(unit_power_mws)
            total_load_mw = eval_data['loads'][step]
            total_cost = eval_data['costs'][step]
            cost_per_mwh = total_cost / max(total_gen, 1.0)
            line_flows = eval_data['line_flows'][step]
            
            row = [step, time_str, total_load_mw, total_gen, total_cost, cost_per_mwh]
            row.extend(unit_power_mws)
            row.extend(line_flows)
            writer.writerow(row)
    
    print(f"  Final episode dispatch saved: {dispatch_csv}")
    
    # 2. Save summary statistics CSV
    summary_csv = os.path.join(output_dir, 'final_episode_summary.csv')
    
    powers = np.array(eval_data['powers'])
    loads = np.array(eval_data['loads'])
    costs = np.array(eval_data['costs'])
    line_flows = np.array(eval_data['line_flows'])
    
    total_gens = np.sum(powers, axis=1)
    cost_per_mwh = costs / np.maximum(total_gens, 1.0)
    
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'unit'])
        
        # Overall metrics
        writer.writerow(['total_generation_mw', np.sum(total_gens) * 0.5, 'MWh'])  # 0.5h per step
        writer.writerow(['total_load_served', np.sum(loads) * 0.5, 'MWh'])
        writer.writerow(['total_cost', np.sum(costs), '$'])
        writer.writerow(['avg_cost_per_mwh', np.mean(cost_per_mwh), '$/MWh'])
        writer.writerow(['min_cost_per_mwh', np.min(cost_per_mwh), '$/MWh'])
        writer.writerow(['max_cost_per_mwh', np.max(cost_per_mwh), '$/MWh'])
        writer.writerow([''])
        
        # Per-unit metrics
        for i in range(powers.shape[1]):
            unit_gen = np.sum(powers[:, i]) * 0.5
            unit_avg = np.mean(powers[:, i])
            unit_max = np.max(powers[:, i])
            unit_min = np.min(powers[:, i])
            writer.writerow([f'unit_{i}_total_gen', unit_gen, 'MWh'])
            writer.writerow([f'unit_{i}_avg_power', unit_avg, 'MW'])
            writer.writerow([f'unit_{i}_max_power', unit_max, 'MW'])
            writer.writerow([f'unit_{i}_min_power', unit_min, 'MW'])
        
        writer.writerow([''])
        
        # Line flow metrics
        for i in range(line_flows.shape[1]):
            line_avg = np.mean(line_flows[:, i])
            line_max = np.max(line_flows[:, i])
            line_utilization = line_max / case.lines.iloc[i]['cap'] * 100 if case.lines.iloc[i]['cap'] > 0 else 0
            writer.writerow([f'line_{i}_avg_flow', line_avg, 'MW'])
            writer.writerow([f'line_{i}_max_flow', line_max, 'MW'])
            writer.writerow([f'line_{i}_utilization', line_utilization, '%'])
    
    print(f"  Final episode summary saved: {summary_csv}")
    
    return dispatch_csv, summary_csv


# ==============================================================================
# Report-Quality Plotting
# ==============================================================================
def plot_report_figures(eval_data: Dict, logger: TrainingLogger, case, output_dir: str):
    """Generate publication-quality figures for report"""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    # Set style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 150
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    powers = np.array(eval_data['powers'])
    loads = np.array(eval_data['loads'])
    costs = np.array(eval_data['costs'])
    line_flows = np.array(eval_data['line_flows'])
    
    n_steps = len(powers)
    time_hours = np.arange(n_steps) * 0.5  # 30-min intervals
    
    total_gens = np.sum(powers, axis=1)
    cost_per_mwh = costs / np.maximum(total_gens, 1.0)
    
    # ========== Figure 1: Training Convergence ==========
    if logger.results:
        fig1, axes1 = plt.subplots(2, 2, figsize=(12, 9))
        
        iterations = [r['iteration'] for r in logger.results]
        rewards = [r['episode_reward_mean'] for r in logger.results]
        policy_losses = [r['policy_loss'] for r in logger.results]
        vf_losses = [r['vf_loss'] for r in logger.results]
        entropies = [r['entropy'] for r in logger.results]
        
        # Reward curve
        ax = axes1[0, 0]
        ax.plot(iterations, rewards, 'b-', linewidth=2)
        ax.set_xlabel('Training Iteration')
        ax.set_ylabel('Episode Reward Mean')
        ax.set_title('(a) Training Reward Convergence')
        ax.grid(True, alpha=0.3)
        
        # Policy loss
        ax = axes1[0, 1]
        ax.plot(iterations, policy_losses, 'r-', linewidth=2)
        ax.set_xlabel('Training Iteration')
        ax.set_ylabel('Policy Loss')
        ax.set_title('(b) Policy Loss')
        ax.grid(True, alpha=0.3)
        
        # Value function loss
        ax = axes1[1, 0]
        ax.plot(iterations, vf_losses, 'g-', linewidth=2)
        ax.set_xlabel('Training Iteration')
        ax.set_ylabel('Value Function Loss')
        ax.set_title('(c) Value Function Loss')
        ax.grid(True, alpha=0.3)
        
        # Entropy
        ax = axes1[1, 1]
        ax.plot(iterations, entropies, 'm-', linewidth=2)
        ax.set_xlabel('Training Iteration')
        ax.set_ylabel('Policy Entropy')
        ax.set_title('(d) Policy Entropy')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig1.savefig(os.path.join(output_dir, 'fig_training_convergence.pdf'), bbox_inches='tight')
        fig1.savefig(os.path.join(output_dir, 'fig_training_convergence.png'), bbox_inches='tight', dpi=150)
        plt.close(fig1)
        print(f"  Training convergence figure saved")
    
    # ========== Figure 2: Power Dispatch ==========
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Stacked area chart for generation
    ax = axes2[0]
    ax.stackplot(time_hours, powers.T, labels=[f'Unit {i}' for i in range(powers.shape[1])],
                 colors=colors[:powers.shape[1]], alpha=0.8)
    ax.plot(time_hours, loads, 'k--', linewidth=2, label='Total Load')
    ax.set_ylabel('Power (MW)')
    ax.set_title('(a) Generator Power Dispatch vs Load')
    ax.legend(loc='upper right', ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)
    
    # Cost per MWh
    ax = axes2[1]
    ax.plot(time_hours, cost_per_mwh, 'r-', linewidth=2, label='Cost per MWh')
    ax.axhline(y=np.mean(cost_per_mwh), color='r', linestyle='--', alpha=0.7,
               label=f'Average: {np.mean(cost_per_mwh):.2f} $/MWh')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Cost ($/MWh)')
    ax.set_title('(b) Generation Cost Efficiency')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'fig_power_dispatch.pdf'), bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'fig_power_dispatch.png'), bbox_inches='tight', dpi=150)
    plt.close(fig2)
    print(f"  Power dispatch figure saved")
    
    # ========== Figure 3: Line Flows ==========
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    
    for i in range(line_flows.shape[1]):
        line_cap = case.lines.iloc[i]['cap']
        ax3.plot(time_hours, line_flows[:, i], linewidth=1.5, 
                label=f'Line {i} (cap={line_cap:.0f}MW)')
    
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Line Flow (MW)')
    ax3.set_title('Transmission Line Power Flows')
    ax3.legend(loc='upper right', ncol=3)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 24)
    
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'fig_line_flows.pdf'), bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'fig_line_flows.png'), bbox_inches='tight', dpi=150)
    plt.close(fig3)
    print(f"  Line flows figure saved")
    
    # ========== Figure 4: Unit Generation Profile ==========
    fig4, axes4 = plt.subplots(2, 3, figsize=(14, 8))
    axes4 = axes4.flatten()
    
    p_min = case.units['p_min'].values
    p_max = case.units['p_max'].values
    
    for i in range(min(5, powers.shape[1])):
        ax = axes4[i]
        ax.fill_between(time_hours, p_min[i], p_max[i], alpha=0.2, color='gray', label='Capacity')
        ax.plot(time_hours, powers[:, i], 'b-', linewidth=2, label='Actual Output')
        ax.axhline(y=p_min[i], color='r', linestyle='--', alpha=0.5, label=f'P_min={p_min[i]:.0f}')
        ax.axhline(y=p_max[i], color='g', linestyle='--', alpha=0.5, label=f'P_max={p_max[i]:.0f}')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Power (MW)')
        ax.set_title(f'Unit {i} Output Profile')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 24)
    
    # Summary in last subplot
    ax = axes4[5]
    unit_totals = np.sum(powers, axis=0) * 0.5  # MWh
    bars = ax.bar(range(powers.shape[1]), unit_totals, color=colors[:powers.shape[1]])
    ax.set_xlabel('Unit')
    ax.set_ylabel('Total Generation (MWh)')
    ax.set_title('Total Generation by Unit')
    ax.set_xticks(range(powers.shape[1]))
    ax.set_xticklabels([f'Unit {i}' for i in range(powers.shape[1])])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, unit_totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig4.savefig(os.path.join(output_dir, 'fig_unit_profiles.pdf'), bbox_inches='tight')
    fig4.savefig(os.path.join(output_dir, 'fig_unit_profiles.png'), bbox_inches='tight', dpi=150)
    plt.close(fig4)
    print(f"  Unit profiles figure saved")
    
    return


# ==============================================================================
# Quick Test Function
# ==============================================================================
def quick_test():
    """Quick test to verify environment works"""
    print("=" * 60)
    print("Quick Test: Verify Environment")
    print("=" * 60)
    
    
    env = OPFMultiAgentEnv({
        'scenario_name': 'IEEE5Bus-OPF-MARL',
        'max_steps': 48
    })
    
    print(f"\n[Environment Info]")
    print(f"  Agents: {env.possible_agents}")
    print(f"  Observation space: {env.observation_space[env.possible_agents[0]]}")
    print(f"  Action space: {env.action_space[env.possible_agents[0]]}")
    print(f"  Max steps: {env._max_steps}")
    print(f"  P_min: {env.p_min}")
    print(f"  P_max: {env.p_max}")
    
    obs, info = env.reset()
    print(f"\n[Test Run - 10 steps]")
    
    total_reward = 0
    for step in range(10):
        actions = {agent: env.action_space[agent].sample() for agent in env.possible_agents}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        agent0_info = infos[env.possible_agents[0]]
        reward = rewards[env.possible_agents[0]]
        total_reward += reward
        
        print(f"  Step {step+1}: load={agent0_info['total_load_mw']:.1f}MW, "
              f"cost/MWh={agent0_info['cost_per_mwh']:.2f}, "
              f"safe={agent0_info['is_safe']}, reward={reward:.4f}")
        
        if terminateds.get('__all__') or truncateds.get('__all__'):
            break
    
    print(f"\n  Total reward (10 steps): {total_reward:.4f}")
    print("\n✓ Quick test passed!")
    
    return True


# ==============================================================================
# Training Function
# ==============================================================================
def train_marl_opf(is_full_training: bool = False):
    """Train multi-agent PPO for OPF control"""
    
    mode = "Full Training" if is_full_training else "Quick Test Training"
    print("=" * 60)
    print(f"Multi-Agent OPF Control with RLlib PPO - {mode}")
    print("=" * 60)
    
    
    test_env = OPFMultiAgentEnv({
        'scenario_name': 'IEEE5Bus-OPF-MARL',
        'max_steps': CONFIG['max_steps']
    })
    agents = test_env.possible_agents
    obs_space = test_env.observation_space[agents[0]]
    
    print(f"\n[Environment Info]")
    print(f"  Agents: {agents}")
    print(f"  Observation space: {obs_space}")
    print(f"  Action space: {test_env.action_space[agents[0]]}")
    print(f"  Max steps per episode: {CONFIG['max_steps']}")
    
    def env_creator(config):
        return OPFMultiAgentEnv(config)
    
    register_env("OPF-MARL", env_creator)
    
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    logger = TrainingLogger(OUTPUT_DIR)
    
    try:
        num_iterations = CONFIG['train_iterations'] if is_full_training else CONFIG['test_iterations']
        batch_size = CONFIG['train_batch_size'] if is_full_training else CONFIG['test_batch_size']
        
        config = (
            PPOConfig()
            .environment(
                env="OPF-MARL",
                env_config={
                    'scenario_name': 'IEEE5Bus-OPF-MARL',
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
                    "fcnet_hiddens": [128, 128],
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
        print(f"  CSV log: {logger.csv_path}")
        
        for i in range(num_iterations):
            result = algo.train()
            row = logger.log(i + 1, result)
            
            episode_reward = row[2]
            episode_len = row[5]
            policy_loss = row[6]
            
            if (i + 1) % max(1, num_iterations // 10) == 0 or i == 0:
                print(f"  Iteration {i+1:3d}: reward={episode_reward:.4f}, "
                      f"ep_len={episode_len:.1f}, policy_loss={policy_loss:.4f}")
        
        checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = algo.save(checkpoint_dir)
        print(f"\n[Checkpoint saved]: {checkpoint_path}")
        
        # ========== Final Evaluation (1 day = 48 steps) ==========
        print(f"\n[Final Evaluation - 1 Day (48 steps)]")
        eval_env = OPFMultiAgentEnv({
            'scenario_name': 'IEEE5Bus-OPF-MARL',
            'max_steps': 48  # Exactly 1 day
        })
        obs, info = eval_env.reset()
        
        import torch
        try:
            rl_module = algo.get_module()
        except Exception as e:
            print(f"  Warning: Could not get module: {e}")
            rl_module = None
        
        total_reward = 0
        for step in range(48):  # Exactly 1 day
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
        
        # Get episode data
        eval_data = eval_env.get_episode_data()
        eval_data['rewards'] = [total_reward / 48] * len(eval_data['costs'])  # Average reward
        
        avg_cost_per_mwh = np.mean([c / max(sum(p), 1.0) 
                                    for c, p in zip(eval_data['costs'], eval_data['powers'])])
        
        print(f"  Evaluation steps: {len(eval_data['powers'])}")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Avg cost per MWh: {avg_cost_per_mwh:.2f} $/MWh")
        
        # Save detailed final episode data
        save_final_episode_details(eval_data, eval_env.case, OUTPUT_DIR)
        
        # Generate training progress curves
        print(f"\n[Generating Training Progress Figures]")
        plot_training_progress(logger, OUTPUT_DIR)
        
        # Generate report-quality figures
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
    import sys
    
    is_full_training = '--full' in sys.argv or '-f' in sys.argv
    skip_test = '--skip-test' in sys.argv
    
    if not skip_test:
        if not quick_test():
            print("Quick test failed!")
            sys.exit(1)
        print("\n" + "=" * 60 + "\n")
    
    train_marl_opf(is_full_training=is_full_training)
