"""MARL07: OPF with 10 selected controllable units on GB case.

Policy controls only the selected 10 units.
- Controlled block target output: 60% of selected units' total installed capacity.
- Non-selected units: deterministic dispatch proportional to installed capacity.
"""

import os
import sys
import csv
from datetime import datetime
from typing import Dict, List
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Ensure project root is importable when running this file directly from examples/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SELECT_AGENT_LIST_ONE_BASED = [1564, 93, 1782, 1781, 1908, 111, 112, 109, 27, 28]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "x_MARL07_selected_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GPU_AVAILABLE = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if GPU_AVAILABLE else 0
GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "N/A"

CONFIG = {
    "test_iterations": 5,
    "train_iterations": 200,
    "test_batch_size": 480,
    "train_batch_size": 960,
    "lr": 3e-4,
    "gamma": 0.99,
    "max_steps": 48,
    "fcnet_hiddens": [128, 128],
    "case": "Case552GB",
    "control_ratio": 0.60,
    "num_gpus": 1 if GPU_AVAILABLE else 0,
    "num_env_runners": 4 if GPU_AVAILABLE else 0,
}


class SelectedUnitsOPFEnv(MultiAgentEnv):
    """Wrap marl_opf and expose only selected unit agents."""

    def __init__(self, env_config: Dict):
        super().__init__()
        from powerzoo.tasks import make_task_env

        self.env_config = env_config or {}
        self._max_steps = int(self.env_config.get("max_steps", CONFIG["max_steps"]))
        self._control_ratio = float(self.env_config.get("control_ratio", CONFIG["control_ratio"]))

        self.inner_env = make_task_env(
            "marl_opf",
            case=self.env_config.get("case", CONFIG["case"]),
            max_steps=self._max_steps,
            end_date = '2024-01-01',
            action_mode="score",
        )

        self.n_units = self.inner_env.n_units
        self.p_min = self.inner_env.p_min.astype(np.float32)
        self.p_max = self.inner_env.p_max.astype(np.float32)

        selected_one_based = self.env_config.get("selected_units_one_based", SELECT_AGENT_LIST_ONE_BASED)
        self.selected_idx = SELECT_AGENT_LIST_ONE_BASED
        self.non_selected_idx = np.array([i for i in self.inner_env.units['#id'].values if i not in set(self.selected_idx)], dtype=np.int32)

        self.possible_agents = [f"unit_{i}" for i in self.selected_idx]
        self.agents = self.possible_agents.copy()
        self._agent_ids = set(self.possible_agents)

        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True

        self.observation_space = spaces.Dict(
            {agent: self.inner_env.observation_space[agent] for agent in self.possible_agents}
        )
        self.action_space = spaces.Dict(
            {
                agent: spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
                for agent in self.possible_agents
            }
        )

        self.selected_capacity_total = float(np.sum(self.p_max[self.selected_idx]))
        self.selected_control_target = self._control_ratio * self.selected_capacity_total

    def get_agent_ids(self):
        return self._agent_ids

    @staticmethod
    def _one_based_to_zero_based(unit_ids_one_based: List[int], n_units: int) -> np.ndarray:
        idx = np.array(unit_ids_one_based, dtype=np.int32) - 1
        if np.any(idx < 0) or np.any(idx >= n_units):
            raise ValueError(f"Selected unit id out of range. n_units={n_units}, ids={unit_ids_one_based}")
        return idx

    @staticmethod
    def _allocate_with_bounds(target_total: float, p_min: np.ndarray, p_max: np.ndarray, weights: np.ndarray) -> np.ndarray:
        min_sum = float(np.sum(p_min))
        max_sum = float(np.sum(p_max))
        target = float(np.clip(target_total, min_sum, max_sum))

        out = p_min.copy()
        remaining = target - min_sum
        if remaining <= 1e-9:
            return out

        weights = np.clip(weights.astype(np.float64), 1e-8, None)
        available = (p_max - p_min).astype(np.float64)
        active = available > 1e-9

        for _ in range(16):
            if remaining <= 1e-6 or not np.any(active):
                break
            w = weights * active
            w_sum = float(np.sum(w))
            if w_sum <= 1e-12:
                break
            alloc = remaining * (w / w_sum)
            alloc = np.minimum(alloc, available)
            out += alloc.astype(np.float32)
            used = float(np.sum(alloc))
            remaining -= used
            available = (p_max - out).astype(np.float64)
            active = available > 1e-9
        return out

    def _build_full_dispatch(self, action_dict: Dict[str, np.ndarray], total_load_mw: float) -> np.ndarray:
        unit_power_mw = np.zeros(self.n_units, dtype=np.float32)
        # action_dict = dict(zip([f"unit_{idx}" for idx in SELECT_AGENT_LIST_ONE_BASED], [0] * 10))
        selected_scores = []
        for idx in self.selected_idx:
            key = f"unit_{idx}"
            act = action_dict.get(key, 0.5)
            if isinstance(act, np.ndarray):
                v = float(act.flatten()[0])
            elif isinstance(act, ( list, tuple)):
                v = float(act[0])
            else:
                v = float(act)
            selected_scores.append(np.clip(v, 0.0, 1.0))
        selected_scores = np.array(selected_scores, dtype=np.float32)

        sel_p_min = self.p_min[self.selected_idx]
        sel_p_max = self.p_max[self.selected_idx]
        sel_weights = selected_scores + 1e-6
        # selected_power = self._allocate_with_bounds(self.selected_control_target, sel_p_min, sel_p_max, sel_weights)
        selected_power = sel_p_max * selected_scores

        unit_power_mw[self.selected_idx] = selected_power

        non_p_min = self.p_min[self.non_selected_idx]
        non_p_max = self.p_max[self.non_selected_idx]
        fixed_target = total_load_mw - float(np.sum(selected_power))
        non_weights = np.clip(non_p_max, 1e-6, None)
        non_selected_power = self._allocate_with_bounds(fixed_target, non_p_min, non_p_max, non_weights)
        unit_power_mw[self.non_selected_idx] = non_selected_power
        # unit_power_mw[SELECT_AGENT_LIST_ONE_BASED]
        return unit_power_mw

    def reset(self, *, seed=None, options=None):
        obs_all, _ = self.inner_env.reset(seed=seed, options=options)
        observations = {agent: obs_all[agent] for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}
        # self.inner_env.base_env._start_day_id = 0
        return observations, infos

    def step(self, action_dict: Dict[str, np.ndarray]):

        current_load_mw = self.inner_env._get_total_load()
        unit_power_mw = self._build_full_dispatch(action_dict, current_load_mw)

        _, _, terminated, truncated, info = self.inner_env.base_env.step({"unit_power_mw": unit_power_mw})
        self.inner_env._step_count += 1
        self.inner_env._current_state = self.inner_env.grid._get_state() if hasattr(self.inner_env.grid, "_get_state") else {}

        total_cost = self.inner_env._calculate_cost(unit_power_mw)
        reward = self.inner_env._calculate_reward(unit_power_mw, current_load_mw, total_cost, info)

        if self.inner_env._step_count >= self._max_steps:
            truncated = True

        obs_all = self.inner_env._build_observations()
        observations = {agent: obs_all[agent] for agent in self.possible_agents}
        rewards = {agent: reward for agent in self.possible_agents}
        terminateds = {agent: terminated for agent in self.possible_agents}
        terminateds["__all__"] = terminated
        truncateds = {agent: truncated for agent in self.possible_agents}
        truncateds["__all__"] = truncated

        infos = {
            agent: {
                "unit_power_mw": unit_power_mw,
                "selected_control_target_mw": self.selected_control_target,
                "selected_actual_total_mw": float(np.sum(unit_power_mw[self.selected_idx])),
                "fixed_non_selected_total_mw": float(np.sum(unit_power_mw[self.non_selected_idx])),
                "total_load_mw": float(current_load_mw),
                "total_cost": float(total_cost),
                "is_safe": info.get("is_safe", True),
            }
            for agent in self.possible_agents
        }
        return observations, rewards, terminateds, truncateds, infos

    def close(self):
        if hasattr(self.inner_env, "close"):
            self.inner_env.close()
        elif hasattr(self.inner_env, "base_env") and hasattr(self.inner_env.base_env, "close"):
            self.inner_env.base_env.close()


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




def train_with_task_system(is_full_training: bool = False):
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    print("=" * 70)
    print("MARL07 Selected-Unit OPF Training")
    print("=" * 70)
    print(f"CUDA Available: {GPU_AVAILABLE}")
    if GPU_AVAILABLE:
        print(f"GPU Count: {GPU_COUNT}, GPU Name: {GPU_NAME}")

    env_name = "marl_opf_selected10"

    def make_env(cfg):
        probe_env = SelectedUnitsOPFEnv(
            {
                "case": CONFIG["case"],
                "max_steps": CONFIG["max_steps"],
                "control_ratio": CONFIG["control_ratio"],
                "selected_units_one_based": SELECT_AGENT_LIST_ONE_BASED,
                **cfg,
            }
        )
        # inject high mc_c for selected Units
        case = probe_env.inner_env.case
        case.units.loc[SELECT_AGENT_LIST_ONE_BASED, 'mc_c'] = [111,111,114,115,111,114,191,312,200,1927]
        return probe_env

    register_env(env_name, make_env)

    probe_env = make_env({})

    agents = probe_env.possible_agents
    obs_space = probe_env.observation_space[agents[0]]

    print(f"Case: {CONFIG['case']}")
    print(f"Controlled agents: {len(agents)}")
    print(f"Selected units(one-based): {SELECT_AGENT_LIST_ONE_BASED}")
    print(f"Selected capacity total (MW): {probe_env.selected_capacity_total:.3f}")
    print(f"Selected control target (MW): {probe_env.selected_control_target:.3f}")

    ray.init(ignore_reinit_error=True, log_to_driver=False, include_dashboard=False)
    logger = TrainingLogger(OUTPUT_DIR)

    try:
        num_iterations = CONFIG["train_iterations"] if is_full_training else CONFIG["test_iterations"]
        batch_size = CONFIG["train_batch_size"] if is_full_training else CONFIG["test_batch_size"]

        config = (
            PPOConfig()
            .environment(
                env=env_name,
                env_config={
                    "case": CONFIG["case"],
                    "max_steps": CONFIG["max_steps"],
                    "control_ratio": CONFIG["control_ratio"],
                    "selected_units_one_based": SELECT_AGENT_LIST_ONE_BASED,
                },
            )
            .framework("torch")
            .env_runners(num_env_runners=CONFIG["num_env_runners"])
            .learners(
                num_learners=1 if GPU_AVAILABLE else 0,
                num_gpus_per_learner=CONFIG["num_gpus"],
            )
            .training(
                lr=CONFIG["lr"],
                gamma=CONFIG["gamma"],
                train_batch_size=batch_size,
            )
            .rl_module(
                model_config={
                    "fcnet_hiddens": CONFIG["fcnet_hiddens"],
                    "fcnet_activation": "tanh",
                }
            )
            .multi_agent(
                policies={
                    agent: (None, obs_space, probe_env.action_space[agent], {})
                    for agent in agents
                },
                policy_mapping_fn=lambda agent_id, episode, worker=None, **kw: agent_id,
            )
        )


        algo = config.build_algo()

        for i in range(num_iterations):
            result = algo.train()
            row = logger.log(i + 1, result)

            episode_reward = row[2]
            episode_len = row[5]

            if (i + 1) % max(1, num_iterations // 10) == 0 or i == 0:
                print(f"  Iteration {i+1:3d}/{num_iterations}: reward={episode_reward:.4f}, ep_len={episode_len:.1f}")

        checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = algo.save(checkpoint_dir)
        print(f"Checkpoint saved: {checkpoint_path}")
        algo.stop()
    finally:
        ray.shutdown()
    # Plot training progress
    print(f"\n[Generating Figures]")
    plot_training_progress(logger, OUTPUT_DIR)
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    is_full_training = "--full" in sys.argv or "-f" in sys.argv
    train_with_task_system(is_full_training=is_full_training)
