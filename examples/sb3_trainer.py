"""Example: Training with Stable-Baselines3

Demonstrates how to plug PowerZoo tasks into SB3 without any binding to
the benchmark internals.  This script is intentionally minimal.

Usage
-----
    # Single-agent battery arbitrage (train split)
    python examples/sb3_trainer.py --task battery_arbitrage --split train
"""

from __future__ import annotations

import argparse
import sys

# ---------------------------------------------------------------------------
# Dependency check (SB3 is optional)
# ---------------------------------------------------------------------------
try:
    from stable_baselines3 import SAC, PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    sys.exit(
        "stable-baselines3 is required for this example.\n"
        "Install: pip install powerzoo[rl]"
    )

import numpy as np
from powerzoo.tasks import make_task_env
from powerzoo.benchmarks.policies import RandomPolicy, evaluate
from powerzoo.wrappers import GymnasiumWrapper


def ensure_gymnasium_env(raw_env):
    """Wrap legacy envs only; task envs may already be Gymnasium-compatible."""
    return raw_env if hasattr(raw_env, 'observation_space') else GymnasiumWrapper(raw_env)


def main(task_name: str, split: str, timesteps: int = 50_000):
    print(f"\n{'='*60}")
    print(f"  PowerZoo SB3 Example")
    print(f"  task={task_name}  split={split}  timesteps={timesteps:,}")
    print('='*60)

    # ------------------------------------------------------------------
    # 1. Create training environment (Gymnasium-compliant single-agent)
    # ------------------------------------------------------------------
    env = ensure_gymnasium_env(make_task_env(task_name, split=split))
    check_env(env, warn=True)

    # ------------------------------------------------------------------
    # 2. Baseline: random policy before training
    # ------------------------------------------------------------------
    rand_result = evaluate(RandomPolicy(env.action_space), env, n_episodes=5, verbose=True)
    print(f"\nRandom baseline:  mean_reward = {rand_result['mean_reward']:.2f}")

    # ------------------------------------------------------------------
    # 3. Train with SAC
    # ------------------------------------------------------------------
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps, progress_bar=True)

    # ------------------------------------------------------------------
    # 4. Evaluate on test split
    # ------------------------------------------------------------------
    test_env = ensure_gymnasium_env(make_task_env(task_name, split='test'))

    class SB3Policy:
        def __init__(self, model):
            self.model = model
        def act(self, obs, info=None):
            action, _ = self.model.predict(obs, deterministic=True)
            return action

    test_result = evaluate(SB3Policy(model), test_env, n_episodes=10, verbose=True)
    print(f"\nTest result:  mean_reward = {test_result['mean_reward']:.2f}")

    return test_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="battery_arbitrage", help="Task name")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--timesteps", type=int, default=50_000)
    args = parser.parse_args()

    main(args.task, args.split, args.timesteps)
