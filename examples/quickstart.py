"""Quickstart: train a SAC agent on battery_arbitrage in < 10 lines.

Usage
-----
    python examples/quickstart.py

Requirements: pip install powerzoo[rl]
"""

import argparse

from powerzoo.rl import make_env, Trainer


parser = argparse.ArgumentParser(description="PowerZoo quickstart")
parser.add_argument("--timesteps", type=int, default=1_000,
                    help="SAC training timesteps; use 10000 for a longer run.")
args = parser.parse_args()

# Create a Gymnasium env (single-line)
env = make_env('battery_arbitrage', split='train')
obs, info = env.reset(seed=42)
print(f"Observation shape: {obs.shape}  Action shape: {env.action_space.shape}")

# Train with SAC
trainer = Trainer('battery_arbitrage', total_timesteps=args.timesteps)
trainer.train()

# Evaluate on the test split
results = trainer.evaluate(split='test')
print(f"Test mean_reward: {results['mean_reward']:.3f}")
