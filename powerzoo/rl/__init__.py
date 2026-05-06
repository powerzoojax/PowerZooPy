"""powerzoo.rl — unified RL entry point for PowerZoo.

Public API::

    from powerzoo.rl import make_env, RLConfig, RewardWrapper, info, describe
    from powerzoo.rl import Trainer   # lazy: requires stable-baselines3

Usage::

    # One-liner environment creation
    env = make_env('battery_arbitrage', split='train')

    # Full training workflow
    trainer = Trainer('battery_arbitrage')
    trainer.train(total_timesteps=200_000)
    results = trainer.evaluate(split='test')

    # Task introspection
    print(describe('marl_opf'))

"""

from powerzoo.rl.config import RLConfig
from powerzoo.rl.env import make_env
from powerzoo.rl.reward import RewardWrapper, MDPFallbackRewardWrapper
from powerzoo.rl.describe import info, describe


def __getattr__(name: str):
    if name == 'Trainer':
        from powerzoo.rl.trainer import Trainer
        return Trainer
    raise AttributeError(f"module 'powerzoo.rl' has no attribute '{name}'")


__all__ = [
    'RLConfig',
    'make_env',
    'RewardWrapper',
    'MDPFallbackRewardWrapper',
    'info',
    'describe',
    'Trainer',
]
