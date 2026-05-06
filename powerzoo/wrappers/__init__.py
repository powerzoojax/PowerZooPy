"""PowerZoo Wrappers

Gymnasium-compatible wrappers that adapt PowerZoo environments for standard
RL libraries (Stable-Baselines3, RLlib, CleanRL, …) without modifying the
underlying physics simulation.

Quick start
-----------
Single-agent, flat observation::

    from powerzoo.envs.grid.trans import TransGridEnv
    from powerzoo.wrappers import GymnasiumWrapper, NormalizationWrapper

    env = TransGridEnv()
    env = NormalizationWrapper(GymnasiumWrapper(env))

    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Multi-agent (PettingZoo Parallel)::

    from powerzoo.wrappers import MARLWrapper
    env = MARLWrapper(TransGridEnv(), agent_type='generators')
"""

from powerzoo.wrappers.gym_wrappers import GymnasiumWrapper, NormalizationWrapper
from powerzoo.wrappers.marl_wrapper import MARLWrapper, TaskPettingZooWrapper
from powerzoo.wrappers.flatten import FlattenObservation, FlattenAction, FlattenWrapper
from powerzoo.wrappers.safe_rl_wrapper import (
    CMDPWrapper,
    GymnasiumSafeWrapper,
    SafeRLWrapper,
    TaskCMDPWrapper,
)
from powerzoo.wrappers.forecast_wrapper import ForecastWrapper

__all__ = [
    'GymnasiumWrapper',
    'NormalizationWrapper',
    'MARLWrapper',
    'TaskPettingZooWrapper',
    'FlattenObservation',
    'FlattenAction',
    'FlattenWrapper',
    'TaskCMDPWrapper',
    'CMDPWrapper',
    'SafeRLWrapper',
    'GymnasiumSafeWrapper',
    'ForecastWrapper',
]
