"""Gymnasium environment registration for PowerZoo.

After ``import powerzoo``, all environments below are available via::

    import gymnasium as gym
    env = gym.make("PowerZoo/TransGrid-easy-v0")

Environment ID convention
-------------------------
``PowerZoo/<GridType>-<difficulty>-v<N>``

  GridType   : TransGrid | DistGrid | BatteryArbitrage | MARL-OPF | MARL-DER | MARL-EV
  difficulty : easy | medium | hard  (omitted for task-based envs)
  v<N>       : API version

All factory functions accept an optional ``split`` keyword
(``'train'``, ``'val'``, ``'test'``) so researchers can do::

    env = gym.make("PowerZoo/MARL-OPF-v0", split="test")
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_trans_grid(difficulty: str, **kwargs):
    from powerzoo.envs.grid.trans import TransGridEnv
    from powerzoo.wrappers.gym_wrappers import GymnasiumWrapper
    return GymnasiumWrapper(TransGridEnv(difficulty=difficulty, **kwargs))


def _make_dist_grid(difficulty: str, **kwargs):
    from powerzoo.envs.grid.dist import DistGridEnv
    from powerzoo.wrappers.gym_wrappers import GymnasiumWrapper
    return GymnasiumWrapper(DistGridEnv(difficulty=difficulty, **kwargs))


def _make_task(name: str, **kwargs):
    from powerzoo.tasks.registry import make_task_env
    return make_task_env(name, **kwargs)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _register_all():
    """Register all PowerZoo environments with Gymnasium.

    Called once from ``powerzoo/__init__.py``.  Safe to call multiple times
    (duplicate registrations are skipped with a debug log).
    """
    from gymnasium.envs.registration import register, registry

    # -----------------------------------------------------------------------
    # Raw grid environments (single-agent, difficulty-aware)
    # -----------------------------------------------------------------------
    _raw_envs = [
        # Transmission grid  (DCOPF / ACOPF / DCPF / ACPF, IEEE 5-bus)
        ("PowerZoo/TransGrid-easy-v0",   _make_trans_grid, {"difficulty": "easy"}),
        ("PowerZoo/TransGrid-medium-v0", _make_trans_grid, {"difficulty": "medium"}),
        ("PowerZoo/TransGrid-hard-v0",   _make_trans_grid, {"difficulty": "hard"}),
        # Distribution grid  (FBS, IEEE 33-bus)
        ("PowerZoo/DistGrid-easy-v0",    _make_dist_grid,  {"difficulty": "easy"}),
        ("PowerZoo/DistGrid-medium-v0",  _make_dist_grid,  {"difficulty": "medium"}),
        ("PowerZoo/DistGrid-hard-v0",    _make_dist_grid,  {"difficulty": "hard"}),
    ]

    for env_id, factory, kwargs in _raw_envs:
        if env_id in registry:
            logger.debug("Already registered: %s", env_id)
            continue
        _kwargs = dict(kwargs)  # snapshot
        # Use a closure to capture kwargs correctly
        register(
            id=env_id,
            entry_point=factory,
            kwargs=_kwargs,
        )
        logger.debug("Registered: %s", env_id)

    # -----------------------------------------------------------------------
    # Task-based environments (use powerzoo.tasks system)
    # -----------------------------------------------------------------------
    _task_envs = [
        # Multi-agent OPF — IEEE 5-bus, each generator is an agent
        ("PowerZoo/MARL-OPF-v0",         "marl_opf"),
        ("PowerZoo/MARL-OPF-7d-v0",      "marl_opf_7d"),
        ("PowerZoo/MARL-OPF-118-v0",     "marl_opf_118"),
        # DER storage arbitrage — multiple batteries
        ("PowerZoo/MARL-DER-v0",         "marl_der_arbitrage"),
        ("PowerZoo/MARL-DER-7d-v0",      "marl_der_arbitrage_7d"),
        # EV V2G/G2V — multiple EVs, IEEE 33-bus distribution grid
        ("PowerZoo/MARL-EV-v0",          "marl_ev_v2g"),
        ("PowerZoo/MARL-EV-1d-v0",       "marl_ev_v2g_1d"),
    ]

    for env_id, task_name in _task_envs:
        if env_id in registry:
            logger.debug("Already registered: %s", env_id)
            continue
        _name = task_name  # closure capture
        register(
            id=env_id,
            entry_point=_make_task,
            kwargs={"name": _name},
        )
        logger.debug("Registered: %s", env_id)

    logger.info(
        "PowerZoo: registered %d environments with gymnasium.",
        len(_raw_envs) + len(_task_envs),
    )
