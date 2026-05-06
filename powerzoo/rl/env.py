"""make_env() — unified single-line environment entry point.

Accepts a task name, inline config dict, YAML config path, or RLConfig object
and returns a ready-to-use Gymnasium (single-agent) or PettingZoo/RLlib
(multi-agent) environment.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import gymnasium as gym

logger = logging.getLogger(__name__)


def make_env(
    config: Union[str, Dict[str, Any], 'RLConfig', Path],
    *,
    split: Optional[str] = None,
    framework: str = 'auto',
    reward: Optional[Union[Dict[str, Any], Callable]] = None,
    normalize: bool = False,
    forecast_horizon: int = 0,
    safe_rl: bool = False,
    cost_threshold: Optional[float] = None,
    seed: Optional[int] = None,
    **task_kwargs,
) -> Any:
    """Create a PowerZoo environment in one line.

    Args:
        config: One of:

            - ``str`` task name: ``make_env('battery_arbitrage')``
            - ``str`` YAML path ending in ``'.yaml'`` or ``'.yml'``:
              ``make_env('experiment.yaml')``
            - ``dict`` inline task config:
              ``make_env({'grid': {...}, 'resources': [...]})``
            - :class:`~powerzoo.rl.config.RLConfig` instance

        split:  Data split — ``'train'`` (default), ``'val'``, or ``'test'``.
                Ignored for anonymous config dicts (a warning is emitted).
        framework: ``'auto'`` (default), ``'rllib'``, or ``'pettingzoo'``.
                   Multi-agent tasks only.
        reward: Optional reward override — a reward-type dict
                (``{'type': 'lmp_arbitrage'}``) or any callable
                ``(state, info) -> float``.  Single-agent only.
        normalize: Wrap the env in
                   :class:`~powerzoo.wrappers.NormalizationWrapper`.
                   Single-agent only.
        forecast_horizon: Append a load-forecast window to the observation
                          via :class:`~powerzoo.wrappers.ForecastWrapper`.
                          Single-agent only.
        safe_rl: Wrap in
                 :class:`~powerzoo.wrappers.GymnasiumSafeWrapper` for
                 CMDP-style cost exposure.  Single-agent only.
        cost_threshold: Cost threshold forwarded to
                        :class:`~powerzoo.wrappers.GymnasiumSafeWrapper`
                        when ``safe_rl=True``.
        seed:   Optional seed passed to ``env.reset(seed=seed)`` immediately.
                The seeded env is returned in a *reset* state.
        **task_kwargs: Extra kwargs forwarded to the Task constructor when
                       ``config`` is a string name.

    Returns:
        - Single-agent tasks: a :class:`gymnasium.Env`
        - Multi-agent tasks: an RLlib ``MultiAgentEnv`` or PettingZoo
          ``ParallelEnv``, depending on ``framework``.

    Raises:
        ValueError: If ``config`` type is not recognised.
        TypeError: If single-agent-only wrappers are requested for a MARL env.

    Example::

        from powerzoo.rl import make_env

        # Single-agent battery task
        env = make_env('battery_arbitrage', split='train')

        # MARL-OPF via PettingZoo
        env = make_env('marl_opf', framework='pettingzoo')

        # Anonymous config with reward override
        env = make_env(
            {'grid': {'type': 'transmission', 'case': 'case5'}, 'resources': []},
            reward={'type': 'zero'},
        )

        # From a YAML config file
        env = make_env('my_experiment.yaml')

    """
    from powerzoo.rl.config import RLConfig

    # ── Step 1: resolve config to a Task object ──────────────────────────
    effective_split = split or 'train'

    if isinstance(config, RLConfig):
        return _make_env_from_rlconfig(
            config,
            split=split,
            framework=framework,
            reward=reward,
            normalize=normalize,
            forecast_horizon=forecast_horizon,
            safe_rl=safe_rl,
            cost_threshold=cost_threshold,
            seed=seed,
            **task_kwargs,
        )

    if isinstance(config, dict):
        from powerzoo.tasks.registry import make_task
        # For MARL tasks, a dict reward is merged into the config before task
        # creation so that the adapter's reward-routing picks it up.  A
        # callable reward cannot be serialised through the config system and
        # is handled (with a warning) in Step 3 below.
        if reward is not None and not callable(reward) and isinstance(reward, dict):
            config = {**config, 'reward': reward}
            reward = None   # consumed here; skip the single-agent wrapper path
        task = make_task(config, split=effective_split)

    elif isinstance(config, (str, Path)):
        config_str = str(config)
        if config_str.endswith('.yaml') or config_str.endswith('.yml'):
            rlcfg = RLConfig.from_yaml(config_str)
            return make_env(
                rlcfg,
                split=split,
                framework=framework,
                reward=reward,
                normalize=normalize,
                forecast_horizon=forecast_horizon,
                safe_rl=safe_rl,
                cost_threshold=cost_threshold,
                seed=seed,
                **task_kwargs,
            )
        else:
            from powerzoo.tasks.registry import make_task
            task = make_task(config_str, split=effective_split, **task_kwargs)

    else:
        raise ValueError(
            f"make_env() expects a task name (str), config dict, YAML path, "
            f"or RLConfig; got {type(config).__name__}."
        )

    # ── Step 2: build the env from the Task ──────────────────────────────
    from powerzoo.tasks.adapters import create_task_env
    env = create_task_env(task, framework=framework)

    # ── Step 3: apply wrapper stack (single-agent only) ──────────────────
    is_single_agent = task.agent_mode == 'single'

    if reward is not None:
        if not is_single_agent:
            warnings.warn(
                "callable reward is not supported for multi-agent envs via "
                "make_env(). Pass a reward dict in config['reward'] or as the "
                "reward= keyword argument (dict only) to override at the task level.",
                UserWarning,
                stacklevel=2,
            )
        else:
            from powerzoo.rl.reward import RewardWrapper
            env = RewardWrapper(env, reward)

    _sa_wrappers_requested = forecast_horizon > 0 or normalize or safe_rl
    if _sa_wrappers_requested and not is_single_agent:
        warnings.warn(
            "normalize / forecast_horizon / safe_rl wrappers are not "
            "applicable to multi-agent envs and will be skipped.",
            UserWarning,
            stacklevel=2,
        )
    elif is_single_agent:
        if forecast_horizon > 0:
            from powerzoo.wrappers import ForecastWrapper
            env = ForecastWrapper(env, horizon=forecast_horizon)

        if normalize:
            from powerzoo.wrappers import NormalizationWrapper
            env = NormalizationWrapper(env)

        if safe_rl:
            from powerzoo.wrappers import GymnasiumSafeWrapper
            kw: Dict[str, Any] = {}
            if cost_threshold is not None:
                kw['cost_threshold'] = cost_threshold
            env = GymnasiumSafeWrapper(env, **kw)

    # ── Step 4: optional initial seed ─────────────────────────────────────
    if seed is not None and is_single_agent:
        env.reset(seed=seed)

    return env


def _make_env_from_rlconfig(
    rlcfg: 'RLConfig',
    *,
    split: Optional[str],
    framework: str,
    reward,
    normalize: bool,
    forecast_horizon: int,
    safe_rl: bool,
    cost_threshold,
    seed,
    **task_kwargs,
) -> Any:
    """Delegate to make_env() after unpacking the RLConfig fields."""
    effective_split = split or rlcfg.split

    effective_reward = reward if reward is not None else rlcfg.reward
    if rlcfg.custom_reward_fn is not None and effective_reward is None:
        effective_reward = rlcfg.custom_reward_fn

    effective_normalize = normalize or rlcfg.normalize
    effective_horizon = max(forecast_horizon, rlcfg.forecast_horizon)
    effective_safe_rl = safe_rl or rlcfg.safe_rl
    effective_cost_threshold = cost_threshold or rlcfg.cost_threshold
    effective_seed = seed if seed is not None else rlcfg.seed
    effective_framework = framework if framework != 'auto' else rlcfg.framework

    if rlcfg.task_name is not None:
        source = rlcfg.task_name
    elif rlcfg.task_config is not None:
        source = rlcfg.task_config
    else:
        raise ValueError("RLConfig must have task_name or task_config set.")

    return make_env(
        source,
        split=effective_split,
        framework=effective_framework,
        reward=effective_reward,
        normalize=effective_normalize,
        forecast_horizon=effective_horizon,
        safe_rl=effective_safe_rl,
        cost_threshold=effective_cost_threshold,
        seed=effective_seed,
        **task_kwargs,
    )
