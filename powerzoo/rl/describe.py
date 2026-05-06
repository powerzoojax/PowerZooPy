"""info() and describe() — AI-friendly task introspection.

These functions build a structured summary of a registered or anonymous
PowerZoo task, including space shapes, reward/cost contracts, and a
YAML-ready config template.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union


def info(
    task: Union[str, Dict[str, Any], 'Task'],
    *,
    format: str = 'dict',
) -> Union[Dict[str, Any], str]:
    """Return structured metadata about a task.

    Args:
        task:   Task name (``str``), inline config dict, or a
                :class:`~powerzoo.tasks.base.Task` instance.
        format: ``'dict'`` (default) returns a Python dict; ``'json'``
                returns a formatted JSON string.

    Returns:
        Dict with keys: ``task_id``, ``description``, ``agent_mode``,
        ``difficulty``, ``observation_space``, ``action_space``, ``reward``,
        ``cost``, ``splits``, ``eval_protocol``, ``config_template``.

    Example::

        from powerzoo.rl import info

        d = info('battery_arbitrage')
        print(d['observation_space'])   # {'type': 'Box', 'shape': [24], ...}
        print(d['reward'])              # {'default': 'battery_lmp_arbitrage', ...}

    """
    result = _build_info_dict(task)
    if format == 'json':
        return json.dumps(result, indent=2, default=str)
    return result


def describe(task: Union[str, Dict[str, Any], 'Task']) -> str:
    """Return a human-readable multi-line description of a task.

    Args:
        task: Task name, inline config dict, or Task instance.

    Returns:
        A plain-text summary string suitable for printing.

    Example::

        from powerzoo.rl import describe

        print(describe('marl_opf'))

    """
    d = _build_info_dict(task)
    lines = [
        f"Task:        {d['task_id']}",
        f"Description: {d.get('description', 'N/A')}",
        f"Agent mode:  {d['agent_mode']}",
        f"Difficulty:  {d['difficulty']}",
        "",
    ]

    obs = d.get('observation_space')
    if obs:
        lines.append(f"Observation: {obs.get('type', '?')} shape={obs.get('shape', '?')}")
    act = d.get('action_space')
    if act:
        lines.append(f"Action:      {act.get('type', '?')} shape={act.get('shape', '?')}")

    reward = d.get('reward')
    if reward:
        lines.append(f"Reward:      {reward.get('default', 'N/A')}")
        avail = reward.get('available')
        if avail:
            lines.append(f"             (available: {', '.join(avail)})")

    cost = d.get('cost')
    if cost:
        has_cmdp = cost.get('has_cmdp', False)
        threshold = cost.get('threshold')
        threshold_vec = cost.get('thresholds')
        if has_cmdp:
            lines.append(f"Cost (CMDP): threshold={threshold}")
            names = cost.get('constraint_names') or []
            if names:
                lines.append(f"             constraints={', '.join(names)}")
            if threshold_vec:
                lines.append(f"             thresholds={threshold_vec}")
            training_contract = cost.get('training_contract')
            if training_contract:
                lines.append(f"             training={training_contract}")
        else:
            lines.append("Cost (CMDP): none")

    splits = d.get('splits')
    if splits:
        lines.append("")
        lines.append("Data splits:")
        for k, v in splits.items():
            if v:
                lines.append(f"  {k}: {v}")

    ep = d.get('eval_protocol')
    if ep:
        lines.append("")
        lines.append(f"Eval protocol: {ep}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────

def _build_info_dict(task: Union[str, Dict[str, Any], Any]) -> Dict[str, Any]:
    """Build the canonical info dict regardless of input type."""
    from powerzoo.tasks.registry import make_task, get_task_info
    from powerzoo.tasks.public import PUBLIC_TASKS, get_public_task_info
    from powerzoo.tasks.rewards.registry import list_reward_types
    from powerzoo.tasks.base import Task, ConfigTask, ConfigMultiAgentTask

    # ── resolve to a Task instance ────────────────────────────────────────
    task_instance: Optional[Any] = None
    task_id: str = 'unknown'

    if isinstance(task, str):
        task_id = task
        task_instance = make_task(task)
    elif isinstance(task, dict):
        task_id = task.get('name', '_config_task')
        task_instance = make_task(task)
    elif isinstance(task, Task):
        task_instance = task
        task_id = getattr(task, 'name', 'unknown')
    else:
        raise TypeError(
            f"info() expects a task name, config dict, or Task instance; "
            f"got {type(task).__name__}."
        )

    # ── base metadata ──────────────────────────────────────────────────────
    result: Dict[str, Any] = {
        'task_id': task_id,
        'description': getattr(task_instance, 'description', ''),
        'agent_mode': getattr(task_instance, 'agent_mode', 'single'),
        'difficulty': getattr(task_instance, 'difficulty', 'unknown'),
    }

    # ── merge public benchmark card data if available ─────────────────────
    if isinstance(task, str) and task in PUBLIC_TASKS:
        try:
            card = get_public_task_info(task)
            for key in ('grid_type', 'grid_case', 'route', 'frameworks',
                        'benchmark_family', 'reward_contract', 'cost_contract',
                        'action_contract', 'supported_observation_modes'):
                if key in card:
                    result[key] = card[key]
        except Exception:
            pass

    # ── observation / action spaces ───────────────────────────────────────
    try:
        env = task_instance.create_single_agent_env()
        result['observation_space'] = _describe_space(env.observation_space)
        result['action_space'] = _describe_space(env.action_space)
        env.close()
    except Exception:
        result['observation_space'] = None
        result['action_space'] = None

    # ── reward info ───────────────────────────────────────────────────────
    scenario_reward = task_instance.get_scenario_config().get('reward') or {}
    default_reward = scenario_reward.get('type', 'zero')
    result['reward'] = {
        'default': default_reward,
        'available': list_reward_types(),
    }

    # ── cost / CMDP info ──────────────────────────────────────────────────
    spec = task_instance.constraint_spec() if hasattr(task_instance, 'constraint_spec') else None
    threshold = getattr(task_instance, 'effective_cost_threshold', None)
    ep = getattr(task_instance.__class__, 'eval_protocol', None) or {}
    has_cmdp = (threshold is not None) or bool(ep.get('cost_threshold'))
    result['cost'] = {
        'has_cmdp': has_cmdp,
        'threshold': threshold,
        'thresholds': list(spec.thresholds) if spec is not None else (
            list(ep.get('cost_thresholds', ())) or None
        ),
        'constraint_names': list(spec.selected_names) if spec is not None else (
            list(ep.get('constraint_names', ())) or None
        ),
        'fallback_weights': list(spec.fallback_weights) if spec is not None else None,
        'training_contract': getattr(task_instance, 'training_contract', 'legacy'),
    }

    # ── data splits ───────────────────────────────────────────────────────
    split_dates = getattr(task_instance.__class__, 'SPLIT_DATES', None)
    if split_dates:
        result['splits'] = split_dates
    else:
        result['splits'] = None

    # ── eval protocol ──────────────────────────────────────────────────────
    result['eval_protocol'] = ep if ep else None

    # ── YAML config template ──────────────────────────────────────────────
    scenario_cfg = task_instance.get_scenario_config()
    result['config_template'] = {
        'task': {
            'name': task_id if isinstance(task, str) else None,
            'split': 'train',
        },
        'wrappers': {
            'normalize': False,
            'forecast_horizon': 0,
            'safe_rl': False,
        },
        'reward': {'type': default_reward},
        'trainer': {
            'algorithm': 'SAC',
            'total_timesteps': 100_000,
        },
    }

    return result


def _describe_space(space: Any) -> Optional[Dict[str, Any]]:
    """Summarize a gymnasium space as a plain dict."""
    try:
        import gymnasium as gym
        import numpy as np
        if isinstance(space, gym.spaces.Box):
            return {
                'type': 'Box',
                'shape': list(space.shape),
                'low': float(np.min(space.low)),
                'high': float(np.max(space.high)),
                'dtype': str(space.dtype),
            }
        if isinstance(space, gym.spaces.Discrete):
            return {'type': 'Discrete', 'n': int(space.n)}
        if isinstance(space, gym.spaces.Dict):
            return {
                'type': 'Dict',
                'keys': list(space.spaces.keys()),
            }
        return {'type': type(space).__name__}
    except Exception:
        return None
