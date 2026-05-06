"""Thin shared helpers for task adapter step/output handling."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np


def coerce_scalar_action(action: Any) -> float:
    """Extract a scalar float from a scalar, array, or sequence action."""
    if isinstance(action, (int, float)):
        return float(action)
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr[0])


def build_parallel_done_dicts(
    agents: Iterable[str],
    *,
    terminated: bool,
    truncated: bool,
) -> Tuple[Dict[str, bool], Dict[str, bool]]:
    """Build RLlib-style per-agent done dicts with ``__all__``."""
    agent_list = list(agents)
    terminateds = {agent: terminated for agent in agent_list}
    terminateds['__all__'] = terminated
    truncateds = {agent: truncated for agent in agent_list}
    truncateds['__all__'] = truncated
    return terminateds, truncateds


def maybe_share_rewards(
    rewards: Mapping[str, float],
    *,
    reward_type: str,
) -> Dict[str, float]:
    """Broadcast a shared reward when the task is cooperative."""
    result = {agent: float(value) for agent, value in rewards.items()}
    if reward_type != 'shared' or not result:
        return result
    shared_reward = sum(result.values()) / len(result)
    return {agent: shared_reward for agent in result}


def make_agent_info(
    *,
    extra: Mapping[str, Any] | None = None,
    cost: float = 0.0,
    costs: Mapping[str, float] | None = None,
    constraint_names: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """Build a normalized task-agent info payload."""
    payload = dict(extra or {})
    payload['cost'] = float(cost)
    cost_dict = {
        key: float(value)
        for key, value in (costs or {}).items()
    }
    payload['costs'] = cost_dict
    names = tuple(constraint_names) if constraint_names is not None else tuple(cost_dict.keys())
    payload['constraint_names'] = names
    payload['constraint_costs'] = np.asarray(
        [cost_dict.get(name, 0.0) for name in names],
        dtype=np.float32,
    )
    return payload
