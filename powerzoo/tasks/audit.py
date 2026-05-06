"""Small audit helpers for benchmark-facing task environments."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from powerzoo.tasks.public import list_public_tasks
from powerzoo.tasks.registry import make_task_env


def _looks_multi_agent(env: Any) -> bool:
    return hasattr(env, 'possible_agents') or hasattr(env, 'get_agent_ids')


def _current_agents(env: Any) -> List[str]:
    agents = list(getattr(env, 'agents', []) or getattr(env, 'possible_agents', []))
    if agents:
        return agents
    if hasattr(env, 'get_agent_ids'):
        return sorted(env.get_agent_ids())
    return []


def _sample_action(env: Any, agent: Optional[str] = None) -> Any:
    action_space = getattr(env, 'action_space')
    if callable(action_space):
        if agent is None:
            raise ValueError("Agent ID is required for callable action_space(...)")
        return action_space(agent).sample()
    if agent is None:
        return action_space.sample()
    return action_space[agent].sample()


def _sample_step_actions(env: Any) -> Any:
    if not _looks_multi_agent(env):
        return _sample_action(env)
    return {
        agent: _sample_action(env, agent)
        for agent in _current_agents(env)
    }


def audit_env(env: Any, *, seed: int = 0, step_once: bool = True) -> Dict[str, Any]:
    """Reset and optionally step an env once, returning a compact audit record."""
    result: Dict[str, Any] = {
        'env_type': type(env).__name__,
        'agent_mode': 'multi' if _looks_multi_agent(env) else 'single',
        'reset_ok': False,
        'step_ok': False,
    }

    try:
        observations, infos = env.reset(seed=seed)
        result['reset_ok'] = True
        result['initial_agents'] = _current_agents(env)
        result['info_type'] = type(infos).__name__

        if step_once:
            actions = _sample_step_actions(env)
            step_out = env.step(actions)
            result['step_arity'] = len(step_out)
            result['step_ok'] = True
    finally:
        if hasattr(env, 'close'):
            env.close()

    return result


def audit_task_env(
    name: str,
    *,
    split: Optional[str] = 'train',
    framework: str = 'auto',
    seed: int = 0,
    step_once: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Instantiate a registered task env and run a small lifecycle audit."""
    env = make_task_env(name, split=split, framework=framework, **kwargs)
    result = audit_env(env, seed=seed, step_once=step_once)
    result.update({
        'task_id': name,
        'framework': framework,
    })
    return result


def audit_task_collection(
    task_names: Iterable[str],
    *,
    split: Optional[str] = 'train',
    framework: str = 'auto',
    seed: int = 0,
    step_once: bool = True,
    fail_fast: bool = False,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Audit a collection of registered tasks."""
    results: List[Dict[str, Any]] = []
    for name in task_names:
        try:
            results.append(
                audit_task_env(
                    name,
                    split=split,
                    framework=framework,
                    seed=seed,
                    step_once=step_once,
                    **kwargs,
                )
            )
        except Exception as exc:
            if fail_fast:
                raise
            results.append({
                'task_id': name,
                'framework': framework,
                'reset_ok': False,
                'step_ok': False,
                'error': f'{type(exc).__name__}: {exc}',
            })
    return results


def audit_public_tasks(
    *,
    split: Optional[str] = 'train',
    framework: str = 'auto',
    seed: int = 0,
    step_once: bool = True,
    fail_fast: bool = False,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Audit the explicit public benchmark surface."""
    return audit_task_collection(
        list_public_tasks(),
        split=split,
        framework=framework,
        seed=seed,
        step_once=step_once,
        fail_fast=fail_fast,
        **kwargs,
    )


__all__ = [
    'audit_env',
    'audit_task_env',
    'audit_task_collection',
    'audit_public_tasks',
]
