"""CMDP benchmark wrappers for PowerZoo's Gymnasium environments.

Wrapper layers
--------------
``TaskCMDPWrapper``
    Pass-through 5-tuple wrapper that attaches task-level CMDP metadata:
    full ``constraint_costs`` from the core env, plus task-selected
    ``selected_constraint_costs`` from a :class:`ConstraintSpec`.

``CMDPWrapper``
    Benchmark-facing 6-tuple wrapper returning vector costs directly:
    ``(obs, reward, costs, terminated, truncated, info)``.

``SafeRLWrapper`` / ``GymnasiumSafeWrapper``
    Compatibility projections for libraries that still expect a scalar cost.
    They reduce the selected vector cost to ``selected_cost_sum`` and expose
    the scalar alias as ``info['cost']``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

import gymnasium as gym


def _maybe_call(obj: Any, name: str) -> Any:
    attr = getattr(obj, name, None)
    return attr() if callable(attr) else attr


def _resolve_constraint_spec(env: gym.Env, explicit: Any = None) -> Any:
    if explicit is not None:
        return explicit
    return _maybe_call(env, 'constraint_spec')


def _resolve_full_constraint_names(
    env: gym.Env,
    info: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, ...]:
    names = _maybe_call(env, 'full_constraint_names')
    if names:
        return tuple(names)

    names = _maybe_call(env, 'constraint_names')
    if names:
        return tuple(names)

    if info:
        if info.get('constraint_names'):
            return tuple(info['constraint_names'])
        if info.get('selected_constraint_names'):
            return tuple(info['selected_constraint_names'])
        if isinstance(info.get('costs'), Mapping):
            return tuple(info['costs'].keys())
    return ()


def _constraint_value_from_info(info: Mapping[str, Any], name: str) -> float:
    key = f'cost_{name}'
    if key in info:
        return max(0.0, float(info[key]))
    if isinstance(info.get('costs'), Mapping) and name in info['costs']:
        return max(0.0, float(info['costs'][name]))
    if name == 'resource' and 'cost_resource_violation' in info:
        return max(0.0, float(info['cost_resource_violation']))
    return 0.0


def _assemble_constraint_costs(
    env: gym.Env,
    info: Mapping[str, Any],
    names: Sequence[str],
) -> np.ndarray:
    if not names:
        return np.zeros((0,), dtype=np.float32)

    assemble = getattr(env, 'assemble_constraint_costs', None)
    if callable(assemble):
        return np.asarray(assemble(info, names=tuple(names)), dtype=np.float32).reshape(-1)

    if 'constraint_costs' in info:
        arr = np.asarray(info['constraint_costs'], dtype=np.float32).reshape(-1)
        if arr.shape == (len(names),):
            return np.maximum(arr, 0.0)

    return np.asarray(
        [_constraint_value_from_info(info, name) for name in names],
        dtype=np.float32,
    )


def _annotate_constraint_info(
    env: gym.Env,
    info: Dict[str, Any],
    *,
    constraint_spec: Any = None,
) -> Dict[str, Any]:
    full_names = _resolve_full_constraint_names(env, info)
    full_costs = _assemble_constraint_costs(env, info, full_names)

    info['constraint_names'] = full_names
    info['constraint_costs'] = full_costs
    if full_names:
        info['cost_sum'] = float(full_costs.sum())
    else:
        info.setdefault('cost_sum', max(0.0, float(info.get('cost', 0.0))))

    spec = _resolve_constraint_spec(env, constraint_spec)
    if spec is not None:
        selected_names = tuple(spec.selected_names)
        missing = [name for name in selected_names if name not in full_names]
        if missing:
            raise KeyError(
                "Task CMDP spec selected unknown constraint names. "
                f"Missing={missing}, available={list(full_names)}"
            )
        name_to_idx = {name: idx for idx, name in enumerate(full_names)}
        selected_costs = np.asarray(
            [full_costs[name_to_idx[name]] for name in selected_names],
            dtype=np.float32,
        )
        info['constraint_spec'] = spec
    else:
        selected_names = tuple(info.get('selected_constraint_names', full_names))
        if 'selected_constraint_costs' in info:
            selected_costs = np.asarray(
                info['selected_constraint_costs'],
                dtype=np.float32,
            ).reshape(-1)
        elif selected_names == full_names:
            selected_costs = full_costs
        else:
            selected_costs = _assemble_constraint_costs(env, info, selected_names)

    info['selected_constraint_names'] = selected_names
    info['selected_constraint_costs'] = selected_costs
    info['selected_cost_sum'] = float(selected_costs.sum())
    return info


def _selected_cost_dict(info: Mapping[str, Any]) -> Dict[str, float]:
    names = tuple(info.get('selected_constraint_names', ()))
    costs = np.asarray(info.get('selected_constraint_costs', ()), dtype=np.float32).reshape(-1)
    return {
        name: float(costs[idx])
        for idx, name in enumerate(names)
        if idx < costs.shape[0]
    }


class TaskCMDPWrapper(gym.Wrapper):
    """Attach task-level CMDP selection metadata without changing the 5-tuple API."""

    metadata = {"render_modes": []}

    def __init__(self, env: gym.Env, constraint_spec: Any):
        super().__init__(env)
        self._constraint_spec = constraint_spec
        self.cost_thresholds = tuple(constraint_spec.thresholds)
        self.cost_threshold = constraint_spec.scalar_threshold
        self.fallback_weights = tuple(constraint_spec.fallback_weights)

    def constraint_spec(self):
        return self._constraint_spec

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def full_constraint_names(self) -> Tuple[str, ...]:
        return _resolve_full_constraint_names(self.env)

    def constraint_names(self) -> Tuple[str, ...]:
        return self.full_constraint_names()

    def selected_constraint_names(self) -> Tuple[str, ...]:
        return tuple(self._constraint_spec.selected_names)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, _annotate_constraint_info(self.env, info, constraint_spec=self._constraint_spec)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = _annotate_constraint_info(self.env, info, constraint_spec=self._constraint_spec)
        return obs, reward, terminated, truncated, info


class CMDPWrapper(gym.Wrapper):
    """Expose vector costs directly as a 6-tuple benchmark interface."""

    metadata = {"render_modes": []}

    def __init__(self, env: gym.Env):
        super().__init__(env)
        spec = _resolve_constraint_spec(env)
        self.cost_thresholds = tuple(spec.thresholds) if spec is not None else tuple()
        self.cost_threshold = (
            spec.scalar_threshold
            if spec is not None
            else getattr(env, 'cost_threshold', None)
        )

    def constraint_names(self) -> Tuple[str, ...]:
        names = _maybe_call(self.env, 'selected_constraint_names')
        if names:
            return tuple(names)
        return _resolve_full_constraint_names(self.env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = _annotate_constraint_info(self.env, info)
        return obs, info

    def step(self, action):
        step_out = self.env.step(action)
        if len(step_out) == 6:
            obs, reward, _costs, terminated, truncated, info = step_out
        else:
            obs, reward, terminated, truncated, info = step_out
        info = _annotate_constraint_info(self.env, info)
        costs = np.asarray(
            info.get('selected_constraint_costs', info.get('constraint_costs', ())),
            dtype=np.float32,
        ).reshape(-1)
        return obs, reward, costs, terminated, truncated, info


class SafeRLWrapper(gym.Wrapper):
    """Wrap a Gymnasium env to emit ``(obs, reward, cost, terminated, truncated, info)``."""

    metadata = {"render_modes": []}

    def __init__(self, env: gym.Env, cost_threshold: Optional[float] = None):
        super().__init__(env)
        derived = getattr(env, 'cost_threshold', None)
        self.cost_threshold = (
            float(cost_threshold)
            if cost_threshold is not None
            else (float(derived) if derived is not None else 25.0)
        )

    @staticmethod
    def _extract_cost(info: Dict[str, Any]) -> float:
        if 'selected_cost_sum' in info:
            return max(0.0, float(info['selected_cost_sum']))
        if 'cost_sum' in info:
            return max(0.0, float(info['cost_sum']))
        if 'cost' in info:
            return max(0.0, float(info['cost']))
        return 0.0

    @staticmethod
    def _extract_costs_vector(info: Dict[str, Any]) -> Dict[str, float]:
        if 'selected_constraint_costs' in info and info.get('selected_constraint_names'):
            return _selected_cost_dict(info)
        if isinstance(info.get('costs'), Mapping):
            return {k: max(0.0, float(v)) for k, v in info['costs'].items()}
        return {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = _annotate_constraint_info(self.env, info)
        return obs, info

    def step(self, action):
        step_out = self.env.step(action)
        if len(step_out) == 6:
            obs, reward, _costs, terminated, truncated, info = step_out
        else:
            obs, reward, terminated, truncated, info = step_out
        info = _annotate_constraint_info(self.env, info)
        info['costs'] = self._extract_costs_vector(info)
        info['cost'] = self._extract_cost(info)
        return obs, reward, info['cost'], terminated, truncated, info


class GymnasiumSafeWrapper(gym.Wrapper):
    """Keep Gymnasium's 5-tuple API while projecting vector costs to ``info['cost']``."""

    metadata = {"render_modes": []}

    def __init__(self, env: gym.Env, cost_threshold: Optional[float] = None):
        super().__init__(env)
        derived = getattr(env, 'cost_threshold', None)
        self.cost_threshold = (
            float(cost_threshold)
            if cost_threshold is not None
            else (float(derived) if derived is not None else 25.0)
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = _annotate_constraint_info(self.env, info)
        info['costs'] = _selected_cost_dict(info)
        info['cost'] = SafeRLWrapper._extract_cost(info)
        return obs, info

    def step(self, action):
        step_out = self.env.step(action)
        if len(step_out) == 6:
            obs, reward, _costs, terminated, truncated, info = step_out
        else:
            obs, reward, terminated, truncated, info = step_out
        info = _annotate_constraint_info(self.env, info)
        info['costs'] = _selected_cost_dict(info)
        info['cost'] = SafeRLWrapper._extract_cost(info)
        return obs, reward, terminated, truncated, info


__all__ = [
    'TaskCMDPWrapper',
    'CMDPWrapper',
    'SafeRLWrapper',
    'GymnasiumSafeWrapper',
]
