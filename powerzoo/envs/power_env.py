"""PowerEnv - unified benchmark-facing environment facade.

This module keeps the benchmark-facing entry point separate from the lower-level
grid and resource implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from gymnasium import spaces

from powerzoo.envs.base import BaseEnv
from powerzoo.envs.factories import (
    attach_resources,
    create_grid,
)
from powerzoo.tasks.rewards import get_reward_function


def _flattened_space_size(space: spaces.Space) -> int:
    """Return the 1-D size of a Box or Dict space after flattening."""
    if isinstance(space, spaces.Box):
        return int(np.prod(space.shape)) if len(space.shape) > 0 else 1
    if isinstance(space, spaces.Dict):
        return sum(
            _flattened_space_size(subspace)
            for key, subspace in sorted(space.spaces.items())
        )
    raise ValueError(
        f"Unsupported observation space type: {type(space).__name__}. "
        "PowerEnv supports spaces.Box and spaces.Dict."
    )


def _flattened_space_bounds(space: spaces.Space) -> Tuple[np.ndarray, np.ndarray]:
    """Return flattened ``(low, high)`` arrays that match the chosen layout."""
    if isinstance(space, spaces.Box):
        return (
            np.asarray(space.low, dtype=np.float32).reshape(-1),
            np.asarray(space.high, dtype=np.float32).reshape(-1),
        )
    if isinstance(space, spaces.Dict):
        lows: List[np.ndarray] = []
        highs: List[np.ndarray] = []
        for _, subspace in sorted(space.spaces.items()):
            low, high = _flattened_space_bounds(subspace)
            lows.append(low)
            highs.append(high)
        if lows:
            return np.concatenate(lows), np.concatenate(highs)
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    raise ValueError(
        f"Unsupported observation space type: {type(space).__name__}. "
        "PowerEnv supports spaces.Box and spaces.Dict."
    )


def _flatten_observation_value(obs: Any) -> np.ndarray:
    """Flatten a nested observation payload using PowerEnv's sorted-key layout."""
    if isinstance(obs, Mapping):
        parts = [_flatten_observation_value(obs[key]) for key in sorted(obs.keys())]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _collect_resource_violation_costs(
    resource_statuses: Mapping[str, Mapping[str, Any]],
) -> Dict[str, float]:
    """Collect per-resource ``cost_*`` fields into PowerEnv's CMDP cost payload."""
    costs: Dict[str, float] = {}
    total = 0.0

    for res_id, res_info in resource_statuses.items():
        for key, val in res_info.items():
            if key.startswith('cost_') and isinstance(val, (int, float)):
                value = float(max(val, 0.0))
                costs[f'{res_id}/{key}'] = value
                total += value

    costs['cost_resource'] = total
    costs['cost_resource_violation'] = total
    return costs


@dataclass
class _EpisodeClock:
    """Own PowerEnv's episode-day bookkeeping and cyclic time encoding."""

    steps_per_day: int
    max_steps: int
    step: int = 0
    start_day_id: Optional[int] = None

    def day_window(self, total_days: int) -> Tuple[int, int]:
        """Return ``(required_days, max_start_day)`` for the current episode length."""
        required_days = max(1, int(np.ceil(self.max_steps / self.steps_per_day)))
        max_start_day = max(total_days - required_days, 0)
        return required_days, max_start_day

    def validate_start_day(self, day_id: int, total_days: int) -> int:
        """Validate a requested episode start day against available data."""
        start_day = int(day_id)
        if start_day < 0:
            raise ValueError(f"day_id must be >= 0, got {start_day}.")

        required_days, max_start_day = self.day_window(total_days)
        if start_day > max_start_day:
            raise ValueError(
                "day_id is out of range for the available data and episode length. "
                f"Got {start_day}, valid range is [0, {max_start_day}] for "
                f"{total_days} available day(s) and episode max_steps="
                f"{self.max_steps} ({required_days} day window)."
            )

        return start_day

    def choose_start_day(
        self,
        *,
        np_random: np.random.Generator,
        total_days: int,
        requested_day_id: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Choose or validate the episode's starting day."""
        day_id = requested_day_id
        if day_id is None and options is not None:
            day_id = options.get('day_id')
        if day_id is not None:
            return self.validate_start_day(day_id, total_days)

        _, max_start_day = self.day_window(total_days)
        return int(np_random.integers(0, max_start_day + 1))

    def reset(self, *, start_day_id: int) -> None:
        """Start a new episode from a validated day offset."""
        self.step = 0
        self.start_day_id = int(start_day_id)

    def require_started(self) -> int:
        """Return ``start_day_id`` once reset has happened, else fail loudly."""
        if self.start_day_id is None:
            raise RuntimeError("PowerEnv.reset() must be called before step().")
        return self.start_day_id

    @property
    def episode_day(self) -> int:
        """Zero-based day offset within the current episode."""
        return self.step // self.steps_per_day

    @property
    def step_within_day(self) -> int:
        """Zero-based intra-day step used to index time series and time features."""
        return self.step % self.steps_per_day

    @property
    def target_day_id(self) -> int:
        """Absolute day index currently selected inside the dataset window."""
        return self.require_started() + self.episode_day

    @property
    def is_truncated(self) -> bool:
        """Whether the episode has reached the configured step budget."""
        return self.step >= self.max_steps

    def advance(self) -> None:
        """Advance the master clock by one environment step."""
        self.step += 1

    def encode_time_features(self) -> np.ndarray:
        """Encode the current intra-day phase as ``[sin, cos]``."""
        phase = 2.0 * np.pi * self.step_within_day / self.steps_per_day
        return np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)


class PowerEnv(BaseEnv):
    """Unified scenario environment built from one grid plus attached resources."""

    metadata = {'render_modes': []}

    # ====== Initialization ======

    def __init__(self, config: Dict[str, Any], reward_fn: Any = None):
        grid_cfg = config.get('grid', {})
        super().__init__(delta_t_minutes=grid_cfg.get('delta_t_minutes', 30.0))

        self.config = config
        self.name = config.get('name', 'CustomScenario')

        self.grid = create_grid(config.get('grid', {}))
        self.delta_t_minutes = self.grid.delta_t_minutes
        if 1440 % int(self.delta_t_minutes) != 0:
            raise ValueError(
                f"grid.delta_t_minutes={self.delta_t_minutes} must be a "
                "positive integer divisor of 1440."
            )

        self._resource_metadata = self._create_resources(config.get('resources', []))
        self.reward_function = self._resolve_reward_function(reward_fn)
        steps_per_day = max(int(self.grid.steps_per_day), 1)

        episode_config = config.get('episode', {})
        # max_steps is a *step count*, not a physical duration.  The same
        # value corresponds to very different physical windows depending on
        # delta_t_minutes (e.g. 480 steps = 8 h @1 min, 5 d @15 min).
        # Users should adjust max_steps (or use max_hours) for their
        # chosen time resolution.
        if 'max_hours' in episode_config:
            self.max_steps_per_episode = int(
                episode_config['max_hours'] * 60 / self.delta_t_minutes
            )
        else:
            self.max_steps_per_episode = episode_config.get('max_steps', 480)

        self._current_obs: Optional[Dict[str, np.ndarray]] = None
        self._current_state: Optional[Dict[str, Any]] = None
        self._current_info: Optional[Dict[str, Any]] = None
        self._clock = _EpisodeClock(
            steps_per_day=steps_per_day,
            max_steps=self.max_steps_per_episode,
        )
        # Flattened spaces and runtime observations must share one stable
        # resource order; freeze it once during construction.
        self._resource_ids_in_order: Tuple[str, ...] = ()

        self._initialize_spaces(
            config.get('observation', {}),
            config.get('action', {}),
        )

    def _resolve_reward_function(self, reward_fn: Any):
        """Resolve the reward source for this facade.

        Preferred path: reward is injected by a Task.
        Compatibility path: fall back to ``config['reward']`` when present.
        """
        if reward_fn is not None:
            return reward_fn
        reward_config = self.config.get('reward')
        if reward_config is None:
            logging.getLogger(__name__).warning(
                "No reward function configured — falling back to zero reward. "
                "The RL agent will receive reward=0 every step and will not learn."
            )
            return get_reward_function({'type': 'zero'})
        return get_reward_function(reward_config)

    @property
    def resources(self) -> Dict[str, Any]:
        """Direct reference to the grid's attached resources."""
        return self.grid.sub_resources

    def _create_resources(
        self,
        resources_config: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        _, metadata = attach_resources(self.grid, resources_config)
        return metadata

    def _freeze_resource_order(self) -> None:
        """Capture the resource order encoded into observation/action spaces."""
        self._resource_ids_in_order = tuple(sorted(self.resources.keys()))

    def _resource_ids_in_space_order(self) -> Tuple[str, ...]:
        """Return the resource order used by spaces, failing if the registry drifted."""
        current_ids = tuple(sorted(self.resources.keys()))
        if current_ids != self._resource_ids_in_order:
            raise RuntimeError(
                "Resource registry changed after PowerEnv spaces were initialized. "
                "Recreate the environment or rebuild its spaces before stepping."
            )
        return self._resource_ids_in_order

    def _available_step_budget(self) -> int:
        """Return the shortest non-cyclical trace length available to the env.

        Some benchmark splits end with an incomplete tail day. In that case
        ``grid.n_days`` is still computed via ``ceil(T / steps_per_day)`` and
        overstates the set of valid episode start days by one. Use the true
        trace length instead of the nominal day count when validating
        benchmark episodes.
        """
        candidates: List[int] = []

        time_df = getattr(self.grid, '_time_series_data', None)
        if time_df is not None:
            try:
                candidates.append(int(len(time_df)))
            except TypeError:
                pass

        for attr in ('_node_loads_p', '_node_loads_q'):
            arr = getattr(self.grid, attr, None)
            if arr is None:
                continue
            try:
                candidates.append(int(len(arr)))
            except TypeError:
                pass

        for resource in self.resources.values():
            arr = getattr(resource, '_available_cf', None)
            if arr is None or bool(getattr(resource, '_cf_cyclical', False)):
                continue
            try:
                candidates.append(int(len(arr)))
            except TypeError:
                pass

        positive = [n for n in candidates if n > 0]
        if positive:
            return min(positive)

        total_days = max(int(getattr(self.grid, 'n_days', 0)), 1)
        return total_days * max(int(self._clock.steps_per_day), 1)

    def _episode_day_window(self) -> Tuple[int, int, int, int]:
        """Return the valid day window for the configured episode length."""
        steps_per_day = max(int(self._clock.steps_per_day), 1)
        available_steps = max(self._available_step_budget(), 1)
        total_days = max(int(np.ceil(available_steps / steps_per_day)), 1)
        required_days = max(1, int(np.ceil(self._clock.max_steps / steps_per_day)))
        if available_steps < int(self._clock.max_steps):
            max_start_day = 0
        else:
            max_start_day = int((available_steps - int(self._clock.max_steps)) // steps_per_day)
        return total_days, steps_per_day, required_days, max_start_day

    def _validate_start_day(self, day_id: int) -> int:
        """Validate a requested episode start day against available data."""
        start_day = int(day_id)
        if start_day < 0:
            raise ValueError(f"day_id must be >= 0, got {start_day}.")

        total_days, steps_per_day, required_days, max_start_day = self._episode_day_window()
        available_steps = max(self._available_step_budget(), 1)
        if available_steps < int(self._clock.max_steps):
            raise ValueError(
                "Episode max_steps exceeds the available trace length. "
                f"Got max_steps={self._clock.max_steps}, available_steps={available_steps}."
            )
        if start_day > max_start_day:
            raise ValueError(
                "day_id is out of range for the available data and episode length. "
                f"Got {start_day}, valid range is [0, {max_start_day}] for "
                f"{available_steps} available step(s) "
                f"({total_days} nominal day(s), {steps_per_day} steps/day) and "
                f"episode max_steps={self._clock.max_steps} ({required_days} day window)."
            )
        return start_day

    def _build_step_info(
        self,
        info: Dict[str, Any],
        *,
        state: Optional[Dict[str, Any]] = None,
        episode_day: Optional[int] = None,
        step_within_day: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Attach PowerEnv-level benchmark metadata to grid info."""
        info['episode_step'] = self._clock.step
        if episode_day is not None:
            info['episode_day'] = episode_day
        if step_within_day is not None:
            info['step_within_day'] = step_within_day
        info['start_day_id'] = self._clock.start_day_id
        info['delta_t_minutes'] = self.grid.delta_t_minutes
        # Build one resource-status snapshot so the info payload and aggregated
        # cost fields describe the same physical state.
        resource_statuses = self.get_resource_status()
        info['resources'] = resource_statuses
        self._merge_resource_costs(info, resource_statuses)

        if state is not None and 'lmp' in state:
            info['lmp'] = state['lmp']

        return info

    def _merge_resource_costs(
        self,
        info: Dict[str, Any],
        resource_statuses: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Merge resource-level safety costs into the unified env info contract.

        Two aggregated scalars are written for standard single-constraint RL:

        * ``cost_resource`` — total resource constraint cost this step
          (sum of all ``cost_*`` fields across all resources, ≥ 0).
        * ``cost_sum`` — total grid + resource cost for this step.

        For multi-constraint Safe RL algorithms (e.g. CPO, FOCOPS) that require
        a separate cost signal per constraint, wrap the environment and read the
        fine-grained ``<res_id>/cost_<name>`` entries directly from ``info``
        rather than relying on the aggregated scalar.
        """
        resource_costs = _collect_resource_violation_costs(resource_statuses)
        resource_cost = float(resource_costs.get('cost_resource', 0.0))

        info.update(resource_costs)
        info['cost_sum'] = float(info.get('cost_sum', 0.0)) + resource_cost
        info['is_safe'] = bool(info.get('is_safe', True) and resource_cost <= 0.0)
        info['goal_met'] = bool(info.get('goal_met', info['is_safe']) and resource_cost <= 0.0)
        self.attach_constraint_costs(info)

    def constraint_names(self) -> Tuple[str, ...]:
        """Return the full benchmark constraint order for the wrapped scenario."""
        grid_names = tuple(self.grid.constraint_names()) if hasattr(self.grid, 'constraint_names') else ()
        has_resources = bool(self.resources)
        if has_resources and 'resource' not in grid_names:
            return grid_names + ('resource',)
        return grid_names

    # ====== Spaces & Observation ======

    def _resource_observation_size(self, resource: Any) -> int:
        """Return the flattened size of one resource's observation contract."""
        observation_space = getattr(resource, 'observation_space', None)
        if observation_space is None:
            raise ValueError(
                f"Resource {type(resource).__name__} must define observation_space "
                "to be used by PowerEnv."
            )
        return _flattened_space_size(observation_space)

    def _initialize_spaces(self, obs_config: Dict[str, Any], action_config: Dict[str, Any]) -> None:
        """Build the combined benchmark-facing spaces once during construction."""
        self._freeze_resource_order()
        self.observation_space = self._build_observation_space(obs_config)
        self.action_space = self._build_action_space(action_config)

    def _build_observation_space(self, obs_config: Dict[str, Any]) -> spaces.Dict:
        obs_type = obs_config.get('type', 'dict')
        if obs_type != 'dict':
            raise ValueError(f"Unknown observation type: {obs_type}")

        if getattr(self.grid, 'observation_space', None) is None:
            raise ValueError(
                f"{type(self.grid).__name__} must define observation_space to be used by PowerEnv."
            )

        # Iterate in sorted key order so the resource obs vector layout is
        # deterministic regardless of dict insertion order. The same frozen
        # layout is reused when assembling runtime observations. Collect real
        # low/high bounds from each resource's observation_space rather than
        # using ±inf, which benefits RL observation normalisation.
        # Both spaces.Box and spaces.Dict resource spaces are supported.
        resource_lows: List[np.ndarray] = []
        resource_highs: List[np.ndarray] = []
        for res_id in self._resource_ids_in_order:
            resource = self.resources[res_id]
            self._resource_observation_size(resource)
            lo, hi = _flattened_space_bounds(resource.observation_space)
            resource_lows.append(lo)
            resource_highs.append(hi)

        if resource_lows:
            resources_low = np.concatenate(resource_lows)
            resources_high = np.concatenate(resource_highs)
        else:
            resources_low = np.zeros(0, dtype=np.float32)
            resources_high = np.zeros(0, dtype=np.float32)

        return spaces.Dict({
            'grid': self.grid.observation_space,
            'resources': spaces.Box(
                low=resources_low,
                high=resources_high,
                dtype=np.float32,
            ),
            'time': spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            ),
        })

    def _build_action_space(self, action_config: Dict[str, Any]) -> spaces.Dict:
        action_type = action_config.get('type', 'dict')
        if action_type != 'dict':
            raise ValueError(f"Unknown action type: {action_type}")

        action_spaces: Dict[str, spaces.Space] = {}
        grid_action_space = getattr(self.grid, 'action_space', None)
        if grid_action_space is not None:
            grid_shape = getattr(grid_action_space, 'shape', ()) or ()
            if int(np.prod(grid_shape)) > 0:
                action_spaces['unit_power_mw'] = grid_action_space

        for res_id in self._resource_ids_in_order:
            resource = self.resources[res_id]
            resource_space = getattr(resource, 'action_space', None)
            if resource_space is not None:
                action_spaces[res_id] = resource_space

        return spaces.Dict(action_spaces)

    def _choose_start_day(self, day_id: Optional[int], options: Optional[Dict[str, Any]]) -> int:
        requested = day_id
        if requested is None and options is not None:
            requested = options.get('day_id')
        if requested is not None:
            return self._validate_start_day(int(requested))

        _total_days, _steps_per_day, _required_days, max_start_day = self._episode_day_window()
        return int(self.np_random.integers(0, max_start_day + 1))

    # ====== RL Interface Methods ======

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        day_id: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._clock.reset(start_day_id=self._choose_start_day(day_id, options))

        # Derive a child seed from our own RNG so that grid's RNG
        # sequence is independent of PowerEnv's (but still reproducible).
        grid_seed = int(self.np_random.integers(2**31)) if seed is not None else None
        self._current_state, grid_info = self.grid.reset(
            seed=grid_seed,
            options=options,
            day_id=self._clock.start_day_id,
        )
        self.reward_function.reset()

        self._current_obs = self._build_agent_observation(self._current_state)
        self._current_info = self._build_step_info(grid_info, state=self._current_state)

        return self._current_obs, self._current_info

    def _current_state_or_raise(self) -> Dict[str, Any]:
        if self._current_state is None:
            raise RuntimeError(
                "PowerEnv.reset() must be called before accessing the current observation."
            )
        return self._current_state

    def _coerce_resource_action(self, resource: Any, value: Any) -> Any:
        """Convert convenience array/scalar inputs into the resource's dict action shape."""
        if value is None or isinstance(value, dict):
            return value

        action_names = list(getattr(resource, 'action_names', []))
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if len(action_names) == 1:
            return {action_names[0]: float(arr[0])}
        if action_names and len(action_names) == arr.size:
            return {
                key: float(arr[idx])
                for idx, key in enumerate(action_names)
            }
        return value

    def _normalize_env_action(self, action: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize the facade action into the dict structure expected by the grid."""
        if action is None:
            return {}
        if not isinstance(action, dict):
            raise TypeError(
                "PowerEnv.step expects a dict action. "
                f"Got {type(action).__name__}."
            )

        grid_action: Dict[str, Any] = {}
        for key, value in action.items():
            if key in self.resources:
                grid_action[key] = self._coerce_resource_action(self.resources[key], value)
                continue

            grid_action[key] = value

        return grid_action

    def step(self, action: Optional[Dict[str, Any]]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Apply a Dict action and return ``(obs, reward, terminated, truncated, info)``.

        Action structure
        ----------------
        The native action space is ``gymnasium.spaces.Dict`` keyed by resource
        ID (e.g. ``{'battery_0': {'charge_rate_mw': 0.5}}``), plus an optional
        ``'unit_power_mw'`` key for direct generator setpoints.

        .. note::
            Standard RL libraries that do not support ``Dict`` action spaces
            (e.g. Stable-Baselines3) require an ``ActionWrapper`` such as
            :class:`powerzoo.wrappers.FlattenActionWrapper` to flatten the Dict
            into a contiguous array and unflatten it back before calling this
            method.  Passing a flat array directly will raise ``TypeError``.
        """
        self._clock.require_started()

        grid_action = self._normalize_env_action(action)

        current_episode_day = self._clock.episode_day
        step_within_day = self._clock.step_within_day
        target_day_id = self._clock.target_day_id

        # PowerEnv is the master clock.  We explicitly set both grid attributes
        # before every grid.step() call so the correct time-series slice is used
        # regardless of the grid's own internal counter.  Within a day the
        # assignment is a no-op (the value already matches); at a day boundary
        # time_step resets to 0 for the new day while day_id advances by 1.
        self.grid.day_id = target_day_id
        self.grid.time_step = step_within_day

        self._current_state, _grid_reward, terminated, _grid_truncated, self._current_info = self.grid.step(grid_action)

        self._clock.advance()
        truncated = self._clock.is_truncated

        self._current_info = self._build_step_info(
            self._current_info,
            state=self._current_state,
            episode_day=current_episode_day,
            step_within_day=step_within_day,
        )

        reward = self.reward_function.compute(self._current_state, self._current_info)
        self._current_obs = self._build_agent_observation(self._current_state)

        return self._current_obs, reward, terminated, truncated, self._current_info

    def _build_resource_observation_vector(self) -> np.ndarray:
        """Assemble the flat resource-observation block used in ``obs['resources']``."""
        parts: List[np.ndarray] = []
        for res_id in self._resource_ids_in_space_order():
            parts.append(_flatten_observation_value(self.resources[res_id].obs()))
        result = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

        expected_dim = self.observation_space['resources'].shape[0]
        if result.shape[0] != expected_dim:
            raise RuntimeError(
                f"Resource observation dimension mismatch: assembled {result.shape[0]} "
                f"values but observation_space['resources'] expects {expected_dim}. "
                "Ensure each resource's obs() output matches its observation_space."
            )
        return result

    def obs(self, state: Any = None) -> Dict[str, np.ndarray]:
        """Return the current observation dict.

        If *state* is provided it is forwarded to ``_build_agent_observation``;
        otherwise the cached ``_current_state`` is used.
        """
        if state is None:
            state = self._current_state_or_raise()
        return self._build_agent_observation(state)

    def _encode_time_features(self) -> np.ndarray:
        """Encode the master clock as cyclic time features for the agent."""
        return self._clock.encode_time_features()

    def _build_agent_observation(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Build the benchmark-facing observation dict from the current simulator state."""
        grid_obs = np.asarray(self.grid.obs(state), dtype=np.float32).reshape(-1)

        # Use PowerEnv's master episode clock for time encoding instead of
        # grid.time_step, so the feature is independent of whether grid.step()
        # internally increments its own counter.
        time_obs = self._encode_time_features()
        # TODO: extend with multi-day periodic features (day-of-week, holiday
        # flags, etc.) once the data layer exposes reliable calendar metadata
        # per day_id.  Currently day_id is a plain integer index with no
        # calendar information attached.

        return {
            'grid': grid_obs,
            'resources': self._build_resource_observation_vector(),
            'time': time_obs,
        }

    def render(self, mode: str = 'human'):
        if hasattr(self.grid, 'render'):
            return self.grid.render(mode=mode)
        return None

    def close(self) -> None:
        if hasattr(self.grid, 'close'):
            self.grid.close()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PowerEnv':
        import yaml

        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return cls(config)

    # ====== Status & Diagnostics ======

    def get_resource_metadata(self, resource_id: str) -> Dict[str, Any]:
        return self._resource_metadata.get(resource_id, {})

    def get_resource_status(self) -> Dict[str, Dict[str, Any]]:
        return {
            res_id: resource.status()
            for res_id, resource in self.resources.items()
        }

    def __repr__(self) -> str:
        resource_types = [
            self._resource_metadata.get(rid, {}).get('type', 'unknown')
            for rid in self.resources.keys()
        ]
        return (
            f"PowerEnv(name='{self.name}', "
            f"case={self.grid.case.__class__.__name__}, "
            f"resources={resource_types})"
        )
