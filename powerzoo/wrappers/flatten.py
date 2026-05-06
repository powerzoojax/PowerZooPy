"""Flatten Wrappers for PowerZoo Environments

Provides wrappers to flatten Dict observation/action spaces to Box spaces,
making PowerZoo environments compatible with standard RL libraries.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces


class FlattenObservation(gym.ObservationWrapper):
    """Flatten Dict observation space to Box space

    Converts dict observations to a single flat vector, making the environment
    compatible with standard RL algorithms (PPO, SAC, etc.) from libraries like
    Stable-Baselines3, RLlib, CleanRL.

    Args:
        env: Environment with Dict observation space
        custom_flatten_fn: Optional custom flattening function
                          If None, uses default concatenation

    Example:
        >>> from powerzoo.envs.power_env import PowerEnv
        >>> from powerzoo.wrappers.flatten import FlattenObservation
        >>>
        >>> env = PowerEnv(config)
        >>> env = FlattenObservation(env)
        >>>
        >>> obs, info = env.reset()  # Now returns np.array instead of dict
        >>> print(obs.shape)   # (20,) - flattened vector
    """

    def __init__(self, env: gym.Env, custom_flatten_fn: Optional[Callable] = None):
        super().__init__(env)

        self.custom_flatten_fn = custom_flatten_fn

        if isinstance(env.observation_space, spaces.Dict):
            self._key_slices = {}
            start_idx = 0

            for key, space in env.observation_space.spaces.items():
                if isinstance(space, spaces.Box):
                    size = int(np.prod(space.shape))
                    self._key_slices[key] = (start_idx, start_idx + size)
                    start_idx += size
                else:
                    raise ValueError(f"Unsupported space type for key '{key}': {type(space)}")

            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(start_idx,),
                dtype=np.float32
            )

    def observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten dict observation to vector"""
        if self.custom_flatten_fn is not None:
            return self.custom_flatten_fn(obs)

        flat_obs = []
        for key in self.env.observation_space.spaces.keys():
            if key in obs:
                flat_obs.append(obs[key].flatten())

        return np.concatenate(flat_obs).astype(np.float32)


class FlattenAction(gym.ActionWrapper):
    """Flatten Box action space to Dict action space

    Converts flat action vectors to dict actions, allowing standard RL algorithms
    to control PowerZoo environments with dict action spaces.

    Args:
        env: Environment with Dict action space
        action_keys: List of action keys to include (in order)
        action_shapes: Dict of {key: shape} for each action component

    Example:
        >>> from powerzoo.envs.power_env import PowerEnv
        >>> from powerzoo.wrappers.flatten import FlattenAction
        >>>
        >>> env = PowerEnv(config)
        >>> env = FlattenAction(env,
        ...                     action_keys=['p_mw'],
        ...                     action_shapes={'p_mw': (1,)})
        >>>
        >>> action = env.action_space.sample()  # Now samples from Box space
        >>> obs, reward, done, truncated, info = env.step(action)
    """

    def __init__(self, env: gym.Env, action_keys: List[str],
                 action_shapes: Dict[str, Tuple[int, ...]],
                 action_bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        super().__init__(env)

        self.action_keys = action_keys
        self.action_shapes = action_shapes
        self.action_bounds = action_bounds or {}

        total_size = sum(int(np.prod(action_shapes[key])) for key in action_keys)

        low = np.full(total_size, -np.inf, dtype=np.float32)
        high = np.full(total_size, np.inf, dtype=np.float32)

        self._key_slices = {}
        start_idx = 0
        for key in action_keys:
            size = int(np.prod(action_shapes[key]))
            if key in self.action_bounds:
                low[start_idx:start_idx+size] = self.action_bounds[key][0]
                high[start_idx:start_idx+size] = self.action_bounds[key][1]
            self._key_slices[key] = (start_idx, start_idx + size)
            start_idx += size

        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(total_size,),
            dtype=np.float32
        )

    def action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert flat action to dict action"""
        action_dict = {}

        for key in self.action_keys:
            start, end = self._key_slices[key]
            action_dict[key] = action[start:end].reshape(self.action_shapes[key])

        return action_dict


class FlattenWrapper(gym.Wrapper):
    """Combined wrapper for flattening both observation and action spaces

    Intelligently flattens dict spaces based on controlled resources.

    Args:
        env: Environment to wrap (should be PowerEnv instance)
        resource_names: List of resource names to control (e.g., ['bat0', 'wind_0'])
                       If None, controls all resources
        obs_keys: List of observation keys to include (e.g., ['grid', 'resources', 'time'])
                 If None, includes all observations
        custom_obs_fn: Optional custom observation flattening function
        custom_action_fn: Optional custom action mapping function

    Example:
        >>> from powerzoo.envs.power_env import PowerEnv
        >>> from powerzoo.wrappers.flatten import FlattenWrapper
        >>>
        >>> env = PowerEnv(config)
        >>> env = FlattenWrapper(env, resource_names=['bat0'])
        >>>
        >>> from stable_baselines3 import PPO
        >>> model = PPO('MlpPolicy', env, verbose=1)
        >>> model.learn(total_timesteps=10000)
    """

    def __init__(self, env: gym.Env,
                 resource_names: Optional[List[str]] = None,
                 obs_keys: Optional[List[str]] = None,
                 custom_obs_fn: Optional[Callable] = None,
                 custom_action_fn: Optional[Callable] = None):
        super().__init__(env)

        self.resource_names = resource_names
        self.obs_keys = obs_keys
        self.custom_obs_fn = custom_obs_fn
        self.custom_action_fn = custom_action_fn
        self._obs_space_initialized = False

        self.base_env = self._get_base_env()

        if self.resource_names is None and hasattr(self.base_env, 'resources'):
            self.resource_names = list(self.base_env.resources.keys())

        self._build_action_space()
        self._build_observation_space_estimate()

    def _get_base_env(self):
        """Get the underlying PowerEnv instance"""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env

    def _build_action_space(self):
        """Build flattened action space based on controlled resources"""
        if not self.resource_names:
            self.action_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32
            )
            self._action_mapping = []
            return

        self._action_mapping = []
        action_dims = []
        action_lows = []
        action_highs = []

        for res_id in self.resource_names:
            if not hasattr(self.base_env, 'resources') or res_id not in self.base_env.resources:
                raise ValueError(f"Resource '{res_id}' not found in environment")

            resource = self.base_env.resources[res_id]
            res_class = resource.__class__.__name__

            res_action_space = getattr(resource, 'action_space', None)

            if 'Battery' in res_class:
                power_mw = getattr(resource, 'power_mw', 20.0)
                if getattr(resource, 'enable_q_control', False):
                    s_rated = getattr(resource, 's_rated_mva', power_mw)
                    action_key = 'pq'
                    action_dim = 2
                    action_bound = max(power_mw, s_rated)
                    action_low = -action_bound
                    action_high = action_bound
                else:
                    action_key = 'p_mw'
                    action_dim = 1
                    action_low = -power_mw
                    action_high = power_mw
            elif 'Wind' in res_class or 'Solar' in res_class:
                action_key = 'curtailment'
                action_dim = 1
                action_low = 0.0
                action_high = 1.0
            elif res_action_space is not None and hasattr(res_action_space, 'shape'):
                # Generic path: use the resource's own action_space
                action_dim = int(np.prod(res_action_space.shape))
                if action_dim == 1:
                    action_names = getattr(resource, 'action_names', None) or ['p_mw']
                    action_key = action_names[0]
                else:
                    action_key = res_id
                action_low = float(res_action_space.low.min())
                action_high = float(res_action_space.high.max())
            else:
                action_key = 'p'
                action_dim = 1
                action_low = -100.0
                action_high = 100.0

            start_idx = sum(action_dims)
            end_idx = start_idx + action_dim
            self._action_mapping.append({
                'resource_id': res_id,
                'action_key': action_key,
                'action_dim': action_dim,
                'slice': (start_idx, end_idx),
                'bounds': (action_low, action_high)
            })

            action_dims.append(action_dim)
            action_lows.extend([action_low] * action_dim)
            action_highs.extend([action_high] * action_dim)

        total_dims = sum(action_dims)
        if total_dims > 0:
            self.action_space = spaces.Box(
                low=np.array(action_lows, dtype=np.float32),
                high=np.array(action_highs, dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32
            )

    def _build_observation_space_estimate(self):
        """Build estimated flattened observation space (updated on first reset)"""
        if isinstance(self.env.observation_space, spaces.Dict):
            total_size = 0
            obs_keys = self.obs_keys if self.obs_keys else self.env.observation_space.spaces.keys()

            for key in obs_keys:
                if key in self.env.observation_space.spaces:
                    space = self.env.observation_space.spaces[key]
                    if isinstance(space, spaces.Box):
                        total_size += int(np.prod(space.shape))

            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_size,) if total_size > 0 else (1,),
                dtype=np.float32
            )
        else:
            self.observation_space = self.env.observation_space

    def _update_observation_space(self, obs: Dict[str, np.ndarray]):
        """Update observation space based on actual observation (called on first reset)"""
        self._obs_key_slices = {}
        start_idx = 0

        obs_keys = self.obs_keys if self.obs_keys else obs.keys()

        for key in obs_keys:
            if key in obs:
                size = int(np.prod(obs[key].shape))
                self._obs_key_slices[key] = (start_idx, start_idx + size)
                start_idx += size

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(start_idx,),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        """Reset and flatten observation"""
        obs, info = self.env.reset(**kwargs)

        if not self._obs_space_initialized:
            self._update_observation_space(obs)
            self._obs_space_initialized = True

        return self._flatten_obs(obs), info

    def step(self, action):
        """Convert flat action to dict and step"""
        dict_action = self._unflatten_action(action)
        obs, reward, terminated, truncated, info = self.env.step(dict_action)
        return self._flatten_obs(obs), reward, terminated, truncated, info

    def _flatten_obs(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten dict observation to vector"""
        if self.custom_obs_fn is not None:
            return self.custom_obs_fn(obs)

        flat_obs = []
        obs_keys = self.obs_keys if self.obs_keys else self.env.observation_space.spaces.keys()

        for key in obs_keys:
            if key in obs:
                flat_obs.append(obs[key].flatten())

        return np.concatenate(flat_obs).astype(np.float32) if flat_obs else np.array([], dtype=np.float32)

    def _unflatten_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Convert flat action to dict with resource-specific actions"""
        if self.custom_action_fn is not None:
            return self.custom_action_fn(action)

        action_dict = {}

        for mapping in self._action_mapping:
            res_id = mapping['resource_id']
            action_key = mapping['action_key']
            start, end = mapping['slice']
            action_dim = mapping['action_dim']

            if action_dim == 1:
                action_value = float(action[start]) if start < len(action) else 0.0
                action_dict[res_id] = {action_key: action_value}
            else:
                # Multi-dimensional action: pass array directly to the resource
                action_dict[res_id] = action[start:end]

        return action_dict

def make_flat(env_name: str, **kwargs) -> gym.Env:
    """Deprecated: the old scenario-name registry was removed.

    Use ``make_task_env(...)`` for benchmark tasks, or build a
    :class:`powerzoo.envs.power_env.PowerEnv` from a config dict and wrap with
    :class:`FlattenWrapper` / :class:`FlattenObservation` as needed.

    Args:
        env_name: Ignored (kept for signature compatibility).
        **kwargs: Ignored.

    Raises:
        NotImplementedError: Always — registry-based ``make()`` no longer exists.
    """
    raise NotImplementedError(
        "make_flat() depended on the removed powerzoo.scenarios registry. "
        "Use make_task_env('marl_opf', ...) or PowerEnv(config) plus "
        "powerzoo.wrappers.flatten.FlattenWrapper. "
        "See docs/en/getting-started.md."
    )
