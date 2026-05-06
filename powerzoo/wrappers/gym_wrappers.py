"""Gymnasium-compatible wrappers for PowerZoo grid and resource environments.

``GymnasiumWrapper``
    Converts any PowerZoo ``GridEnv`` (which returns a *state dict* from
    ``step``) into a fully Gymnasium-compliant environment that returns flat
    numpy observation arrays.  Just add resources, then wrap.

``NormalizationWrapper``
    Wraps a ``GymnasiumWrapper`` (or any ``gym.Env`` with a ``Box`` obs space)
    and normalises observations to the range ``[-1, 1]`` using pre-computed
    bounds derived from the case data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces


class GymnasiumWrapper(gym.Wrapper):
    """Wrap a PowerZoo ``GridEnv`` to produce a standard Gymnasium interface.

    The inner env's ``step()`` returns a *state dict* as observation; this
    wrapper calls ``env.obs(state)`` to obtain a flat numpy array and passes
    it through as the Gymnasium observation.

    It also exposes:
    - ``env.observation_space`` / ``env.action_space`` (from inner env)
    - ``env.obs_names`` / ``env.action_names`` (human-readable labels)
    - Correct ``(obs, info)`` return from ``reset()``

    Args:
        env: Any ``GridEnv`` subclass (``TransGridEnv``, ``DistGridEnv``, …).

    Example::

        from powerzoo.envs.grid.trans import TransGridEnv
        from powerzoo.wrappers import GymnasiumWrapper

        raw = TransGridEnv()
        env = GymnasiumWrapper(raw)

        obs, info = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    """

    metadata = {"render_modes": []}

    def __init__(self, env):
        super().__init__(env)
        # Sanity: inner env must implement obs()
        if not hasattr(env, 'obs'):
            raise TypeError(
                f"{type(env).__name__} does not implement obs().  "
                "Only PowerZoo GridEnv / ResourceEnv subclasses are supported."
            )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset inner env and return (obs_array, info)."""
        state, info = self.env.reset(seed=seed, options=options, **kwargs)
        obs = self.env.obs(state)
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step inner env; convert state dict → flat obs array.

        Numpy array actions are auto-converted to the dict format expected by
        GridEnv (e.g. a unit-dispatch vector → ``{'unit_power_mw': action}``).
        """
        grid_action = self._to_grid_action(action)
        state, reward, terminated, truncated, info = self.env.step(grid_action)
        obs = self.env.obs(state)
        return obs, reward, terminated, truncated, info

    def _to_grid_action(self, action: Any) -> Any:
        """Convert a flat numpy array action to grid-compatible dict if needed."""
        if not isinstance(action, np.ndarray):
            return action
        inner = self.env
        while hasattr(inner, 'env'):
            inner = inner.env
        # TransGridEnv: array of length n_units → {'unit_power_mw': ...}
        if hasattr(inner, 'case') and hasattr(inner.case, 'units'):
            n_units = len(inner.case.units)
            if action.size == n_units:
                return {'unit_power_mw': action.flatten()}
        # Fallback: empty dict (pure observation env)
        return {}

    # ------------------------------------------------------------------
    # Label passthrough
    # ------------------------------------------------------------------

    @property
    def obs_names(self) -> List[str]:
        return getattr(self.env, 'obs_names', [])

    @property
    def action_names(self) -> List[str]:
        return getattr(self.env, 'action_names', [])


# ---------------------------------------------------------------------------
# NormalizationWrapper
# ---------------------------------------------------------------------------

class NormalizationWrapper(gym.ObservationWrapper):
    """Normalise observations to ``[-1, 1]`` using fixed bounds.

    Bounds are derived automatically from the inner environment's case data
    (physical limits), and are stable across episodes — safe to use during
    training.

    Args:
        env: A ``GymnasiumWrapper`` or any ``gym.Env`` with a ``Box`` obs space.
        clip: Whether to clip the normalised observation to ``[-1, 1]``.
              Default ``True``.

    The raw bounds can be inspected via ``env.obs_low`` and ``env.obs_high``.

    Example::

        from powerzoo.wrappers import GymnasiumWrapper, NormalizationWrapper

        env = NormalizationWrapper(GymnasiumWrapper(TransGridEnv()))
        obs, info = env.reset(seed=0)   # obs ∈ [-1, 1]
    """

    def __init__(self, env: gym.Env, clip: bool = True):
        super().__init__(env)
        self.clip = clip

        # Derive bounds from inner env (best-effort)
        self.obs_low, self.obs_high = self._derive_bounds(env)

        # Normalised observation space is always Box([-1,1], [1,1])
        obs_dim = int(np.prod(env.observation_space.shape))
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -1.0, dtype=np.float32),
            high=np.full(obs_dim,  1.0, dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalise obs to [-1, 1]."""
        span = self.obs_high - self.obs_low
        span = np.where(span > 1e-8, span, 1.0)  # avoid div-by-zero
        norm = 2.0 * (obs - self.obs_low) / span - 1.0
        if self.clip:
            norm = np.clip(norm, -1.0, 1.0)
        return norm.astype(np.float32)

    # ------------------------------------------------------------------
    # Bound derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_bounds(env: gym.Env) -> Tuple[np.ndarray, np.ndarray]:
        """Infer reasonable physical bounds for normalisation.

        Strategy (in priority order):
        1. If the inner env is a ``GymnasiumWrapper`` around a ``GridEnv``,
           read the obs_names and case data to assign meaningful limits.
        2. Fall back to the observation_space's declared low/high (may be ±inf).
        3. Replace ±inf with ±10 as a safe default.

        Handles both single-phase names (``node_0_v_norm``) and three-phase
        names (``node_0_V_A_norm``) via case-insensitive matching with
        explicit per-phase suffixes.
        """
        # Unwrap to find the GridEnv
        inner = env
        while hasattr(inner, 'env'):
            inner = inner.env

        obs_space = env.observation_space
        obs_dim = int(np.prod(obs_space.shape))
        default_low = np.full(obs_dim, -10.0, dtype=np.float32)
        default_high = np.full(obs_dim,  10.0, dtype=np.float32)

        # Try to read physics-based bounds from the GridEnv
        try:
            case = getattr(inner, 'case', None)
            obs_names: List[str] = getattr(inner, 'obs_names', [])
            if case is None or not obs_names:
                raise ValueError("no case/obs_names")

            _PH = ('a', 'b', 'c')

            bounds_low = np.empty(len(obs_names), dtype=np.float32)
            bounds_high = np.empty(len(obs_names), dtype=np.float32)

            for i, name in enumerate(obs_names):
                nl = name.lower()

                # Load features must be checked first: "p_load_norm" contains
                # neither "_p_norm" nor "_v_norm" as contiguous substrings,
                # but checking load first avoids any future overlap.
                is_load = ('load_norm' in nl
                           or any(f'load_{p}_norm' in nl for p in _PH))

                # Voltage: _v_norm (1ph) or _v_{a,b,c}_norm (3ph)
                is_voltage = ('_v_norm' in nl
                              or any(f'_v_{p}_norm' in nl for p in _PH))

                # Flow / power: flow_norm, _p_norm, _q_norm (1ph)
                #               _{p,q}_{a,b,c}_norm (3ph)
                is_flow = ('flow_norm' in nl
                           or '_p_norm' in nl
                           or '_q_norm' in nl
                           or any(f'_{pq}_{p}_norm' in nl
                                  for pq in ('p', 'q') for p in _PH))

                if is_load:
                    bounds_low[i], bounds_high[i] = 0.0, 1.5
                elif is_voltage:
                    # normalised voltage (v-1)/0.1 → ±2 covers ±0.2 p.u.
                    bounds_low[i], bounds_high[i] = -2.0, 2.0
                elif is_flow:
                    bounds_low[i], bounds_high[i] = -1.5, 1.5
                elif 'time_sin' in nl or 'time_cos' in nl:
                    bounds_low[i], bounds_high[i] = -1.0, 1.0
                elif 'soc' in nl:
                    bounds_low[i], bounds_high[i] = 0.0, 1.0
                else:
                    bounds_low[i], bounds_high[i] = -10.0, 10.0

            return bounds_low, bounds_high

        except Exception:
            pass

        # Fallback: replace ±inf with ±10
        low = np.where(np.isfinite(obs_space.low), obs_space.low, default_low).astype(np.float32)
        high = np.where(np.isfinite(obs_space.high), obs_space.high, default_high).astype(np.float32)
        return low, high
