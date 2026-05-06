"""ForecastWrapper — look-ahead observation augmentation for PowerZoo envs.

Wraps any Gymnasium env that has an underlying ``GridEnv`` (accessible via
``env.env`` or ``env.grid``) and appends a *forecast window* to every
observation.  The forecast contains the next ``horizon`` values of the
``load.actual_mw`` time series, enabling model-based RL and multi-step planning
baselines.

End-of-data handling
--------------------
Look-ahead indices that exceed ``len(_time_series_data) - 1`` are **clamped**
to the last available row (repeat-last-value).  This applies both to normal
episode boundaries and to the absolute last row of the dataset.

Forecast modes
--------------
``'perfect'``
    Future demand values are taken directly from the stored time series.
    Used for oracle / upper-bound comparisons.

``'noisy'``
    Gaussian noise ``N(0, noise_std)`` (as a fraction of the value) is added
    to each future step.  Simulates a realistic short-term forecast.

``'none'``
    Forecast is all-zeros (equivalent to no look-ahead).  Keeps the
    observation shape consistent so the same policy architecture can be used
    with or without forecasts.

Usage::

    from powerzoo.envs.grid.trans import TransGridEnv
    from powerzoo.wrappers import GymnasiumWrapper
    from powerzoo.wrappers.forecast_wrapper import ForecastWrapper

    env = ForecastWrapper(GymnasiumWrapper(TransGridEnv()), horizon=6)
    obs, info = env.reset(seed=0)
    # obs now has 6 extra elements at the end: normalised demand forecast
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces


class ForecastWrapper(gym.ObservationWrapper):
    """Augment observations with a look-ahead demand forecast.

    Parameters
    ----------
    env : gym.Env
        A Gymnasium-wrapped PowerZoo ``GridEnv``.
    horizon : int
        Number of future time steps to append to the observation.  Default 6.
    mode : str
        ``'perfect'``, ``'noisy'``, or ``'none'``.  Default ``'perfect'``.
    noise_std : float
        Fractional Gaussian noise std for ``mode='noisy'`` (e.g. 0.02 = 2 %).
        Ignored for other modes.  Default 0.02.
    normalize : bool
        Whether to normalise forecast values by the maximum demand in the
        dataset (so each value is in [0, ~1]).  Default ``True``.
    """

    def __init__(self, env: gym.Env, horizon: int = 6,
                 mode: str = 'perfect', noise_std: float = 0.02,
                 normalize: bool = True):
        if mode not in ('perfect', 'noisy', 'none'):
            raise ValueError(f"mode must be 'perfect', 'noisy', or 'none', got '{mode}'")
        super().__init__(env)

        self.horizon = int(horizon)
        self.mode = mode
        self.noise_std = float(noise_std)
        self.normalize = normalize

        # Locate the inner GridEnv and its time-series data
        self._grid_env = self._find_grid_env(env)
        self._demand_series: Optional[np.ndarray] = None  # populated in reset()
        self._demand_max: float = 1.0

        # Extend observation space
        base_dim = int(np.prod(env.observation_space.shape))
        self.observation_space = spaces.Box(
            low=np.full(base_dim + horizon, -np.inf, dtype=np.float32),
            high=np.full(base_dim + horizon,  np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gymnasium API overrides
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        self._refresh_demand_series()
        return self.observation(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Append forecast to base observation."""
        forecast = self._build_forecast()
        return np.concatenate([obs.flatten(), forecast]).astype(np.float32)

    # ------------------------------------------------------------------
    # Forecast construction
    # ------------------------------------------------------------------

    def _build_forecast(self) -> np.ndarray:
        """Build a ``horizon``-length forecast vector."""
        if self.mode == 'none' or self._demand_series is None:
            return np.zeros(self.horizon, dtype=np.float32)

        grid = self._grid_env
        # Current absolute index into the full time series
        current_idx = grid._get_current_time_index() if hasattr(grid, '_get_current_time_index') else -1

        n = len(self._demand_series)
        forecast = np.empty(self.horizon, dtype=np.float32)

        for k in range(1, self.horizon + 1):
            look_idx = min(current_idx + k, n - 1)  # clamp at last row
            if look_idx < 0:
                value = self._demand_series[-1]
            else:
                value = self._demand_series[look_idx]

            if self.normalize and self._demand_max > 0:
                value = value / self._demand_max

            if self.mode == 'noisy' and value != 0.0:
                noise = self.np_random.normal(0.0, self.noise_std * abs(value))
                value = value + noise

            forecast[k - 1] = float(value)

        return forecast

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_demand_series(self) -> None:
        """Cache the load demand numpy array from the inner GridEnv."""
        from powerzoo.data import signals as S
        _LOAD = S.LOAD_ACTUAL_MW

        grid = self._grid_env
        if grid is None:
            self._demand_series = None
            return

        ts = getattr(grid, '_time_series_data', None)
        if ts is not None and hasattr(ts, 'columns'):
            col = None
            if _LOAD in ts.columns:
                col = _LOAD
            elif 'ActualDemand' in ts.columns:
                col = 'ActualDemand'
            if col is not None:
                arr = ts[col].values.astype(np.float64)
                self._demand_series = arr
                arr_max = float(arr.max()) if len(arr) > 0 else 0.0
                self._demand_max = arr_max if arr_max > 0 else 1.0
                return
        self._demand_series = None
        self._demand_max = 1.0

    @staticmethod
    def _find_grid_env(env: gym.Env) -> Any:
        """Walk the wrapper stack to find the first object with ``_time_series_data``."""
        current = env
        while current is not None:
            if hasattr(current, '_time_series_data'):
                return current
            if hasattr(current, 'grid'):
                return current.grid
            current = getattr(current, 'env', None)
        return None
