from typing import Any, Dict, Optional, List, Tuple
from datetime import timedelta
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from powerzoo.envs.base import BaseEnv
from powerzoo.data import DataLoader
from powerzoo.data import signals as S

logger = logging.getLogger(__name__)


class GridEnv(BaseEnv):
    """Base power grid environment (inner physical simulation engine).

    ``GridEnv`` acts as the **physical state machine** layer of the PowerZoo
    stack.  Its ``reset()`` and ``step()`` return a raw *state dict* (the full
    power-flow solution) rather than a flat numpy observation array, because the
    state is consumed by upper-layer components that each need different subsets
    of it:

    * ``PowerEnv`` — the benchmark-facing RL facade.  Calls ``grid.step()`` to
      advance the simulation, then passes the returned state dict to
      ``grid.obs(state)`` and to each resource's ``obs()`` to assemble the
      agent observation.
    * ``ObsWrapper`` / ``gym_wrappers.GymWrapper`` — thin Gym-compatibility
      wrappers that call ``env.obs(state)`` directly on a ``GridEnv`` subclass
      for single-grid use-cases that don't need ``PowerEnv``.

    Because of this design, passing a bare ``GridEnv`` subclass to standard Gym
    validation tools (e.g. ``gymnasium.utils.env_checker.check_env``) will
    produce an observation-type warning — the raw state dict does not conform to
    ``observation_space``.  Always wrap the env in ``PowerEnv`` or a Gym wrapper
    before running a standard RL training loop or check.

    Subclasses must implement
    -------------------------
    * ``_run_power_flow(action)`` — run the solver; return ``True`` on success.
    * ``_get_state()`` — build and return the full state dict after a solve.
    * ``obs(state)`` — project the state dict to a flat ``float32`` array
      matching ``observation_space``.
    * ``build_info(state)`` — build the step info dict.
    """

    # Reward applied when the power-flow solver fails and the episode is
    # terminated immediately.  Subclasses may override this class attribute to
    # tune the penalty to their reward scale.
    _PF_FAILURE_REWARD: float = -1.0

    def __init__(self, case: Any = None, solver: Any = None, delta_t_minutes: float = 30.0,
                 data_loader: Optional[DataLoader] = None,
                 start_date: str = '2024-01-01',
                 end_date: str = '2024-01-31',
                 load_columns: Optional[List[str]] = None,
                 max_load_ratio: float = 0.9,
                 min_load_ratio: Optional[float] = None,
                 time_series: Any = None,
                 max_episode_steps: Optional[int] = None,
                 randomize_start_time: bool = False,
                 time_alignment: Optional[Dict[str, str]] = None):
        """Initialize grid environment

        Args:
            case: Power system case
            solver: Power flow solver
            delta_t_minutes: Time step in minutes (default: 30)
            data_loader: DataLoader instance. If None, creates default DataLoader.
                         Ignored when ``time_series`` is provided.
            start_date: Start date for data loading (format: 'YYYY-MM-DD').
                        Ignored when ``time_series`` is provided.
            end_date: End date for data loading (format: 'YYYY-MM-DD').
                      Ignored when ``time_series`` is provided.
            load_columns: Semantic signal names to load.
                Default: ``[signals.LOAD_ACTUAL_MW, signals.SOLAR_AVAILABLE_MW,
                signals.WIND_AVAILABLE_MW]``.
                Distribution envs may also request
                ``signals.LOAD_REACTIVE_MVAR`` to provide an explicit feeder
                reactive-demand time series.
                Legacy raw column names (``'ActualDemand'``, ``'Wind'``, …)
                are auto-mapped with a deprecation warning.
                Ignored when ``time_series`` is provided.
            max_load_ratio: Maximum load as ratio of total generation capacity (default: 0.9)
            min_load_ratio: Minimum load as ratio of total generation capacity (default: None)
            time_series: Custom time-series data supplied directly by the user.
                         Accepted formats:
                           - ``pandas.DataFrame``: must have a DatetimeIndex (or 'datetime'
                             column) and at least a ``signals.LOAD_ACTUAL_MW`` (or legacy
                             ``'ActualDemand'``) column. An optional
                             ``signals.LOAD_REACTIVE_MVAR`` (or legacy
                             ``'ReactiveDemand'``) column is used by
                             distribution envs as an explicit Q-demand series.
                           - ``numpy.ndarray`` of shape ``(T,)`` or ``(T, 1)``: treated as
                             a single-column ``load.actual_mw`` time series.  A synthetic
                             DatetimeIndex starting at ``start_date`` is created automatically.
                         When provided, data_loader / start_date / end_date / load_columns
                         are NOT used for loading data (max_load_ratio and min_load_ratio
                         still apply for scaling).
            max_episode_steps: Maximum steps per episode before ``truncated=True``.
                               Defaults to one full day (``steps_per_day``).
            randomize_start_time: When True, choose a random intra-day starting
                                  offset at each reset instead of always beginning at
                                  ``time_step=0``.  Increases initial-state diversity
                                  from O(n_days) to O(n_timesteps).  Default False
                                  to preserve backward compatibility (F5 fix).
            time_alignment: Per-signal time-alignment overrides for
                cross-period data.  Example:
                ``{"solar.available_mw": "2024-01-01"}`` maps the solar
                source's 2024 data onto the simulation's ``start_date``.
                Profile-mode datasets (e.g. data-center traces) are tiled
                automatically and do not need an override.
        """
        super().__init__(delta_t_minutes=delta_t_minutes)
        self.case = case
        self.solver = solver
        self.sub_resources = {}                        # registered sub-resources (batteries, EVs, etc.)
        self.nodes_resources_map = None                 # (n_nodes × n_resources) resource-bus incidence
        self._resource_counters = {}                    # per-type resource instance counter
        self._resource_col_index: Dict[str, int] = {}  # resource name → column index in mapping
        self.day_id = None

        self._episode_reward: float = 0.0
        self._episode_steps: int = 0
        self._last_valid_state: Optional[Dict[str, Any]] = None

        self.data_loader = data_loader or DataLoader()

        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)

        default_load_columns = [S.LOAD_ACTUAL_MW, S.SOLAR_AVAILABLE_MW, S.WIND_AVAILABLE_MW]
        self.load_columns = self._normalize_load_columns(
            load_columns if load_columns is not None else default_load_columns
        )

        self.max_load_ratio = max_load_ratio
        self.min_load_ratio = min_load_ratio
        self.steps_per_day = 1440 // int(delta_t_minutes)
        self.max_episode_steps: int = (
            max_episode_steps if max_episode_steps is not None else self.steps_per_day
        )
        self.randomize_start_time: bool = randomize_start_time
        self.time_offset: int = 0
        self.time_alignment: Dict[str, str] = time_alignment or {}

        self._time_series_data = None
        self._node_loads_p = None
        self._node_loads_q = None
        self._regular_time_index: bool = False
        self.n_days = 0

        if time_series is not None:
            self._load_user_time_series(time_series)
        else:
            self._initialize_time_series_data()

        if self.n_days == 0:
            date_span = (self.end_date - self.start_date).days + 1
            self.n_days = max(date_span, 1)

    # ── Legacy Column-Name Compatibility ─────────────────────────────────────

    @staticmethod
    def _normalize_load_columns(columns: List[str]) -> List[str]:
        """Map legacy raw column names to semantic signals."""
        result: List[str] = []
        for col in columns:
            mapped = S._LEGACY_COLUMN_MAP.get(col)
            if mapped is not None:
                warnings.warn(
                    f"load_columns: '{col}' is a legacy raw column name. "
                    f"Use signal name '{mapped}' instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                if mapped not in result:
                    result.append(mapped)
            else:
                if col not in result:
                    result.append(col)
        return result

    # ── User-Supplied Time Series ─────────────────────────────────────────────

    def _load_user_time_series(self, time_series: Any) -> None:
        """Accept user-supplied time series and run the same preprocessing pipeline.

        Args:
            time_series: pandas.DataFrame or numpy array.
                - DataFrame: needs DatetimeIndex or 'datetime' column,
                  at least a ``load.actual_mw`` column (or legacy ``ActualDemand``).
                - ndarray shape (T,) or (T, 1): treated as ``load.actual_mw``.
                  A DatetimeIndex starting at self.start_date is synthesised.
        """
        _LOAD = S.LOAD_ACTUAL_MW
        _Q = S.LOAD_REACTIVE_MVAR
        try:
            if isinstance(time_series, np.ndarray):
                arr = time_series.flatten()
                n_steps = len(arr)
                idx = pd.date_range(
                    start=self.start_date,
                    periods=n_steps,
                    freq=f'{int(self.delta_t_minutes)}min',
                    tz='UTC',
                )
                df = pd.DataFrame({_LOAD: arr}, index=idx)
            elif isinstance(time_series, pd.DataFrame):
                df = time_series.copy()
                if S.DATETIME in df.columns:
                    df.set_index(S.DATETIME, inplace=True)
                elif 'datetime' in df.columns:
                    df.set_index('datetime', inplace=True)
                if not isinstance(df.index, pd.DatetimeIndex):
                    raise ValueError(
                        "time_series DataFrame must have a DatetimeIndex "
                        "or a 'datetime' column."
                    )
                # Accept legacy demand columns and rename
                if 'ActualDemand' in df.columns and _LOAD not in df.columns:
                    df = df.rename(columns={'ActualDemand': _LOAD})
                if 'ReactiveDemand' in df.columns and _Q not in df.columns:
                    df = df.rename(columns={'ReactiveDemand': _Q})
                if _LOAD not in df.columns:
                    raise ValueError(
                        f"time_series DataFrame must contain a "
                        f"'{_LOAD}' (or legacy 'ActualDemand') column."
                    )
            else:
                raise TypeError(
                    f"time_series must be a pandas DataFrame or numpy ndarray, "
                    f"got {type(time_series).__name__}"
                )

            n_timesteps = len(df)
            self._finalize_time_series(df, update_date_bounds=True)
            logger.info("Loaded user-supplied time series: %d steps, "
                        "%d days, columns=%s", n_timesteps, self.n_days, list(df.columns))

        except Exception as e:
            logger.warning("Failed to load user time_series: %s", e, exc_info=True)
            self._time_series_data = None

    # ── Data Loading and Time Management ─────────────────────────────────────

    def _finalize_time_series(self, df: pd.DataFrame, *, update_date_bounds: bool = False) -> None:
        """Commit a validated DataFrame as the active time series and run preprocessing.

        Called by both loading paths once they have produced a clean DataFrame.
        Centralises the operations that are identical between the DataLoader path
        and the user-supplied path: persisting the data, computing ``n_days``,
        enabling the O(1) time-index fast path, and triggering load scaling.

        Per-path logging and error handling remain in each caller.

        Args:
            df: Validated, clean time-series DataFrame with DatetimeIndex.
            update_date_bounds: When True, overwrite ``start_date`` / ``end_date``
                from the DataFrame's index (needed for user-supplied series whose
                date range may differ from the constructor defaults).
        """
        self._time_series_data = df
        self.n_days = int(np.ceil(len(df) / self.steps_per_day))
        self._detect_regular_time_index()
        if update_date_bounds:
            self.start_date = df.index[0].tz_localize(None) if df.index.tz else df.index[0]
            self.end_date = df.index[-1].tz_localize(None) if df.index.tz else df.index[-1]
        if S.LOAD_ACTUAL_MW in df.columns and self.case is not None:
            self._scale_and_distribute_load()

    def _initialize_time_series_data(self):
        """Load time series data via semantic signals and preprocess loads."""
        _LOAD = S.LOAD_ACTUAL_MW
        try:
            # === Step 1: Load data via semantic API ===
            df = self.data_loader.load_signals(
                signals=self.load_columns,
                start_date=self.start_date,
                end_date=self.end_date,
                resample=f'{int(self.delta_t_minutes)}min',
                interpolation='linear',
                time_alignment=self.time_alignment,
            )

            # === Step 2: Clean and prepare columns ===
            if S.DATETIME in df.columns:
                df.set_index(S.DATETIME, inplace=True)

            if df.isna().any().any():
                nan_counts = df.isna().sum()
                logger.warning("Found NaN in %s", dict(nan_counts[nan_counts > 0]))
                df = df.ffill().bfill().fillna(0)

            n_timesteps = len(df)
            if n_timesteps == 0:
                logger.warning("DataLoader returned empty time series, "
                               "will use default case load data")
                self._time_series_data = None
                return

            case_name = type(self.case).__name__ if self.case is not None else 'None'
            grid_type = type(self).__name__
            logger.info("Grid: %s | Case: %s", grid_type, case_name)
            logger.info("  Time series: %d steps, %s → %s",
                        n_timesteps, df.index[0], df.index[-1])
            logger.info("  Columns: %s", list(df.columns))

            # === Step 3: Persist, configure index fast path, scale and distribute ===
            self._finalize_time_series(df)
            if _LOAD not in df.columns or self.case is None:
                logger.info("  No load data to preprocess, will use case default values")

        except Exception as e:
            logger.warning("Failed to initialize time series data: %s", e, exc_info=True)
            logger.warning("  Grid will use default case load data")
            self._time_series_data = None
    
    def _get_load_scaling_capacity(self) -> float:
        """Return the total capacity (MW) used to normalise the demand time series.

        Default: generator p_max if available, else loads d_max.
        Distribution subclasses should override this to use feeder d_max × baseMVA
        so that load scaling stays within a physically meaningful p.u. range.
        """
        if hasattr(self.case, 'units') and hasattr(self.case.units, 'p_max'):
            return float(self.case.units['p_max'].sum())
        return float(self.case.loads['d_max'].sum())

    @staticmethod
    def _compute_node_load_matrix(load_series: np.ndarray, d_max: np.ndarray) -> np.ndarray:
        """Distribute a scalar load time series to a per-node load matrix.

        Each node's share is proportional to its rated peak demand (``d_max``).

        Args:
            load_series: Shape ``(T,)`` total active load in MW.
            d_max: Shape ``(n_nodes,)`` rated peak demand per load bus.

        Returns:
            Shape ``(T, n_nodes)`` per-node active load matrix in MW.
        """
        total = d_max.sum()
        ratio = d_max / total if total > 0 else np.ones_like(d_max) / len(d_max)
        return load_series[:, None] * ratio[None, :]

    def _scale_and_distribute_load(self):
        """Scale load to case capacity and pre-compute per-node load matrices."""
        _LOAD = S.LOAD_ACTUAL_MW
        _Q = S.LOAD_REACTIVE_MVAR
        if not hasattr(self.case, 'units') or not hasattr(self.case, 'loads'):
            return

        try:
            case_name = type(self.case).__name__
            n_buses = len(self.case.nodes) if hasattr(self.case, 'nodes') else '?'
            n_units = len(self.case.units) if hasattr(self.case, 'units') else '?'
            logger.info("Scaling demand → %s (%s-bus, %s units)...", case_name, n_buses, n_units)

            # ── 1. Sanitise load data ─────────────────────────────────────────
            load_data = self._time_series_data[_LOAD].values
            if np.isnan(load_data).any():
                load_data = np.nan_to_num(load_data, nan=0.0)
                self._time_series_data[_LOAD] = load_data

            total_capacity = self._get_load_scaling_capacity()
            logger.info("  Case total gen capacity: %.2f MW", total_capacity)
            logger.info("  Original demand range: %.2f – %.2f MW",
                        load_data.min(), load_data.max())

            # ── 2. Scale to max_load_ratio ────────────────────────────────────
            max_target = total_capacity * self.max_load_ratio
            load_scale = max_target / load_data.max() if load_data.max() > 0 else 1.0
            self._time_series_data[_LOAD] *= load_scale

            # ── 3. Clamp to min_load_ratio (optional) ─────────────────────────
            if self.min_load_ratio is not None:
                min_target = total_capacity * self.min_load_ratio
                current_min = self._time_series_data[_LOAD].min()
                current_max = self._time_series_data[_LOAD].max()
                if current_min < min_target:
                    old_range = current_max - current_min
                    new_range = max_target - min_target
                    self._time_series_data[_LOAD] = (
                        min_target
                        + (self._time_series_data[_LOAD] - current_min)
                        * (new_range / old_range)
                    )

            scaled_min = self._time_series_data[_LOAD].min()
            scaled_max = self._time_series_data[_LOAD].max()
            logger.info("  Scaled load range: %.2f – %.2f MW", scaled_min, scaled_max)
            logger.info("  Load ratio: %.1f%% – %.1f%% of capacity",
                        scaled_min / total_capacity * 100, scaled_max / total_capacity * 100)

            # ── 4. Distribute active load proportionally to nodes ─────────────
            d_max = self.case.loads['d_max'].values
            load_series = self._time_series_data[_LOAD].values
            self._node_loads_p = self._compute_node_load_matrix(load_series, d_max)

            # ── 5. Scale Q series (if present) and build reactive cache ────────
            if _Q in self._time_series_data.columns:
                q_data = self._time_series_data[_Q].to_numpy(dtype=float)
                if np.isnan(q_data).any():
                    q_data = np.nan_to_num(q_data, nan=0.0)
                self._time_series_data[_Q] = q_data * load_scale
            self._node_loads_q = self._build_reactive_load_cache(self._node_loads_p, d_max)

            n_timesteps = len(load_series)
            self.n_days = int(np.ceil(n_timesteps / self.steps_per_day))
            logger.info("  Pre-computed node loads: %s (%.2f MB)",
                        self._node_loads_p.shape, self._node_loads_p.nbytes / 1024 / 1024)
            logger.info("  Total days available: %d", self.n_days)

        except Exception as e:
            logger.warning("Failed to scale/distribute load: %s", e, exc_info=True)
            self._node_loads_p = None
            self._node_loads_q = None
            self.n_days = 0

    def _build_reactive_load_cache(
        self,
        active_loads: np.ndarray,
        base_active_load: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Precompute reactive load cache from explicit Q data or case PF baseline.

        If the time series includes ``signals.LOAD_REACTIVE_MVAR``, that
        explicit feeder-level Q trajectory is distributed first. Otherwise the
        cache preserves the case's default load-level power factor:

            Q_i(t) = P_i(t) * (Q_base_i / P_base_i)

        Transmission cases that only expose ``d_max`` (without ``Qd``) keep
        ``_node_loads_q`` as ``None`` because there is no reactive baseline to
        preserve.
        """
        if not hasattr(self.case, 'loads'):
            return None

        loads_df = self.case.loads
        explicit_q_col = S.LOAD_REACTIVE_MVAR
        if self._time_series_data is not None and explicit_q_col in self._time_series_data.columns:
            q_series = self._time_series_data[explicit_q_col].to_numpy(dtype=float)
            if len(q_series) != active_loads.shape[0]:
                return None

            if 'Qd' in loads_df.columns:
                q_base = loads_df['Qd'].to_numpy(dtype=float)
            else:
                q_base = np.asarray(base_active_load, dtype=float)

            q_total = float(q_base.sum())
            if len(q_base) != active_loads.shape[1]:
                return None

            if q_total > 1e-12:
                q_ratio = q_base / q_total
            else:
                q_ratio = np.ones_like(q_base, dtype=float) / len(q_base)
            return q_series[:, None] * q_ratio[None, :]

        if 'Qd' not in loads_df.columns:
            return None

        base_q = loads_df['Qd'].to_numpy(dtype=float)
        if 'Pd' in loads_df.columns:
            q_scale_base = loads_df['Pd'].to_numpy(dtype=float)
        else:
            q_scale_base = np.asarray(base_active_load, dtype=float)

        if len(base_q) != active_loads.shape[1] or len(q_scale_base) != active_loads.shape[1]:
            return None

        q_over_p = np.divide(
            base_q,
            q_scale_base,
            out=np.zeros_like(base_q, dtype=float),
            where=np.abs(q_scale_base) > 1e-12,
        )
        return active_loads * q_over_p[None, :]

    def _detect_regular_time_index(self) -> None:
        """Detect whether the time series has a uniform interval and cache the result.

        When the index is a contiguous DatetimeIndex with exactly
        ``delta_t_minutes`` spacing, sets ``_regular_time_index = True`` so that
        :meth:`_get_current_time_index` can use O(1) integer arithmetic instead
        of a pandas ``get_indexer`` call.
        """
        self._regular_time_index = False
        if self._time_series_data is None or len(self._time_series_data) == 0:
            return

        index = self._time_series_data.index
        if not isinstance(index, pd.DatetimeIndex):
            return

        expected = pd.date_range(
            start=index[0],
            periods=len(index),
            freq=f'{int(self.delta_t_minutes)}min',
        )
        self._regular_time_index = index.equals(expected)
    
    def _get_datetime_from_day_and_step(self, day_id: int, time_step: int) -> pd.Timestamp:
        """Convert day_id and time_step to datetime
        
        Args:
            day_id: Day ID (0 = first day of start_date)
            time_step: Time step within the day (0 = 00:00)
            
        Returns:
            Timestamp for the given day and time step
        """
        target_date = self.start_date + timedelta(days=day_id)
        minutes_offset = time_step * self.delta_t_minutes
        target_datetime = target_date + timedelta(minutes=minutes_offset)
        
        # Match timezone of data if available
        data_tz = None
        if self._time_series_data is not None:
            data_tz = getattr(self._time_series_data.index, 'tz', None)
        if data_tz is not None and target_datetime.tz is None:
            target_datetime = target_datetime.tz_localize(data_tz)
        
        return target_datetime
    
    def _get_current_time_index(self) -> int:
        """Get current time index for day_id and time_step."""
        if self._time_series_data is None:
            return -1
        try:
            if self._regular_time_index:
                n_steps = len(self._time_series_data)
                if n_steps == 0:
                    return -1
                day_id = int(self.day_id) if self.day_id is not None else 0
                raw_idx = day_id * self.steps_per_day + int(self.time_step)
                if raw_idx < 0 or raw_idx >= n_steps:
                    logger.debug(
                        "Time index %d out of range [0, %d); clipping.",
                        raw_idx, n_steps,
                    )
                return int(np.clip(raw_idx, 0, n_steps - 1))

            current_datetime = self._get_datetime_from_day_and_step(self.day_id, self.time_step)
            return self._time_series_data.index.get_indexer([current_datetime], method='nearest')[0]
        except Exception:
            return -1

    def _get_current_load_data(self) -> Dict[str, float]:
        """Get load/renewable data dict for current time step"""
        if self._time_series_data is None:
            return {}
        try:
            idx = self._get_current_time_index()
            return self._time_series_data.iloc[idx].to_dict() if idx >= 0 else {}
        except Exception:
            return {}
    
    def _get_node_loads_p_current(self) -> np.ndarray:
        """Get pre-computed node loads for current time step (fast O(1) indexing)"""
        if self._node_loads_p is not None:
            idx = self._get_current_time_index()
            if 0 <= idx < len(self._node_loads_p):
                return self._node_loads_p[idx]
        
        # Fallback to case default
        return (self.case.loads['d_max'].values.copy() 
                if self.case and hasattr(self.case, 'loads') 
                else np.array([0.0]))
    
    def _get_day_profile(self, day_id: int, column: str) -> np.ndarray:
        """Get normalized [0,1] profile for entire day (length = steps_per_day)"""
        if self._time_series_data is None or column not in self._time_series_data.columns:
            return np.zeros(self.steps_per_day)
        
        try:
            # Calculate day boundaries with timezone awareness
            day_start = self.start_date + timedelta(days=day_id)
            day_end = day_start + timedelta(days=1)
            if (getattr(self._time_series_data.index, 'tz', None) is not None
                    and day_start.tz is None):
                day_start = day_start.tz_localize('UTC')
                day_end = day_end.tz_localize('UTC')
            
            # Extract and pad/trim day data
            mask = (self._time_series_data.index >= day_start) & (self._time_series_data.index < day_end)
            day_data = self._time_series_data.loc[mask, column].values
            
            if len(day_data) != self.steps_per_day:
                day_data = (day_data[:self.steps_per_day] if len(day_data) > self.steps_per_day
                           else np.pad(day_data, (0, self.steps_per_day - len(day_data)), mode='edge'))
            
            # Normalize
            max_val = day_data.max()
            return day_data / max_val if max_val > 0 else day_data
            
        except Exception as e:
            logger.warning("Failed to get day profile for %s: %s", column, e)
            return np.zeros(self.steps_per_day)
    
    # ── Episode Lifecycle ─────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict] = None,
              day_id: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset grid environment and all sub-resources.

        Follows the Gymnasium v26+ convention: ``seed`` is the primary
        reproducibility parameter; ``day_id`` (legacy kwarg) or
        ``options={'day_id': N}`` both work for selecting a specific day.

        Args:
            seed: Random seed. Passed through ``super().reset(seed=seed,
                  options=options)`` so Gymnasium manages ``self.np_random``.
            options: Optional dict.  Recognised keys:
                       ``day_id`` (int) – override which day to simulate.
            day_id: Legacy kwarg; takes precedence over ``options['day_id']``.
        """
        # ── Re-seed RNG via Gymnasium and reset the base clock ───────────────
        super().reset(seed=seed, options=options)

        # ── Episode counters ──────────────────────────────────────────────────
        self._episode_reward = 0.0
        self._episode_steps = 0

        # ── Resolve day_id (legacy kwarg > options dict > random) ────────────
        if day_id is None and options is not None:
            day_id = options.get('day_id')
        num_days = (
            max(1, self.n_days) if self.n_days > 0
            else max(1, (self.end_date - self.start_date).days)
        )
        if day_id is None:
            self.day_id = int(self.np_random.integers(0, num_days))
        else:
            self.day_id = int(day_id)
            if not (0 <= self.day_id < num_days):
                raise ValueError(
                    f"day_id={self.day_id} out of range [0, {num_days - 1}]."
                )

        # ── Intra-day start offset (F5: expands initial-state diversity) ──────
        # Only effective when max_episode_steps < steps_per_day.
        # Multi-day episodes always start at offset 0.
        if self.randomize_start_time:
            max_offset = max(1, self.steps_per_day - self.max_episode_steps)
            if max_offset <= 1:
                logger.debug(
                    "randomize_start_time has no effect: max_episode_steps=%d >= "
                    "steps_per_day=%d", self.max_episode_steps, self.steps_per_day,
                )
            self.time_offset = int(self.np_random.integers(0, max_offset))
        else:
            self.time_offset = 0
        self.time_step = self.time_offset

        # ── Reset sub-resources ───────────────────────────────────────────────
        self._reset_sub_resources()

        # Return initial state (subclass may overwrite this after power flow)
        state = self._get_state()
        self._last_valid_state = state
        return state, self.build_info(state)

    def _reset_sub_resources(self) -> None:
        """Broadcast reset to all registered sub-resources with reproducible child seeds.

        Spawns one child seed per resource from the current episode's RNG so
        that resets are fully reproducible given the same parent seed.
        Handles three resource call signatures with graceful fallbacks for
        backward compatibility with older resource implementations.
        """
        if not self.sub_resources:
            return
        child_seeds = self.np_random.integers(0, 2**31, size=len(self.sub_resources))
        options = {
            'time_step': int(self.time_step),
            'time_offset': int(self.time_offset),
        }
        for i, resource in enumerate(self.sub_resources.values()):
            if not hasattr(resource, 'reset'):
                continue
            try:
                resource.reset(
                    seed=int(child_seeds[i]),
                    options=dict(options),
                    day_id=self.day_id,
                )
            except TypeError:
                try:
                    resource.reset(seed=int(child_seeds[i]), day_id=self.day_id)
                except TypeError:
                    resource.reset()

    def step(self, action: Any = None):
        """Execute one time step
        
        Args:
            action: Dict containing:
                - resource actions (key=resource_id, value=resource action)
                - grid control parameters (subclass-specific, e.g., unit_power_mw, node_load_mw)
            
        Returns:
            state, reward, done, truncated, info
        """
        if action is None:
            action = {}

        # Dispatch actions to sub-resources
        # If resource is in action dict, pass the action; otherwise pass None (for auto resources like renewables)
        for resource_id, resource in self.sub_resources.items():
            if resource_id in action:
                resource.step(action[resource_id])
            else:
                # Auto-step for resources that don't need explicit actions (e.g., renewables)
                resource.step(None)

        # Run power flow (implemented in subclass); returns False if solver fails
        pf_ok = self._run_power_flow(action)
        if pf_ok is None:
            pf_ok = True  # backward compat: subclasses that return None treated as success

        # Update time step
        self.time_step += 1
        self._episode_steps += 1

        # Get state.  Subclasses that handle solver failure internally (e.g.
        # via a _pf_failed flag) may return a state with sentinel/NaN values;
        # that is fine — their obs() will sanitise the output.  If _get_state()
        # itself raises, fall back to the last known-good state.
        try:
            state = self._get_state()
        except Exception:
            if self._last_valid_state is None:
                raise RuntimeError(
                    f"{type(self).__name__}.step() raised before any valid state "
                    "was saved. Call reset() before step(), or ensure that "
                    "_get_state() succeeds on the first episode step."
                ) from None
            state = self._last_valid_state

        if pf_ok:
            # Successful solve — save as last-known-good for future failure fallback.
            self._last_valid_state = state
            reward = self._compute_reward(state)
        else:
            # PF failure: skip _compute_reward to avoid NaN propagation.
            # Subclasses that already guard against NaN in _compute_reward can
            # override _PF_FAILURE_REWARD or override this branch entirely.
            reward = self._PF_FAILURE_REWARD

        # Protect against non-finite reward regardless of the path taken above
        # (e.g. a subclass _compute_reward that doesn't fully sanitise NaN).
        if not np.isfinite(reward):
            reward = self._PF_FAILURE_REWARD

        # Episode tracking
        self._episode_reward += reward

        # Check termination/truncation.
        # Use _episode_steps (not time_step) so that a randomised intra-day
        # start offset does not shorten the episode below max_episode_steps.
        terminated = not pf_ok
        truncated = self._episode_steps >= self.max_episode_steps

        # Build info
        info = self.build_info(state)
        info['pf_converged'] = pf_ok
        info['cost_exception'] = 0.0 if pf_ok else 1.0

        # Gymnasium convention: expose episode summary when episode ends
        if terminated or truncated:
            info['episode'] = {
                'r': float(self._episode_reward),
                'l': int(self._episode_steps),
                'metrics': self._get_episode_metrics(state),
            }

        return state, reward, terminated, truncated, info

    def _run_power_flow(self, action: Dict[str, Any]) -> bool:
        """Run power flow with given action (to be implemented in subclass).

        Returns:
            True if the power flow solved successfully, False if the solver
            failed or encountered an infeasible problem.  Returning False
            causes the episode to be terminated immediately (F1 fix).
        """
        raise NotImplementedError

    def _get_state(self) -> Dict[str, Any]:
        """Get current state (to be implemented in subclass)"""
        raise NotImplementedError

    def _compute_reward(self, state: Dict[str, Any]) -> float:
        """Compute reward based on state (to be implemented in subclass)"""
        return 0.0

    def build_info(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build info dict (to be implemented in subclass)"""
        return {}

    def _get_episode_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Override in subclass to expose episode-level KPIs inside info['episode']['metrics'].

        Default returns an empty dict; subclasses can add domain-specific
        metrics (e.g., total violations, average cost, total loss).
        """
        return {}

    def obs(self, state: Any = None) -> np.ndarray:
        """Convert a raw *state* dict to a flat ``float32`` observation array.

        This method is part of ``GridEnv``'s public API and is called by:

        * ``PowerEnv._build_agent_observation(state)`` — assembles the combined
          grid + resource + time observation for the RL agent.
        * ``GymWrapper`` (``powerzoo.wrappers.gym_wrappers``) — provides a
          thin Gym-compliant wrapper for direct single-grid use.

        Subclasses **must** implement this and define ``self.observation_space``
        to match the returned array's shape and dtype.  The default raises
        ``NotImplementedError``.
        """
        raise NotImplementedError("Subclass must implement obs()")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _ensure_case_init(self) -> None:
        """Ensure case data structures are initialized (idempotent)."""
        if self.case and (not hasattr(self.case, 'init_flag') or not self.case.init_flag):
            self.case.init()

    # ── Resource Registry ─────────────────────────────────────────────────────

    def register_resource(self, resource: Any, bus_id: int, name: Optional[str] = None) -> str:
        """Register a resource and assign unique ID.

        Calls ``_on_resource_changed()`` after updating internal state so that
        subclasses can rebuild observation_space and action_space.

        Args:
            resource: Resource instance to register
            bus_id: Bus ID where resource is connected
            name: Optional custom name. If None, auto-generated (e.g., 'solar_0')

        Returns:
            resource_id: The assigned resource ID
        """
        if name is not None:
            resource_id = name
            if resource_id in self.sub_resources:
                raise ValueError(f"Resource name '{name}' already exists. Please use a unique name.")
            self._sync_resource_counter(resource_id)
        else:
            resource_name = getattr(resource.__class__, 'name', None)
            if resource_name is None:
                resource_name = resource.__class__.__name__.lower().replace('env', '')
            resource_id = self._allocate_resource_id(resource_name)

        # Store resource
        self.sub_resources[resource_id] = resource

        # Incremental map update: append one column
        self._append_resource_to_map(resource_id, resource)

        self._on_resource_changed()
        return resource_id

    def _allocate_resource_id(self, resource_name: str) -> str:
        """Allocate the next free ``{resource_name}_{counter}`` identifier."""
        counter = self._resource_counters.get(resource_name, -1)
        while True:
            counter += 1
            resource_id = f"{resource_name}_{counter}"
            if resource_id not in self.sub_resources:
                self._resource_counters[resource_name] = counter
                return resource_id

    def _sync_resource_counter(self, resource_id: str) -> None:
        """Advance a type counter when a custom ID already uses that suffix."""
        prefix, sep, suffix = resource_id.rpartition('_')
        if not sep or not suffix.isdigit():
            return

        counter = int(suffix)
        current = self._resource_counters.get(prefix, -1)
        if counter > current:
            self._resource_counters[prefix] = counter

    def unregister_resource(self, resource_id: str) -> None:
        """Unregister a resource.

        Calls ``_on_resource_changed()`` after updating internal state.
        """
        if resource_id in self.sub_resources:
            # Remove from map index then rebuild without this resource
            self._resource_col_index.pop(resource_id, None)
            self._rebuild_nodes_resources_map(exclude=resource_id)
            del self.sub_resources[resource_id]
            self._on_resource_changed()

    def _on_resource_changed(self) -> None:
        """Called after register_resource / unregister_resource.

        Default implementation calls ``_build_spaces()`` so that subclasses
        only need to implement that single method to keep spaces consistent
        with the current resource set.
        """
        self._build_spaces()

    def _build_spaces(self) -> None:
        """Rebuild observation_space, action_space, obs_names, action_names.

        Called automatically after any resource register/unregister.
        Override in subclasses; the default is a no-op so that base-class
        instances (e.g. used in unit tests) do not fail.
        """

    # ── Nodes–Resources Incidence Map ─────────────────────────────────────────

    def _get_internal_bus_id(self, bus_id: int) -> int:
        """Convert external bus_id to internal node index via case.get_nodes_id."""
        self._ensure_case_init()
        return int(self.case.get_nodes_id([bus_id])[0][0])

    def _append_resource_to_map(self, resource_id: str, resource: Any) -> None:
        """Add one column to nodes_resources_map for a newly registered resource.

        If the map doesn't exist yet (first resource or case not ready), a full
        rebuild is performed instead so that the map is always consistent.
        """
        if not self.case:
            return

        self._ensure_case_init()

        num_nodes = len(self.case.nodes)

        try:
            internal_idx = self._get_internal_bus_id(resource.bus_id)
        except Exception:
            # Fall back to full rebuild if bus lookup fails
            self._rebuild_nodes_resources_map()
            return

        new_col = np.zeros(num_nodes)
        new_col[internal_idx] = 1.0

        if self.nodes_resources_map is None or self.nodes_resources_map.shape[0] != num_nodes:
            # First resource or shape mismatch — start fresh
            self.nodes_resources_map = new_col.reshape(num_nodes, 1)
            self._resource_col_index = {resource_id: 0}
        else:
            col_idx = self.nodes_resources_map.shape[1]
            self.nodes_resources_map = np.column_stack(
                [self.nodes_resources_map, new_col]
            )
            self._resource_col_index[resource_id] = col_idx

    def _rebuild_nodes_resources_map(self, exclude: str = None) -> None:
        """Full rebuild of nodes_resources_map (O(n) fallback).

        Args:
            exclude: resource_id to skip (used during unregister before dict cleanup).
        """
        resource_ids = [rid for rid in self.sub_resources if rid != exclude]
        if not self.case or not resource_ids:
            self.nodes_resources_map = None
            self._resource_col_index = {}
            return

        self._ensure_case_init()

        bus_ids = [self.sub_resources[rid].bus_id for rid in resource_ids]
        internal_bus_ids = self.case.get_nodes_id(bus_ids)[0]

        num_nodes = len(self.case.nodes)
        num_resources = len(resource_ids)
        self.nodes_resources_map = coo_matrix(
            (np.ones(num_resources), (internal_bus_ids, np.arange(num_resources))),
            shape=(num_nodes, num_resources)
        ).toarray()
        self._resource_col_index = {rid: i for i, rid in enumerate(resource_ids)}

    def _update_nodes_resources_map(self) -> None:
        """Full rebuild (kept for backward-compatibility; prefer incremental helpers)."""
        self._rebuild_nodes_resources_map()

    # ── Abstract Physics Interface ────────────────────────────────────────────

    def cal_pf(self, *args, **kwargs) -> Dict[str, Any]:
        """Run power flow and return result dict (must be implemented).

        If the solver does not converge, implementations should return a
        result dict with ``converged=False`` (or equivalent flag) rather
        than raising an exception, so that downstream ``safety_check`` and
        reward/cost logic can handle divergence gracefully.
        """
        raise NotImplementedError

    def safety_check(self, *args, **kwargs) -> Dict[str, Any]:
        """Check physical constraints (must be implemented).

        Implementations must be robust to non-converged or NaN-laden states
        (e.g. when ``cal_pf`` did not converge).  In such cases the method
        should report maximum violation / unsafe status instead of crashing.
        """
        raise NotImplementedError
