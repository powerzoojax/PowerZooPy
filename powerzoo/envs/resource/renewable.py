"""Renewable energy resources (Solar and Wind)"""
from __future__ import annotations

import logging
import warnings
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .base import ResourceEnv
from powerzoo.data import signals as S

logger = logging.getLogger(__name__)

from gymnasium import spaces as _spaces


class RenewableEnv(ResourceEnv):
    """Renewable energy resource (physical sub-component, not standalone RL env).

    For the full CMDP interface, use a Task which wraps this inside PowerEnv.

    Time series data is loaded and aligned in ``attach()``, not at construction:
    if a ``custom_data_loader`` is provided it is used; otherwise the parent
    grid's ``_time_series_data`` is used.  The result is stored as a
    pre-aligned numpy array for O(1) step access.
    """

    # ====== Initialization ======

    def __init__(self, parent: Any = None, bus_id: int = -1,
                 capacity_mw: float = 100.0,
                 profile_column: Optional[str] = None,
                 custom_data_loader: Any = None,
                 cf_array: Optional[np.ndarray] = None,
                 delta_t_minutes: float = 30.0,
                 normalize_actions: bool = True,
                 curtailment_penalty_per_mwh: float = 0.0,
                 enable_q_control: bool = False,
                 s_rated_mva: Optional[float] = None):
        """Initialize renewable resource

        Args:
            parent: Parent grid environment.
            bus_id: Bus ID to connect to.
            capacity_mw: Installed capacity in MW.
            profile_column: Semantic signal name (e.g. ``signals.SOLAR_AVAILABLE_MW``).
                Legacy raw names (``'Solar'``, ``'Wind'``) are auto-mapped.
                If None, uses default signal based on resource type.
            custom_data_loader: Custom DataLoader instance. If None, uses parent grid's data.
            cf_array: **Direct capacity-factor array** (third data path; mirrors
                JAX ``RenewableBundle.profiles``).  When provided, the resource
                bypasses the parent / loader-based data loading entirely and
                drives output from this array.  Useful for behind-the-meter
                envs (e.g. ``DCMicrogridEnv``) where there is no grid parent
                with a ``_time_series_data`` DataFrame.  Shape ``(T,)`` where
                ``T`` is any positive length; indexing is cyclical
                (``arr[t % T]``).  Values are clipped to ``[0, 1]``.
            delta_t_minutes: Time step in minutes
            normalize_actions: When True (default), action_space is [-1, 1] and
                step() maps ``-1 → full curtailment``, ``1 → full output``.
                When False, action_space is [0, 1] (physical curtailment fraction).
            curtailment_penalty_per_mwh: Economic penalty for curtailed renewable
                energy [$/MWh].  Returned via ``econ_components()`` as the
                ``'curtailment'`` key each step.  Default 0.0 (no penalty).
            enable_q_control: If True, action is 2-D ``[curtailment_norm, q_norm]``
                and obs includes ``q_mvar_norm``.  PQ circle constraint is applied
                (P-priority), matching ``BatteryBundle`` semantics.  Default False.
            s_rated_mva: Inverter apparent power rating [MVA].  Only used when
                ``enable_q_control=True``.  Defaults to ``capacity_mw``.
        """
        if capacity_mw <= 0:
            raise ValueError(f"capacity_mw must be > 0, got {capacity_mw}")

        self.capacity_mw = capacity_mw
        self.profile_column = self._normalize_profile_column(profile_column)
        self.custom_data_loader = custom_data_loader
        self._available_cf = None      # Pre-aligned numpy array: available capacity factor per step
        self._capacity_factor = 0.0    # Current available capacity factor (weather-driven)
        self._cf_cyclical: bool = False  # When True, _cf_at_current_step wraps via modulo
        self.normalize_actions = normalize_actions
        self.curtailment_penalty_per_mwh = float(curtailment_penalty_per_mwh)
        self.enable_q_control = enable_q_control
        self.s_rated_mva = float(s_rated_mva if s_rated_mva is not None else capacity_mw)

        # Direct CF array path (bypasses attach-based data loading)
        if cf_array is not None:
            self.set_cf_array(cf_array)

        super().__init__(parent=parent, bus_id=bus_id, delta_t_minutes=delta_t_minutes)

        self.action_space = self._build_action_space()
        if enable_q_control:
            self.observation_space = _spaces.Box(
                low=np.array([0.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0,  1.0,  1.0,  1.0], dtype=np.float32),
                shape=(5,), dtype=np.float32,
            )
            self.action_names: List[str] = ['curtailment', 'q_control']
        else:
            self.observation_space = _spaces.Box(
                low=np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0,  1.0,  1.0], dtype=np.float32),
                shape=(4,), dtype=np.float32,
            )
            self.action_names: List[str] = ['curtailment']

        self._complete_resource_init()

    def _build_action_space(self) -> _spaces.Box:
        """Build action space based on normalization and Q-control mode.

        P-only (enable_q_control=False):
            normalize_actions=True  → 1-D ``[-1, 1]``
            normalize_actions=False → 1-D ``[0, 1]`` (physical curtailment)
        P+Q (enable_q_control=True):
            Always 2-D ``[-1, 1]²``: ``[curtailment_norm, q_norm]``
        """
        if self.enable_q_control:
            return _spaces.Box(
                low=-np.ones(2, dtype=np.float32),
                high=np.ones(2, dtype=np.float32),
                shape=(2,), dtype=np.float32,
            )
        if self.normalize_actions:
            return _spaces.Box(
                low=-np.ones(1, dtype=np.float32),
                high=np.ones(1, dtype=np.float32),
                shape=(1,), dtype=np.float32,
            )
        return _spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,), dtype=np.float32,
        )

    @staticmethod
    def _normalize_profile_column(col: Optional[str]) -> Optional[str]:
        if col is None:
            return None
        mapped = S._LEGACY_COLUMN_MAP.get(col)
        if mapped is not None:
            warnings.warn(
                f"profile_column='{col}' is a legacy name. "
                f"Use signal name '{mapped}' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            return mapped
        return col

    def attach(self, parent: Any, bus_id: int = None, name: str = None) -> str:
        """Attach resource to parent and load time series data.

        When ``_available_cf`` is already populated (via the ``cf_array``
        constructor argument or ``set_cf_array()``) the parent / loader-based
        data load is skipped so the direct CF path takes precedence and no
        warning is emitted for the missing parent data.
        """
        resource_id = super().attach(parent, bus_id, name)
        if self._available_cf is None:
            self._load_time_series_data()
        return resource_id

    def set_cf_array(self, cf_array: np.ndarray) -> None:
        """Inject a capacity-factor array directly (bypassing attach loading).

        This is the standalone / behind-the-meter data path used by
        ``DCMicrogridEnv``: the microgrid owns the PV profile (synthetic
        clip-sin or loaded externally) and feeds it to the SolarEnv without
        a grid parent.  Mirrors JAX ``RenewableBundle.profiles``.

        Args:
            cf_array: ``(T,)`` numpy array of capacity factors; clipped to
                ``[0, 1]``.  Indexing is cyclical (``arr[t % T]``).
        """
        arr = np.asarray(cf_array, dtype=np.float64).ravel()
        if arr.size == 0:
            raise ValueError("cf_array must be non-empty")
        self._available_cf = np.clip(arr, 0.0, 1.0)
        # Direct CF arrays are intended to be cyclical (mirrors the JAX
        # bundle's ``arr[t % T]`` semantics).  The legacy parent-loaded path
        # keeps its strict bounds-checking behaviour.
        self._cf_cyclical = True

    def _cf_at(self, t: int) -> float:
        """Return capacity factor at a specific time index (cyclical lookup).

        Used by composite envs (``DCMicrogridEnv``) that need to peek at
        ``cf(t)`` for a given step without driving their own internal
        ``time_step`` cursor.
        """
        if self._available_cf is None:
            return 0.0
        return float(self._available_cf[t % len(self._available_cf)])

    # ====== RL Interface Methods ======

    def reset(self, *, seed=None, options=None, day_id: int = None) -> dict:
        """Reset renewable resource and return initial observation."""
        super().reset(seed=seed, options=options, day_id=day_id)
        self.current_p_mw = 0.0
        self.current_q_mvar = 0.0
        # Initialise CF from data so the first obs reflects the true weather.
        self._capacity_factor = self._cf_at_current_step()
        return self.obs()

    @property
    def available_cf(self) -> float:
        """Weather-driven available capacity factor before curtailment."""
        return float(self._capacity_factor)

    @property
    def available_p_mw(self) -> float:
        """Weather-driven available active power before curtailment (MW)."""
        return float(self.available_cf * self.capacity_mw)

    # ====== Time Series Access ======

    def _cf_at_current_step(self) -> float:
        """Read capacity factor for the current (day_id, time_step) position.

        For the legacy parent-loaded path, returns 0.0 when the flat index
        falls outside the data range (with a warning).  For the direct
        ``cf_array`` path (``_cf_cyclical=True``) the lookup wraps via
        modulo so callers can run arbitrarily long episodes against a
        short CF profile.
        """
        if self._available_cf is None:
            return 0.0
        day = self.day_id if self.day_id is not None else 0
        idx = day * self.steps_per_day + self.time_step
        if self._cf_cyclical:
            return float(self._available_cf[idx % len(self._available_cf)])
        if not (0 <= idx < len(self._available_cf)):
            logger.warning(
                "RenewableEnv: flat_idx=%d out of data range [0, %d). "
                "Setting capacity_factor=0.0.",
                idx, len(self._available_cf),
            )
            return 0.0
        return float(self._available_cf[idx])

    # ====== Internal Data Loading ======

    def _load_time_series_data(self):
        """Load time series data and pre-compute capacity factor array.

        Called once during attach().  The raw MW values are converted to
        capacity factors (clipped to [0, 1]) using ``capacity_mw`` and stored
        as a flat numpy array for O(1) access during step().
        """
        signal = self.profile_column if self.profile_column else self._get_default_column()

        if signal is None:
            logger.warning("No signal specified for %s", self.__class__.__name__)
            return

        raw_series = self._load_raw_series(signal)

        if raw_series is None:
            return

        # Convert raw MW values → capacity factor using physical capacity
        cf_array = np.asarray(raw_series.values, dtype=np.float64) / self.capacity_mw
        self._available_cf = np.clip(cf_array, 0.0, 1.0)
        logger.info("Loaded %s: %d steps, peak CF=%.3f",
                     signal, len(self._available_cf), float(self._available_cf.max()))

    def _load_raw_series(self, signal: str) -> Optional[pd.Series]:
        """Dispatch to the appropriate data source.

        Raises on critical errors (missing signal, loader failure).
        Returns None only when no data source is available (no parent, no loader).
        """
        if self.custom_data_loader is None:
            return self._load_from_parent(signal)
        return self._load_from_custom_loader(signal)

    def _load_from_parent(self, signal: str) -> Optional[pd.Series]:
        """Load signal from the parent grid's time series DataFrame."""
        if self._parent is None:
            logger.warning("No parent for %s", self.__class__.__name__)
            return None

        parent_data = getattr(self._parent, '_time_series_data', None)
        if parent_data is None:
            logger.warning("Parent has no _time_series_data")
            return None

        if signal not in parent_data.columns:
            raise ValueError(
                f"Signal '{signal}' not in parent's data. "
                f"Available: {list(parent_data.columns)}"
            )
        return parent_data[signal].copy()

    def _load_from_custom_loader(self, signal: str) -> pd.Series:
        """Load signal via the custom data loader."""
        start_date, end_date, delta_t = self._resolve_date_range()
        resample_freq = f'{int(delta_t)}min'
        data = self.custom_data_loader.load_signals(
            signals=[signal],
            start_date=start_date,
            end_date=end_date,
            resample=resample_freq,
            interpolation='linear',
        )

        if S.DATETIME in data.columns:
            data.set_index(S.DATETIME, inplace=True)

        if signal not in data.columns:
            raise ValueError(
                f"Signal '{signal}' not returned by custom loader. "
                f"Available: {list(data.columns)}"
            )
        return data[signal].copy()

    def _resolve_date_range(self) -> tuple[pd.Timestamp, pd.Timestamp, float]:
        """Derive time range from the parent grid or fall back to defaults."""
        if self._parent is not None:
            return (
                getattr(self._parent, 'start_date', pd.Timestamp('2024-01-01')),
                getattr(self._parent, 'end_date', pd.Timestamp('2024-01-31')),
                getattr(self._parent, 'delta_t_minutes', 30),
            )
        return pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-31'), self.delta_t_minutes

    def _get_default_column(self) -> Optional[str]:
        """Return the default signal name for this resource type.

        Override in subclasses to provide a default signal.  Returning None
        disables data loading when no ``profile_column`` is given.
        """
        return None

    def step(self, action: Any = None) -> None:
        """Update renewable output based on time series data.

        Args:
            action: Optional curtailment control.  Accepted forms:

                - ``None``             → no curtailment (default)
                - ``float`` in [0, 1]  → curtailment fraction (physical)
                - ``numpy.ndarray``    → first element used, same rules as float
                - ``dict``             → reads ``'curtailment'`` key; **always
                  physical [0, 1]**, regardless of ``normalize_actions``

                When ``normalize_actions=True``, float / ndarray actions are in
                ``[-1, 1]`` where ``+1 → no curtailment`` (full output) and
                ``-1 → full curtailment``.
        """
        if self._available_cf is None:
            self.current_p_mw = 0.0
            self.current_q_mvar = 0.0
            self.time_step += 1
            return

        curtailment = self._parse_curtailment(action)
        self._capacity_factor = self._cf_at_current_step()
        self.current_p_mw = self._capacity_factor * self.capacity_mw * (1.0 - curtailment)

        # Reactive power: PQ circle constraint, P-priority (matches BatteryBundle)
        if self.enable_q_control:
            q_norm = self._parse_q_norm(action)
            q_max = float(np.sqrt(max(self.s_rated_mva ** 2 - self.current_p_mw ** 2, 0.0)))
            self.current_q_mvar = float(np.clip(q_norm * self.s_rated_mva, -q_max, q_max))
        else:
            self.current_q_mvar = 0.0

        self.time_step += 1

    def _extract_raw_action(self, action: Any) -> tuple[float, bool]:
        """Extract a scalar value from any accepted action form.

        Returns:
            (raw, is_physical): ``raw`` is the scalar value; ``is_physical``
                is True when the value is already a physical curtailment
                fraction in [0, 1] (dict input), False when normalization
                may still apply (float / ndarray input).
        """
        if action is None:
            return 0.0, True
        if isinstance(action, dict):
            return float(action.get('curtailment', 0.0)), True
        if isinstance(action, np.ndarray):
            return float(action.flat[0]), False
        return float(action), False

    def _parse_q_norm(self, action: Any) -> float:
        """Extract Q normalised value ∈ [-1, 1] from action.

        Accepted forms when ``enable_q_control=True``:
            - ``None``          → 0.0 (no reactive injection)
            - ``dict``          → reads ``'q_norm'`` key ∈ [-1, 1]
            - ``np.ndarray``    → second element (``action.flat[1]``)
            - ``float``         → 0.0 (scalar implies P-only intent)
        """
        if action is None:
            return 0.0
        if isinstance(action, dict):
            return float(np.clip(action.get('q_norm', 0.0), -1.0, 1.0))
        if isinstance(action, np.ndarray) and action.size >= 2:
            return float(np.clip(action.flat[1], -1.0, 1.0))
        return 0.0

    def _parse_curtailment(self, action: Any) -> float:
        """Convert any accepted action form to physical curtailment ∈ [0, 1].

        Delegates type extraction to ``_extract_raw_action``, then applies
        denormalization when ``normalize_actions=True`` for non-dict actions.
        """
        raw, is_physical = self._extract_raw_action(action)

        if is_physical:
            return float(np.clip(raw, 0.0, 1.0))

        if self.normalize_actions:
            # [-1, 1] → curtailment: +1 = no curtailment, -1 = full curtailment
            curtailment = (1.0 - raw) / 2.0
        else:
            curtailment = raw

        return float(np.clip(curtailment, 0.0, 1.0))

    def obs(self, state: Any = None) -> dict:
        """Observation dict matching ``self.observation_space``.

        Keys (alphabetical order — defines flattening order in PowerEnv):

        P-only (enable_q_control=False):
            ``available_cf``, ``p_mw_norm``, ``time_cos``, ``time_sin``
        P+Q (enable_q_control=True):
            ``available_cf``, ``p_mw_norm``, ``q_mvar_norm``, ``time_cos``, ``time_sin``
        """
        p_norm = self.current_p_mw / self.capacity_mw if self.capacity_mw > 0 else 0.0
        phase = 2.0 * np.pi * self.time_step / max(self.steps_per_day, 1)
        d = {
            'available_cf': float(self.available_cf),
            'p_mw_norm': float(p_norm),
            'time_cos': float(np.cos(phase)),
            'time_sin': float(np.sin(phase)),
        }
        if self.enable_q_control:
            q_norm = self.current_q_mvar / self.s_rated_mva if self.s_rated_mva > 0 else 0.0
            d['q_mvar_norm'] = float(q_norm)
        return d

    def grid_obs(self) -> np.ndarray:
        """Grid-embedded observation (strips time encoding — grid provides its own).

        P-only: ``[available_cf, p_mw_norm]``
        P+Q:    ``[available_cf, p_mw_norm, q_mvar_norm]``
        """
        p_norm = self.current_p_mw / self.capacity_mw if self.capacity_mw > 0 else 0.0
        if self.enable_q_control:
            q_norm = self.current_q_mvar / self.s_rated_mva if self.s_rated_mva > 0 else 0.0
            return np.array([float(self.available_cf), float(p_norm), float(q_norm)], dtype=np.float32)
        return np.array([float(self.available_cf), float(p_norm)], dtype=np.float32)

    def grid_obs_names(self, rid: str) -> list:
        if self.enable_q_control:
            return [f'{rid}_available_cf', f'{rid}_p_mw_norm', f'{rid}_q_mvar_norm']
        return [f'{rid}_available_cf', f'{rid}_p_mw_norm']

    def grid_action_from_normalized(self, raw: float) -> float:
        """Curtailment semantics: +1 → no curtailment (0.0), -1 → full curtailment (1.0)."""
        low, high = self.grid_action_bounds()   # (0.0, 1.0)
        return float(np.clip((low + high) / 2 - raw * (high - low) / 2, low, high))

    # ====== Status & Diagnostics ======

    def status(self):
        """Return current renewable resource status.

        Includes all base fields (``current_p_mw``, ``current_q_mvar``,
        ``time_step``, ``bus_id``, ``local_v``) plus:

        ``capacity_mw``: installed capacity in MW.
        ``available_cf``: weather-driven available capacity factor ∈ [0, 1]
            (fraction of capacity available before curtailment).
        ``available_p_mw``: maximum power output before curtailment (MW).
        ``curtailed_p_mw``: power curtailed this step (MW), i.e.
            ``available_p_mw - current_p_mw``.
        ``output_ratio``: actual output / installed capacity ∈ [0, 1]
            (after curtailment).
        """
        avail = self.available_p_mw
        d = {
            'current_p_mw': self.current_p_mw,
            'current_q_mvar': self.current_q_mvar,
            'capacity_mw': self.capacity_mw,
            'available_cf': self.available_cf,
            'available_p_mw': avail,
            'curtailed_p_mw': avail - self.current_p_mw,
            'output_ratio': self.current_p_mw / self.capacity_mw if self.capacity_mw > 0 else 0.0,
            'time_step': self.time_step,
            'bus_id': int(self._bus_id),
            'local_v': self._get_local_voltage(),
        }
        if self.enable_q_control:
            d['s_rated_mva'] = self.s_rated_mva
            d['q_mvar_norm'] = self.current_q_mvar / self.s_rated_mva if self.s_rated_mva > 0 else 0.0
        return d

    def econ_components(self, dt_hours: float) -> dict:
        """Economic curtailment penalty for this step.

        Returns ``{'curtailment': -penalty_per_mwh * curtailed_mwh}`` when
        ``curtailment_penalty_per_mwh > 0``, otherwise ``{}``.
        """
        if self.curtailment_penalty_per_mwh <= 0:
            return {}
        curtailed_mwh = max(0.0, self.available_p_mw - self.current_p_mw) * dt_hours
        return {'curtailment': -self.curtailment_penalty_per_mwh * curtailed_mwh}


class SolarEnv(RenewableEnv):
    """Solar PV resource"""

    name = 'solar'

    def _get_default_column(self) -> Optional[str]:
        return S.SOLAR_AVAILABLE_MW


class WindEnv(RenewableEnv):
    """Wind turbine resource"""

    name = 'wind'

    def _get_default_column(self) -> Optional[str]:
        return S.WIND_AVAILABLE_MW
