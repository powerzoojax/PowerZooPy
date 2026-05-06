"""FlexLoad — Flexible Load Demand Response Resource
==================================================

Physically accurate model of a controllable load for use in Power System RL
environments. Supports:

    - **Curtailment** (interruptible load, IL): permanent demand reduction.
    - **Demand Shifting** (time-shiftable load, TL): energy-conserving deferral.

Physical Model
--------------
At each discrete step t with interval dt [h], actual consumption is:

    D(t) = D_base(t) - c(t) - s_out(t) + s_in(t)           [MW]

where:
    D_base(t)  : exogenous baseline (uncontrolled) demand     [MW]
    c(t)       in [0, C_max]   curtailment (permanent shed)   [MW]
    s_out(t)   in [0, S_max]   demand shifted out to buffer   [MW]
    s_in(t)    >= 0            deferred demand released now    [MW]

Energy conservation for shifting over horizon H:
    sum_{tau=t+1}^{t+H} s_in(tau)  =  s_out(t)              [MWh equality]

Net injection change at bus b due to DR (generator-positive convention):
    dP_inj[b] = c(t) + s_out(t) - s_in(t)                   [MW]

Action Space (2-dimensional)
----------------------------
Two independent non-negative control variables:

    action[0]  curtailment   in [0, C_max]
    action[1]  shift-out     in [0, S_max]

Mutual exclusivity (c(t) * s_out(t) = 0) is NOT hard-enforced, reflecting
the soft complementarity approach used in MILP-based SCUC (via a big-M
constraint or a penalty term). The ``cost_simultaneous`` field in ``status()``
provides a CMDP cost signal for the RL agent.

Action Scaling Modes (``action_scale`` parameter)
-------------------------------------------------
``'physical'``   action_space = Box([0, 0], [C_max, S_max])   raw MW
``'unit'``       action_space = Box([0, 0], [1, 1])           x capacity  (PPO-friendly)
``'tanh'``       action_space = Box([-1,-1], [1, 1])          (a+1)/2 x capacity  (SAC-friendly)

Observation Vector (8-dim)
--------------------------
Index  Name                  Formula                     Range
  0    curtail_norm          c(t) / C_max                [0, 1]
  1    shift_out_norm        s_out(t) / S_max            [0, 1]
  2    shift_in_norm         s_in(t) / S_max             [0, 1]
  3    buffer_fill_ratio     |buffer| / H                [0, 1]
  4    buffer_energy_norm    sum(buffer) / (S_max * H)   [0, 1]
  5    time_sin              sin(2*pi*t/T)               [-1, 1]
  6    time_cos              cos(2*pi*t/T)               [-1, 1]
  7    price_norm            LMP / price_ref             [0, 2] clipped

CMDP Cost Signals (in ``status()``)
------------------------------------
``cost_curtailment``      c(t)*dt * curtail_cost_per_mwh      [$]
``cost_shift_discomfort`` sum(buffer)*dt * shift_cost_per_mwh  [$]  (holding cost)
``cost_buffer_overflow``  energy not released in time          [MWh]
``cost_simultaneous``     min(c(t), s_out(t)) * penalty       [MW] (complementarity)

References
----------
[1] Albadi & El-Saadany (2008). A summary of demand response in electricity
    markets. Electric Power Systems Research, 78(11), 1989-1996.
[2] Wang et al. (2019). Optimal joint-dispatch of energy and reserve for
    CCHP-based microgrids. IET GTD.
[3] Lu & Hong (2019). Incentive-based demand response for smart grid with
    reinforcement learning and deep neural network. Applied Energy, 236.
[4] Ye et al. (2022). Model-free real-time autonomous control for a
    residential multi-energy system using DRL. IEEE Trans. Smart Grid, 13(2).
"""

from __future__ import annotations

import warnings
from collections import deque
from itertools import islice
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from gymnasium import spaces as _spaces

from .base import ResourceEnv

ActionScale = Literal['physical', 'unit', 'tanh']


class _DeferralBuffer:
    """FIFO buffer that manages deferred demand across a rolling horizon.

    Each shift-out of ``shift_mw`` MW spreads the energy uniformly over
    the next ``horizon`` release slots (one slot released per step).

    Responsibilities:
        - enqueue new deferred demand (``add``)
        - release the slot due this step (``release``)
        - report aggregate state (``total_mw``, ``fill_ratio``, ``overflow_mwh``)

    Time-discretisation (``dt_hours``) is passed at query time rather than at
    construction so the buffer stays correct after ``ResourceEnv.attach()``
    syncs the parent grid's ``delta_t_minutes``.
    """

    def __init__(self, horizon: int) -> None:
        self.horizon = int(horizon)
        self._slots: deque = deque()

    # -- Mutation --

    def add(self, shift_mw: float) -> None:
        """Enqueue ``shift_mw`` MW spread uniformly over ``horizon`` slots."""
        per_step_mw = shift_mw / self.horizon
        for _ in range(self.horizon):
            self._slots.append(per_step_mw)

    def release(self) -> float:
        """Pop and return the MW scheduled for release this step (0.0 if empty)."""
        return float(self._slots.popleft()) if self._slots else 0.0

    def clear(self) -> None:
        """Discard all pending slots (call on episode reset)."""
        self._slots.clear()

    # -- Read-only state --

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def total_mw(self) -> float:
        """Sum of all pending release slots [MW]."""
        return float(sum(self._slots))

    @property
    def fill_ratio(self) -> float:
        """Fraction of the horizon window currently occupied [0, 1+]."""
        return len(self._slots) / self.horizon

    @property
    def is_over_horizon(self) -> bool:
        """True when the buffer holds more than one full horizon of entries."""
        return len(self._slots) > self.horizon

    def overflow_mwh(self, dt_hours: float) -> float:
        """Energy held in slots beyond the shift horizon — unreleasable in time [MWh]."""
        excess_mw = sum(islice(self._slots, self.horizon, None))
        return float(excess_mw) * dt_hours


class FlexLoad(ResourceEnv):
    """Flexible Load Demand Response Resource (physical sub-component, not standalone RL env).

    Implements a two-action (curtailment + shift-out) demand response asset.

    Sign convention: curtailment and shift-out both reduce net load at the bus,
    which is equivalent to *injecting* power (positive ``current_p_mw``).
    Shift-in returns deferred load, reducing the injection.

    For the full CMDP interface, use a Task which wraps this inside PowerEnv.

    Parameters
    ----------
    curtail_cap_mw : float
        Maximum curtailment capacity [MW]. Default: 10.0.
    shift_cap_mw : float
        Maximum per-step demand shift capacity [MW]. Default: 10.0.
    shift_horizon : int
        Number of future steps over which deferred demand is uniformly
        released. Longer horizons reduce rebound severity. Default: 4.
    baseline_mw : float
        Nominal baseline demand [MW]. Reserved for profile normalisation;
        not currently used in the observation vector. Default: 50.0.
    curtail_cost_per_mwh : float
        Discomfort / compensation cost for curtailment [$/MWh]. Default: 50.0.
    shift_cost_per_mwh : float
        Holding / discomfort cost rate for buffered deferred demand [$/MWh].
        Default: 10.0.
    complementarity_penalty : float
        Penalty coefficient [$/MW] for co-activating curtailment and shift-out.
        Default: 100.0.
    price_ref : float
        Reference LMP for price normalisation in observation [$/MWh]. Default: 100.0.
    action_scale : {'physical', 'unit', 'tanh'}
        Action space normalisation mode. Default: 'unit'.
    normalize_actions : bool or None
        Legacy compatibility parameter. If provided, overrides action_scale:
        True -> 'unit', False -> 'physical'. Default: None (use action_scale).
    parent : GridEnv or None
        Parent grid environment.
    bus_id : int
        Bus index where this asset is connected.
    delta_t_minutes : float
        Duration of each RL time step [min]. Default: 15.0.
    """

    name = 'flexload'

    def __init__(
        self,
        curtail_cap_mw: float = 10.0,
        shift_cap_mw: float = 10.0,
        shift_horizon: int = 4,
        baseline_mw: float = 50.0,
        curtail_cost_per_mwh: float = 50.0,
        shift_cost_per_mwh: float = 10.0,
        complementarity_penalty: float = 100.0,
        price_ref: float = 100.0,
        action_scale: ActionScale = 'unit',
        normalize_actions: Optional[bool] = None,
        parent: Any = None,
        bus_id: int = -1,
        delta_t_minutes: float = 15.0,
    ):
        super().__init__(parent=parent, bus_id=bus_id, delta_t_minutes=delta_t_minutes)

        # -- Physical parameters --
        self.curtail_cap_mw = float(curtail_cap_mw)
        self.shift_cap_mw = float(shift_cap_mw)
        self.shift_horizon = max(int(shift_horizon), 1)
        self.baseline_mw = float(baseline_mw) if baseline_mw > 0 else 1.0
        # NOTE: use self.dt_hours (synced by attach()) instead of a
        # private copy, so cost calculations stay correct when the parent
        # grid has a different delta_t_minutes.

        # -- Cost / market parameters --
        self.curtail_cost_per_mwh = float(curtail_cost_per_mwh)
        self.shift_cost_per_mwh = float(shift_cost_per_mwh)
        self.complementarity_penalty = float(complementarity_penalty)
        self.price_ref = float(price_ref) if price_ref > 0 else 100.0

        # -- Action scaling mode --
        self.action_scale: ActionScale = self._resolve_action_scale(action_scale, normalize_actions)

        # -- Internal state --
        self._curtailed_mw: float = 0.0
        self._shift_out_mw: float = 0.0
        self._shift_in_mw: float = 0.0
        self._buffer = _DeferralBuffer(self.shift_horizon)
        self._current_lmp: float = 0.0

        # -- Gymnasium spaces --
        self.action_space = self._build_action_space()

        # Obs: [curtail_norm, shift_out_norm, shift_in_norm,
        #       buffer_fill_ratio, buffer_energy_norm,
        #       time_sin, time_cos, price_norm]
        self.observation_space = _spaces.Box(
            low=np.array([0., 0., 0., 0., 0., -1., -1., 0.], dtype=np.float32),
            high=np.array([1., 1., 1., 1., 1.,  1.,  1., 2.], dtype=np.float32),
            shape=(8,),
            dtype=np.float32,
        )

        self.action_names: List[str] = ['curtail_mw', 'shift_out_mw']
        self.obs_names: List[str] = [
            'curtail_norm', 'shift_out_norm', 'shift_in_norm',
            'buffer_fill_ratio', 'buffer_energy_norm',
            'time_sin', 'time_cos', 'price_norm',
        ]

        self._complete_resource_init()

    # ====== RL Interface ======

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
        day_id: Optional[int] = None,
    ) -> None:
        """Reset to zero-action, empty-buffer state."""
        super().reset(seed=seed, options=options, day_id=day_id)
        self._curtailed_mw = 0.0
        self._shift_out_mw = 0.0
        self._shift_in_mw = 0.0
        self._buffer.clear()
        self._current_lmp = 0.0
        self.current_p_mw = 0.0
        self.current_q_mvar = 0.0

    def step(
        self,
        action: Optional[Union[float, np.ndarray, Dict]] = None,
    ) -> None:
        """Apply flexibility action for one time step.

        Execution order:
        1. Release any deferred demand due this step  -> s_in(t)
        2. Parse and scale the RL action              -> c(t), s_out(t)
        3. Clip to physical capacities
        4. Buffer s_out(t) for future release
        5. Compute net injection: dP = c(t) + s_out(t) - s_in(t)
        6. Advance time step

        Parameters
        ----------
        action : None | float | ndarray | dict
            - None or 0           -> no flexibility (idle)
            - ndarray shape (2,)  -> [curtail_mw, shift_out_mw] (scaled)
            - dict                -> keys 'curtail_mw', 'shift_out_mw'
        """
        # Step 1: release deferred demand scheduled for this step
        self._shift_in_mw = self._buffer.release()

        # Step 2: parse + scale action to physical MW
        c_raw, s_raw = self._parse_action(action)

        # Step 3: clip to capacity
        self._curtailed_mw = float(np.clip(c_raw, 0.0, self.curtail_cap_mw))
        self._shift_out_mw = float(np.clip(s_raw, 0.0, self.shift_cap_mw))

        # Step 4: buffer the shift-out demand for future release
        if self._shift_out_mw > 0.0:
            self._buffer.add(self._shift_out_mw)

        # Step 5: net injection at bus (generator-positive convention)
        self.current_p_mw = (
            self._curtailed_mw + self._shift_out_mw - self._shift_in_mw
        )
        self.current_q_mvar = 0.0

        # Step 6: advance
        self.time_step += 1

    def obs(self, state: Any = None) -> Dict[str, float]:
        """Return observation dict matching ``self.observation_space``."""
        curtail_cap = self.curtail_cap_mw or 1.0
        shift_cap   = self.shift_cap_mw or 1.0
        buffer_cap_mw = max(self.shift_cap_mw * self.shift_horizon, 1e-6)

        return {
            'curtail_norm':       float(np.clip(self._curtailed_mw / curtail_cap, 0.0, 1.0)),
            'shift_out_norm':     float(np.clip(self._shift_out_mw / shift_cap,   0.0, 1.0)),
            'shift_in_norm':      float(np.clip(self._shift_in_mw  / shift_cap,   0.0, 1.0)),
            'buffer_fill_ratio':  float(np.clip(self._buffer.fill_ratio,           0.0, 1.0)),
            'buffer_energy_norm': float(np.clip(self._buffer.total_mw / buffer_cap_mw, 0.0, 1.0)),
            'time_sin':           float(np.sin(self._time_phase)),
            'time_cos':           float(np.cos(self._time_phase)),
            'price_norm':         float(np.clip(self._current_lmp / self.price_ref, 0.0, 2.0)),
        }

    def status(self) -> Dict[str, Any]:
        """Return full status dict with physical state and CMDP cost signals.

        CMDP cost fields:
        - ``cost_curtailment``: discomfort/compensation for curtailed energy [$/step]
        - ``cost_shift_discomfort``: holding cost for buffered demand [$/step]
        - ``cost_buffer_overflow``: energy beyond shift horizon [MWh]
        - ``cost_simultaneous``: complementarity violation penalty [$/step]
        """
        overflow_mwh = self._buffer.overflow_mwh(self.dt_hours)

        return {
            # Physical state
            'current_p_mw':        self.current_p_mw,
            'current_q_mvar':      self.current_q_mvar,
            'curtailed_mw':        self._curtailed_mw,
            'shift_out_mw':        self._shift_out_mw,
            'shift_in_mw':         self._shift_in_mw,
            'buffer_size':         len(self._buffer),
            'buffer_total_mw':     self._buffer.total_mw,
            'buffer_overflow_mwh': overflow_mwh,
            'current_lmp':         self._current_lmp,
            'time_step':           self.time_step,
            'bus_id':              int(self._bus_id),
            'local_v':             self._get_local_voltage(),
            # CMDP cost signals
            'cost_curtailment':      self._curtailed_mw * self.dt_hours * self.curtail_cost_per_mwh,
            'cost_shift_discomfort': self._buffer.total_mw * self.dt_hours * self.shift_cost_per_mwh,
            'cost_buffer_overflow':  overflow_mwh,
            'cost_simultaneous':     min(self._curtailed_mw, self._shift_out_mw) * self.complementarity_penalty,
        }

    # ====== External State Injection ======

    def set_lmp(self, lmp: float) -> None:
        """Inject current Locational Marginal Price [$/MWh] into the observation."""
        self._current_lmp = float(lmp)

    def get_bid(self) -> Dict[str, float]:
        """Return a three-part bid structure compatible with SCUC/SCED frameworks.

        Returns
        -------
        dict with keys:
            'curtail_cap_mw', 'curtail_price_per_mwh',
            'shift_cap_mw', 'shift_price_per_mwh', 'shift_horizon'
        """
        available_shift = 0.0 if self._buffer.is_over_horizon else self.shift_cap_mw

        return {
            'curtail_cap_mw':        self.curtail_cap_mw,
            'curtail_price_per_mwh': self.curtail_cost_per_mwh,
            'shift_cap_mw':          available_shift,
            'shift_price_per_mwh':   self.shift_cost_per_mwh,
            'shift_horizon':         self.shift_horizon,
        }

    # ====== Private Properties ======

    @staticmethod
    def _resolve_action_scale(
        action_scale: ActionScale,
        normalize_actions: Optional[bool],
    ) -> ActionScale:
        """Resolve the effective action scale, handling the legacy ``normalize_actions`` flag."""
        if normalize_actions is not None:
            warnings.warn(
                "normalize_actions is deprecated; use action_scale='unit' or 'physical' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            return 'unit' if normalize_actions else 'physical'
        if action_scale not in ('physical', 'unit', 'tanh'):
            raise ValueError(
                f"action_scale must be 'physical', 'unit', or 'tanh'; got '{action_scale}'"
            )
        return action_scale

    @property
    def _time_phase(self) -> float:
        """Current time step expressed as a phase angle [rad] within the daily cycle."""
        return 2.0 * np.pi * self.time_step / max(self.steps_per_day, 1)

    # ====== Internal Dynamics ======

    def _build_action_space(self) -> _spaces.Box:
        """Construct the Gymnasium action space for the chosen scale mode."""
        if self.action_scale == 'physical':
            return _spaces.Box(
                low=np.zeros(2, dtype=np.float32),
                high=np.array([self.curtail_cap_mw, self.shift_cap_mw], dtype=np.float32),
                shape=(2,), dtype=np.float32,
            )
        elif self.action_scale == 'unit':
            return _spaces.Box(
                low=np.zeros(2, dtype=np.float32),
                high=np.ones(2, dtype=np.float32),
                shape=(2,), dtype=np.float32,
            )
        else:  # 'tanh'
            return _spaces.Box(
                low=-np.ones(2, dtype=np.float32),
                high=np.ones(2, dtype=np.float32),
                shape=(2,), dtype=np.float32,
            )

    def _parse_action(
        self,
        action: Optional[Union[float, np.ndarray, Dict]],
    ) -> Tuple[float, float]:
        """Dispatch to the correct parsing path and return physical (curtail_mw, shift_out_mw).

        Dict actions carry physical MW values directly (no scaling).
        Array/scalar actions are first extracted then passed through ``_scale_to_physical``.
        """
        if action is None:
            return 0.0, 0.0
        if isinstance(action, dict):
            return (
                max(float(action.get('curtail_mw', 0.0)), 0.0),
                max(float(action.get('shift_out_mw', 0.0)), 0.0),
            )
        a0, a1 = self._extract_array_action(action)
        return self._scale_to_physical(a0, a1)

    def _extract_array_action(
        self,
        action: Union[float, np.ndarray],
    ) -> Tuple[float, float]:
        """Flatten an array/scalar action into a raw (a0, a1) pair before capacity scaling."""
        arr = np.asarray(action, dtype=np.float64).flatten()
        if arr.size == 0:
            return 0.0, 0.0
        if arr.size == 1:
            warnings.warn(
                "FlexLoad received a scalar action. Interpreting as curtailment only. "
                "Pass a 2-element array [curtail, shift_out] for full control.",
                stacklevel=3,
            )
            return float(arr[0]), 0.0
        return float(arr[0]), float(arr[1])

    def _scale_to_physical(self, a0: float, a1: float) -> Tuple[float, float]:
        """Map raw action components to physical MW using ``self.action_scale``."""
        if self.action_scale == 'physical':
            return max(a0, 0.0), max(a1, 0.0)
        if self.action_scale == 'unit':
            return (
                float(np.clip(a0, 0.0, 1.0) * self.curtail_cap_mw),
                float(np.clip(a1, 0.0, 1.0) * self.shift_cap_mw),
            )
        # 'tanh': map [-1, 1] → [0, capacity]
        return (
            float((np.clip(a0, -1.0, 1.0) + 1.0) / 2.0 * self.curtail_cap_mw),
            float((np.clip(a1, -1.0, 1.0) + 1.0) / 2.0 * self.shift_cap_mw),
        )

    def __repr__(self) -> str:
        return (
            f"FlexLoad("
            f"curtail={self.curtail_cap_mw:.1f}MW, "
            f"shift={self.shift_cap_mw:.1f}MW, "
            f"horizon={self.shift_horizon}steps, "
            f"scale='{self.action_scale}'"
            f")"
        )
