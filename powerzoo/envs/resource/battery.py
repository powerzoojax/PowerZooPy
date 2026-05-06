"""Battery Energy Storage System (BESS) Resource

Provides battery storage with charge/discharge control for grid services.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import ResourceEnv

from gymnasium import spaces as _spaces

_DEFAULT_ONEWAY_ETA = 0.95


def _check_eta(name: str, value: float) -> None:
    """Raise ValueError if *value* is not a valid one-way efficiency ∈ (0, 1]."""
    if not (0.0 < value <= 1.0):
        raise ValueError(f"{name} must be in (0, 1], got {value}")


def _resolve_battery_efficiencies(
    eta_charge: Optional[float],
    eta_discharge: Optional[float],
    eta_roundtrip: Optional[float],
    efficiency: Optional[float],
) -> Tuple[float, float]:
    """Resolve one-way charge/discharge efficiencies.

    - If neither ``eta_roundtrip`` nor deprecated ``efficiency`` is set: any
      missing ``eta_charge`` / ``eta_discharge`` defaults to
      ``_DEFAULT_ONEWAY_ETA`` (0.95).
    - If ``eta_roundtrip`` is set (or ``efficiency`` is passed): symmetric
      decomposition uses ``sqrt(eta_roundtrip)`` for unspecified sides; when both
      one-way values are explicit, ``eta_roundtrip`` is ignored for assignment.
    """
    if efficiency is not None and eta_roundtrip is not None:
        raise ValueError(
            "Pass only one of 'efficiency' (deprecated) and 'eta_roundtrip'"
        )
    if efficiency is not None:
        warnings.warn(
            "BatteryEnv parameter 'efficiency' is deprecated; use "
            "'eta_roundtrip' instead (same meaning).",
            DeprecationWarning,
            stacklevel=3,
        )
        eta_roundtrip = float(efficiency)

    if eta_roundtrip is None:
        ec = _DEFAULT_ONEWAY_ETA if eta_charge is None else float(eta_charge)
        ed = _DEFAULT_ONEWAY_ETA if eta_discharge is None else float(eta_discharge)
        _check_eta('eta_charge', ec)
        _check_eta('eta_discharge', ed)
        return ec, ed

    _check_eta('eta_roundtrip', eta_roundtrip)
    eta_rt_sqrt = float(np.sqrt(eta_roundtrip))

    if eta_charge is not None and eta_discharge is not None:
        ec, ed = float(eta_charge), float(eta_discharge)
        _check_eta('eta_charge', ec)
        _check_eta('eta_discharge', ed)
        return ec, ed

    if eta_charge is not None:
        ec = float(eta_charge)
        _check_eta('eta_charge', ec)
        return ec, eta_rt_sqrt

    if eta_discharge is not None:
        ed = float(eta_discharge)
        _check_eta('eta_discharge', ed)
        return eta_rt_sqrt, ed

    return eta_rt_sqrt, eta_rt_sqrt


class BatteryEnv(ResourceEnv):
    """Battery Energy Storage System (physical sub-component, not standalone RL env).

    For the full CMDP interface (obs, reward, cost, terminated, truncated, info),
    use a Task (e.g. ``battery_arbitrage``) which wraps this resource inside PowerEnv.

    Models a battery with:
    - State of Charge (SOC) dynamics
    - Separate charge / discharge efficiency
    - Charge/discharge power limits
    - SOC constraints

    Sign convention (matches all other resources):
        current_p > 0: discharging (injecting power to the grid)
        current_p < 0: charging (drawing power from the grid)

    Efficiency model:
        Discharging: grid receives `p` MW, battery loses `p / eta_discharge` MWh per hour
        Charging:    grid supplies `|p|` MW, battery gains `|p| * eta_charge` MWh per hour
        Round-trip:  eta_rt = eta_charge * eta_discharge

    Default one-way efficiencies are 0.95 each (round-trip ≈ 0.9025).  To use a
    round-trip shorthand with symmetric ``sqrt(η_rt)`` decomposition, pass
    ``eta_roundtrip``.  For the legacy default that matched η_rt = 0.95 with equal
    legs, use ``eta_roundtrip=0.95``.

    Args:
        capacity_mwh: Energy capacity in MWh
        power_mw: Maximum charge/discharge power in MW
        eta_charge: One-way charging efficiency ∈ (0, 1].  Default 0.95 if omitted
            and ``eta_roundtrip`` is not set.
        eta_discharge: One-way discharging efficiency ∈ (0, 1].  Default 0.95 if
            omitted and ``eta_roundtrip`` is not set.
        eta_roundtrip: Optional round-trip η_rt ∈ (0, 1].  When set, unspecified
            one-way sides use ``sqrt(eta_roundtrip)``; when both one-way values
            are explicit, this is ignored for assignment.
        efficiency: Deprecated alias for ``eta_roundtrip``.
        soc_min: Minimum SOC (0-1)
        soc_max: Maximum SOC (0-1)
        initial_soc: Initial SOC (0-1)
    """

    name = 'battery'  # Resource type name

    # ====== Initialization ======

    def __init__(self, capacity_mwh: float = 50.0, power_mw: float = 20.0,
                 eta_charge: Optional[float] = None,
                 eta_discharge: Optional[float] = None,
                 eta_roundtrip: Optional[float] = None,
                 efficiency: Optional[float] = None,
                 soc_min: float = 0.1, soc_max: float = 0.9,
                 initial_soc: float = 0.5,
                 parent: Any = None, bus_id: int = -1,
                 normalize_actions: bool = True,
                 delta_t_minutes: float = 15.0,
                 cycle_cost_per_mwh: float = 0.0,
                 enable_q_control: bool = False,
                 s_rated_mva: Optional[float] = None):
        """Initialize battery storage system

        Args:
            capacity_mwh: Energy capacity in MWh (default: 50)
            power_mw: Maximum power in MW (default: 20)
            eta_charge: One-way charging efficiency ∈ (0, 1].  Omitted sides default
                to 0.95 unless ``eta_roundtrip`` is set (then ``sqrt(eta_roundtrip)``).
            eta_discharge: One-way discharging efficiency ∈ (0, 1].
            eta_roundtrip: Round-trip η_rt ∈ (0, 1] for sqrt decomposition (optional).
            efficiency: Deprecated; same as ``eta_roundtrip``.
            soc_min: Minimum SOC 0-1 (default: 0.1)
            soc_max: Maximum SOC 0-1 (default: 0.9)
            initial_soc: Initial SOC 0-1 (default: 0.5)
            parent: Parent grid or hub to attach to (optional)
            bus_id: Bus ID where this battery is connected (default: -1)
            normalize_actions: When True (default), action_space is [-1, 1] and
                step() maps ``-1 → -power_mw`` (charge), ``0 → idle``,
                ``1 → +power_mw`` (discharge).  When False, action_space
                is [-power_mw, power_mw] in physical MW.
            delta_t_minutes: Time step duration in minutes (default: 15).
                Overridden by parent grid's value after attach().
            cycle_cost_per_mwh: Cycle-degradation cost per MWh of energy
                throughput [$/MWh].  Returned via ``econ_components()`` as the
                ``'cycle_degradation'`` key each step.  Default 0.0 (no cost).
            enable_q_control: When True, action becomes 2D [P, Q] with PQ
                circle constraint P² + Q² ≤ S_rated².  Default False.
            s_rated_mva: Inverter apparent power rating [MVA].  Defaults to
                ``power_mw`` (inverter sized to active power rating).
        """
        super().__init__(parent=parent, bus_id=bus_id, delta_t_minutes=delta_t_minutes)
        self._randomize_soc_on_reset = False
        self._soc_init_low = soc_min
        self._soc_init_high = soc_max

        # --- Parameter validation ---
        if capacity_mwh <= 0:
            raise ValueError(f"capacity_mwh must be > 0, got {capacity_mwh}")
        if power_mw <= 0:
            raise ValueError(f"power_mw must be > 0, got {power_mw}")
        if not (0.0 <= soc_min <= soc_max <= 1.0):
            raise ValueError(
                f"Required: 0 <= soc_min <= soc_max <= 1, "
                f"got soc_min={soc_min}, soc_max={soc_max}"
            )

        # --- Parameters ---
        self.capacity_mwh = capacity_mwh
        self.power_mw = power_mw
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.initial_soc = np.clip(initial_soc, soc_min, soc_max)

        self.eta_charge, self.eta_discharge = _resolve_battery_efficiencies(
            eta_charge, eta_discharge, eta_roundtrip, efficiency
        )

        # Backward-compatible attribute: round-trip efficiency (product)
        self.efficiency = self.eta_charge * self.eta_discharge

        # --- Q control ---
        self.enable_q_control = enable_q_control
        self.s_rated_mva = float(s_rated_mva if s_rated_mva is not None else power_mw)

        # --- State ---
        self.soc = self.initial_soc  # State of charge (0-1)
        self.current_p_mw = 0.0
        self.current_q_mvar = 0.0
        self._clipped_power_mw = 0.0
        self._soc_history = []
        self.cycle_cost_per_mwh = float(cycle_cost_per_mwh)

        # --- Action normalization & Gymnasium spaces ---
        self.normalize_actions = normalize_actions
        action_dim = 2 if enable_q_control else 1
        self._action_phys_low = np.full(action_dim, -self.power_mw, dtype=np.float32)
        self._action_phys_high = np.full(action_dim, self.power_mw, dtype=np.float32)
        if enable_q_control:
            self._action_phys_low[1] = -self.s_rated_mva
            self._action_phys_high[1] = self.s_rated_mva

        if self.normalize_actions:
            self.action_space = _spaces.Box(
                low=-np.ones(action_dim, dtype=np.float32),
                high=np.ones(action_dim, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = _spaces.Box(
                low=self._action_phys_low,
                high=self._action_phys_high,
                dtype=np.float32,
            )

        if enable_q_control:
            self.observation_space = _spaces.Box(
                low=np.array([0.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0,  1.0,  1.0, 1.0, 1.0,  1.0,  1.0], dtype=np.float32),
                dtype=np.float32,
            )
            self.action_names: List[str] = ['p_mw', 'q_mvar']
        else:
            self.observation_space = _spaces.Box(
                low=np.array([0.0, -1.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0,  1.0, 1.0, 1.0,  1.0,  1.0], dtype=np.float32),
                dtype=np.float32,
            )
            self.action_names: List[str] = ['p_mw']

        # --- Diagnostics ---
        self.throughput_mwh = 0.0  # cumulative |P|·Δt since last reset

        self._complete_resource_init()

    # ====== RL Interface Methods ======

    def configure_soc_randomization(
        self, enable: bool, low: float = 0.3, high: float = 0.7
    ) -> None:
        """Configure automatic SOC randomization on every reset.

        Args:
            enable: When True, randomize SOC in ``[low, high]`` at each reset.
            low: Lower bound of randomization range.
            high: Upper bound of randomization range.
        """
        self._randomize_soc_on_reset = enable
        self._soc_init_low = low
        self._soc_init_high = high

    def reset(self, *, seed=None, options=None, day_id: Optional[int] = None):
        """Reset battery to initial state.

        Args:
            seed: RNG seed passed to the parent class.
            options: Optional dict of reset overrides:
                - ``randomize_soc`` (bool): when ``True``, draw initial SOC
                  uniformly from ``[soc_init_low, soc_init_high]``.
            day_id: Episode day index passed to the parent class.
        """
        super().reset(seed=seed, options=options, day_id=day_id)
        opts = options or {}
        if opts.get('randomize_soc', False) or self._randomize_soc_on_reset:
            self.soc = float(
                self.np_random.uniform(self._soc_init_low, self._soc_init_high)
            )
        else:
            self.soc = self.initial_soc
        self.current_p_mw = 0.0
        self.current_q_mvar = 0.0
        self._clipped_power_mw = 0.0
        self.throughput_mwh = 0.0
        self._soc_history = [self.soc]

    def _parse_action(self, action: Optional[Union[Dict[str, Any], float, np.ndarray]]) -> Tuple[float, float]:
        """Parse and denormalise an action to physical (P_MW, Q_MVAr).

        Returns:
            (desired_p_mw, desired_q_mvar).  Q is 0 when ``enable_q_control=False``.
        """
        if action is None:
            return 0.0, 0.0
        if isinstance(action, dict):
            return float(action.get('p_mw', 0.0)), float(action.get('q_mvar', 0.0))

        arr = np.atleast_1d(np.asarray(action, dtype=np.float32)).flatten()

        if self.enable_q_control and arr.shape[0] >= 2:
            raw_p, raw_q = float(arr[0]), float(arr[1])
            if self.normalize_actions:
                return raw_p * self.power_mw, raw_q * self.s_rated_mva
            return raw_p, raw_q

        raw_p = float(arr[0])
        if self.normalize_actions:
            return raw_p * self.power_mw, 0.0
        return raw_p, 0.0

    def step(self, action: Optional[Union[Dict[str, Any], float, np.ndarray]] = None) -> None:
        """Execute battery control action.

        Sign convention: positive P = discharge (inject to grid), negative = charge.
        When ``enable_q_control=True``, Q is limited by PQ circle: P²+Q² ≤ S².
        """
        desired_p, desired_q = self._parse_action(action)
        desired_p = np.clip(desired_p, -self.power_mw, self.power_mw)
        feasible_p = self._compute_feasible_power(desired_p)

        if self.enable_q_control:
            q_headroom = np.sqrt(max(self.s_rated_mva ** 2 - feasible_p ** 2, 0.0))
            feasible_q = np.clip(desired_q, -q_headroom, q_headroom)
        else:
            feasible_q = 0.0

        self._clipped_power_mw = abs(desired_p - feasible_p)
        self._update_soc(feasible_p)
        self.throughput_mwh += abs(feasible_p) * self.dt_hours
        self.current_p_mw = feasible_p
        self.current_q_mvar = feasible_q
        self.time_step += 1
        self._soc_history.append(self.soc)

    def obs(self, state: Any = None) -> dict:
        """Observation dict matching ``self.observation_space``.

        When ``enable_q_control=True`` adds ``q_mvar_norm`` (7-dim total).
        """
        p_discharge_limit, p_charge_limit = self._soc_power_limits()
        inv_power = 1.0 / self.power_mw
        p_norm = self.current_p_mw * inv_power
        p_discharge_max_norm = p_discharge_limit * inv_power
        p_charge_max_norm = p_charge_limit * inv_power
        phase = 2.0 * np.pi * self.time_step / max(self.steps_per_day, 1)

        d = {
            'soc': float(self.soc),
            'p_mw_norm': float(p_norm),
            'p_discharge_max_norm': float(p_discharge_max_norm),
            'p_charge_max_norm': float(p_charge_max_norm),
            'time_sin': float(np.sin(phase)),
            'time_cos': float(np.cos(phase)),
        }
        if self.enable_q_control:
            inv_s = 1.0 / max(self.s_rated_mva, 1e-8)
            d['q_mvar_norm'] = float(self.current_q_mvar * inv_s)
        return d

    def grid_obs(self) -> np.ndarray:
        """Grid-embedded observation (excludes time encoding).

        P-only: [soc, p_discharge_max_norm, p_charge_max_norm, p_mw_norm]  (4D)
        P+Q:    [soc, p_discharge_max_norm, p_charge_max_norm, p_mw_norm, q_mvar_norm]  (5D)
        """
        d = self.obs()
        vals = [d['soc'], d['p_discharge_max_norm'], d['p_charge_max_norm'], d['p_mw_norm']]
        if self.enable_q_control:
            vals.append(d['q_mvar_norm'])
        return np.array(vals, dtype=np.float32)

    def grid_obs_names(self, rid: str) -> list:
        names = [f'{rid}_soc', f'{rid}_p_discharge_max_norm',
                 f'{rid}_p_charge_max_norm', f'{rid}_p_mw_norm']
        if self.enable_q_control:
            names.append(f'{rid}_q_mvar_norm')
        return names

    # ====== SOC Dynamics (Internal) ======

    def _soc_power_limits(self) -> Tuple[float, float]:
        """SOC- and rated-power-capped limits at the current state.

        Returns:
            (p_discharge_limit_mw, p_charge_limit_mw): both non-negative MW.
            p_discharge_limit_mw: max grid-side injection (discharge direction).
            p_charge_limit_mw:    max grid-side absorption magnitude (charge direction).

        Both values incorporate:
          - rated power cap (``power_mw``)
          - SOC energy availability (guards against floating-point drift past soc_min / soc_max)

        Physics:
          Discharge limit: (soc − soc_min) × C × η_d / Δt  capped at power_mw
          Charge limit:    (soc_max − soc) × C / (η_c × Δt)  capped at power_mw
        """
        dt = self.dt_hours
        p_discharge = max(0.0, self.soc - self.soc_min) * self.capacity_mwh * self.eta_discharge / dt
        p_charge = max(0.0, self.soc_max - self.soc) * self.capacity_mwh / (self.eta_charge * dt)
        return min(p_discharge, self.power_mw), min(p_charge, self.power_mw)

    def _compute_feasible_power(self, desired_power: float) -> float:
        """Clip desired grid-side power to physically realisable bounds.

        Args:
            desired_power: Desired grid-side power in MW (positive = discharge).

        Returns:
            Feasible grid-side power respecting SOC and rated-power limits.
        """
        if desired_power == 0.0:
            return 0.0  # fast path: skip _soc_power_limits() call
        p_discharge_limit, p_charge_limit = self._soc_power_limits()
        if desired_power > 0:
            return min(desired_power, p_discharge_limit)
        else:
            return max(desired_power, -p_charge_limit)

    def _update_soc(self, power: float) -> None:
        """Update state of charge from a feasible grid-side power setpoint.

        Args:
            power: Feasible grid-side power in MW (positive = discharge, negative = charge).

        Physics (per time-step Δt hours):
          Discharging (power > 0):
            Battery energy drawn = power / η_d  →  SOC decreases.
          Charging (power < 0):
            Battery energy stored = |power| × η_c  →  SOC increases.
        """
        dt = self.dt_hours
        if power > 0:
            delta_soc = -(power / self.eta_discharge) * dt / self.capacity_mwh
        else:
            delta_soc = (-power * self.eta_charge) * dt / self.capacity_mwh
        self.soc = np.clip(self.soc + delta_soc, self.soc_min, self.soc_max)

    # ====== Status & Diagnostics ======

    def status(self) -> Dict[str, Any]:
        """Return current battery status.

        Includes all base fields (``current_p_mw``, ``current_q_mvar``,
        ``time_step``, ``bus_id``, ``local_v``) plus:

        ``soc``, ``soc_percent``, ``energy_stored_mwh``, ``capacity_mwh``,
        ``power_mw``, ``eta_charge``, ``eta_discharge``, ``efficiency_rt``:
            physical state and configuration parameters.

        ``p_discharge_headroom`` (MW): rated-power headroom toward discharge,
            computed as ``power_mw - max(current_p_mw, 0)``.
            This is the distance from the current grid-side setpoint to the
            rated discharge limit — **not** the SOC-constrained feasible margin.
        ``p_charge_headroom`` (MW): rated-power headroom toward charge,
            computed as ``power_mw + min(current_p_mw, 0)``.
            Same caveat: rated-power margin only, not energy-availability margin.

        ``p_max_feasible_mw`` (MW): true SOC + rated-power constrained maximum
            discharge power at the current state (same constraint as
            ``_compute_feasible_power``).  Non-negative.  Useful for diagnosing
            action-clipping in RL experiments.
        ``p_min_feasible_mw`` (MW): true SOC + rated-power constrained minimum
            power (most negative = maximum charge magnitude).  Non-positive.

        ``cost_clipped_power`` (MW, ≥ 0): absolute difference between the
            desired action and the feasible power after SOC / power-limit
            clipping.  Zero when the action was fully feasible.
            Prefixed ``cost_`` so PowerEnv automatically aggregates it into the
            CMDP safety cost channel (see ResourceEnv base class convention).

        ``throughput_mwh`` (MWh, ≥ 0): cumulative energy throughput (|P|·Δt
            summed over all steps since last reset).  Outer environments can
            use this to approximate degradation cost (e.g. multiply by a $/MWh
            wear coefficient).
        """
        p_discharge_limit, p_charge_limit = self._soc_power_limits()
        return {
            'current_p_mw': self.current_p_mw,
            'current_q_mvar': self.current_q_mvar,
            'soc': self.soc,
            'soc_percent': self.soc * 100,
            'energy_stored_mwh': self.soc * self.capacity_mwh,
            'capacity_mwh': self.capacity_mwh,
            'power_mw': self.power_mw,
            'eta_charge': self.eta_charge,
            'eta_discharge': self.eta_discharge,
            'efficiency_rt': self.eta_charge * self.eta_discharge,
            'p_discharge_headroom': float(self.power_mw - max(self.current_p_mw, 0.0)),
            'p_charge_headroom': float(self.power_mw + min(self.current_p_mw, 0.0)),
            'p_max_feasible_mw': float(p_discharge_limit),
            'p_min_feasible_mw': float(-p_charge_limit),
            'cost_clipped_power': self._clipped_power_mw,
            'throughput_mwh': self.throughput_mwh,
            'time_step': self.time_step,
            'bus_id': int(self._bus_id),
            'local_v': self._get_local_voltage(),
        }

    def econ_components(self, dt_hours: float) -> dict:
        """Cycle-degradation cost for this step.

        Returns ``{'cycle_degradation': -cycle_cost_per_mwh * throughput_mwh}``
        when ``cycle_cost_per_mwh > 0``, otherwise ``{}``.
        """
        if self.cycle_cost_per_mwh <= 0:
            return {}
        throughput = abs(self.current_p_mw) * dt_hours
        return {'cycle_degradation': -self.cycle_cost_per_mwh * throughput}

    def get_soc_history(self) -> np.ndarray:
        """Get SOC history for current episode

        Returns:
            Array of SOC values over time
        """
        return np.array(self._soc_history)

    def __repr__(self) -> str:
        return (f"BatteryEnv(capacity={self.capacity_mwh}MWh, "
                f"power={self.power_mw}MW, SOC={self.soc:.2%})")
