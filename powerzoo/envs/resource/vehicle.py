"""Electric Vehicle (EV) resource environment."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .base import ResourceEnv

from gymnasium import spaces as _spaces


@dataclass(frozen=True)
class CommuteTrip:
    """One commute leg in a daily schedule."""
    departure_hour: float
    arrival_hour: float
    energy_mwh: float


@dataclass(frozen=True)
class PowerLimits:
    """SOC- and availability-constrained power bounds for the current step."""
    charge_max_mw: float
    discharge_max_mw: float


# Observation keys in obs() insertion order; must match observation_space bounds.
OBS_NAMES = (
    'soc',
    'p_mw_norm',
    'is_home',
    'departure_ready',
    'time_to_departure_norm',
    'time_to_arrival_norm',
    'time_sin',
    'time_cos',
    'soc_departure_min',
)


class VehicleEnv(ResourceEnv):
    """Electric Vehicle resource (physical sub-component, not standalone RL env).

    For the full CMDP interface, use a Task which wraps this inside PowerEnv.

    Models an EV with multiple commute trips, charging when at home, and
    departure SOC requirements.  Positive ``current_p`` means V2G
    (vehicle-to-grid); negative means G2V (grid-to-vehicle).

    Supports flexible commute patterns in a day:
    - Simple: home -> work -> home
    - Complex: home -> work -> lunch -> work -> errands -> home
    """
    name = 'vehicle'

    # ====== Initialization ======

    def __init__(
            self,
            parent: Any = None,
            bus_id: int = -1,
            E_max_kWh: float = 60.0,
            soc_init: float = 0.8,
            soc_min: float = 0.1,
            soc_max: float = 0.95,
            soc_departure_min: float = 0.8,
            p_charge_max_kW: float = 7.0,
            p_discharge_max_kW: float = 7.0,
            eta_charge: float = 0.95,
            eta_discharge: float = 0.95,
            commute_schedule: List[Dict[str, float]] = None,
            delta_t_minutes: float = 15.0,
            normalize_actions: bool = True,
    ):
        """Initialize electric vehicle environment.

        Args:
            parent: Parent grid or hub.
            bus_id: Bus ID where EV is connected.
            E_max_kWh: Battery capacity (kWh), typical: 40–80 kWh.
            soc_init: Initial SOC [0, 1].
            soc_min: Minimum SOC [0, 1], typical: 0.1–0.2.
            soc_max: Maximum SOC [0, 1], typical: 0.9–0.95.
            soc_departure_min: Minimum SOC required at departure.
            p_charge_max_kW: Maximum charging power (kW), typical: 3.3–7.7.
            p_discharge_max_kW: Maximum V2G discharge power (kW).
            eta_charge: Charging efficiency (0, 1].
            eta_discharge: Discharging efficiency (0, 1].
            commute_schedule: List of daily trips, each dict with keys:
                - 'departure': hour [0, 24)
                - 'arrival':   hour [0, 24)
                - 'energy_kWh': energy consumed (kWh)
                Defaults to a single trip (8 am–6 pm, 15 kWh).
            delta_t_minutes: Time step duration (minutes).
            normalize_actions: If True, action space is normalised to [-1, 1].
        """
        super().__init__(parent=parent, bus_id=bus_id, delta_t_minutes=delta_t_minutes)

        # --- Battery specifications ---
        self.capacity_mwh = float(E_max_kWh) / 1000.0
        self.soc_init = float(soc_init)
        self.soc = float(soc_init)
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.soc_departure_min = float(soc_departure_min)

        # --- Power limits (kW → MW) ---
        self.p_charge_max_mw = float(p_charge_max_kW) / 1000.0
        self.p_discharge_max_mw = float(p_discharge_max_kW) / 1000.0

        # --- Efficiencies ---
        self.eta_charge = float(eta_charge)
        self.eta_discharge = float(eta_discharge)

        # --- Parameter invariant checks ---
        if self.capacity_mwh <= 0:
            raise ValueError(f"E_max_kWh must be > 0, got {E_max_kWh}")
        if not (0.0 <= self.soc_min <= self.soc_init <= self.soc_max <= 1.0):
            raise ValueError(
                f"SOC parameters must satisfy 0 ≤ soc_min ≤ soc_init ≤ soc_max ≤ 1, "
                f"got soc_min={soc_min}, soc_init={soc_init}, soc_max={soc_max}"
            )
        if not (0.0 < self.eta_charge <= 1.0):
            raise ValueError(f"eta_charge must be in (0, 1], got {eta_charge}")
        if not (0.0 < self.eta_discharge <= 1.0):
            raise ValueError(f"eta_discharge must be in (0, 1], got {eta_discharge}")
        if self.p_charge_max_mw <= 0:
            raise ValueError(f"p_charge_max_kW must be > 0, got {p_charge_max_kW}")
        if self.p_discharge_max_mw <= 0:
            raise ValueError(f"p_discharge_max_kW must be > 0, got {p_discharge_max_kW}")

        # --- Commute schedule (normalised into CommuteTrip objects) ---
        if commute_schedule is None:
            commute_schedule = [{'departure': 8.0, 'arrival': 18.0, 'energy_kWh': 15.0}]

        self._trips: List[CommuteTrip] = sorted(
            [
                CommuteTrip(
                    departure_hour=float(t['departure']),
                    arrival_hour=float(t['arrival']),
                    energy_mwh=float(t['energy_kWh']) / 1000.0,
                )
                for t in commute_schedule
            ],
            key=lambda trip: trip.departure_hour,
        )

        # Validate trip durations against step size.
        for trip in self._trips:
            duration = (trip.arrival_hour - trip.departure_hour) % 24.0
            if duration < self.dt_hours:
                raise ValueError(
                    f"Trip duration {duration:.2f}h "
                    f"(depart {trip.departure_hour:.1f} → arrive {trip.arrival_hour:.1f}) "
                    f"is shorter than step size {self.dt_hours:.2f}h."
                )

        # --- State tracking ---
        self.is_home: bool = True
        self.time_of_day: float = 0.0
        self._clipped_power_mw: float = 0.0
        self.unmet_energy_mwh: float = 0.0
        self._schedule_cursor: int = 0

        self.normalize_actions = normalize_actions
        self._action_phys_low = np.array([-self.p_charge_max_mw], dtype=np.float32)
        self._action_phys_high = np.array([self.p_discharge_max_mw], dtype=np.float32)

        if self.normalize_actions:
            self.action_space = _spaces.Box(
                low=-np.ones(1, dtype=np.float32),
                high=np.ones(1, dtype=np.float32),
                shape=(1,), dtype=np.float32,
            )
        else:
            self.action_space = _spaces.Box(
                low=self._action_phys_low,
                high=self._action_phys_high,
                shape=(1,), dtype=np.float32,
            )

        # Bounds ordered by OBS_NAMES insertion order.
        self.observation_space = _spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0,  1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0, 1.0], dtype=np.float32),
            shape=(len(OBS_NAMES),),
            dtype=np.float32,
        )
        self.action_names: List[str] = ['p_mw']

        self._complete_resource_init()

    # ====== RL Interface Methods ======

    def reset(self, *, seed=None, options=None, day_id: Optional[int] = None):
        """Reset vehicle to initial state."""
        super().reset(seed=seed, options=options, day_id=day_id)
        self.soc = self.soc_init
        self.time_of_day = (self.time_step * self.dt_hours) % 24.0
        self._clipped_power_mw = 0.0
        self.unmet_energy_mwh = 0.0
        self._sync_schedule_state_to_time_of_day()
        return self.obs()

    def step(self, action: Optional[Union[Dict[str, Any], float, np.ndarray]] = None) -> None:
        """Apply charging/discharging action and update SOC and availability.

        **Execution order**: state transitions (arrival / departure) are
        resolved *before* the charging action is applied.  This means a
        vehicle arriving at time *T* can begin charging in the same step.

        Action can be:
        - ``None``                 → idle (0 MW)
        - ``dict`` with ``'p_mw'`` → physical MW; positive = discharge (V2G),
          negative = charge (G2V).  Always physical regardless of
          ``normalize_actions``.
        - ``float`` / ``ndarray``  → when ``normalize_actions=True`` (default),
          treated as a normalised value and de-normalised via
          ``grid_action_from_normalized()``; when ``False``, treated as MW.
        """
        desired_p_mw = self._parse_action(action)
        self.unmet_energy_mwh = 0.0

        # 1. Resolve commute events (arrival / departure) before applying power.
        #    A vehicle arriving this step can charge in the same step.
        self._update_availability()

        # 2. Determine feasible power (away → both limits are 0 from available_power).
        power_limits = self.available_power()
        feasible_p_mw = float(np.clip(
            desired_p_mw,
            -power_limits.charge_max_mw,
            power_limits.discharge_max_mw,
        ))
        self._clipped_power_mw = abs(desired_p_mw - feasible_p_mw)

        # 3. SOC update — single branch, no home/away split needed.
        if feasible_p_mw > 0.0:
            battery_delta_mwh = -(feasible_p_mw / self.eta_discharge) * self.dt_hours
        else:
            battery_delta_mwh = (-feasible_p_mw * self.eta_charge) * self.dt_hours

        self.soc = float(np.clip(
            self.soc + battery_delta_mwh / self.capacity_mwh,
            self.soc_min,
            self.soc_max,
        ))
        self.current_p_mw = feasible_p_mw
        self.current_q_mvar = 0.0

        # 4. Advance clock.
        self.time_of_day = (self.time_of_day + self.dt_hours) % 24.0
        self.time_step += 1

    def obs(self, state: Any = None) -> dict:
        """Return observation dict matching ``self.observation_space`` (9-dim).

        Keys are defined by ``OBS_NAMES``.
        """
        power_scale = max(self.p_charge_max_mw, self.p_discharge_max_mw, 1e-6)
        phase = 2.0 * np.pi * self.time_of_day / 24.0
        return {
            'soc': float(self.soc),
            'p_mw_norm': float(np.clip(self.current_p_mw / power_scale, -1.0, 1.0)),
            'is_home': 1.0 if self.is_home else 0.0,
            'departure_ready': 1.0 if self.check_departure_ready() else 0.0,
            'time_to_departure_norm': min(self._time_to_next_departure() / 24.0, 1.0),
            'time_to_arrival_norm': min(self._time_to_next_arrival() / 24.0, 1.0),
            'time_sin': float(np.sin(phase)),
            'time_cos': float(np.cos(phase)),
            'soc_departure_min': float(self.soc_departure_min),
        }

    # ====== Vehicle-Specific Methods ======

    def available_power(self) -> PowerLimits:
        """Return available charge/discharge power considering SOC and availability."""
        if not self.is_home:
            return PowerLimits(charge_max_mw=0.0, discharge_max_mw=0.0)

        energy_from_cell_max = max(0.0, (self.soc - self.soc_min) * self.capacity_mwh)
        p_discharge_soc = energy_from_cell_max * self.eta_discharge / self.dt_hours

        energy_to_cell_max = max(0.0, (self.soc_max - self.soc) * self.capacity_mwh)
        p_charge_soc = energy_to_cell_max / self.eta_charge / self.dt_hours

        return PowerLimits(
            charge_max_mw=min(self.p_charge_max_mw, p_charge_soc),
            discharge_max_mw=min(self.p_discharge_max_mw, p_discharge_soc),
        )

    @property
    def commute_schedule(self) -> List[Dict[str, float]]:
        """Backward-compatible view of ``_trips`` as a list of dicts.

        Keys: ``'departure'``, ``'arrival'``, ``'energy'`` (MWh).
        """
        return [
            {
                'departure': trip.departure_hour,
                'arrival': trip.arrival_hour,
                'energy': trip.energy_mwh,
            }
            for trip in self._trips
        ]

    def _parse_action(self, action: Optional[Union[Dict[str, Any], float, np.ndarray]]) -> float:
        if action is None:
            return 0.0
        is_dict = isinstance(action, dict)
        if is_dict:
            raw = float(action.get("p_mw", 0.0))
        elif isinstance(action, np.ndarray):
            raw = float(action.flat[0])
        else:
            raw = float(action)

        if self.normalize_actions and not is_dict:
            # Piecewise linear: RL action 0 → physical 0 (idle).
            # [0, 1] → [0, p_discharge_max],  [-1, 0] → [-p_charge_max, 0]
            if raw >= 0:
                raw = float(raw * self._action_phys_high[0])
            else:
                raw = float(raw * (-self._action_phys_low[0]))
        return raw

    def _update_availability(self) -> None:
        """Resolve commute events at the current time step.

        Uses ``_schedule_cursor`` to track progress through the daily schedule,
        which correctly handles back-to-back trips where one trip's arrival
        coincides with the next trip's departure.
        """
        n = len(self._trips)
        if n == 0:
            return

        idx = self._schedule_cursor % n

        if self.is_home:
            if self._triggers_this_step(self._trips[idx].departure_hour):
                self._depart(self._trips[idx].energy_mwh)
        else:
            if self._triggers_this_step(self._trips[idx].arrival_hour):
                self.is_home = True
                self._schedule_cursor = (self._schedule_cursor + 1) % n
                next_trip = self._trips[self._schedule_cursor]
                if self._triggers_this_step(next_trip.departure_hour):
                    self._depart(next_trip.energy_mwh)

    def _sync_schedule_state_to_time_of_day(self) -> None:
        """Align commute state to the current intra-day reset offset."""
        n = len(self._trips)
        if n == 0:
            self.is_home = True
            self._schedule_cursor = 0
            return

        current_time = self.time_of_day % 24.0
        for idx, trip in enumerate(self._trips):
            if self._in_trip_interval(current_time, trip.departure_hour, trip.arrival_hour):
                self.is_home = False
                self._schedule_cursor = idx
                return

        self.is_home = True
        self._schedule_cursor = min(
            range(n),
            key=lambda i: (self._trips[i].departure_hour - current_time) % 24.0,
        )

    @staticmethod
    def _in_trip_interval(current_time: float, start: float, end: float) -> bool:
        """Return whether ``current_time`` lies in the cyclic trip interval ``[start, end)``."""
        if start <= end:
            return start <= current_time < end
        return current_time >= start or current_time < end

    def _triggers_this_step(self, event_hour: float) -> bool:
        """Return whether a scheduled event at ``event_hour`` fires during the current step.

        Checks if ``event_hour`` falls in ``[time_of_day, time_of_day + dt_hours)`` mod 24.
        """
        t_start = self.time_of_day
        t_end = (self.time_of_day + self.dt_hours) % 24.0
        if t_start < t_end:
            return t_start <= event_hour < t_end
        return event_hour >= t_start or event_hour < t_end

    def _depart(self, energy_mwh: float) -> None:
        """Handle vehicle departure.

        Deducts commute energy from SOC.  If SOC would drop below ``soc_min``,
        the deficit is recorded in ``unmet_energy_mwh`` (exposed as
        ``cost_unmet_energy`` in ``status()``).
        """
        self.is_home = False
        new_soc = self.soc - energy_mwh / self.capacity_mwh

        if new_soc < self.soc_min:
            self.unmet_energy_mwh = (self.soc_min - new_soc) * self.capacity_mwh
            self.soc = self.soc_min
        else:
            self.unmet_energy_mwh = 0.0
            self.soc = new_soc

        self.current_p_mw = 0.0

    def _time_to_next_departure(self) -> float:
        """Return hours until the next scheduled departure."""
        if not self._trips:
            return 24.0
        return min((trip.departure_hour - self.time_of_day) % 24.0 for trip in self._trips)

    def _time_to_next_arrival(self) -> float:
        """Return hours until the vehicle arrives home.

        When at home, returns 0.0.  When away, returns time to the current
        trip's scheduled arrival.
        """
        if self.is_home or not self._trips:
            return 0.0
        arrival = self._trips[self._schedule_cursor % len(self._trips)].arrival_hour
        return (arrival - self.time_of_day) % 24.0

    def check_departure_ready(self) -> bool:
        """Check if vehicle meets departure SOC requirement."""
        return self.soc >= self.soc_departure_min

    def status(self) -> Dict[str, Any]:
        """Return current vehicle status.

        Base fields (``current_p_mw``, ``current_q_mvar``, ``time_step``,
        ``bus_id``, ``local_v``) are provided by the base class.

        Additional fields:

        ``soc``, ``capacity_mwh``, ``soc_min``, ``soc_max``,
        ``soc_departure_min``, ``is_home``, ``time_of_day``,
        ``departure_ready``: physical state and configuration.

        ``time_to_departure`` (h): hours until next scheduled departure.

        ``time_to_arrival`` (h): hours until home arrival (0 when at home).

        ``cost_clipped_power`` (MW, ≥ 0): |desired − feasible| power.

        ``cost_unmet_energy`` (MWh, ≥ 0): energy shortfall at departure.
        """
        base = super().status()
        base.update({
            "soc": self.soc,
            "capacity_mwh": self.capacity_mwh,
            "is_home": self.is_home,
            "time_of_day": self.time_of_day,
            "soc_min": self.soc_min,
            "soc_max": self.soc_max,
            "soc_departure_min": self.soc_departure_min,
            "departure_ready": self.check_departure_ready(),
            "time_to_departure": float(self._time_to_next_departure()),
            "time_to_arrival": float(self._time_to_next_arrival()),
            "cost_clipped_power": self._clipped_power_mw,
            "cost_unmet_energy": self.unmet_energy_mwh,
        })
        return base
