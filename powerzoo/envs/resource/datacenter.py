"""AI Data Center Resource

Models an AI data center as a controllable load on the grid.  RL controls
GPU resource scheduling (training / finetuning) and cooling setpoint;
power consumption is the natural outcome of those scheduling decisions.

Physical model (3 layers):
    1. IT power  — GPU-count × per-GPU power, driven by task scheduling
    2. Cooling   — COP-based, depends on IT load and outdoor temperature
    3. Thermal   — first-order zone temperature dynamics

Task types:
    - inference   : non-deferrable, follows a diurnal profile (exogenous)
    - training    : deferrable, high GPU demand, long duration
    - finetuning  : deferrable, moderate GPU demand, shorter duration

Sign convention (consistent with all ResourceEnv):
    current_p < 0  : absorbing power from the grid (load)
    current_p > 0  : injecting power to the grid  (never for a DC)

References:
    - Chen et al. (2025) — AI DC power structure & PUE
    - Latif et al. (2024, arXiv:2412.08602) — H100 node measurements
    - SustainDC (HPE) — COP / thermal model inspiration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .base import ResourceEnv

from gymnasium import spaces as _spaces


# ---------------------------------------------------------------------------
# Task data structure
# ---------------------------------------------------------------------------

@dataclass
class _Task:
    """A compute task waiting in or running on the data center."""
    arrive_step: int     # step when the task arrived
    duration: int        # total steps required to complete
    gpus: int            # number of GPUs required
    deadline: int        # step by which the task must *complete* (or be dropped)
    task_type: str       # 'training' | 'finetuning'
    remaining: Optional[int] = None  # steps left; None → initialised to duration on schedule
    gpu_eta: float = 1.0             # GPU utilisation efficiency for this task type

    def __post_init__(self):
        if self.remaining is None:
            self.remaining = self.duration


# ---------------------------------------------------------------------------
# Default task-generation configs
# ---------------------------------------------------------------------------

_DEFAULT_TRAIN_CFG: Dict[str, Any] = {
    'arrival_interval': 8,       # mean steps between arrivals (Poisson)
    'gpu_range': (50, 200),      # uniform [lo, hi]
    'duration_range': (10, 50),  # uniform [lo, hi] steps
    'deadline_slack': 2.0,       # completion deadline = arrive_step + duration * slack
    'gpu_eta': 0.90,             # GPU utilisation efficiency
}

_DEFAULT_FINETUNE_CFG: Dict[str, Any] = {
    'arrival_interval': 4,
    'gpu_range': (10, 50),
    'duration_range': (5, 20),
    'deadline_slack': 3.0,
    'gpu_eta': 0.75,
}

# Zone temperature physical bounds [°C] — shared by clip and normalisation
_T_ZONE_MIN: float = 15.0
_T_ZONE_MAX: float = 45.0


# ---------------------------------------------------------------------------
# DataCenterEnv
# ---------------------------------------------------------------------------

class DataCenterEnv(ResourceEnv):
    """AI Data Center resource (physical sub-component, not standalone RL env).

    For the full CMDP interface, use a Task which wraps this inside PowerEnv.

    RL action (3-D continuous):
        [r_train, r_finetune, cooling_setpoint]
        When ``normalize_actions=True`` (default) the action space is ``[-1, 1]``
        and each element is linearly mapped to its physical range.  When
        ``normalize_actions=False`` the action space is ``[0, 1]``.

        - r_train          : fraction of *available* GPUs allocated to training
        - r_finetune       : fraction of *available* GPUs allocated to finetuning
          (training and finetuning share the available pool proportionally, not
          sequentially — r_finetune is *not* a fraction of what remains after
          training is subtracted)
        - cooling_setpoint : normalised setpoint mapped to [t_set_min, t_set_max]

    Parameters
    ----------
    n_gpus : int
        Total GPU count in the data center (default 1000).
    gpu_idle_w : float
        Per-GPU idle power in watts (default 55, H100 measured).
    gpu_active_w : float
        Per-GPU *system-level* active power in watts (default 1100,
        approximating H100 SXM node total power divided by 8 GPUs).
        Includes CPU, NVSwitch, memory, and board losses.
    p_base_mw : float
        Baseline non-GPU IT power — networking, storage, mgmt nodes (MW).
    infer_gpu_peak : int
        Peak GPU count consumed by inference (diurnal profile amplitude).
    cop_ref : float
        Reference COP at t_ref degrees (default 5.0).
    cop_decay : float
        COP fractional decay per degree above t_ref (default 0.04).
    t_ref : float
        Reference outdoor temperature for COP (°C, default 20).
    c_thermal : float
        Thermal capacitance of the DC zone (kWh/°C, default 500).
    ua_cooling : float
        Cooling system heat-transfer coefficient (kW/°C, default 200).
    h_wall : float
        Building envelope heat-transfer coefficient (kW/°C, default 5).
    t_set_min, t_set_max : float
        Cooling setpoint range (°C, default 18–27, ASHRAE).
    t_critical : float
        Over-temperature safety threshold (°C, default 35).
    p_aux_frac : float
        Auxiliary power as fraction of IT power (default 0.05).
    infer_gpu_eta : float
        GPU utilisation factor for inference workloads (default 0.5).
    air_heat_fraction : float
        Fraction of IT heat entering the air zone (default 1.0 = pure
        air-cooled; set ~0.2 for high-density liquid-cooled clusters).
    p_cool_min_mw : float
        Minimum standby cooling power in MW (default 0.05).  Real cooling
        plants (fans, pumps, controls) never draw zero even when the zone
        is below setpoint.
    train_cfg, finetune_cfg : dict or None
        Task generation parameters (see _DEFAULT_TRAIN_CFG).
    """

    name = 'datacenter'

    # ====== Initialization ======

    def __init__(
        self,
        # IT subsystem
        n_gpus: int = 1000,
        gpu_idle_w: float = 55.0,
        gpu_active_w: float = 1100.0,   # system-level node power (CPU + NVSwitch + board)
        p_base_mw: float = 0.5,
        infer_gpu_peak: int = 400,
        infer_gpu_eta: float = 0.5,     # GPU utilisation factor for inference workloads
        p_aux_frac: float = 0.05,
        # Cooling subsystem
        cop_ref: float = 5.0,
        cop_decay: float = 0.04,
        t_ref: float = 20.0,
        ua_cooling: float = 200.0,
        air_heat_fraction: float = 1.0, # fraction of IT heat entering the air zone (1.0=air-cooled, ~0.2=DLC)
        p_cool_min_mw: float = 0.05,    # standby floor: real cooling plants never draw zero
        # Thermal zone
        c_thermal: float = 500.0,
        h_wall: float = 5.0,
        t_set_min: float = 18.0,
        t_set_max: float = 27.0,
        t_critical: float = 35.0,
        # Task generation
        train_cfg: Optional[Dict[str, Any]] = None,
        finetune_cfg: Optional[Dict[str, Any]] = None,
        # Environment plumbing
        parent: Any = None,
        bus_id: int = -1,
        normalize_actions: bool = True,
        delta_t_minutes: float = 15.0,
    ):
        super().__init__(parent=parent, bus_id=bus_id, delta_t_minutes=delta_t_minutes)
        self.normalize_actions = normalize_actions

        # IT subsystem
        self.n_gpus = int(n_gpus)
        self.gpu_idle_w = float(gpu_idle_w)
        self.gpu_active_w = float(gpu_active_w)
        self.p_base_mw = float(p_base_mw)
        self.infer_gpu_peak = int(infer_gpu_peak)
        self.infer_gpu_eta = float(infer_gpu_eta)
        self.p_aux_frac = float(p_aux_frac)

        # Cooling subsystem
        self.cop_ref = float(cop_ref)
        self.cop_decay = float(cop_decay)
        self.t_ref = float(t_ref)
        self.ua_cooling = float(ua_cooling)
        self.air_heat_fraction = float(air_heat_fraction)
        self.p_cool_min_mw = float(p_cool_min_mw)

        # Thermal zone
        self.c_thermal = float(c_thermal)
        self.h_wall = float(h_wall)
        self.t_set_min = float(t_set_min)
        self.t_set_max = float(t_set_max)
        self.t_critical = float(t_critical)

        # Task generation configs
        self.train_cfg = {**_DEFAULT_TRAIN_CFG, **(train_cfg or {})}
        self.finetune_cfg = {**_DEFAULT_FINETUNE_CFG, **(finetune_cfg or {})}

        # State (initialised in reset)
        self.t_zone: float = 22.0
        self.t_setpoint: float = 22.0
        self.t_outdoor: float = 20.0
        self.p_it_mw: float = 0.0
        self.p_cool_mw: float = 0.0
        self.p_dc_mw: float = 0.0
        self.gpus_infer: int = 0
        self.gpus_active: int = 0
        self.sla_violations: int = 0
        self.step_sla_violations: int = 0  # per-step count, used in the reward signal
        self.is_overtemp: bool = False

        # External override hooks (used by DCMicrogridEnv to inject profile values).
        # When not None, these take priority over the internal synthetic computation.
        self._override_t_outdoor: Optional[float] = None
        self._override_gpus_infer: Optional[int] = None

        self._wait_queue: List[_Task] = []
        self._running: List[_Task] = []

        # Gymnasium spaces
        if self.normalize_actions:
            self.action_space = _spaces.Box(
                low=-np.ones(3, dtype=np.float32),
                high=np.ones(3, dtype=np.float32),
                shape=(3,), dtype=np.float32,
            )
        else:
            self.action_space = _spaces.Box(
                low=np.zeros(3, dtype=np.float32),
                high=np.ones(3, dtype=np.float32),
                shape=(3,), dtype=np.float32,
            )
        self.observation_space = _spaces.Box(
            low=np.array([0, 0, 0, 0, -1, 0, 0, 0, 0, -1, -1], dtype=np.float32),
            #                        ↑ urgency       ↑↑ sin/cos ∈ [-1,1]
            high=np.ones(11, dtype=np.float32),
            shape=(11,),
            dtype=np.float32,
        )
        self.action_names: List[str] = ['r_train', 'r_finetune', 'cooling_setpoint']

        self._complete_resource_init()

    # ====== RL Interface Methods ======

    def reset(self, *, seed=None, options=None, day_id: Optional[int] = None):
        super().reset(seed=seed, options=options, day_id=day_id)
        self.t_zone = 22.0
        self.t_setpoint = 22.0
        self.t_outdoor = 20.0
        self.p_it_mw = 0.0
        self.p_cool_mw = 0.0
        self.p_dc_mw = 0.0
        self.gpus_infer = 0
        self.gpus_active = 0
        self.sla_violations = 0
        self.step_sla_violations = 0
        self.is_overtemp = False
        self.current_p_mw = 0.0
        self.current_q_mvar = 0.0
        self._wait_queue.clear()
        self._running.clear()
        # Clear override hooks on reset so stale values don't leak across episodes.
        self._override_t_outdoor = None
        self._override_gpus_infer = None

    def step(self, action=None, *, t_outdoor_override=None, gpus_infer_override=None) -> None:
        """Execute one time step.

        Args:
            action: None, ndarray(3,), or dict with keys
                    'r_train', 'r_finetune', 'cooling_setpoint'.
            t_outdoor_override: Optional [°C] outdoor temperature to inject for
                this step (formal API used by composite envs like
                ``DCMicrogridEnv`` to feed exogenous weather profiles).  Falls
                back to ``self._override_t_outdoor`` (deprecated private hook)
                or ``_get_outdoor_temp()`` if not provided.
            gpus_infer_override: Optional integer override for the inference
                GPU count this step (for injecting CPU-utilisation profiles).
                Falls back to ``self._override_gpus_infer`` or the synthetic
                diurnal curve.
        """
        r_train, r_ft, cool_norm = self._parse_action(action)

        # Map normalised cooling setpoint to temperature
        self.t_setpoint = self.t_set_min + cool_norm * (self.t_set_max - self.t_set_min)

        # Outdoor temperature: formal kwarg → deprecated private hook → synthetic
        if t_outdoor_override is not None:
            self.t_outdoor = float(t_outdoor_override)
        elif self._override_t_outdoor is not None:
            self.t_outdoor = float(self._override_t_outdoor)
        else:
            self.t_outdoor = self._get_outdoor_temp()

        # --- 1. Inference load (exogenous, diurnal: peak 14:00, trough 04:00) ---
        if gpus_infer_override is not None:
            self.gpus_infer = int(gpus_infer_override)
        elif self._override_gpus_infer is not None:
            self.gpus_infer = int(self._override_gpus_infer)
        else:
            hour = (self.time_step % self.steps_per_day) / self.steps_per_day * 24.0
            diurnal = float(np.clip(0.5 + 0.5 * np.sin(2 * np.pi * (hour - 8.0) / 24.0), 0.1, 1.0))
            self.gpus_infer = int(self.infer_gpu_peak * diurnal)

        # --- 2. New deferrable tasks arrive ---
        self._generate_arrivals()

        # --- 3. Force-schedule urgent tasks (slack <= 0) ---
        self._schedule_urgent()

        # --- 4. RL-controlled scheduling ---
        # r_train and r_ft are independent fractions of the shared available pool,
        # not sequential deductions — allocate proportionally so neither starves.
        gpus_in_flight = sum(t.gpus for t in self._running)
        total_avail = max(0, self.n_gpus - self.gpus_infer - gpus_in_flight)

        denom = max(r_train + r_ft, 1e-6)
        gpu_budget_train = int(total_avail * r_train / denom)
        gpu_budget_ft = int(total_avail * r_ft / denom)

        self._schedule_by_budget('training', gpu_budget_train)
        self._schedule_by_budget('finetuning', gpu_budget_ft)

        # --- 5. Update running pool ---
        for task in self._running:
            task.remaining -= 1
        self._running = [t for t in self._running if t.remaining > 0]

        # --- 6. Check deadline violations ---
        prev_violations = self.sla_violations
        expired = [t for t in self._wait_queue if self.time_step >= t.deadline]
        self.sla_violations += len(expired)
        self._wait_queue = [t for t in self._wait_queue if self.time_step < t.deadline]
        self.step_sla_violations = self.sla_violations - prev_violations

        # --- 7. IT power ---
        gpus_in_flight = sum(t.gpus for t in self._running)
        self.gpus_active = min(self.gpus_infer + gpus_in_flight, self.n_gpus)

        # Weighted active power: inference GPUs at infer_gpu_eta, task GPUs at their per-task eta
        gpus_infer_eff = min(self.gpus_infer, self.n_gpus)
        p_active_w = gpus_infer_eff * self.gpu_active_w * self.infer_gpu_eta
        gpus_remaining = self.n_gpus - gpus_infer_eff
        for t in self._running:
            gpus_for_task = min(t.gpus, gpus_remaining)
            per_gpu_w = self.gpu_idle_w + (self.gpu_active_w - self.gpu_idle_w) * t.gpu_eta
            p_active_w += gpus_for_task * per_gpu_w
            gpus_remaining = max(0, gpus_remaining - gpus_for_task)
        p_idle_w = max(0, self.n_gpus - self.gpus_active) * self.gpu_idle_w

        self.p_it_mw = (p_active_w + p_idle_w) / 1e6 + self.p_base_mw

        # --- 8. Cooling power ---
        # Heat removed by cooling system, driven by zone-setpoint gap
        q_cool_kw = self.ua_cooling * max(self.t_zone - self.t_setpoint, 0.0)

        # COP degrades linearly above t_ref; clip to a physical range [0.4, 1.2] × cop_ref
        cop = self.cop_ref * np.clip(
            1.0 - self.cop_decay * max(self.t_outdoor - self.t_ref, 0.0),
            0.4, 1.2,
        )
        self.p_cool_mw = max(q_cool_kw / (cop * 1e3), self.p_cool_min_mw)  # kW → MW

        self.p_dc_mw = self.p_it_mw + self.p_cool_mw + self.p_aux_frac * self.p_it_mw

        # --- 9. Thermal zone dynamics ---
        # Only the air-side fraction of IT heat enters the zone (remainder removed by liquid)
        p_it_air_kw = self.p_it_mw * 1e3 * self.air_heat_fraction
        q_wall = self.h_wall * (self.t_outdoor - self.t_zone)
        self.t_zone += self.dt_hours * (p_it_air_kw - q_cool_kw + q_wall) / max(self.c_thermal, 1e-6)
        self.t_zone = float(np.clip(self.t_zone, _T_ZONE_MIN, _T_ZONE_MAX))
        self.is_overtemp = self.t_zone > self.t_critical

        # --- 10. Grid power output ---
        self.current_p_mw = -self.p_dc_mw
        self.current_q_mvar = 0.0
        self.time_step += 1

    def status(self) -> Dict[str, Any]:
        """Return current data center status.

        ``cost_overtemp`` (°C, ≥ 0): zone temperature above critical
        threshold.  Zero when within safe limits.
        """
        overtemp = max(self.t_zone - self.t_critical, 0.0)
        return {
            'current_p_mw': self.current_p_mw,
            'current_q_mvar': self.current_q_mvar,
            'p_it_mw': self.p_it_mw,
            'p_cool_mw': self.p_cool_mw,
            'p_dc_mw': self.p_dc_mw,
            'pue': self.p_dc_mw / max(self.p_it_mw, 1e-9),
            'gpu_util': self.gpus_active / max(self.n_gpus, 1),
            'gpus_infer': self.gpus_infer,
            'gpus_active': self.gpus_active,
            'n_running': len(self._running),
            'n_queued': len(self._wait_queue),
            'queue_gpu_demand': sum(t.gpus for t in self._wait_queue),
            't_zone': self.t_zone,
            't_setpoint': self.t_setpoint,
            't_outdoor': self.t_outdoor,
            't_critical': self.t_critical,
            'is_overtemp': self.is_overtemp,
            'overtemp_violation': overtemp,    # backward compat
            'cost_overtemp': overtemp,         # CMDP cost convention
            'sla_violations': self.sla_violations,              # cumulative (for stats)
            'step_sla_violations': self.step_sla_violations,   # per-step (for reward)
            'cost_sla_violations': float(self.step_sla_violations),  # CMDP cost convention
            'time_step': self.time_step,
            'bus_id': int(self._bus_id),
            'local_v': self._get_local_voltage(),
        }

    def obs(self, state: Any = None) -> dict:
        """Observation dict (11 fields)."""
        gpu_util = self.gpus_active / max(self.n_gpus, 1)
        infer_util = self.gpus_infer / max(self.n_gpus, 1)

        train_demand = sum(t.gpus for t in self._wait_queue if t.task_type == 'training')
        ft_demand = sum(t.gpus for t in self._wait_queue if t.task_type == 'finetuning')
        q_train_fill = min(train_demand / max(self.n_gpus, 1), 1.0)
        q_ft_fill = min(ft_demand / max(self.n_gpus, 1), 1.0)

        # Urgency mirrors _schedule_urgent: slack = deadline - now - duration.
        # Negative values are kept so the agent can sense overdue severity.
        if self._wait_queue:
            slacks = [
                (t.deadline - self.time_step - t.duration) / max(t.duration, 1)
                for t in self._wait_queue
            ]
            urgency = float(np.clip(min(slacks) / 5.0, -1.0, 1.0))
        else:
            urgency = 1.0

        p_total = self.p_it_mw + self.p_cool_mw
        cool_ratio = self.p_cool_mw / max(p_total, 1e-9)

        zone_norm    = (self.t_zone - _T_ZONE_MIN) / (_T_ZONE_MAX - _T_ZONE_MIN)  # maps clip range [15, 45] → [0, 1]
        outdoor_norm = (self.t_outdoor - 10.0) / 20.0                             # maps operating range [10, 30] → [0, 1]
        setpoint_norm = ((self.t_setpoint - self.t_set_min)
                         / max(self.t_set_max - self.t_set_min, 1e-6))

        phase = 2.0 * np.pi * self.time_step / max(self.steps_per_day, 1)

        return {
            'gpu_util': float(np.clip(gpu_util, 0, 1)),
            'infer_util': float(np.clip(infer_util, 0, 1)),
            'queue_train_fill': float(np.clip(q_train_fill, 0, 1)),
            'queue_ft_fill': float(np.clip(q_ft_fill, 0, 1)),
            'queue_urgency': float(np.clip(urgency, -1, 1)),
            'cool_ratio': float(np.clip(cool_ratio, 0, 1)),
            'zone_temp_norm': float(np.clip(zone_norm, 0, 1)),
            'outdoor_temp_norm': float(np.clip(outdoor_norm, 0, 1)),
            'setpoint_norm': float(np.clip(setpoint_norm, 0, 1)),
            'time_sin': float(np.sin(phase)),
            'time_cos': float(np.cos(phase)),
        }

    # ====== Internal Dynamics ======

    def _parse_action(self, action):
        if action is None:
            return 0.5, 0.5, 0.5
        if isinstance(action, dict):
            return (
                float(action.get('r_train', 0.5)),
                float(action.get('r_finetune', 0.5)),
                float(action.get('cooling_setpoint', 0.5)),
            )
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        # Denormalize [-1, 1] → [0, 1]
        if self.normalize_actions:
            arr = (arr + 1.0) / 2.0
        r_train = float(np.clip(arr[0], 0, 1)) if len(arr) > 0 else 0.5
        r_ft = float(np.clip(arr[1], 0, 1)) if len(arr) > 1 else 0.5
        cool = float(np.clip(arr[2], 0, 1)) if len(arr) > 2 else 0.5
        return r_train, r_ft, cool

    def _get_outdoor_temp(self) -> float:
        """Get outdoor temperature from parent time-series or synthetic curve."""
        if self._parent is not None:
            ts_data = getattr(self._parent, '_time_series_data', None)
            if ts_data is not None:
                for col in ('Temperature', 'temperature', 'T_outdoor'):
                    if col in ts_data.columns:
                        idx = self.time_step % len(ts_data)
                        return float(ts_data[col].iloc[idx])
        # Synthetic: sinusoidal daily cycle, mean 20°C, amplitude 8°C, peak ~14:00.
        # Phase (hour - 8) places the peak at h=14 (afternoon), matching the
        # PowerZooJax JAX reference (powerzoojax/envs/resource/datacenter.py).
        hour = (self.time_step % self.steps_per_day) / self.steps_per_day * 24.0
        return 20.0 + 8.0 * np.sin(2 * np.pi * (hour - 8.0) / 24.0)

    def _generate_arrivals(self) -> None:
        """Generate new deferrable tasks via Poisson process."""
        for cfg, ttype in [(self.train_cfg, 'training'), (self.finetune_cfg, 'finetuning')]:
            lam = 1.0 / max(cfg['arrival_interval'], 1)
            n_arrive = self.np_random.poisson(lam)
            lo_g, hi_g = cfg['gpu_range']
            lo_d, hi_d = cfg['duration_range']
            for _ in range(n_arrive):
                gpus = int(self.np_random.integers(lo_g, hi_g + 1))
                dur = int(self.np_random.integers(lo_d, hi_d + 1))
                deadline = self.time_step + int(dur * cfg['deadline_slack'])
                self._wait_queue.append(_Task(
                    arrive_step=self.time_step,
                    duration=dur,
                    gpus=gpus,
                    deadline=deadline,
                    task_type=ttype,
                    gpu_eta=cfg['gpu_eta'],
                ))

    def _schedule_urgent(self) -> None:
        """Force-schedule tasks whose slack has run out, respecting GPU capacity."""
        gpus_used = self.gpus_infer + sum(t.gpus for t in self._running)
        urgent = []
        remaining = []
        for t in self._wait_queue:
            slack = t.deadline - self.time_step - t.duration
            if slack <= 0 and gpus_used + t.gpus <= self.n_gpus:
                urgent.append(t)
                gpus_used += t.gpus
            else:
                remaining.append(t)
        self._wait_queue = remaining
        self._running.extend(urgent)

    def _schedule_by_budget(self, task_type: str, gpu_budget: int) -> None:
        """Pick tasks from queue by EDF until GPU budget is exhausted."""
        if gpu_budget <= 0:
            return
        candidates = [t for t in self._wait_queue if t.task_type == task_type]
        # EDF: sort by slack ascending (most urgent first)
        candidates.sort(key=lambda t: t.deadline - self.time_step - t.duration)

        scheduled_ids = set()
        gpus_used = 0
        for t in candidates:
            if gpus_used + t.gpus <= gpu_budget:
                scheduled_ids.add(id(t))
                gpus_used += t.gpus
                self._running.append(t)

        if scheduled_ids:
            self._wait_queue = [t for t in self._wait_queue if id(t) not in scheduled_ids]

    def __repr__(self) -> str:
        return (
            f"DataCenterEnv(gpus={self.n_gpus}, "
            f"active={self.gpus_active}, "
            f"queued={len(self._wait_queue)}, "
            f"P_DC={self.p_dc_mw:.2f}MW)"
        )
