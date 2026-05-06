"""DC Microgrid Environment — self-contained behind-the-meter microgrid (Python).

Mirrors ``powerzoojax.envs.resource.DataCenterMicrogridEnv``.  Physics for
each device lives in its own ``ResourceEnv`` subclass and the microgrid env
composes them as sub-resources — paralleling how the JAX twin attaches
``BatteryBundle`` / ``SolarBundle`` / ``DieselBundle`` via the ResourceBundle
protocol.

Topology
--------
No external grid connection.  Power balance is explicit::

    p_supply       = p_pv + p_dg + p_batt
    residual       = p_supply - p_load
    power_deficit  = max(-residual, 0)   → cost_power_deficit
    power_spill    = max(+residual, 0)   → info only

Sub-resources (all ``ResourceEnv`` subclasses)::

    self._dc    DataCenterEnv     (existing — IT + cooling + thermal)
    self._batt  BatteryEnv        (existing — SOC dynamics)
    self._pv    SolarEnv          (existing — fed via cf_array; no curtail)
    self._dg    DieselResource    (NEW — dispatchable diesel)

Action (5-D Box) — unchanged from the previous public API::

    action[0] : train_sched_rate        ∈ [0, 1]
    action[1] : ft_sched_rate           ∈ [0, 1]
    action[2] : cooling_setpoint_norm   ∈ [0, 1]
    action[3] : battery_power_norm      ∈ [-1, 1]  positive = discharge
    action[4] : dg_power_norm           ∈ [0, 1]

Observation (18-D Box) — layout preserved exactly.

Reward / Cost / units — numerically aligned with the JAX twin::

    r_energy  = -(p_load * dt_h)
    r_cost    = -(fuel_cost + |p_batt|*dt_h*battery_deg_cost_per_mwh)
    r_carbon  = -carbon_kg
    reward    = r_energy + w_cost*r_cost + w_carbon*r_carbon
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from powerzoo.envs.base import BaseEnv
from powerzoo.envs.resource.datacenter import DataCenterEnv
from powerzoo.envs.resource.battery import BatteryEnv
from powerzoo.envs.resource.diesel import DieselResource
from powerzoo.envs.resource.renewable import SolarEnv


_T_ZONE_MIN: float = 15.0
_T_ZONE_MAX: float = 45.0


# Synthetic profile helpers (kept as module-level functions for backward
# compat with tests that import them directly).
def _make_synthetic_solar(n_steps: int = 288) -> np.ndarray:
    """Synthetic solar CF curve, peak at noon.  Matches JAX ``synthetic_solar_cf``."""
    t = np.arange(n_steps, dtype=np.float32)
    hour = t / n_steps * 24.0
    cf = np.where(
        (hour >= 6.0) & (hour <= 18.0),
        np.sin(np.pi * (hour - 6.0) / 12.0),
        0.0,
    )
    return np.clip(cf, 0.0, 1.0).astype(np.float32)


def _make_synthetic_temp(n_steps: int = 288) -> np.ndarray:
    """Synthetic outdoor temperature curve [°C], peak at ~14:00."""
    t = np.arange(n_steps, dtype=np.float32)
    hour = t / n_steps * 24.0
    return (20.0 + 8.0 * np.sin(2.0 * np.pi * (hour - 8.0) / 24.0)).astype(np.float32)


class DCMicrogridEnv(BaseEnv):
    """Self-contained behind-the-meter data-center microgrid environment.

    A standalone Gymnasium-compatible environment that holds four
    ``ResourceEnv`` sub-components and drives them together each step.

    Args:
        n_gpus, gpu_idle_w, ...:
            DataCenter parameters (forwarded to ``DataCenterEnv``).
        battery_capacity_mwh, battery_power_mw, ...:
            Battery parameters (forwarded to ``BatteryEnv``).
        battery_deg_cost_per_mwh:
            Degradation cost [$/MWh of throughput].  Applied at the env-level
            reward layer (NOT inside BatteryEnv) to mirror the JAX twin.
        pv_capacity_mw:
            PV nameplate capacity [MW].
        dg_max_mw, dg_fuel_cost_per_mwh, dg_emission_factor:
            Diesel generator parameters.
        w_cost, w_carbon:
            Reward scalarisation weights.
        max_steps:
            Episode length in steps (default 288 = 24 h × 5 min).
        delta_t_minutes:
            Step duration (default 5 min).
        cpu_profile, solar_profile, outdoor_temp_profile:
            Optional external profiles (1-D float32 ndarrays, cyclically
            indexed at each step).  ``solar_profile=None`` → synthetic;
            ``cpu_profile=None`` → DataCenterEnv internal diurnal curve;
            ``outdoor_temp_profile=None`` → synthetic.
        dg_derating_factor:
            Multiplier applied to ``dg_max_mw`` for OOD scenarios.
        sla_tighten_factor:
            Multiplier applied to DC deadline slack for OOD scenarios.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        # Datacenter
        n_gpus: int = 1000,
        gpu_idle_w: float = 55.0,
        gpu_active_w: float = 580.0,
        p_base_mw: float = 0.4,
        infer_gpu_peak: int = 400,
        cop_ref: float = 5.0,
        cop_decay: float = 0.04,
        t_ref: float = 20.0,
        c_thermal: float = 500.0,
        ua_cooling: float = 200.0,
        h_wall: float = 5.0,
        t_set_min: float = 18.0,
        t_set_max: float = 27.0,
        t_critical: float = 35.0,
        p_aux_frac: float = 0.05,
        train_cfg: Optional[Dict[str, Any]] = None,
        finetune_cfg: Optional[Dict[str, Any]] = None,
        # Battery
        battery_capacity_mwh: float = 2.0,
        battery_power_mw: float = 0.5,
        battery_eta_charge: float = 0.95,
        battery_eta_discharge: float = 0.95,
        battery_soc_min: float = 0.1,
        battery_soc_max: float = 0.9,
        battery_soc_init: float = 0.5,
        battery_deg_cost_per_mwh: float = 5.0,
        # PV
        pv_capacity_mw: float = 0.4,
        # Diesel generator
        dg_max_mw: float = 0.6,
        dg_fuel_cost_per_mwh: float = 300.0,
        dg_emission_factor: float = 0.80,
        dg_p_min_norm: float = 0.0,
        # Reward weights
        w_cost: float = 0.5,
        w_carbon: float = 0.3,
        # Episode config
        max_steps: int = 288,
        delta_t_minutes: float = 5.0,
        # External profiles (None → synthetic)
        cpu_profile: Optional[np.ndarray] = None,
        solar_profile: Optional[np.ndarray] = None,
        outdoor_temp_profile: Optional[np.ndarray] = None,
        # OOD parameter overrides
        dg_derating_factor: float = 1.0,
        sla_tighten_factor: float = 1.0,
    ):
        super().__init__(delta_t_minutes=delta_t_minutes)
        self.max_steps = int(max_steps)
        self.delta_t_minutes = float(delta_t_minutes)
        self._dt_h = delta_t_minutes / 60.0
        self._steps_per_day = 288

        self._w_cost = float(w_cost)
        self._w_carbon = float(w_carbon)
        self._battery_deg_cost_per_mwh = float(battery_deg_cost_per_mwh)
        self._sla_tighten_factor = float(sla_tighten_factor)

        # Build DC sub-component.  We mirror the JAX reference's lack of a
        # cooling-power floor (p_cool_min_mw=0.0) so per-step p_load matches.
        dc_train_cfg = dict(train_cfg or {})
        dc_ft_cfg = dict(finetune_cfg or {})
        if self._sla_tighten_factor != 1.0:
            dc_train_cfg.setdefault('deadline_slack', 2.0)
            dc_ft_cfg.setdefault('deadline_slack', 3.0)
            dc_train_cfg['deadline_slack'] *= self._sla_tighten_factor
            dc_ft_cfg['deadline_slack']    *= self._sla_tighten_factor

        self._dc = DataCenterEnv(
            n_gpus=n_gpus,
            gpu_idle_w=gpu_idle_w,
            gpu_active_w=gpu_active_w,
            p_base_mw=p_base_mw,
            infer_gpu_peak=infer_gpu_peak,
            cop_ref=cop_ref,
            cop_decay=cop_decay,
            t_ref=t_ref,
            c_thermal=c_thermal,
            ua_cooling=ua_cooling,
            h_wall=h_wall,
            t_set_min=t_set_min,
            t_set_max=t_set_max,
            t_critical=t_critical,
            p_aux_frac=p_aux_frac,
            p_cool_min_mw=0.0,
            train_cfg=dc_train_cfg if dc_train_cfg else None,
            finetune_cfg=dc_ft_cfg if dc_ft_cfg else None,
            normalize_actions=False,
            delta_t_minutes=delta_t_minutes,
        )

        # Battery sub-component.  Battery degradation cost is computed once at
        # the env-level reward layer (see ``r_cost`` in ``step()``) — do NOT
        # pass cycle_cost_per_mwh here, otherwise it would double-count.
        self._batt = BatteryEnv(
            capacity_mwh=battery_capacity_mwh,
            power_mw=battery_power_mw,
            eta_charge=battery_eta_charge,
            eta_discharge=battery_eta_discharge,
            soc_min=battery_soc_min,
            soc_max=battery_soc_max,
            initial_soc=battery_soc_init,
            normalize_actions=False,
            delta_t_minutes=delta_t_minutes,
        )

        # PV sub-component: existing ``SolarEnv`` (subclass of RenewableEnv)
        # fed through its third data path — ``cf_array=`` — which bypasses
        # the parent / loader-based attach loading and accepts the synthetic
        # or externally-supplied capacity-factor profile directly.  Mirrors
        # JAX side ``RenewableBundle(profiles=...)``.  Action is left at
        # default (None per step) so PV runs at MPPT (no curtailment).
        effective_solar = (
            np.asarray(solar_profile, dtype=np.float32)
            if solar_profile is not None
            else _make_synthetic_solar(self._steps_per_day)
        )
        # Handle the "PV disabled" benchmark scenario (``pv_capacity_mw=0``)
        # without violating ``RenewableEnv``'s ``capacity_mw > 0`` invariant:
        # use a tiny positive capacity that yields p_pv ≈ 0 down-stream.
        # ``effective_pv_capacity`` is also stored on the env for diagnostics.
        self._pv_capacity_mw_user = float(pv_capacity_mw)
        self._pv_capacity_mw = float(pv_capacity_mw)
        effective_pv_capacity = (
            float(pv_capacity_mw) if float(pv_capacity_mw) > 0.0 else 1e-9
        )
        self._pv = SolarEnv(
            capacity_mw=effective_pv_capacity,
            cf_array=effective_solar,
            normalize_actions=False,
            delta_t_minutes=delta_t_minutes,
        )

        # Diesel sub-component.
        self._dg = DieselResource(
            p_dg_max_mw=float(dg_max_mw) * float(dg_derating_factor),
            fuel_cost_per_mwh=dg_fuel_cost_per_mwh,
            emission_factor=dg_emission_factor,
            p_min_norm=dg_p_min_norm,
            normalize_actions=True,
            delta_t_minutes=delta_t_minutes,
        )
        self._dg_max_mw = float(dg_max_mw) * float(dg_derating_factor)

        # Sub-resource registry (Python parallel of JAX ``params.resources``).
        # Keeps the OO "grid step calls sub-resource step" pattern explicit.
        self.sub_resources: Dict[str, Any] = {
            'datacenter': self._dc,
            'battery':    self._batt,
            'pv':         self._pv,
            'dg':         self._dg,
        }

        # Profiles
        self._cpu_profile = (
            np.asarray(cpu_profile, dtype=np.float32)
            if cpu_profile is not None else None
        )
        self._temp_profile = (
            np.asarray(outdoor_temp_profile, dtype=np.float32)
            if outdoor_temp_profile is not None
            else _make_synthetic_temp(self._steps_per_day)
        )

        # Episode state
        self._step_count: int = 0
        self._last_action: np.ndarray = np.zeros(5, dtype=np.float32)

        # Gymnasium spaces (preserved exactly)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0,  1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0, 0, 0, -1,
                 0, 0, 0,
                 0, 0, 0,
                 0, 0, 0, -1, 0,
                 -1, -1],
                dtype=np.float32,
            ),
            high=np.ones(18, dtype=np.float32),
            dtype=np.float32,
        )

        self.reward_range = (-np.inf, 0.0)
        self.spec = None

    def constraint_names(self) -> Tuple[str, ...]:
        """Return the fixed benchmark cost-channel order."""
        return ('sla', 'overtemp', 'power_deficit')

    # ------------------------------------------------------------------
    # Profile injection API
    # ------------------------------------------------------------------

    def set_profiles(
        self,
        cpu: Optional[np.ndarray] = None,
        solar: Optional[np.ndarray] = None,
        temp: Optional[np.ndarray] = None,
    ) -> None:
        """Inject external profiles.  Call before ``reset()``.

        ``cpu``: workload (CPU/GPU utilisation) — ``None`` keeps current.
        ``solar``: PV CF profile — forwarded to ``SolarResource.set_profile``.
        ``temp``: outdoor temperature [°C] — ``None`` keeps current.
        """
        if cpu is not None:
            self._cpu_profile = np.asarray(cpu, dtype=np.float32)
        if solar is not None:
            self._pv.set_cf_array(np.asarray(solar, dtype=np.float32))
        if temp is not None:
            self._temp_profile = np.asarray(temp, dtype=np.float32)

    # ------------------------------------------------------------------
    # Profile accessors (current step, PV phase aligned: use t NOT t+1)
    # ------------------------------------------------------------------

    def _outdoor_temp(self, t: int) -> float:
        return float(self._temp_profile[t % len(self._temp_profile)])

    def _cpu_cf(self, t: int) -> float:
        if self._cpu_profile is None:
            raise RuntimeError(
                "_cpu_cf called without a cpu_profile; this should never happen "
                "because step() guards on `cpu_profile is not None`."
            )
        return float(np.clip(self._cpu_profile[t % len(self._cpu_profile)], 0.0, 1.0))

    def _solar_cf(self, t: int) -> float:
        """Backward-compatible solar profile accessor used by tests and analysis."""
        return float(self._pv._cf_at(t))

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        # Reset every sub-resource (Python parallel of JAX bundle reset loop).
        for resource in self.sub_resources.values():
            resource.reset(seed=seed)
        self._step_count = 0
        self.time_step = 0
        self._last_action = np.zeros(5, dtype=np.float32)

        obs = self._get_obs()
        info = {'step': 0, 'reward_vector': [0.0, 0.0, 0.0]}
        return obs, self.attach_constraint_costs(info)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one 5-min microgrid step.

        Args:
            action: float32 array of shape (5,).
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action_clipped = np.clip(action, self.action_space.low, self.action_space.high)

        t = self._step_count   # step index BEFORE increment

        # Profile values for the CURRENT step
        t_outdoor = self._outdoor_temp(t)
        gpus_infer_override = (
            int(self._dc.infer_gpu_peak * self._cpu_cf(t))
            if self._cpu_profile is not None else None
        )

        # ---------- Step DataCenter sub-resource ----------
        # Use the formal profile-injection API on DataCenterEnv (no private
        # attribute hacks): t_outdoor_override + gpus_infer_override.
        dc_action = np.array([
            float(action_clipped[0]),
            float(action_clipped[1]),
            float(action_clipped[2]),
        ], dtype=np.float32)
        self._dc.step(
            dc_action,
            t_outdoor_override=t_outdoor,
            gpus_infer_override=gpus_infer_override,
        )

        dc_status = self._dc.status()
        p_load = abs(dc_status['p_dc_mw'])

        # ---------- Step PV sub-resource (zero action) ----------
        self._pv.step()
        p_pv = float(self._pv.current_p_mw)
        solar_cf = float(self._pv.available_cf)

        # ---------- Step Battery sub-resource ----------
        # action[3] ∈ [-1, 1]: positive = discharge; map to physical MW
        batt_desired_mw = float(action_clipped[3]) * self._batt.power_mw
        self._batt.step(batt_desired_mw)
        batt_status = self._batt.status()
        p_batt = float(batt_status['current_p_mw'])

        # ---------- Step Diesel sub-resource ----------
        self._dg.step([float(action_clipped[4])])
        dg_status = self._dg.status()
        p_dg = float(dg_status['current_p_mw'])
        fuel_cost = float(dg_status['fuel_cost_step'])
        carbon_kg = float(dg_status['carbon_kg_step'])

        # ---------- Explicit power balance ----------
        residual      = p_pv + p_dg + p_batt - p_load
        power_deficit = max(-residual, 0.0)
        power_spill   = max(residual, 0.0)

        # ---------- Reward (scalarised multi-objective) ----------
        dt_h = self._dt_h
        battery_deg = self._battery_deg_cost_per_mwh * abs(p_batt) * dt_h

        r_energy = -(p_load * dt_h)
        r_cost   = -(fuel_cost + battery_deg)
        r_carbon = -carbon_kg
        reward   = r_energy + self._w_cost * r_cost + self._w_carbon * r_carbon

        # ---------- CMDP cost channels ----------
        n_expired = float(dc_status.get('step_sla_violations', 0))
        cost_sla  = n_expired / max(self._dc.n_gpus, 1)
        overtemp_excess = max(float(dc_status.get('t_zone', 22.0)) - self._dc.t_critical, 0.0)
        cost_overtemp   = overtemp_excess / max(_T_ZONE_MAX - self._dc.t_critical, 1e-6)
        cost_power_deficit = power_deficit / max(p_load, 1e-6)
        total_cost = cost_sla + cost_overtemp + cost_power_deficit

        # ---------- Advance step counter ----------
        self._step_count += 1
        self.time_step = self._step_count
        self._last_action = action_clipped.copy()

        terminated = False
        truncated  = (self._step_count >= self.max_steps)

        obs  = self._get_obs()
        info = self._build_info(
            t=t,
            p_load=p_load, p_pv=p_pv, p_batt=p_batt, p_dg=p_dg,
            power_deficit=power_deficit, power_spill=power_spill,
            solar_cf=solar_cf, t_outdoor=t_outdoor,
            fuel_cost=fuel_cost, battery_deg=battery_deg,
            carbon_kg=carbon_kg,
            r_energy=r_energy, r_cost=r_cost, r_carbon=r_carbon,
            reward=reward,
            cost_sla=cost_sla,
            cost_overtemp=cost_overtemp,
            cost_power_deficit=cost_power_deficit,
            total_cost=total_cost,
            dc_status=dc_status, batt_status=batt_status,
        )

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation builder (18-D, layout preserved)
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        t = self._step_count
        dc_obs = self._dc.obs()
        batt_obs = self._batt.obs()

        cpu_util  = float(dc_obs.get('gpu_util',         0.0))
        mem_util  = float(dc_obs.get('infer_util',       0.0))
        q_train   = float(dc_obs.get('queue_train_fill', 0.0))
        q_ft      = float(dc_obs.get('queue_ft_fill',    0.0))
        urgency   = float(dc_obs.get('queue_urgency',    1.0))
        zone_norm = float(dc_obs.get('zone_temp_norm',   0.5))
        out_norm  = float(dc_obs.get('outdoor_temp_norm', 0.5))

        # COP ratio (matches JAX DataCenterMicrogridEnv._get_obs)
        cop_factor = float(np.clip(
            1.0 - self._dc.cop_decay * max(self._dc.t_outdoor - self._dc.t_ref, 0.0),
            0.4, 1.2,
        ))
        cop_ratio = float(np.clip((cop_factor - 0.4) / 0.8, 0.0, 1.0))

        # Resource obs sourced from sub-resources directly.
        solar_cf  = float(self._pv._cf_at(t))   # phase-aligned to current t
        soc       = float(batt_obs.get('soc', 0.5))
        dg_margin = float(np.clip(
            (self._dg.p_dg_max_mw - self._dg.current_p_mw)
            / max(self._dg.p_dg_max_mw, 1e-8),
            0.0, 1.0,
        ))

        # Last action (normalised to [0, 1] for obs even though action[3]∈[-1,1])
        la = self._last_action
        la_norm = np.array(
            [la[0], la[1], la[2], (la[3] + 1.0) / 2.0, la[4]],
            dtype=np.float32,
        )

        phase  = 2.0 * np.pi * (t % self._steps_per_day) / self._steps_per_day
        sin_t  = float(np.sin(phase))
        cos_t  = float(np.cos(phase))

        obs = np.array([
            cpu_util, mem_util, q_train, q_ft, urgency,
            zone_norm, out_norm, cop_ratio,
            solar_cf, soc, dg_margin,
            float(la_norm[0]), float(la_norm[1]), float(la_norm[2]),
            float(la_norm[3]), float(la_norm[4]),
            sin_t, cos_t,
        ], dtype=np.float32)

        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def obs(self, state: Any = None) -> np.ndarray:
        """Gym-compatible observation hook required by BaseEnv."""
        return self._get_obs()

    def _build_info(self, *, t, p_load, p_pv, p_batt, p_dg,
                    power_deficit, power_spill, solar_cf, t_outdoor,
                    fuel_cost, battery_deg, carbon_kg,
                    r_energy, r_cost, r_carbon, reward,
                    cost_sla, cost_overtemp, cost_power_deficit, total_cost,
                    dc_status, batt_status) -> Dict[str, Any]:
        return self.attach_constraint_costs({
            # CMDP cost channels
            'cost_sla':             cost_sla,
            'cost_overtemp':        cost_overtemp,
            'cost_power_deficit':   cost_power_deficit,
            'cost_sum':             total_cost,

            # Reward decomposition
            'reward_vector':        [r_energy, r_cost, r_carbon],

            # Power balance
            'p_load_mw':            p_load,
            'p_pv_mw':              p_pv,
            'p_batt_mw':            p_batt,
            'p_dg_mw':              p_dg,
            'power_deficit_mw':     power_deficit,
            'power_spill_mw':       power_spill,

            # Economics
            'fuel_cost':            fuel_cost,
            'battery_deg_cost':     battery_deg,
            'carbon_kg':            carbon_kg,

            # Exogenous
            'solar_cf':             solar_cf,
            't_outdoor':            t_outdoor,

            # DC sub-status (pass-through)
            'sla_violations':       int(dc_status.get('sla_violations', 0)),
            'step_sla_violations':  int(dc_status.get('step_sla_violations', 0)),
            't_zone':               dc_status.get('t_zone', 22.0),
            'pue':                  dc_status.get('pue', 2.0),
            'gpu_util':             dc_status.get('gpu_util', 0.0),

            # Battery sub-status
            'soc':                  batt_status.get('soc', 0.5),

            # Episode meta
            'step':                 t,
        })

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------

    def render(self, mode: str = 'human'):
        pass

    def close(self):
        pass

    def __repr__(self) -> str:
        return (
            f"DCMicrogridEnv(max_steps={self.max_steps}, "
            f"dt={self.delta_t_minutes}min, "
            f"step={self._step_count})"
        )


__all__ = ['DCMicrogridEnv', '_make_synthetic_solar', '_make_synthetic_temp']
