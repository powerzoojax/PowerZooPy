"""Diesel Generator Resource (numpy / gymnasium).

Mirrors ``powerzoojax.envs.resource.diesel`` in numerical defaults and physics.
A single-device ``ResourceEnv`` subclass that exposes the same physics through
the OO sub-resource protocol used by ``DCMicrogridEnv`` (and any future
behind-the-meter env that wants a dispatchable generator).

Action (1-D in [0, 1]):
    dg_norm = clip(action, 0, 1) → p_dg = dg_norm × p_dg_max_mw

Sign convention:
    current_p_mw > 0 : injecting (DG can only inject, never absorbs)

Per-step economics (computed eagerly; read via ``status()``):
    fuel_cost_step = p_dg × dt_h × fuel_cost_per_mwh   [$]
    carbon_kg_step = p_dg × dt_h × 1e3 × emission_factor  [kgCO2]
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
from gymnasium import spaces as _spaces

from .base import ResourceEnv


class DieselResource(ResourceEnv):
    """Single-device diesel generator following the ResourceEnv contract.

    Args:
        p_dg_max_mw: Nameplate capacity [MW] (default 0.6).
        fuel_cost_per_mwh: Variable fuel cost [$/MWh] (default 300.0).
        emission_factor: Carbon intensity [kgCO2 / kWh_e] (default 0.80).
        p_min_norm: Minimum loading fraction in ``[0, 1)``.  Default 0.0
            keeps the legacy unconstrained behaviour.  When > 0 a soft
            deadband at ``p_min_norm / 2`` is applied: requested setpoints
            below the deadband shut the generator OFF (p = 0); setpoints
            above the deadband are clamped to ``[p_min_norm, 1]``.  Real
            diesel gensets should not run below ~30 % of rated load (wet
            stacking damages the engine), so a typical opt-in value is 0.3.
        normalize_actions: If True, action space is [0, 1]; otherwise [0, p_max]
            in physical MW.  Default True (matches DCMicrogridEnv conventions).
        delta_t_minutes: Step duration in minutes.  Default 15 (legacy default
            of ResourceEnv); the host env should pass its own dt.
        parent / bus_id: Optional grid attachment (forwarded to ResourceEnv).
    """

    def __init__(
        self,
        *,
        p_dg_max_mw: float = 0.6,
        fuel_cost_per_mwh: float = 300.0,
        emission_factor: float = 0.80,
        p_min_norm: float = 0.0,
        normalize_actions: bool = True,
        delta_t_minutes: float = 15.0,
        parent: Any = None,
        bus_id: int = -1,
    ):
        super().__init__(parent=parent, bus_id=bus_id, delta_t_minutes=delta_t_minutes)

        self.p_dg_max_mw: float = float(p_dg_max_mw)
        self.fuel_cost_per_mwh: float = float(fuel_cost_per_mwh)
        self.emission_factor: float = float(emission_factor)
        self.p_min_norm: float = float(p_min_norm)
        if not (0.0 <= self.p_min_norm < 1.0):
            raise ValueError(f"p_min_norm must be in [0, 1), got {self.p_min_norm}")
        self.normalize_actions: bool = bool(normalize_actions)

        # Per-step diagnostics (refreshed every step)
        self.fuel_cost_step: float = 0.0
        self.carbon_kg_step: float = 0.0

        # Gymnasium spaces
        if self.normalize_actions:
            self.action_space = _spaces.Box(
                low=np.zeros(1, dtype=np.float32),
                high=np.ones(1, dtype=np.float32),
                shape=(1,), dtype=np.float32,
            )
        else:
            self.action_space = _spaces.Box(
                low=np.zeros(1, dtype=np.float32),
                high=np.array([self.p_dg_max_mw], dtype=np.float32),
                shape=(1,), dtype=np.float32,
            )
        # Obs: [p_norm, dg_margin_norm]
        self.observation_space = _spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            shape=(2,), dtype=np.float32,
        )
        self.action_names = ['dg_power_norm']

        self._complete_resource_init() if hasattr(self, '_complete_resource_init') else None

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None, day_id: Optional[int] = None):
        super().reset(seed=seed, options=options, day_id=day_id)
        self.current_p_mw = 0.0
        self.current_q_mvar = 0.0
        self.fuel_cost_step = 0.0
        self.carbon_kg_step = 0.0

    def step(self, action: Optional[Union[float, np.ndarray, Dict[str, Any]]] = None) -> None:
        """Execute one DG step.

        Args:
            action: float, ndarray(1,), or dict with key ``'dg_power_norm'``.
                When ``normalize_actions=True``: clipped to [0, 1] then scaled
                by ``p_dg_max_mw``.  Otherwise: clipped to [0, p_dg_max_mw].
        """
        norm = self._parse_action(action)
        if self.normalize_actions:
            norm_clipped = float(np.clip(norm, 0.0, 1.0))
        else:
            norm_clipped = float(np.clip(norm, 0.0, self.p_dg_max_mw)) / self.p_dg_max_mw

        # Minimum loading soft constraint (no-op when p_min_norm == 0).
        deadband = 0.5 * self.p_min_norm
        if norm_clipped <= deadband:
            effective_norm = 0.0
        else:
            effective_norm = max(norm_clipped, self.p_min_norm)
        p_dg = effective_norm * self.p_dg_max_mw

        dt_h = self.delta_t_minutes / 60.0
        self.fuel_cost_step = p_dg * dt_h * self.fuel_cost_per_mwh
        self.carbon_kg_step = p_dg * dt_h * 1000.0 * self.emission_factor

        self.current_p_mw = p_dg
        self.current_q_mvar = 0.0
        self.time_step += 1

    def _parse_action(self, action) -> float:
        if action is None:
            return 0.0
        if isinstance(action, dict):
            return float(action.get('dg_power_norm', 0.0))
        arr = np.atleast_1d(np.asarray(action, dtype=np.float32)).flatten()
        return float(arr[0])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            'current_p_mw': float(self.current_p_mw),
            'current_q_mvar': float(self.current_q_mvar),
            'time_step': int(self.time_step),
            'fuel_cost_step': float(self.fuel_cost_step),
            'carbon_kg_step': float(self.carbon_kg_step),
            'p_dg_max_mw': float(self.p_dg_max_mw),
        }

    def obs(self) -> Dict[str, float]:
        safe = max(self.p_dg_max_mw, 1e-9)
        p_norm = float(np.clip(self.current_p_mw / safe, 0.0, 1.0))
        dg_margin = float(np.clip(1.0 - p_norm, 0.0, 1.0))
        return {'p_norm': p_norm, 'dg_margin_norm': dg_margin}


__all__ = ['DieselResource']
