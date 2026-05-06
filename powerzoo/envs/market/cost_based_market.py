"""Battery arbitrage with LMPs from marginal-cost DC-OPF (``mc_c @ p``).

Single-agent env: LMPs are dual variables of the cost-based dispatch; no
bid–cost separation.  Reward is LMP × net power × Δt.

For piecewise offer curves and bid–cost separation, see :class:`BidBasedMarketEnv`.

LMP derivation
--------------
DC-OPF dual variables (shadow prices on nodal power-balance constraints)
give the LMP at each bus.  When the system is uncongested, LMP equals the
system marginal cost; under congestion, buses on the constrained side are
priced higher.

Observation space (flat)
------------------------
    [soc, lmp_norm, time_sin, time_cos, total_demand_norm]

Action space
------------
    Battery power setpoint in [-power_mw, power_mw] MW.

Reward
------
    revenue = LMP × P_net × Δt_h

Safety penalties flow only through the CMDP cost channel
(``info['cost_sum']``, ``info['cost_thermal_overload']``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from powerzoo.envs.grid.trans import TransGridEnv


class CostBasedMarketEnv(gym.Env):
    """Battery arbitrage on LMPs from ``TransGridEnv`` marginal-cost DC-OPF.

    Offer-based clearing: :class:`BidBasedMarketEnv`.

    Parameters
    ----------
    case : ClearCase, optional
        Power system case.  Defaults to Case5.
    battery_bus_id : int, optional
        Bus to attach the default battery (default: 2).
        Set to ``None`` to skip auto-creating a battery (attach your own).
    battery_capacity_mwh : float
        Battery energy capacity.  Default 200 MWh.
    battery_power_mw : float
        Battery power rating.  Default 50 MW.
    lmp_scale : float
        Divide raw LMP values by this factor for normalisation in the obs.
        Default 100 $/MWh.
    difficulty : str or None
        Passed to ``TransGridEnv``.  ``'easy'``, ``'medium'``, ``'hard'``.
    **grid_kwargs :
        Any remaining kwargs forwarded to ``TransGridEnv.__init__``.

    Example::

        from powerzoo import CostBasedMarketEnv

        env = CostBasedMarketEnv(difficulty='medium')
        obs, info = env.reset(seed=42)
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    """

    metadata = {"render_modes": ["human"]}

    def constraint_names(self) -> Tuple[str, ...]:
        """Market-core CMDP exposes only the thermal overload channel."""
        return ('thermal_overload',)

    def _attach_market_constraint_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        cost_thermal = max(0.0, float(info.get('cost_thermal_overload', 0.0)))
        info['constraint_names'] = self.constraint_names()
        info['constraint_costs'] = np.asarray([cost_thermal], dtype=np.float32)
        info['cost_sum'] = cost_thermal
        return info

    # ====== Initialization ======

    def __init__(
        self,
        case=None,
        battery_bus_id: int = 2,
        battery_capacity_mwh: float = 200.0,
        battery_power_mw: float = 50.0,
        lmp_scale: float = 100.0,
        difficulty: Optional[str] = None,
        normalize_actions: bool = True,
        **grid_kwargs,
    ):
        super().__init__()

        self.lmp_scale = lmp_scale
        self.normalize_actions = normalize_actions
        self._dt_h = grid_kwargs.get('delta_t_minutes', 30.0) / 60.0

        # Build the underlying grid env
        self.grid = TransGridEnv(
            case=case,
            difficulty=difficulty,
            solver_type=grid_kwargs.pop('solver_type', 'auto'),
            normalize_actions=normalize_actions,
            **grid_kwargs,
        )

        # Optionally attach a default battery.
        # normalize_actions is always False here: this wrapper fully owns
        # denormalization and always passes physical MW via a dict action.
        # Keeping the battery in physical mode prevents accidental
        # double-scaling if the call path is ever refactored.
        self._battery = None
        if battery_bus_id is not None:
            from powerzoo.envs.resource.battery import BatteryEnv
            self._battery = BatteryEnv(
                parent=self.grid,
                bus_id=battery_bus_id,
                capacity_mwh=battery_capacity_mwh,
                power_mw=battery_power_mw,
                normalize_actions=False,
            )

        # Cached constants (static across an episode)
        self._p_max_sum: float = float(self.grid.case.units['p_max'].sum()) or 1.0
        self._time_phase_scale: float = 2.0 * np.pi / max(self.grid.steps_per_day, 1)

        # Action normalization
        self._action_phys_low = np.array([-battery_power_mw], dtype=np.float32)
        self._action_phys_high = np.array([battery_power_mw], dtype=np.float32)

        # Spaces
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0,  np.inf,  1.0,  1.0, np.inf], dtype=np.float32),
            shape=(5,), dtype=np.float32,
        )
        if self.normalize_actions:
            self.action_space = spaces.Box(
                low=-np.ones(1, dtype=np.float32),
                high=np.ones(1, dtype=np.float32),
                shape=(1,), dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=self._action_phys_low,
                high=self._action_phys_high,
                shape=(1,), dtype=np.float32,
            )

        self.obs_names: List[str] = [
            'soc', 'lmp_norm', 'time_sin', 'time_cos', 'total_demand_norm'
        ]
        self.action_names: List[str] = ['p_mw']

        # Internal state cache
        self._last_lmp: Optional[np.ndarray] = None
        self._last_state: Optional[Dict] = None

    # ====== RL Interface Methods ======

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset grid and battery; return initial observation.

        ``self.grid.reset()`` cascades to all registered sub-resources
        (including ``self._battery``) via ``GridEnv.reset()`` → each resource's
        ``reset()`` is called, which resets battery SOC to ``initial_soc``.
        No separate battery reset is required here.
        """
        state, grid_info = self.grid.reset(seed=seed, options=options)

        # Validate that the underlying solver provides benchmark-grade LMP
        if not state.get('lmp_available', False):
            solver = state.get('solver_backend', 'unknown')
            raise RuntimeError(
                f"CostBasedMarketEnv requires benchmark-grade LMP but the "
                f"'{solver}' solver does not provide nodal LMPs. "
                f"Use solver_type='scipy' or 'gurobi'."
            )

        self._last_state = state
        self._last_lmp = state.get('lmp', np.zeros(len(self.grid.case.nodes)))
        obs = self._build_obs()
        info = self._build_market_info(state)
        info.update(grid_info)
        return obs, self._attach_market_constraint_info(info)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the market environment.

        Args:
            action: Battery power setpoint.  When ``normalize_actions=True``
                (default), in [-1, 1] where -1 = max charge, 0 = idle,
                1 = max discharge.  When False, in [-power_mw, power_mw] MW.

        Returns:
            Standard Gymnasium 5-tuple.  ``info`` includes:

            ``requested_p_mw`` (float): physical MW requested by the agent
                after denormalization and rated-power clipping.
            ``realized_p_mw`` (float): physical MW actually dispatched by the
                battery after SOC constraints.  Differs from ``requested_p_mw``
                when the battery is near a SOC limit.  Useful for diagnosing
                policy saturation and SOC boundary effects.

        Raises:
            RuntimeError: if no battery is attached (``battery_bus_id=None``).
                Without a physical battery, the environment cannot settle
                actions and would compute reward on an unconstrained phantom
                power value.
        """
        if self._battery is None:
            raise RuntimeError(
                "CostBasedMarketEnv.step() called without an attached battery. "
                "Pass battery_bus_id != None at construction, or attach a battery "
                "manually before calling step()."
            )

        # Denormalize [-1, 1] → [-power_mw, +power_mw]
        # Clip first: stochastic policies (e.g. PPO/SAC) can emit values slightly
        # outside [-1, 1] during early exploration.
        raw = float(np.atleast_1d(action)[0])
        if self.normalize_actions:
            raw = float(np.clip(raw, -1.0, 1.0))
            lo, hi = self._action_phys_low[0], self._action_phys_high[0]
            raw = float((lo + hi) / 2 + raw * (hi - lo) / 2)
        # Clip to physical bounds unconditionally: guards non-normalized mode
        # against extreme values that could destabilize the OPF solver.
        raw = float(np.clip(raw, self._action_phys_low[0], self._action_phys_high[0]))

        # Build grid action dict (pass physical MW via dict key, skipping
        # the battery's own normalize_actions since we already converted)
        grid_action = {self._battery.resource_id: {'p_mw': raw}}

        state, _grid_reward, terminated, truncated, info = self.grid.step(grid_action)
        self._last_state = state
        self._last_lmp = state.get('lmp', np.zeros(len(self.grid.case.nodes)))

        # Market reward: LMP × realized_power × Δt.
        # Use the battery's actual dispatched power (SOC-feasible) rather than
        # the requested `raw`, so that reward tracks what was physically injected.
        realized = self._battery.current_p_mw
        reward = self._compute_market_reward(np.array([realized]), state, info)

        obs = self._build_obs()
        market_info = self._build_market_info(state)
        market_info.update(info)  # include grid info
        market_info['requested_p_mw'] = raw
        market_info['realized_p_mw'] = realized
        market_info = self._attach_market_constraint_info(market_info)

        # Episode summary
        if terminated or truncated:
            market_info['episode'] = info.get('episode', {})

        return obs, reward, terminated, truncated, market_info

    # ====== Internal Market Logic ======

    def _get_bus_lmp(self) -> float:
        """Return LMP at the battery bus, or system mean if no battery.

        The LMP array is indexed by 0-based node position (order in
        ``case.nodes``), **not** by the physical bus ID.  ``nodes['#id']``
        holds the 0-based positional index for each bus.
        """
        if self._last_lmp is None or len(self._last_lmp) == 0:
            return 0.0
        if self._battery is not None:
            bus_id = self._battery.bus_id  # public property
            try:
                bus_idx = int(self.grid.case.nodes.loc[bus_id, '#id'])
            except KeyError:
                # bus_id not found in case — fall back to mean
                return float(np.mean(self._last_lmp))
            return float(self._last_lmp[bus_idx])
        return float(np.mean(self._last_lmp))

    def _build_obs(self) -> np.ndarray:
        """Build [soc, lmp_norm, time_sin, time_cos, demand_norm]."""
        soc = 0.5
        if self._battery is not None:
            soc = float(self._battery.soc)

        lmp_norm = self._get_bus_lmp() / max(self.lmp_scale, 1e-6)

        # Time encoding
        phase = self._time_phase_scale * self.grid.time_step
        time_sin, time_cos = float(np.sin(phase)), float(np.cos(phase))

        # Total demand (normalised by generation capacity)
        total_demand = float(self.grid._get_node_loads_p_current().sum())
        demand_norm = total_demand / self._p_max_sum

        return np.array(
            [soc, lmp_norm, time_sin, time_cos, demand_norm],
            dtype=np.float32,
        )

    def _compute_market_reward(self, action: Any, state: Dict,
                                info: Dict) -> float:
        """Revenue-based reward:  r = LMP × P_net × Δt.

        Sign convention (matches BatteryEnv):
            ``power > 0`` — discharging: battery injects into grid → positive revenue
            ``power < 0`` — charging:    battery draws from grid  → negative revenue

        ``action`` should be the **realized** (SOC-feasible) power in physical MW,
        not the requested setpoint.  In ``step()``, this is supplied as
        ``battery.current_p_mw`` after the grid sub-step.

        Safety penalties are excluded from the scalar reward and flow
        only through the CMDP cost channel (``info['cost_sum']``).
        """
        power = float(np.atleast_1d(action)[0])

        lmp = self._get_bus_lmp()

        revenue = lmp * power * self._dt_h

        return float(revenue)

    def _build_market_info(self, state: Dict) -> Dict:
        return {
            'lmp': self._last_lmp,
            'is_safe': state.get('is_safe', True),
            'safety_info': state.get('safety_info', {}),
            'reward_components': state.get('reward_components', {}),
        }

    def render(self) -> None:
        if self._last_state is not None:
            t = self.grid.time_step
            lmp_mean = float(np.mean(self._last_lmp)) if self._last_lmp is not None else 0.0
            soc = float(self._battery.soc) if self._battery else float('nan')
            print(f"t={t:3d} | LMP={lmp_mean:.2f} $/MWh | SOC={soc:.2%} | "
                  f"safe={self._last_state.get('is_safe', '?')}")

    def close(self) -> None:
        pass

    @property
    def steps_per_day(self) -> int:
        return self.grid.steps_per_day

    @property
    def np_random(self):
        return self.grid.np_random
