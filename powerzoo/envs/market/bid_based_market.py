"""Bid-Based LMP Market Environment.

A proper electricity market environment with **piecewise-linear offer
curves** and explicit separation between offer prices, clearing, and
settlement.

Unlike :class:`CostBasedMarketEnv` (which uses flat cost-based
dispatch), this environment models a competitive market:

* **Generators** submit stepped offer curves (price–quantity segments).
  By default the offers are derived from true costs with an optional
  random markup, but they can also be externally supplied.
  **Offer curves are static within an episode** — they are generated
  once at :meth:`reset` and remain fixed until the next episode.
* **Clearing** is a network-constrained SCED (Security-Constrained
  Economic Dispatch) using the submitted offers as the LP objective.
* **LMP** is derived from the offer-based dispatch (LP dual variables),
  so it reflects offer prices, not necessarily true marginal costs.
* **Settlement** is LMP-based: revenue = LMP × P.
* **Profit** = settlement revenue − true generation cost (for
  generators) or settlement revenue alone (for battery, whose "cost" is
  the electricity purchased when charging).

Battery role
-----------
The battery is a **prosumer / demand-side resource** — it does not submit
offer segments, but its charge/discharge power **does influence the LMP**.
The battery's net injection enters the SCED as a nodal net-load offset:
discharging (P > 0) reduces net load and can lower the LMP at its bus;
charging (P < 0) increases net load and can raise the LMP.
In market-structure terms the battery is a price-influencing entity,
not a passive price-taker.

Market structure
----------------
- Single time-step (rolling) day-ahead / real-time market.
- No unit commitment, no start-up / shut-down costs, no uplift.
- Compatible with the ``Market Lite`` boundary in CHARTER.md.

Observation space (battery agent, flat)
---------------------------------------
    [soc, lmp_norm, time_sin, time_cos, demand_norm,
     mean_offer_price_norm]

Action space
------------
    Battery power setpoint in [-power_mw, power_mw] MW.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from powerzoo.envs.grid.trans import TransGridEnv
from powerzoo.envs.grid.cal_dcopf_trans import (
    make_cost_segments,
    solve_piecewise_ed_opf,
)


class BidBasedMarketEnv(gym.Env): # strategic bidding market
    """Competitive electricity market with piecewise-linear offer curves.

    Parameters
    ----------
    case : ClearCase, optional
        Power system case.  Defaults to Case5.
    battery_bus_id : int, optional
        Bus to attach the default battery (default: 2).
    battery_capacity_mwh : float
        Battery energy capacity (default 200 MWh).
    battery_power_mw : float
        Battery power rating (default 50 MW).
    n_segments : int
        Number of offer-curve segments per generator (default 5).
    markup_std : float
        Standard deviation of random markup (fraction) applied to
        cost-based offer prices each episode.  0 = truthful bidding.
        Default 0.05 (5 % noise).
    lmp_scale : float
        Divide raw LMP values by this for observation normalisation
        (default 100 $/MWh).
    difficulty : str or None
        Passed to ``TransGridEnv``.
    normalize_actions : bool
        Whether to normalise the battery action to [-1, 1].
    skip_grid_opf : bool
        When ``True`` (default for RL training), the underlying grid
        skips its internal OPF solve and instead uses the
        market-cleared dispatch computed by :func:`solve_piecewise_ed_opf`.
        This eliminates a redundant LP solve per step (the grid's OPF and
        the market's piecewise ED-OPF would otherwise be solving very similar
        problems with the same net-load input).  Set to ``False`` when
        debugging or when the grid OPF result is needed separately.
    degradation_cost_per_mwh : float
        Monetary penalty per MWh of battery throughput, added to the
        reward as ``-degradation_cost_per_mwh * |power| * dt_h``.
        Models battery wear; suppresses high-frequency charge/discharge
        cycles.  Default 0 (no degradation penalty).
    action_smooth_cost : float
        Penalty coefficient for rapid changes in battery setpoint.
        Added as ``-action_smooth_cost * |power - prev_power|``.
        Helps prevent oscillation between charge and discharge.
        Default 0 (disabled).
    infeasible_penalty : float
        Penalty subtracted from reward when market clearing is infeasible
        (``opf success == False``).  Replaces the hard-coded value.
        Default 1000.0.  Tune together with ``reward_scale`` to keep
        the penalty in a reasonable range relative to the LMP revenue.
    **grid_kwargs
        Forwarded to ``TransGridEnv.__init__``.
    """

    metadata = {"render_modes": ["human"]}

    # ====== Initialization ======

    def __init__(
        self,
        case=None,
        battery_bus_id: int = 2,
        battery_capacity_mwh: float = 200.0,
        battery_power_mw: float = 50.0,
        n_segments: int = 5,
        markup_std: float = 0.05,
        lmp_scale: float = 100.0,
        difficulty: Optional[str] = None,
        normalize_actions: bool = True,
        skip_grid_opf: bool = True,
        degradation_cost_per_mwh: float = 0.0,
        action_smooth_cost: float = 0.0,
        infeasible_penalty: float = 1000.0,
        **grid_kwargs,
    ):
        super().__init__()

        self.n_segments = n_segments
        self.markup_std = markup_std
        self.lmp_scale = lmp_scale
        self.normalize_actions = normalize_actions
        self.skip_grid_opf = skip_grid_opf
        self.degradation_cost_per_mwh = degradation_cost_per_mwh
        self.action_smooth_cost = action_smooth_cost
        self.infeasible_penalty = infeasible_penalty
        self._dt_h = grid_kwargs.get('delta_t_minutes', 30.0) / 60.0

        # Build underlying grid env (solver_type is irrelevant—we call our
        # own piecewise solver, but grid env still needs one for reset).
        self.grid = TransGridEnv(
            case=case,
            difficulty=difficulty,
            solver_type=grid_kwargs.pop('solver_type', 'scipy'),
            normalize_actions=normalize_actions,
            **grid_kwargs,
        )

        # Battery — registered with the grid so its current_p_mw is visible
        # inside _calculate_node_net_load for net-load correction.
        self._battery = None
        if battery_bus_id is not None:
            from powerzoo.envs.resource.battery import BatteryEnv
            self._battery = BatteryEnv(
                parent=self.grid,
                bus_id=battery_bus_id,
                capacity_mwh=battery_capacity_mwh,
                power_mw=battery_power_mw,
                normalize_actions=normalize_actions,
            )

        # Cost segments (generated once from case, markup applied per episode)
        self._base_segments = make_cost_segments(self.grid.case, n_segments)
        self._current_segments: Optional[Dict[str, np.ndarray]] = None

        # Cached constants
        self._p_max_sum = float(self.grid.case.units['p_max'].sum()) or 1.0
        self._time_phase_scale = 2.0 * np.pi / max(self.grid.steps_per_day, 1)

        # Action normalisation
        self._action_phys_low = np.array([-battery_power_mw], dtype=np.float32)
        self._action_phys_high = np.array([battery_power_mw], dtype=np.float32)

        # Spaces — 6-dim obs (adds mean_offer_price_norm vs CostBasedMarketEnv)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -5.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 5.0, 1.0, 1.0, 2.0, 5.0], dtype=np.float32),
            shape=(6,), dtype=np.float32,
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
            'soc', 'lmp_norm', 'time_sin', 'time_cos',
            'total_demand_norm', 'mean_offer_price_norm',
        ]

        # Internal state
        self._last_lmp: Optional[np.ndarray] = None
        self._last_state: Optional[Dict] = None
        self._last_opf_result: Optional[Dict] = None
        self._prev_action: float = 0.0   # for action-smoothing penalty

        # Infeasibility rollback
        self._safe_soc: float = 0.5      # last known-safe SOC

    # ====== RL Interface Methods ======

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        state, grid_info = self.grid.reset(seed=seed, options=options)

        # Record initial safe SOC
        self._safe_soc = float(self._battery.soc) if self._battery else 0.5
        self._prev_action = 0.0

        # Generate offer curves for this episode (cost + random markup).
        # These remain FIXED for the entire episode — there is no intra-episode
        # offer updating.  Change markup_std or call reset() for new offers.
        self._current_segments = self._generate_offer_segments()

        # Run initial piecewise OPF to get LMP
        self._run_market_clearing(state)

        obs = self._build_obs()
        info = self._build_market_info(state)
        info.update(grid_info)
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the market environment.

        Two internal paths:

        ``skip_grid_opf=True`` (default, recommended for RL training):
            The underlying grid's internal OPF solve is bypassed.
            The sequence per step is:

            1. Step the battery (update SOC, current_p_mw).
            2. Advance the grid's time step counter.
            3. Compute net load = gross load − battery injection at its bus.
            4. Run piecewise-linear SCED → market dispatch + LMP.
            5. Update grid's cached line/node state directly from the
               market-cleared OPF result (DC power flow for line flows).
            6. Compute LMP-based settlement reward.

            This avoids one redundant LP solve per step.

        ``skip_grid_opf=False`` (legacy / debug):
            The grid runs its own OPF, then the market overrides the LMP
            with its own piecewise ED-OPF result.  Runs two OPFs per step.

        Returns standard Gymnasium 5-tuple.
        """
        raw = float(np.atleast_1d(action)[0])
        if self.normalize_actions:
            lo, hi = self._action_phys_low[0], self._action_phys_high[0]
            raw = float((lo + hi) / 2 + raw * (hi - lo) / 2)

        if self._battery is not None:
            bat_id = self._battery.resource_id
            bat_action = {bat_id: {'p_mw': raw}}
        else:
            bat_action = {}

        # ---- Fast path: bypass the grid's internal OPF ----
        if self.skip_grid_opf:
            # 1. Step battery (updates soc, current_p_mw, time_step)
            for rid, res in self.grid.sub_resources.items():
                res.step(bat_action.get(rid))
            # NOTE: directly incrementing time counters bypasses any future
            # side effects that GridEnv.step() may add (e.g. rolling stats,
            # price curve advances).  Kept minimal on purpose — review when
            # GridEnv.step() gains new time-step hooks.
            self.grid.time_step += 1
            self.grid._episode_steps += 1

            # 2. Gross node loads for the NEW time step
            node_loads = self.grid._get_node_loads_p_current().copy()

            # 3. Subtract battery injection: discharging (p>0) reduces net load;
            #    charging (p<0) increases net load.
            if self._battery is not None:
                bat_bus = self._battery._bus_id
                if 0 <= bat_bus < len(node_loads):
                    node_loads[bat_bus] -= float(self._battery.current_p_mw)

            # 4. Market clearing
            result = solve_piecewise_ed_opf(
                self.grid.case,
                node_loads,
                offer_segments=self._current_segments,
                commitment=None,
                slack_penalty=1e6,
                verbose=False,
            )
            self._last_opf_result = result

            # 5. Update grid state from market result
            if result['success']:
                self._last_lmp = result['lmp']
                self._update_grid_state_from_opf(result, node_loads)
                self._safe_soc = float(self._battery.soc)
                terminated = False
            else:
                self._last_lmp = np.zeros(len(self.grid.case.nodes))
                self._handle_infeasible()
                terminated = True   # market failure → episode terminates

            truncated = self.grid._episode_steps >= self.grid.max_episode_steps

            # Build state dict for info
            state = self._build_state_dict_from_opf(result, node_loads)

            # Reward (includes penalties) — use realised power (SOC-constrained),
            # not the raw action, so that settlement reflects physical dispatch.
            realized = float(self._battery.current_p_mw) if self._battery else raw
            reward = self._compute_market_reward(np.array([realized]))
            self._prev_action = realized

            # Build info (mimics grid.build_info + market additions)
            info = self._build_grid_info_fast(state, result)
            self._last_state = state

        # ---- Legacy path: let the grid run its own OPF first ----
        else:
            state, _grid_reward, terminated, truncated, info = \
                self.grid.step(bat_action)
            self._last_state = state
            self._run_market_clearing(state)
            realized = float(self._battery.current_p_mw) if self._battery else raw
            reward = self._compute_market_reward(np.array([realized]))
            self._prev_action = realized

        obs = self._build_obs()
        market_info = self._build_market_info(state)
        market_info.update(info)
        market_info = self._attach_market_constraint_info(market_info)

        if terminated or truncated:
            market_info['episode'] = info.get('episode', {})

        return obs, reward, terminated, truncated, market_info

    def _build_state_dict_from_opf(self, opf_result, node_net_load_mw):
        """Construct a state dict from a market-cleared OPF result.

        Mirrors the key fields of ``GridEnv._get_state()`` that the market
        info helpers and observation builder depend on.
        """
        return {
            'lines': self.grid._lines,
            'nodes': self.grid._nodes,
            'is_safe': self.grid._is_safe,
            'safety_info': self.grid._safety_info,
            'time_step': self.grid.time_step,
            'unit_power_mw': self.grid._unit_power_mw,
            'physics': self.grid.physics,
            'solver_mode': self.grid.solver_mode,
            'pf_mode': self.grid.pf_mode,
            'opf_cost': opf_result.get('total_cost', 0.0),
            'opf_slack': opf_result.get('slack_violation', 0.0),
            'lmp': self._last_lmp,
            'lmp_available': opf_result.get('success', False),
        }

    def _update_grid_state_from_opf(self, opf_result, node_net_load_mw):
        """Populate grid's cached state from a solved OPF result."""
        self.grid._opf_result = opf_result
        self.grid._unit_power_mw = opf_result['unit_power_mw']

        lines = self.grid.case.lines.copy()
        lines['line_flow_mw'] = opf_result['line_flow_mw']
        self.grid._lines = lines

        nodes = self.grid.case.nodes.copy()
        nodes['node_inj_mw'] = opf_result['node_net_injection_mw']
        nodes['node_net_load_mw'] = node_net_load_mw
        self.grid._nodes = nodes

        # Safety check
        line_flow_safe, safety_info = self.grid.safety_check(lines, with_info=True)
        self.grid._is_safe = bool(line_flow_safe.all())
        self.grid._safety_info = safety_info
        if opf_result.get('slack_violation', 0.0) > 1e-3:
            self.grid._is_safe = False
            if self.grid._safety_info is not None:
                self.grid._safety_info = dict(self.grid._safety_info)
                self.grid._safety_info['slack_violation'] = float(
                    opf_result['slack_violation'])

    def _build_grid_info_fast(self, state, opf_result) -> Dict[str, Any]:
        """Build info dict for the fast (skip_grid_opf) path.

        Mirrors the key fields of ``GridEnv.build_info``.
        """
        safety_info = state.get('safety_info') or {}

        caps = self.grid.case.lines['cap'].values
        flows = opf_result['line_flow_mw']
        cost_thermal = float(np.sum(np.maximum(0.0, np.abs(flows) - caps)))
        cost_sum = cost_thermal

        info = {
            'is_safe': bool(state.get('is_safe', True)),
            'cost_sum': cost_sum,
            # skip_grid_opf mode does not run a voltage solve, so voltage
            # violation cost is always zero.  For full safety constraints
            # (CMDP), disable skip_grid_opf.
            'cost_voltage_violation': 0.0,
            'cost_thermal_overload': cost_thermal,
            'cost_load_shedding': 0.0,
            'cost_power_balance': 0.0,
            'goal_met': bool(state.get('is_safe', True)),
            'physics': state.get('physics', self.grid.physics),
            'solver_mode': state.get('solver_mode', self.grid.solver_mode),
            'pf_mode': state.get('pf_mode', self.grid.pf_mode),
            'safety_info': safety_info,
            'resource_status': {
                rid: res.status()
                for rid, res in self.grid.sub_resources.items()
            },
            'reward_components': {},
            'pf_converged': opf_result.get('success', False),
            'cost_exception': 0.0 if opf_result.get('success', False) else 1.0,
            'unit_power_mw': opf_result['unit_power_mw'],
            'total_generation_mw': float(opf_result['unit_power_mw'].sum()),
            'opf_cost': opf_result.get('total_cost', 0.0),
            'opf_slack': opf_result.get('slack_violation', 0.0),
            'lmp': self._last_lmp,
            'lmp_available': opf_result.get('success', False),
            'lmp_method': 'nodal_dual_reconstruction',
            'lmp_quality': 'nodal',
        }
        return self._attach_market_constraint_info(info)

    def constraint_names(self) -> Tuple[str, ...]:
        """Market-core CMDP exposes only the thermal overload channel."""
        return ('thermal_overload',)

    def _attach_market_constraint_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        cost_thermal = max(0.0, float(info.get('cost_thermal_overload', 0.0)))
        info['constraint_names'] = self.constraint_names()
        info['constraint_costs'] = np.asarray([cost_thermal], dtype=np.float32)
        info['cost_sum'] = cost_thermal
        return info

    def _handle_infeasible(self):
        """Rollback battery SOC to the last confirmed-safe value.

        Called when market clearing returns ``success == False``.
        After stepping the battery (fast path) the SOC may reflect an
        infeasible dispatch outcome.  Reverting to the last safe SOC
        keeps the battery state internally consistent and prevents the
        RL agent from exploiting infeasible market conditions.
        The battery power is also reset to idle.
        """
        if self._battery is not None:
            self._battery.soc = self._safe_soc
            self._battery.current_p_mw = 0.0

    # ====== Internal Market Logic ======

    def _generate_offer_segments(self) -> Dict[str, np.ndarray]:
        """Create offer curves for this episode.

        Starts from cost-based segments and applies a random multiplicative
        markup (controlled by ``markup_std``).  When ``markup_std == 0``
        the offers exactly equal true costs (truthful bidding).

        .. note::
            Offer curves are **static within an episode**.  They are generated
            once at :meth:`reset` and used unchanged for all subsequent
            :meth:`step` calls.  To expose the agent to offer-price dynamics,
            call ``env.reset()`` to start a new episode with a new offer
            realisation.
        """
        base_w = self._base_segments['seg_widths'].copy()
        base_p = self._base_segments['seg_prices'].copy()

        if self.markup_std > 0:
            rng = self.np_random
            # Multiplicative markup ≥ 1 (generators bid above cost)
            markup = 1.0 + np.abs(rng.normal(0, self.markup_std, size=base_p.shape))
            offer_prices = base_p * markup
            # Re-enforce monotonicity per generator
            for i in range(offer_prices.shape[0]):
                for k in range(1, offer_prices.shape[1]):
                    if offer_prices[i, k] < offer_prices[i, k - 1]:
                        offer_prices[i, k] = offer_prices[i, k - 1] + 0.01
        else:
            offer_prices = base_p

        return {'seg_widths': base_w, 'seg_prices': offer_prices}

    # ------------------------------------------------------------------
    # Market clearing
    # ------------------------------------------------------------------

    def _run_market_clearing(self, state: Dict) -> None:
        """Run piecewise-linear SCED and cache results.

        Net load passed to the SCED is the same net load used by the grid's
        internal OPF (``state['nodes']['node_net_load_mw']``), which already
        includes the battery injection subtraction.  Reusing it here avoids
        double-counting the battery and ensures the market clears at the same
        operating point the grid solved for.

        The battery is a **prosumer** — its net injection enters the SCED as a
        nodal offset and influences the LMP.
        """
        # Use the net load that the grid's OPF actually solved with (not the
        # gross load at the new time step, which would double-count the battery).
        nodes_df = state.get('nodes')
        if nodes_df is not None and (
            hasattr(nodes_df, 'columns') and 'node_net_load_mw' in nodes_df.columns
            or isinstance(nodes_df, dict) and 'node_net_load_mw' in nodes_df
        ):
            if hasattr(nodes_df, 'columns'):
                node_loads = np.asarray(nodes_df['node_net_load_mw'].values, dtype=float).copy()
            else:
                node_loads = np.asarray(nodes_df['node_net_load_mw'], dtype=float).copy()
        else:
            # Fallback: gross load minus battery (should not reach here in normal use)
            node_loads = self.grid._get_node_loads_p_current().copy()
            if self._battery is not None:
                bat_bus = self._battery._bus_id
                bat_p = float(self._battery.current_p_mw)
                if 0 <= bat_bus < len(node_loads):
                    node_loads[bat_bus] -= bat_p

        result = solve_piecewise_ed_opf(
            self.grid.case,
            node_loads,
            offer_segments=self._current_segments,
            commitment=None,
            slack_penalty=1e6,
            verbose=False,
        )
        self._last_opf_result = result
        if result['success']:
            self._last_lmp = result['lmp']
        else:
            self._last_lmp = np.zeros(len(self.grid.case.nodes))
            self._handle_infeasible()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_bus_lmp(self) -> float:
        if self._last_lmp is None or len(self._last_lmp) == 0:
            return 0.0
        if self._battery is not None:
            bus_idx = min(self._battery._bus_id, len(self._last_lmp) - 1)
            return float(self._last_lmp[bus_idx])
        return float(np.mean(self._last_lmp))

    def _build_obs(self) -> np.ndarray:
        soc = float(self._battery.soc) if self._battery is not None else 0.5
        lmp_norm = self._get_bus_lmp() / max(self.lmp_scale, 1e-6)

        phase = self._time_phase_scale * self.grid.time_step
        time_sin = float(np.sin(phase))
        time_cos = float(np.cos(phase))

        total_demand = float(self.grid._get_node_loads_p_current().sum())
        demand_norm = total_demand / self._p_max_sum

        # Mean offer price (weighted by segment width)
        if self._current_segments is not None:
            w = self._current_segments['seg_widths']
            p = self._current_segments['seg_prices']
            total_w = w.sum()
            mean_offer = float((p * w).sum() / max(total_w, 1e-6))
        else:
            mean_offer = 0.0
        mean_offer_norm = mean_offer / max(self.lmp_scale, 1e-6)

        return np.array(
            [soc, lmp_norm, time_sin, time_cos, demand_norm, mean_offer_norm],
            dtype=np.float32,
        )

    def _compute_market_reward(self, action: np.ndarray) -> float:
        """LMP × P_net × Δt  plus optional degradation / action-smoothing.

        Two optional penalty terms discourage unrealistic RL behaviour:

        * ``degradation_cost_per_mwh``: models battery wear; proportional
          to absolute throughput ``|P| × Δt``.
        * ``action_smooth_cost``: penalises rapid charge/discharge switching;
          proportional to ``|P_t − P_{t-1}|``.

        Safety penalties are excluded — they flow through the CMDP cost
        channel via ``info['cost_sum']`` (set by the underlying grid).
        """
        power = float(np.atleast_1d(action)[0])
        lmp = self._get_bus_lmp()
        reward = lmp * power * self._dt_h

        # Degradation cost: proportional to throughput
        if self.degradation_cost_per_mwh > 0 and self._battery is not None:
            degradation = self.degradation_cost_per_mwh * abs(power) * self._dt_h
            reward -= degradation

        # Action-smoothing: penalises rapid charge/discharge switching
        if self.action_smooth_cost > 0:
            delta = abs(power - self._prev_action)
            reward -= self.action_smooth_cost * delta

        # Infeasibility penalty: signal when market fails
        if (self._last_opf_result is not None
                and not self._last_opf_result['success']):
            reward -= self.infeasible_penalty

        return float(reward)

    def _build_market_info(self, state: Dict) -> Dict:
        info: Dict[str, Any] = {
            'lmp': self._last_lmp,
            'is_safe': state.get('is_safe', True),
            'safety_info': state.get('safety_info', {}),
            'reward_components': state.get('reward_components', {}),
            'cost_model': 'piecewise',
        }
        if self._last_opf_result is not None:
            info['offer_cost'] = self._last_opf_result.get('offer_cost', 0.0)
            info['true_cost'] = self._last_opf_result.get('total_cost', 0.0)
            info['opf_success'] = self._last_opf_result.get('success', False)
        return info

    def render(self) -> None:
        if self._last_state is not None:
            t = self.grid.time_step
            lmp_mean = float(np.mean(self._last_lmp)) if self._last_lmp is not None else 0.0
            soc = float(self._battery.soc) if self._battery else float('nan')
            offer_cost = (self._last_opf_result.get('offer_cost', 0.0)
                          if self._last_opf_result else 0.0)
            true_cost = (self._last_opf_result.get('total_cost', 0.0)
                         if self._last_opf_result else 0.0)
            print(f"t={t:3d} | LMP={lmp_mean:.2f} $/MWh | SOC={soc:.2%} | "
                  f"offer_cost={offer_cost:.1f} | true_cost={true_cost:.1f}")

    def close(self) -> None:
        pass

    @property
    def steps_per_day(self) -> int:
        return self.grid.steps_per_day

    @property
    def np_random(self):
        return self.grid.np_random
