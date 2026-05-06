"""GenCos Rolling Market MARL Environment.

5-agent competitive market on case5: each generator is an independent agent
that submits a 3-segment monotone offer curve every time step.

Benchmark semantics (matching powerzoojax MarketMARLEnv):
- 5 agents (genco_0..4), one per generator in case5
- Action:  Box(3) ∈ [-1, 1] — 3 markup scalars; sorted → monotone offer curve
- Obs:     12-dim private obs (own state + aggregate signal, same layout as JAX)
- Reward:  dispatch_profit = LMP[node_i] * P_i * dt - TC(P_i) * dt
- Episode: max_steps=48 (48 × 30 min rolling market)
- Ramp:    dispatch at step t constrains [p_min_rt, p_max_rt] at step t+1
           (enforced at LP level via solve_piecewise_ed_opf p_min_rt/p_max_rt)

Observation layout per agent (12 dims, matches JAX MarketMARLEnv):
    [0]  own_cost_norm        = base_seg_prices[i, 0] / lmp_scale
    [1]  own_p_max_norm       = p_max[i] / max(p_max)
    [2]  own_last_dispatch    = prev_dispatch[i] / p_max[i]
    [3]  own_last_profit      = prev_profit[i] / (lmp_scale * p_max[i] * dt)
    [4]  own_ramp_headroom    = min(p_max[i] - prev_dispatch[i], ramp_up[i]) / p_max[i]
    [5]  demand_forecast_norm = total_load at (t+1) / total_p_max
    [6]  sin(2π * t / steps_per_day)
    [7]  cos(2π * t / steps_per_day)
    [8..11] lmp_history_norm  = lmp_history / lmp_scale (oldest → newest, 4 values)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from powerzoo.envs.grid.cal_dcopf_trans import (
    make_cost_segments,
    solve_piecewise_ed_opf,
)


# ────────────────────────────────────────────────────────────────────────────
# Public factory helpers
# ────────────────────────────────────────────────────────────────────────────

def make_gencos_env(
    case=None,
    load_profiles: Optional[np.ndarray] = None,
    *,
    n_segments: int = 3,
    max_markup: float = 2.0,
    max_steps: int = 48,
    delta_t_hours: float = 0.5,
    lmp_scale: float = 100.0,
    lmp_history_len: int = 4,
    ramp_rate_fraction: float = 0.5,
    ramp_up_mw_per_step: Optional[np.ndarray] = None,
    ramp_down_mw_per_step: Optional[np.ndarray] = None,
    data_source: Optional[str] = None,
    benchmark_split: Optional[str] = None,
    ood_axis: Optional[str] = None,
    profile_window: Optional[Tuple[str, str]] = None,
) -> "GenCosMARLEnv":
    """Create a GenCosMARLEnv for case5 with synthetic flat profiles (dev/smoke).

    This is the Python counterpart of powerzoojax's
    ``_wrap_gencos_case5_dev()`` preset.  Use for local development and CI
    when real load data is not required.

    Args:
        case:        ClearCase instance.  Defaults to Case5.
        load_profiles: (T, n_nodes) array [MW].  When None, a 48-step flat
                       mid-load pool is generated automatically.
        n_segments:  Offer-curve segments per generator (default 3).
        max_markup:  Max fractional markup (offer = base * (1 + m * max_markup)).
        max_steps:   Episode length in steps (default 48).
        delta_t_hours: Time resolution [h] (default 0.5 = 30 min).
        lmp_scale:   LMP normalisation scale [$/MWh] (default 100).
        lmp_history_len: Length of mean-LMP circular buffer in obs (default 4).
        ramp_rate_fraction: Default ramp rate as fraction of p_max per step.
        ramp_up_mw_per_step: Per-unit ramp-up limit [MW/step].
        ramp_down_mw_per_step: Per-unit ramp-down limit [MW/step].

    Returns:
        GenCosMARLEnv instance.
    """
    from powerzoo.case import load_case

    if case is None:
        case = load_case(5)
    if not getattr(case, 'init_flag', False):
        case.init()

    if load_profiles is None:
        # Synthetic flat mid-load pool of 48 rows (same as JAX dev preset)
        n_nodes = len(case.nodes)
        pd = case.nodes['Pd'].values.astype(np.float32)  # (n_nodes,)
        load_profiles = np.tile(pd[None, :], (48, 1))    # (48, n_nodes)
        resolved_source = data_source or "synthetic_flat"
    else:
        resolved_source = data_source or "custom"

    return GenCosMARLEnv(
        case=case,
        load_profiles=load_profiles,
        n_segments=n_segments,
        max_markup=max_markup,
        max_steps=max_steps,
        delta_t_hours=delta_t_hours,
        lmp_scale=lmp_scale,
        lmp_history_len=lmp_history_len,
        ramp_rate_fraction=ramp_rate_fraction,
        ramp_up_mw_per_step=ramp_up_mw_per_step,
        ramp_down_mw_per_step=ramp_down_mw_per_step,
        data_source=resolved_source,
        benchmark_split=benchmark_split,
        ood_axis=ood_axis,
        profile_window=profile_window,
    )


# ────────────────────────────────────────────────────────────────────────────
# Main env class
# ────────────────────────────────────────────────────────────────────────────

class GenCosMARLEnv:
    """GenCos 5-agent rolling market MARL env (PettingZoo Parallel API style).

    Each step all agents simultaneously submit 3-segment monotone offer curves.
    The market clears via a network-constrained SCED and each agent receives
    its dispatch profit as reward.

    This env follows the PettingZoo Parallel API:

        obs, info = env.reset()
        while not all(terminateds.values()):
            actions = {ag: env.action_spaces[ag].sample() for ag in env.agents}
            obs, rewards, terminateds, truncateds, infos = env.step(actions)

    It is also compatible with the gymnasium MARL convention used internally.

    Args:
        case: ClearCase for the grid (default Case5).
        load_profiles: (T, n_nodes) nodal load array [MW].
            At each reset a random episode_start_idx ∈ [0, T-1] is sampled;
            step t accesses row (episode_start_idx + t) % T.
        n_segments: Offer-curve segments per agent (default 3).
        max_markup: Maximum fractional markup (default 2.0).
        max_steps: Episode length (default 48 = 48 × 30 min).
        delta_t_hours: Time resolution in hours (default 0.5).
        lmp_scale: LMP normalisation [$/MWh] (default 100).
        lmp_history_len: Circular LMP history buffer length (default 4).
        ramp_rate_fraction: Default ramp as fraction of p_max per step (0.5).
        ramp_up_mw_per_step: Per-unit ramp-up [MW/step]; None → fraction × p_max.
        ramp_down_mw_per_step: Per-unit ramp-down [MW/step]; None → fraction × p_max.
    """

    metadata: Dict[str, Any] = {}

    def constraint_names(self) -> tuple[str, ...]:
        """Return the fixed benchmark cost-channel order."""
        return ('thermal_overload',)

    def __init__(
        self,
        case,
        load_profiles: np.ndarray,
        *,
        n_segments: int = 3,
        max_markup: float = 2.0,
        max_steps: int = 48,
        delta_t_hours: float = 0.5,
        lmp_scale: float = 100.0,
        lmp_history_len: int = 4,
        ramp_rate_fraction: float = 0.5,
        ramp_up_mw_per_step: Optional[np.ndarray] = None,
        ramp_down_mw_per_step: Optional[np.ndarray] = None,
        data_source: str = "custom",
        benchmark_split: Optional[str] = None,
        ood_axis: Optional[str] = None,
        profile_window: Optional[Tuple[str, str]] = None,
    ):
        if not getattr(case, 'init_flag', False):
            case.init()
        self.case = case
        self.load_profiles = np.asarray(load_profiles, dtype=np.float32)  # (T, n_nodes)
        self.data_source = str(data_source)
        self.load_profile_source = self.data_source
        self.benchmark_split = benchmark_split
        self.ood_axis = ood_axis
        self.profile_window = tuple(profile_window) if profile_window is not None else None
        self.n_segments = n_segments
        self.max_markup = max_markup
        self.max_steps = max_steps
        self.delta_t_hours = delta_t_hours
        self.lmp_scale = lmp_scale
        self.lmp_history_len = lmp_history_len

        n_units = len(case.units)
        self._n_units = n_units
        self._n_nodes = len(case.nodes)

        # Agent metadata
        self._agent_names: List[str] = [f"genco_{i}" for i in range(n_units)]
        self.possible_agents: List[str] = list(self._agent_names)
        self.agents: List[str] = list(self._agent_names)

        # Unit cost parameters
        self._mc_a = case.units['mc_a'].values.astype(np.float32)
        self._mc_b = case.units['mc_b'].values.astype(np.float32)
        self._mc_c = case.units['mc_c'].values.astype(np.float32)
        self._p_min = case.units['p_min'].values.astype(np.float32)
        self._p_max = case.units['p_max'].values.astype(np.float32)
        self._total_p_max = float(self._p_max.sum()) + 1e-6

        # Unit → node index (0-based)
        bus_ids = case.units['bus_id'].values.astype(int) - 1  # convert 1-indexed → 0-indexed
        self._unit_node_idx = bus_ids.astype(int)  # (n_units,)

        # Ramp parameters
        if ramp_up_mw_per_step is None:
            self._ramp_up = (ramp_rate_fraction * self._p_max).astype(np.float32)
        else:
            self._ramp_up = np.asarray(ramp_up_mw_per_step, dtype=np.float32)
        if ramp_down_mw_per_step is None:
            self._ramp_down = (ramp_rate_fraction * self._p_max).astype(np.float32)
        else:
            self._ramp_down = np.asarray(ramp_down_mw_per_step, dtype=np.float32)

        # Base offer segments (cost-derived, agent markup applied at each step)
        self._base_segments = make_cost_segments(case, n_segments)

        # Obs / action dims
        self._obs_dim = 8 + lmp_history_len   # 12 for default lmp_history_len=4

        obs_low  = np.full(self._obs_dim, -np.inf, dtype=np.float32)
        obs_high = np.full(self._obs_dim, +np.inf, dtype=np.float32)
        act_low  = np.full(n_segments, -1.0, dtype=np.float32)
        act_high = np.full(n_segments, +1.0, dtype=np.float32)

        self.observation_spaces: Dict[str, spaces.Box] = {
            ag: spaces.Box(low=obs_low, high=obs_high, shape=(self._obs_dim,), dtype=np.float32)
            for ag in self._agent_names
        }
        self.action_spaces: Dict[str, spaces.Box] = {
            ag: spaces.Box(low=act_low, high=act_high, shape=(n_segments,), dtype=np.float32)
            for ag in self._agent_names
        }

        # Internal state (populated by reset)
        self._time_step: int = 0
        self._episode_start_idx: int = 0
        self._prev_dispatch: np.ndarray = np.zeros(n_units, dtype=np.float32)
        self._prev_profit: np.ndarray = np.zeros(n_units, dtype=np.float32)
        self._lmp_history: deque = deque([0.0] * lmp_history_len, maxlen=lmp_history_len)
        self._rng: np.random.Generator = np.random.default_rng()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def num_agents(self) -> int:
        return self._n_units

    @property
    def agent_names(self) -> List[str]:
        return self._agent_names

    def observation_space(self, agent: str = None) -> spaces.Box:
        return self.observation_spaces[agent or self._agent_names[0]]

    def action_space(self, agent: str = None) -> spaces.Box:
        return self.action_spaces[agent or self._agent_names[0]]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset to a new episode.

        Samples a random episode_start_idx from [0, T-1] so different episodes
        cover different 48-step windows of the load profile pool.

        Returns:
            obs_dict, info_dict (both keyed by agent name).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        T = self.load_profiles.shape[0]
        self._episode_start_idx = int(self._rng.integers(0, T))
        self._time_step = 0

        # Run initial SCED with truthful offers to set prev_dispatch
        node_loads = self._get_node_loads(0)
        result = solve_piecewise_ed_opf(
            self.case, node_loads, self._base_segments,
        )
        if result['success']:
            self._prev_dispatch = result['unit_power_mw'].astype(np.float32)
            mean_lmp = float(np.mean(result['lmp']))
        else:
            self._prev_dispatch = self._p_min.copy()
            mean_lmp = 0.0

        self._prev_profit = np.zeros(self._n_units, dtype=np.float32)
        self._lmp_history = deque([mean_lmp] * self.lmp_history_len, maxlen=self.lmp_history_len)

        self.agents = list(self._agent_names)
        obs = self._build_obs()
        info = {ag: {} for ag in self._agent_names}
        return obs, info

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        """Step the market: all agents bid simultaneously, SCED clears.

        Action mapping:
            a ∈ [-1, 1]^n_segments  →  m = (a + 1) / 2 ∈ [0, 1]
            sorted_m = sort(m)  (enforce monotonicity)
            offer_prices[k] = base_seg_prices[k] * (1 + sorted_m[k] * max_markup)

        Ramp coupling:
            p_min_rt[i] = max(p_min[i], prev_dispatch[i] - ramp_down[i])
            p_max_rt[i] = min(p_max[i], prev_dispatch[i] + ramp_up[i])

        Reward:
            dispatch_profit[i] = LMP[node_i] * P_i * dt - TC(P_i) * dt
            TC(P) = (mc_a/3) P^3 + (mc_b/2) P^2 + mc_c * P

        Returns:
            (obs, rewards, terminateds, truncateds, infos)
        """
        # 1. Build offer prices from actions
        offer_prices = self._base_segments['seg_prices'].copy()  # (n_u, n_seg)
        for i, ag in enumerate(self._agent_names):
            a = np.clip(np.asarray(actions[ag], dtype=np.float32), -1.0, 1.0)
            m = (a + 1.0) / 2.0                    # ∈ [0, 1]
            sorted_m = np.sort(m)                  # enforce monotone curve
            offer_prices[i] = (
                self._base_segments['seg_prices'][i]
                * (1.0 + sorted_m * self.max_markup)
            )

        current_segments = {
            'seg_widths': self._base_segments['seg_widths'].copy(),
            'seg_prices': offer_prices,
        }

        # 2. Ramp-bounded dispatch limits
        p_min_rt = np.maximum(self._p_min, self._prev_dispatch - self._ramp_down)
        p_max_rt = np.minimum(self._p_max, self._prev_dispatch + self._ramp_up)

        # 3. Node loads at current step
        node_loads = self._get_node_loads(self._time_step)

        # 4. Network-constrained SCED
        result = solve_piecewise_ed_opf(
            self.case, node_loads, current_segments,
            p_min_rt=p_min_rt, p_max_rt=p_max_rt,
        )

        if result['success']:
            dispatch = result['unit_power_mw'].astype(np.float32)
            lmp = result['lmp'].astype(np.float32)
        else:
            dispatch = self._prev_dispatch.copy()
            lmp = np.zeros(self._n_nodes, dtype=np.float32)

        # 5. Dispatch profit reward
        TC = (
            (self._mc_a / 3.0) * dispatch ** 3
            + (self._mc_b / 2.0) * dispatch ** 2
            + self._mc_c * dispatch
        )
        profit = (
            lmp[self._unit_node_idx] * dispatch * self.delta_t_hours
            - TC * self.delta_t_hours
        )

        # 6. State update
        self._prev_dispatch = dispatch
        self._prev_profit = profit
        mean_lmp = float(np.mean(lmp))
        self._lmp_history.append(mean_lmp)
        self._time_step += 1

        # 7. Done / truncation
        done = self._time_step >= self.max_steps
        terminateds = {ag: False for ag in self._agent_names}
        terminateds['__all__'] = False
        truncateds = {ag: done for ag in self._agent_names}
        truncateds['__all__'] = done

        # 8. Ramp binding rate — use p_min_rt/p_max_rt from step 2 (computed from
        #    prev_dispatch BEFORE state update), matching JAX market_marl_core
        ramp_binding = float(np.mean((dispatch >= p_max_rt - 0.5) | (dispatch <= p_min_rt + 0.5)))

        # 9. Build obs and info
        obs = self._build_obs()
        shared_info = {
            'lmp': lmp,
            'unit_power': dispatch,
            'gen_cost': float(np.sum(TC)),
            'ramp_binding_rate': ramp_binding,
            'sced_success': result['success'],
            'cost_thermal_overload': float(
                np.sum(np.maximum(0.0, np.abs(result.get('line_flow_mw', np.zeros(1))) -
                                  self.case.lines['cap'].values))
                if result['success'] else 0.0
            ),
        }
        shared_info['constraint_names'] = self.constraint_names()
        shared_info['constraint_costs'] = np.asarray(
            [shared_info['cost_thermal_overload']],
            dtype=np.float32,
        )
        shared_info['cost_sum'] = float(shared_info['constraint_costs'].sum())
        shared_info['cost'] = shared_info['cost_sum']
        shared_info['costs'] = {
            'thermal_overload': shared_info['cost_thermal_overload'],
        }
        rewards = {ag: float(profit[i]) for i, ag in enumerate(self._agent_names)}
        infos = {ag: shared_info for ag in self._agent_names}

        if done:
            self.agents = []

        return obs, rewards, terminateds, truncateds, infos

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_node_loads(self, within_episode_t: int) -> np.ndarray:
        """Return nodal load [MW] for step t = episode_start_idx + within_episode_t."""
        T = self.load_profiles.shape[0]
        row = (self._episode_start_idx + within_episode_t) % T
        return self.load_profiles[row].astype(np.float32)

    def _build_obs(self) -> Dict[str, np.ndarray]:
        """Build per-agent 12-dim private observation."""
        lmp_scale = self.lmp_scale
        pmax_norm = float(np.max(self._p_max)) + 1e-6
        dt = self.delta_t_hours
        t = float(self._time_step)
        spd = float(self.max_steps)  # treat episode length as steps_per_day

        # One-step-ahead demand forecast (obs[5])
        T = self.load_profiles.shape[0]
        next_row = (self._episode_start_idx + self._time_step + 1) % T
        total_load_forecast = float(self.load_profiles[next_row].sum())
        demand_forecast_norm = total_load_forecast / self._total_p_max

        t_sin = float(np.sin(2.0 * np.pi * t / spd))
        t_cos = float(np.cos(2.0 * np.pi * t / spd))

        # LMP history (4 values, oldest→newest), normalised
        lmp_hist_norm = np.array(list(self._lmp_history), dtype=np.float32) / lmp_scale

        obs_dict: Dict[str, np.ndarray] = {}
        for i, ag in enumerate(self._agent_names):
            pmax_i = float(self._p_max[i]) + 1e-6
            # Base offer price proxy: first segment price normalised by lmp_scale
            base_price_i = float(self._base_segments['seg_prices'][i, 0])
            disp_i = float(self._prev_dispatch[i])
            profit_i = float(self._prev_profit[i])
            profit_scale = lmp_scale * pmax_i * dt + 1e-6
            ramp_up_i = float(self._ramp_up[i])
            headroom = min(pmax_i - disp_i, ramp_up_i)
            headroom_norm = headroom / pmax_i

            obs_scalar = np.array([
                base_price_i / lmp_scale,                    # [0] own_cost_norm
                pmax_i / pmax_norm,                          # [1] own_p_max_norm
                disp_i / pmax_i,                             # [2] own_last_dispatch
                profit_i / profit_scale,                     # [3] own_last_profit
                max(0.0, headroom_norm),                     # [4] own_ramp_headroom
                demand_forecast_norm,                        # [5] demand_forecast
                t_sin,                                       # [6]
                t_cos,                                       # [7]
            ], dtype=np.float32)

            obs_dict[ag] = np.concatenate([obs_scalar, lmp_hist_norm])  # (12,)

        return obs_dict
