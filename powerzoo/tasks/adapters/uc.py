"""Unit Commitment Multi-Agent Environment Adapter.

Extends ``TaskOPFMultiAgentEnv`` with commitment decisions (on/off),
startup/shutdown costs, ramp-rate limits, and minimum up/down-time constraints.

Agent action space
------------------
Each agent emits a **2-element** vector:
    ``[score, on_off]``

where:
    ``score``  ∈ [0, 1]  — power allocation score (same as OPF)
    ``on_off`` ∈ [0, 1]  — commitment signal; values ≥ 0.5 → commit, < 0.5 → decommit

Unit Commitment parameters (read from ``case.units`` columns when present,
otherwise assigned defaults):
    ``startup_cost``  ($/start) : one-time cost when a unit is turned on
    ``shutdown_cost`` ($/stop)  : one-time cost when a unit is turned off
    ``ramp_rate``     (MW/step) : maximum change in output per time step
    ``min_up_time``   (steps)   : minimum consecutive on steps before decommitting
    ``min_down_time`` (steps)   : minimum consecutive off steps before committing

Reward
------
``reward = -(generation_cost + startup_cost + shutdown_cost)``

Safety violations (thermal overload, voltage) appear in named cost fields and
fixed-order cost vectors, keeping the CMDP separation between reward
(economic) and cost (physical).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    HAS_RLLIB = True
except ImportError:
    HAS_RLLIB = False
    MultiAgentEnv = object

from powerzoo.tasks.adapters.opf import TaskOPFMultiAgentEnv
from powerzoo.tasks.adapters.common import build_parallel_done_dicts, make_agent_info

# Default UC parameters applied when not present in case.units
_UC_DEFAULTS = {
    'startup_cost':  500.0,   # $/start
    'shutdown_cost': 200.0,   # $/stop
    'ramp_rate':     999.0,   # MW/step — effectively no ramp limit by default
    'min_up_time':   1,       # steps
    'min_down_time': 1,       # steps
}


class TaskUCMultiAgentEnv(TaskOPFMultiAgentEnv):
    """Unit Commitment multi-agent environment.

    Inherits full OPF infrastructure (score allocation, line-flow obs, etc.)
    and adds:
    - Binary commitment decisions per agent
    - Startup/shutdown cost accounting
    - Ramp-rate feasibility clamping
    - Minimum up/down-time enforcement
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, task):
        super().__init__(task)

        # ---- UC parameters (from case or defaults) ----
        self.startup_cost  = self._load_uc_col('startup_cost')
        self.shutdown_cost = self._load_uc_col('shutdown_cost')
        self.ramp_rate     = self._load_uc_col('ramp_rate')
        self.min_up_time   = self._load_uc_col('min_up_time', dtype=int)
        self.min_down_time = self._load_uc_col('min_down_time', dtype=int)

        # ---- Commitment state ----
        self._committed: np.ndarray = np.ones(self.n_units, dtype=bool)  # all on at start
        self._time_on: np.ndarray  = np.zeros(self.n_units, dtype=int)   # steps since last commit
        self._time_off: np.ndarray = np.zeros(self.n_units, dtype=int)   # steps since last decommit
        self._prev_power: np.ndarray = self.p_min.copy()

        # ---- Override action space: [score, on_off] per agent ----
        self.action_space = spaces.Dict({
            agent: spaces.Box(
                low=np.array([0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                shape=(2,), dtype=np.float32,
            )
            for agent in self.possible_agents
        })

        # ---- Extend obs space: +n_units for commitment status ----
        self._observation_fields = tuple(self._observation_fields) + tuple(
            f'commitment_{i}' for i in range(self.n_units)
        )
        new_obs_dim = len(self._observation_fields)
        self._obs_dim = new_obs_dim
        self.observation_space = spaces.Dict({
            agent: spaces.Box(
                low=-np.inf, high=np.inf, shape=(new_obs_dim,), dtype=np.float32
            )
            for agent in self.possible_agents
        })

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        obs, infos = super().reset(seed=seed, options=options)
        self._committed[:] = True
        self._time_on[:]   = 0
        self._time_off[:]  = 0
        self._prev_power   = self.p_min.copy()
        # Re-build obs with commitment appended
        observations = self._build_observations()
        return observations, infos

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action_dict: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        # Parse score + on_off from each agent action
        scores   = np.zeros(self.n_units, dtype=np.float32)
        on_off   = np.zeros(self.n_units, dtype=np.float32)

        for agent, action in action_dict.items():
            idx = int(agent.split('_')[1])
            action_flat = np.atleast_1d(action).flatten()
            scores[idx]  = float(action_flat[0]) if len(action_flat) > 0 else 0.5
            on_off[idx]  = float(action_flat[1]) if len(action_flat) > 1 else 1.0

        # ---- Commitment decision ----
        desired_commit = on_off >= 0.5
        new_committed  = self._apply_commitment_constraints(desired_commit)

        # ---- Startup / shutdown cost ----
        startup_events  = (~self._committed) & new_committed
        shutdown_events = self._committed & (~new_committed)
        startup_total   = float(np.sum(self.startup_cost  * startup_events))
        shutdown_total  = float(np.sum(self.shutdown_cost * shutdown_events))

        # ---- Update commitment state ----
        self._committed = new_committed
        self._time_on[new_committed]   += 1
        self._time_off[~new_committed] += 1
        self._time_on[~new_committed]   = 0
        self._time_off[new_committed]   = 0

        # Force off-units to zero power; on-units use score allocation
        active_scores = scores * self._committed.astype(np.float32)
        active_p_min  = self.p_min * self._committed.astype(np.float32)
        active_p_max  = self.p_max * self._committed.astype(np.float32)

        unit_power_mw = self._allocate_uc_power(active_scores, active_p_min, active_p_max)

        # ---- Ramp-rate clamping ----
        unit_power_mw = self._apply_ramp_constraints(unit_power_mw)

        # ---- Step base env ----
        current_load_mw = self._get_total_load()
        grid_action  = {'unit_power_mw': unit_power_mw}
        _, _, terminated, truncated, info = self.base_env.step(grid_action)

        self._step_count += 1
        if hasattr(self.grid, '_get_state'):
            self._current_state = self.grid._get_state()

        total_gen_cost = self._calculate_cost(unit_power_mw)

        # ---- UC reward: economic cost only ----
        total_cost_reward = total_gen_cost + startup_total + shutdown_total
        reward = -total_cost_reward / 1000.0  # scale to ~[-1, 0]

        # ---- Record episode data ----
        self._episode_data['powers'].append(unit_power_mw.copy())
        self._episode_data['costs'].append(total_cost_reward)
        self._episode_data['loads'].append(current_load_mw)
        self._episode_data['line_flows'].append(self._get_line_flows())

        self._prev_power = unit_power_mw.copy()

        # Truncation
        if self._step_count >= self._max_steps:
            truncated = True

        observations = self._build_observations()
        rewards = {agent: reward for agent in self.possible_agents}

        terminateds, truncateds = build_parallel_done_dicts(
            self.possible_agents,
            terminated=terminated,
            truncated=truncated,
        )

        # Safety violations go to cost signal only
        safety_thermal = float(info.get('cost_thermal_overload', 0.0))
        safety_voltage = float(info.get('cost_voltage_violation', 0.0))
        full_costs = {
            'thermal_overload': safety_thermal,
            'voltage_violation': safety_voltage,
        }
        safety_costs = {
            name: float(full_costs.get(name, 0.0))
            for name in self._constraint_names
        }
        safety_cost = float(sum(safety_costs.values()))
        infos = {}
        for agent in self.possible_agents:
            infos[agent] = make_agent_info(
                extra={
                    'unit_power_mw': unit_power_mw,
                    'committed': self._committed.copy(),
                    'total_load_mw': current_load_mw,
                    'gen_cost': total_gen_cost,
                    'startup_cost': startup_total,
                    'shutdown_cost': shutdown_total,
                    'is_safe': info.get('is_safe', True),
                    'pf_converged': info.get('pf_converged', True),
                },
                cost=safety_cost,
                costs=safety_costs,
                constraint_names=self._constraint_names,
            )

        return observations, rewards, terminateds, truncateds, infos

    # ------------------------------------------------------------------
    # UC-specific helpers
    # ------------------------------------------------------------------

    def _apply_commitment_constraints(self, desired: np.ndarray) -> np.ndarray:
        """Enforce min-up-time and min-down-time constraints."""
        new_committed = desired.copy()
        for i in range(self.n_units):
            if self._committed[i]:
                # Currently on: cannot turn off before min_up_time
                if not desired[i] and self._time_on[i] < self.min_up_time[i]:
                    new_committed[i] = True  # forced to stay on
            else:
                # Currently off: cannot turn on before min_down_time
                if desired[i] and self._time_off[i] < self.min_down_time[i]:
                    new_committed[i] = False  # forced to stay off
        return new_committed

    def _apply_ramp_constraints(self, unit_power_mw: np.ndarray) -> np.ndarray:
        """Clamp each unit's power change to ±ramp_rate."""
        clamped = unit_power_mw.copy()
        mask = self._committed
        clamped[mask] = np.clip(
            clamped[mask],
            self._prev_power[mask] - self.ramp_rate[mask],
            self._prev_power[mask] + self.ramp_rate[mask],
        )
        return clamped

    def _allocate_uc_power(self, scores: np.ndarray,
                           p_min: np.ndarray, p_max: np.ndarray) -> np.ndarray:
        """Score-based allocation respecting per-unit p_min/p_max."""
        total_load_mw     = self._get_total_load()
        renewable_power = self._get_renewable_power()
        net_power       = total_load_mw - np.sum(p_min) - renewable_power

        if net_power <= 0:
            self._last_allocation_mode = 'renewable_surplus'
            return p_min.copy().astype(np.float32)

        self._last_allocation_mode = 'normal'

        scores_clipped = np.clip(scores, 0.01, 1.0)
        exp_scores     = np.exp(scores_clipped * 3)

        available   = p_max - p_min
        unit_power_mw  = p_min.copy()
        remaining   = net_power
        active_mask = available > 0.01

        for _ in range(10):
            if remaining < 0.01 or not np.any(active_mask):
                break
            active_s = exp_scores * active_mask
            if active_s.sum() < 1e-6:
                break
            ratios     = active_s / active_s.sum()
            allocation = ratios * remaining
            new_power  = unit_power_mw + allocation
            overflow   = np.maximum(0, new_power - p_max)
            unit_power_mw  = np.minimum(new_power, p_max)
            remaining   = float(np.sum(overflow))
            active_mask = (p_max - unit_power_mw) > 0.01

        return unit_power_mw.astype(np.float32)

    def _build_observations(self) -> Dict[str, np.ndarray]:
        """Append commitment vector to base observations."""
        base_obs = super()._build_observations()
        commit_vec = self._committed.astype(np.float32)
        return {agent: np.concatenate([obs, commit_vec]) for agent, obs in base_obs.items()}

    # ------------------------------------------------------------------
    # Parameter loading
    # ------------------------------------------------------------------

    def _load_uc_col(self, col: str, dtype=float) -> np.ndarray:
        """Read a UC column from case.units, falling back to defaults."""
        default = _UC_DEFAULTS[col]
        if col in self.units.columns:
            vals = self.units[col].values
        else:
            vals = np.full(self.n_units, default)
        return np.array(vals, dtype=dtype)
