"""PettingZoo wrappers for PowerZoo environments.

Converts a PowerZoo ``GridEnv`` into a PettingZoo ``ParallelEnv`` where each
agent controls one entity (generator unit or DER resource).

Supported ``agent_type`` values
--------------------------------
``'generators'``
    One agent per thermal/hydro generator unit.  The action space for each
    agent is ``Box(p_min_i, p_max_i, (1,))``.  Observations include shared
    global info (line flows, loads) plus agent-local info (unit parameters).

``'resources'``
    One agent per registered DER resource (batteries, EVs).  Each resource
    must already be attached to the grid before wrapping.

Example::

    from powerzoo.envs.grid.trans import TransGridEnv
    from powerzoo.envs.resource.battery import BatteryEnv
    from powerzoo.wrappers import MARLWrapper

    grid = TransGridEnv()
    bat0 = BatteryEnv(parent=grid, bus_id=2, capacity_mwh=100, power_mw=50)
    bat1 = BatteryEnv(parent=grid, bus_id=4, capacity_mwh=80,  power_mw=40)

    env = MARLWrapper(grid, agent_type='resources')

    obs, infos = env.reset(seed=0)
    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from powerzoo.tasks.interfaces import TaskPettingZooWrapper

import gymnasium as gym
from gymnasium import spaces

try:
    from pettingzoo import ParallelEnv
    _HAS_PETTINGZOO = True
except ModuleNotFoundError:
    # Provide a minimal stub so the module can be imported without pettingzoo
    class ParallelEnv:  # type: ignore[no-redef]
        pass
    _HAS_PETTINGZOO = False


class MARLWrapper(ParallelEnv):
    """PettingZoo Parallel wrapper for PowerZoo GridEnv.

    Parameters
    ----------
    env : GridEnv
        An initialised (and optionally resource-populated) PowerZoo grid env.
    agent_type : str
        ``'generators'`` or ``'resources'``.
    render_mode : str or None
        Passed through to ``env.render_mode`` if set.
    """

    metadata = {"render_modes": ["human"], "name": "powerzoo_marl_v0"}

    def __init__(self, env, agent_type: str = 'generators',
                 render_mode: Optional[str] = None):
        if not _HAS_PETTINGZOO:
            raise ImportError(
                "pettingzoo is required for MARLWrapper.  "
                "Install it with: pip install pettingzoo"
            )
        super().__init__()
        if agent_type not in ('generators', 'resources'):
            raise ValueError(
                f"agent_type must be 'generators' or 'resources', got '{agent_type}'"
            )

        self.grid = env
        self.agent_type = agent_type
        self.render_mode = render_mode

        self._setup_agents()
        self._step_count = 0

    # ------------------------------------------------------------------
    # Agent setup
    # ------------------------------------------------------------------

    def _setup_agents(self) -> None:
        case = self.grid.case
        n_lines = len(case.lines)
        n_loads = len(case.loads) if hasattr(case, 'loads') else len(case.nodes)

        # Global obs components shared by all agents
        self._global_obs_dim = n_lines + n_loads + 2  # flows + loads + time

        if self.agent_type == 'generators':
            n_units = len(case.units)
            self.possible_agents = [f"unit_{i}" for i in range(n_units)]
            self._agent_idx = {a: i for i, a in enumerate(self.possible_agents)}

            p_min = case.units['p_min'].values.astype(np.float32)
            p_max = case.units['p_max'].values.astype(np.float32)

            # Local obs per generator: [unit_idx_norm, p_min_norm, p_max_norm,
            #                           mc_a, mc_b_norm, mc_c_norm]
            local_dim = 6
            p_max_sum = float(p_max.sum()) or 1.0

            self._observation_spaces = {
                agent: spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self._global_obs_dim + local_dim,),
                    dtype=np.float32,
                )
                for agent in self.possible_agents
            }
            self._action_spaces = {
                f"unit_{i}": spaces.Box(
                    low=np.array([p_min[i]], dtype=np.float32),
                    high=np.array([p_max[i]], dtype=np.float32),
                    shape=(1,), dtype=np.float32,
                )
                for i in range(n_units)
            }
            self._p_min = p_min
            self._p_max = p_max
            self._p_max_sum = p_max_sum

            # Cost coefficients
            mc_a = case.units['mc_a'].values if 'mc_a' in case.units.columns else np.zeros(n_units)
            mc_b = case.units['mc_b'].values if 'mc_b' in case.units.columns else np.ones(n_units)
            mc_c = case.units['mc_c'].values if 'mc_c' in case.units.columns else np.zeros(n_units)
            self._mc_a = mc_a.astype(np.float32)
            self._mc_b = mc_b.astype(np.float32)
            self._mc_c = mc_c.astype(np.float32)

        else:  # 'resources'
            self._p_max_sum = 1.0  # default for load normalization in _global_obs

            resources = self.grid.sub_resources
            self.possible_agents = list(resources.keys())
            self._agent_idx = {a: i for i, a in enumerate(self.possible_agents)}

            self._observation_spaces = {}
            self._action_spaces = {}
            for res_id, res in resources.items():
                # Use resource's own spaces if available
                res_obs_space = getattr(res, 'observation_space', None)
                res_act_space = getattr(res, 'action_space', None)

                if res_obs_space is not None:
                    # Concatenate global grid summary + local resource obs
                    global_dim = self._global_obs_dim
                    local_dim = int(np.prod(res_obs_space.shape))
                    self._observation_spaces[res_id] = spaces.Box(
                        low=-np.inf, high=np.inf,
                        shape=(global_dim + local_dim,), dtype=np.float32,
                    )
                else:
                    self._observation_spaces[res_id] = spaces.Box(
                        low=-np.inf, high=np.inf,
                        shape=(self._global_obs_dim + 4,), dtype=np.float32,
                    )

                if res_act_space is not None:
                    self._action_spaces[res_id] = res_act_space
                else:
                    power_mw = getattr(res, 'power_mw', 20.0)
                    self._action_spaces[res_id] = spaces.Box(
                        low=np.array([-power_mw], dtype=np.float32),
                        high=np.array([power_mw], dtype=np.float32),
                        shape=(1,), dtype=np.float32,
                    )

        self.agents: List[str] = []
        self._current_state: Dict = {}

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_spaces[agent]

    @lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return self._action_spaces[agent]

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        state, info = self.grid.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self._step_count = 0
        self._current_state = state
        obs = self._build_observations()
        return obs, {agent: dict(info) for agent in self.agents}

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        if self.agent_type == 'generators':
            unit_power_mw = np.zeros(len(self.possible_agents), dtype=np.float32)
            for agent, action in actions.items():
                idx = self._agent_idx[agent]
                unit_power_mw[idx] = float(np.atleast_1d(action)[0])
            grid_action = {'unit_power_mw': unit_power_mw}
        else:
            grid_action = {}
            for res_id, action in actions.items():
                power = float(np.atleast_1d(action)[0])
                grid_action[res_id] = {'p_mw': power}

        state, reward, terminated, truncated, info = self.grid.step(grid_action)
        self._current_state = state
        self._step_count += 1

        obs = self._build_observations()
        rewards = {agent: float(reward) for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: info for agent in self.agents}

        if terminated or truncated:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _global_obs(self) -> np.ndarray:
        """Shared global observation vector for all agents."""
        parts = []
        state = self._current_state

        # 1. Line flows (normalised by capacity)
        lines = state.get('lines', None)
        n_lines = len(self.grid.case.lines)
        if lines is not None and 'line_flow_mw' in lines.columns:
            caps = self.grid.case.lines['cap'].values.astype(np.float32)
            caps = np.where(caps > 0, caps, 1.0)
            parts.append((lines['line_flow_mw'].values / caps).astype(np.float32))
        else:
            parts.append(np.zeros(n_lines, dtype=np.float32))

        # 2. Node loads (normalised by total p_max)
        loads = self.grid._get_node_loads_p_current().astype(np.float32)
        parts.append(loads / self._p_max_sum)

        # 3. Time encoding
        phase = 2.0 * np.pi * self.grid.time_step / max(self.grid.steps_per_day, 1)
        parts.append(np.array([np.sin(phase), np.cos(phase)], dtype=np.float32))

        return np.concatenate(parts)

    def _build_observations(self) -> Dict[str, np.ndarray]:
        global_obs = self._global_obs()
        observations = {}

        if self.agent_type == 'generators':
            for i, agent in enumerate(self.agents):
                local = np.array([
                    float(i) / len(self.possible_agents),
                    float(self._p_min[i]) / (self._p_max_sum or 1.0),
                    float(self._p_max[i]) / (self._p_max_sum or 1.0),
                    float(self._mc_a[i]),
                    float(self._mc_b[i]) / 100.0,
                    float(self._mc_c[i]) / 1000.0,
                ], dtype=np.float32)
                observations[agent] = np.concatenate([global_obs, local])
        else:
            for res_id in self.agents:
                res = self.grid.sub_resources[res_id]
                local = self._resource_obs(res)
                observations[res_id] = np.concatenate([global_obs, local])

        return observations

    @staticmethod
    def _resource_obs(res) -> np.ndarray:
        """Extract a local observation vector from a resource.

        Tries ``res.obs(state=None)`` first (works for BatteryEnv which
        accepts an optional state).  Falls back to manual attribute
        extraction for resources whose ``obs()`` requires positional args
        (e.g. VehicleEnv inherits ``BaseEnv.obs(state)``).
        """
        if hasattr(res, 'obs'):
            sig = inspect.signature(res.obs)
            params = [
                p for p in sig.parameters.values()
                if p.name != 'self'
                and p.default is inspect.Parameter.empty
            ]
            if not params:
                # obs() accepts no required arguments
                result = res.obs()
                if isinstance(result, dict):
                    return np.array(list(result.values()), dtype=np.float32)
                if isinstance(result, np.ndarray):
                    return result

        # Fallback: build a 4-element vector from common resource attributes
        soc = getattr(res, 'soc', 0.5)
        p = getattr(res, 'current_p_mw', 0.0)
        power_mw = getattr(res, 'power_mw', None)
        if power_mw is None:
            # VehicleEnv uses kW-scale attributes
            power_mw = getattr(res, 'p_charge_max_kW', 1.0)
        is_home = float(getattr(res, 'is_home', 1.0))
        return np.array([soc, p / (power_mw or 1.0), is_home, 0.0],
                        dtype=np.float32)

    def render(self) -> None:
        if self.render_mode == 'human':
            print(f"Step {self._step_count} | "
                  f"safe={self._current_state.get('is_safe', '?')}")

    def close(self) -> None:
        if hasattr(self.grid, 'close'):
            self.grid.close()

    def state(self) -> np.ndarray:
        return self._global_obs()
