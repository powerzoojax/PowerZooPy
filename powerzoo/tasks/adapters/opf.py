"""OPF Multi-Agent Environment Adapter.

Each generator is modeled as an independent agent.
"""

from typing import Dict, Any, Optional, List, Tuple, Set, TYPE_CHECKING
import numpy as np
from gymnasium import spaces

from powerzoo.tasks.adapters.common import (
    build_parallel_done_dicts,
    make_agent_info,
)
from powerzoo.tasks.observations import (
    build_opf_observation_fields,
    build_opf_observations,
)

if TYPE_CHECKING:
    from powerzoo.tasks.base import Task

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    HAS_RLLIB = True
except ImportError:
    HAS_RLLIB = False
    MultiAgentEnv = object


class TaskOPFMultiAgentEnv(MultiAgentEnv):
    """OPF multi-agent environment built from a Task.

    Converts Task config into an RLlib MultiAgentEnv.
    Each generator is modeled as an independent agent.

    Notes:
    - Score-based action allocation keeps power balance.
    - Shared reward (cooperative setting).
    - Configurable observation space.
    """

    def __init__(self, task: 'Task'):
        """Initialize the environment.

        Args:
            task: Task instance
        """
        super().__init__()

        self.task = task
        self._config = task.get_config()
        self._scenario_config = self._config['scenario']
        self._agents_config = self._config['agents']
        spec = task.constraint_spec() if hasattr(task, 'constraint_spec') else None
        self._constraint_names = (
            tuple(spec.selected_names)
            if spec is not None
            else ('thermal_overload', 'voltage_violation')
        )

        # Create the base environment directly
        self.base_env = task.create_power_env()
        self.grid = self.base_env.grid
        self.case = self.grid.case

        # Unit information
        self.n_units = len(self.case.units)
        self.units = self.case.units

        # Agent definitions (RLlib expects both agents and possible_agents)
        self.possible_agents = [f"unit_{i}" for i in range(self.n_units)]
        self.agents = self.possible_agents.copy()  # currently active agents
        self._agent_ids: Set[str] = set(self.possible_agents)

        # Unit power limits
        self.p_min = self.units['p_min'].values.astype(np.float32)
        self.p_max = self.units['p_max'].values.astype(np.float32)
        self.p_range = self.p_max - self.p_min

        # Cost coefficients
        self.mc_a = self.units['mc_a'].values.astype(np.float32) if 'mc_a' in self.units.columns else np.zeros(self.n_units, dtype=np.float32)
        self.mc_b = self.units['mc_b'].values.astype(np.float32) if 'mc_b' in self.units.columns else np.ones(self.n_units, dtype=np.float32) * 30
        self.mc_c = self.units['mc_c'].values.astype(np.float32) if 'mc_c' in self.units.columns else np.zeros(self.n_units, dtype=np.float32)

        # Grid topology
        self.n_lines = len(self.case.lines)
        self.n_nodes = len(self.case.nodes)

        # Action mode
        self._action_mode = self._agents_config.get('action_mode', 'score')  # 'score' or 'direct'
        _DEFAULT_OPF_OBS = {
            'mode': 'global',
            'supported_modes': ['global', 'local', 'local_plus_forecast'],
            'global_features': ['total_load_mw', 'line_flows', 'time_features'],
            'local_features': ['bus_load', 'adjacent_line_flows',
                               'unit_idx', 'p_min', 'p_max', 'cost_coeffs'],
            'forecast_features': [],
            'forecast_horizon_steps': 0,
        }
        self._obs_config = task.get_observation_config() or _DEFAULT_OPF_OBS
        self._obs_mode = self._obs_config['mode']
        self._forecast_horizon_steps = self._obs_config.get('forecast_horizon_steps', 0)
        self._observation_fields = self._build_observation_fields()
        self._obs_dim = len(self._observation_fields)

        # Build spaces
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True

        self.observation_space = spaces.Dict({
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        })

        if self._action_mode == 'score':
            # Score mode: each agent outputs a score in [0, 1]
            self.action_space = spaces.Dict({
                agent: spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
                for agent in self.possible_agents
            })
        else:
            # Direct mode: output power directly
            self.action_space = spaces.Dict({
                agent: spaces.Box(
                    low=np.array([self.p_min[i]], dtype=np.float32),
                    high=np.array([self.p_max[i]], dtype=np.float32),
                    shape=(1,),
                    dtype=np.float32
                )
                for i, agent in enumerate(self.possible_agents)
            })

        # State tracking
        self._current_state = None
        self._step_count = 0
        self._max_steps = self._scenario_config.get('episode', {}).get('max_steps', 48)
        # F3: track whether the last allocation was limited by renewable surplus
        self._last_allocation_mode: str = 'normal'

        # Record episode data
        self._episode_data = {
            'powers': [],
            'costs': [],
            'loads': [],
            'line_flows': [],
        }

    def get_agent_ids(self) -> Set[str]:
        return self._agent_ids

    def _build_observation_fields(self) -> Tuple[str, ...]:
        return build_opf_observation_fields(
            mode=self._obs_mode,
            n_lines=self.n_lines,
            forecast_horizon_steps=self._forecast_horizon_steps,
        )

    def get_observation_fields(self) -> Tuple[str, ...]:
        return self._observation_fields

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        obs, info = self.base_env.reset(seed=seed, options=options)

        self._step_count = 0
        self._current_state = self.grid._get_state() if hasattr(self.grid, '_get_state') else {}

        # Reset data record
        self._episode_data = {'powers': [], 'costs': [], 'loads': [], 'line_flows': []}

        observations = self._build_observations()
        infos = {agent: {} for agent in self.possible_agents}

        return observations, infos

    def step(self, action_dict: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        # Extract actions
        actions = np.zeros(self.n_units, dtype=np.float32)
        for agent, action in action_dict.items():
            idx = int(agent.split('_')[1])
            if isinstance(action, (int, float)):
                actions[idx] = float(action)
            elif isinstance(action, np.ndarray):
                actions[idx] = float(action.flatten()[0])
            else:
                actions[idx] = float(action[0])

        # IMPORTANT: Get load BEFORE stepping (action is for current load)
        # _allocate_power_by_score uses this same load to ensure balance
        current_load_mw = self._get_total_load()

        # Compute unit power
        if self._action_mode == 'score':
            unit_power_mw = self._allocate_power_by_score(actions)
        else:
            unit_power_mw = np.clip(actions, self.p_min, self.p_max)

        # Step the base environment (this advances time_step)
        grid_action = {'unit_power_mw': unit_power_mw}
        obs, base_reward, terminated, truncated, info = self.base_env.step(grid_action)

        self._step_count += 1
        self._current_state = self.grid._get_state() if hasattr(self.grid, '_get_state') else {}

        # Record episode data - use current_load_mw (BEFORE step) to match with unit_power_mw
        total_load_mw = current_load_mw  # This is the load that unit_power_mw was dispatched for
        total_cost = self._calculate_cost(unit_power_mw)
        line_flows = self._get_line_flows()

        self._episode_data['powers'].append(unit_power_mw.copy())
        self._episode_data['costs'].append(total_cost)
        self._episode_data['loads'].append(total_load_mw)
        self._episode_data['line_flows'].append(line_flows)

        # Compute reward
        reward = self._calculate_reward(unit_power_mw, total_load_mw, total_cost, info)

        # Check truncation
        if self._step_count >= self._max_steps:
            truncated = True

        # Build outputs
        observations = self._build_observations()
        rewards = {agent: reward for agent in self.possible_agents}

        terminateds, truncateds = build_parallel_done_dicts(
            self.possible_agents,
            terminated=terminated,
            truncated=truncated,
        )

        # CMDP cost signal: physical constraint violations only, separate from reward.
        _cost_thermal = float(info.get('cost_thermal_overload', 0.0))
        _cost_voltage = float(info.get('cost_voltage_violation', 0.0))
        _full_costs = {
            'thermal_overload': _cost_thermal,
            'voltage_violation': _cost_voltage,
        }
        _selected_costs = {
            name: float(_full_costs.get(name, 0.0))
            for name in self._constraint_names
        }
        _cost_scalar = float(sum(_selected_costs.values()))

        infos = {}
        for agent in self.possible_agents:
            infos[agent] = make_agent_info(
                extra={
                    'unit_power_mw': unit_power_mw,
                    'total_load_mw': total_load_mw,
                    'total_cost': total_cost,
                    'cost_per_mwh': total_cost / max(np.sum(unit_power_mw), 1.0),
                    'is_safe': info.get('is_safe', True),
                    'pf_converged': info.get('pf_converged', True),
                    'cost_exception': info.get('cost_exception', 0.0),
                    'allocation_mode': self._last_allocation_mode,
                },
                cost=_cost_scalar,
                costs=_selected_costs,
                constraint_names=self._constraint_names,
            )

        return observations, rewards, terminateds, truncateds, infos

    def _allocate_power_by_score(self, scores: np.ndarray) -> np.ndarray:
        """Allocate power by scores while keeping balance.

        net_power = total_load_mw - sum(p_min) - renewable
        allocation: p_i = p_min_i + ratio_i * net_power

        Uses iterative allocation to handle capacity limits while preserving balance.

        F3 fix: When renewables cover the entire load (net_power <= 0), all units
        are held at p_min and ``self._last_allocation_mode`` is set to
        ``'renewable_surplus'`` so agents can observe they had no effect.
        """
        total_load_mw = self._get_total_load()
        renewable_power = self._get_renewable_power()

        # Net demand = total load - min output - renewables
        net_power = total_load_mw - np.sum(self.p_min) - renewable_power

        if net_power <= 0:
            # Renewable surplus: scores have no effect — signal this to agents
            self._last_allocation_mode = 'renewable_surplus'
            return self.p_min.copy().astype(np.float32)

        self._last_allocation_mode = 'normal'

        # Softmax allocation with temperature
        scores = np.clip(scores, 0.01, 1.0)
        exp_scores = np.exp(scores * 3)  # amplify differences

        # Available capacity for each unit
        available = self.p_max - self.p_min

        # Iterative allocation to handle capacity limits
        unit_power_mw = self.p_min.copy()
        remaining_power = net_power
        active_mask = available > 0.01  # units that can still receive power

        max_iterations = 10
        for _ in range(max_iterations):
            if remaining_power < 0.01 or not np.any(active_mask):
                break

            # Compute allocation ratios for active units only
            active_scores = exp_scores * active_mask
            if np.sum(active_scores) < 1e-6:
                break

            ratios = active_scores / np.sum(active_scores)

            # Allocate remaining power
            allocation = ratios * remaining_power

            # Check for capacity overflows
            new_power = unit_power_mw + allocation
            overflow = np.maximum(0, new_power - self.p_max)

            # Apply allocation with capping
            unit_power_mw = np.minimum(new_power, self.p_max)

            # Recompute remaining power (from overflows)
            remaining_power = np.sum(overflow)

            # Update active mask: units at max capacity can't receive more
            active_mask = (self.p_max - unit_power_mw) > 0.01

        return unit_power_mw.astype(np.float32)

    def _get_total_load(self) -> float:
        """Get current total load."""
        if hasattr(self.grid, '_get_node_loads_p_current'):
            return float(np.sum(self.grid._get_node_loads_p_current()))
        return 500.0  # default fallback

    def _get_renewable_power(self) -> float:
        """Get renewable generation output."""
        renewable_power = 0.0
        if hasattr(self.base_env, 'resources'):
            for res_id, resource in self.base_env.resources.items():
                res_type = resource.__class__.__name__.lower()
                if 'solar' in res_type or 'wind' in res_type:
                    # Renewable injections are negative in the base env
                    renewable_power += abs(float(resource.current_p_mw))
        return renewable_power

    def _get_line_flows(self) -> np.ndarray:
        """Get line flows."""
        if self._current_state and 'lines' in self._current_state:
            lines = self._current_state['lines']
            if 'line_flow_mw' in lines.columns:
                return lines['line_flow_mw'].values.astype(np.float32)
        return np.zeros(self.n_lines, dtype=np.float32)

    def _calculate_cost(self, unit_power_mw: np.ndarray) -> float:
        """Compute generation cost.

        For standard quadratic cost: cost = mc_a*P² + mc_b*P + mc_c

        Special case: If mc_a and mc_b are all zeros (like Case5),
        mc_c represents marginal cost ($/MWh), so: cost = mc_c * P
        """
        # Check if mc_a and mc_b are effectively zero (standard case vs. marginal-only)
        if np.allclose(self.mc_a, 0) and np.allclose(self.mc_b, 0):
            # mc_c is marginal cost in $/MWh (like Case5)
            return float(np.sum(self.mc_c * unit_power_mw))
        else:
            # Standard quadratic cost function
            return float(np.sum(self.mc_a * unit_power_mw**2 + self.mc_b * unit_power_mw + self.mc_c))

    def _calculate_reward(self, unit_power_mw: np.ndarray, total_load_mw: float,
                          total_cost: float, info: Dict) -> float:
        """Compute reward.

        reward = -cost_per_mwh / 100 - imbalance_penalty

        Safety violations are NOT penalised here. They appear in named
        ``cost_*`` fields plus the vector ``constraint_costs`` so that the CMDP
        separation between reward (economic) and cost (physical constraints)
        is maintained.
        """
        total_gen = np.sum(unit_power_mw)

        # Cost per MWh
        cost_per_mwh = total_cost / max(total_gen, 1.0)
        reward = -cost_per_mwh / 100.0  # normalize

        # Power imbalance penalty
        if total_load_mw > 0:
            imbalance_ratio = abs(total_gen - total_load_mw) / total_load_mw
            reward -= 0.5 * imbalance_ratio

        return float(reward)

    def _build_observations(self) -> Dict[str, np.ndarray]:
        return build_opf_observations(
            grid=self.grid,
            case=self.case,
            possible_agents=self.possible_agents,
            n_units=self.n_units,
            n_lines=self.n_lines,
            p_min=self.p_min,
            p_max=self.p_max,
            mc_a=self.mc_a,
            mc_b=self.mc_b,
            mc_c=self.mc_c,
            step_count=self._step_count,
            max_steps=self._max_steps,
            obs_mode=self._obs_mode,
            forecast_horizon_steps=self._forecast_horizon_steps,
            total_load_mw=self._get_total_load(),
            line_flows=self._get_line_flows(),
        )

    def get_episode_data(self) -> Dict[str, List]:
        """Return episode data for analysis."""
        return self._episode_data
