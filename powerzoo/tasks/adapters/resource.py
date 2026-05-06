"""Resource (DER / Battery) Multi-Agent Environment Adapter.

Each controllable resource (battery, EV, etc.) is an independent agent.

Heterogeneous resource support
-------------------------------
``resource_filter`` in agents_config controls which resource types become
agents.  The adapter now accepts any resource type that passes the filter
(battery, solar, wind, flexload, vehicle, …).  Action spaces are derived
from each resource's own ``action_space`` so multi-dimensional actions
(e.g. [P, Q] for battery, [curtail, q] for PV, [curtail, shift_out] for
flexload) are passed through correctly without adapter-side hard-coding.

Observation note: the per-agent observation still uses the battery-centric
``build_resource_observations()`` builder (fields: soc, p_mw_norm, price,
time, …).  For non-battery resources ``soc`` falls back to 0.5 and
``target_soc`` to 0.5; these fields are semantically meaningless for
PV/FlexLoad but keep the observation vector shape uniform across all 12
agents (required for batch training).  This is a **documented residual
mismatch** — it does not affect action correctness or voltage cost signals.
"""

from typing import Dict, Any, Optional, List, Tuple, Set, TYPE_CHECKING
import numpy as np
from gymnasium import spaces

from powerzoo.tasks.adapters.common import (
    build_parallel_done_dicts,
    coerce_scalar_action,
    make_agent_info,
    maybe_share_rewards,
)
from powerzoo.tasks.observations import (
    build_resource_observation_fields,
    build_resource_observations,
    build_ders_observation_fields,
    build_ders_observations,
)
from powerzoo.tasks.observations.resource import _read_local_bus_voltage as _read_bus_v

if TYPE_CHECKING:
    from powerzoo.tasks.base import Task


def _build_resource_info(
    res_id: str,
    res_type: str,
    metadata: Dict[str, Any],
    resource: Any,
) -> Dict[str, Any]:
    """Extract normalized adapter-level info for any resource type.

    Returns a dict with at least:
        type, bus_id, power_mw, capacity_mwh, soc_min, soc_max

    ``power_mw``/``capacity_mwh``/``soc_min``/``soc_max`` are meaningful
    only for batteries; other resource types get safe placeholder values so
    the battery-centric observation builder does not crash.
    """
    bus_id = int(metadata.get('bus_id', 0))
    if 'battery' in res_type:
        power_mw = metadata.get('power_mw', getattr(resource, 'power_mw', 20.0))
        capacity_mwh = metadata.get('capacity_mwh', getattr(resource, 'capacity_mwh', 50.0))
        return {
            'type': res_type,
            'bus_id': bus_id,
            'power_mw': float(power_mw),
            'capacity_mwh': float(capacity_mwh),
            'soc_min': float(metadata.get('soc_min', getattr(resource, 'soc_min', 0.1))),
            'soc_max': float(metadata.get('soc_max', getattr(resource, 'soc_max', 0.9))),
        }
    elif 'vehicle' in res_type:
        power_kw = metadata.get('charge_power_kw')
        power_mw = float(power_kw) / 1000.0 if power_kw is not None else getattr(resource, 'power_mw', 20.0)
        capacity_kwh = metadata.get('capacity_kwh')
        capacity_mwh = float(capacity_kwh) / 1000.0 if capacity_kwh is not None else getattr(resource, 'capacity_mwh', 50.0)
        return {
            'type': res_type,
            'bus_id': bus_id,
            'power_mw': float(power_mw),
            'capacity_mwh': float(capacity_mwh),
            'soc_min': float(metadata.get('soc_min', getattr(resource, 'soc_min', 0.1))),
            'soc_max': float(metadata.get('soc_max', getattr(resource, 'soc_max', 0.9))),
        }
    elif res_type in ('solar', 'wind') or 'renewable' in res_type:
        capacity_mw = float(metadata.get('capacity_mw', getattr(resource, 'capacity_mw', 100.0)))
        return {
            'type': res_type,
            'bus_id': bus_id,
            'power_mw': capacity_mw,
            'capacity_mwh': 1.0,   # placeholder — not physically meaningful for PV/wind
            'soc_min': 0.0,
            'soc_max': 1.0,
        }
    elif 'flex' in res_type or res_type == 'flexload':
        curtail_cap = float(metadata.get('curtail_cap_mw', getattr(resource, 'curtail_cap_mw', 10.0)))
        shift_cap = float(metadata.get('shift_cap_mw', getattr(resource, 'shift_cap_mw', 10.0)))
        return {
            'type': res_type,
            'bus_id': bus_id,
            'power_mw': max(curtail_cap, shift_cap),
            'capacity_mwh': 1.0,   # placeholder — not physically meaningful for FlexLoad
            'soc_min': 0.0,
            'soc_max': 1.0,
        }
    else:
        # Generic fallback for unknown resource types
        power_mw = float(metadata.get('power_mw', getattr(resource, 'power_mw', 20.0)))
        return {
            'type': res_type,
            'bus_id': bus_id,
            'power_mw': power_mw,
            'capacity_mwh': float(metadata.get('capacity_mwh', getattr(resource, 'capacity_mwh', 50.0))),
            'soc_min': float(metadata.get('soc_min', getattr(resource, 'soc_min', 0.0))),
            'soc_max': float(metadata.get('soc_max', getattr(resource, 'soc_max', 1.0))),
        }

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    HAS_RLLIB = True
except ImportError:
    HAS_RLLIB = False
    MultiAgentEnv = object

_DEFAULT_RESOURCE_OBS_CONFIG: Dict[str, Any] = {
    'mode': 'local_plus_voltage',
    'supported_modes': ['global', 'local', 'local_plus_voltage', 'ders_local'],
    'global_features': ['total_load_mw', 'voltage_summary'],
    'local_features': ['soc', 'p_mw', 'time_features',
                       'power_limits', 'capacity', 'local_bus_voltage'],
    'forecast_features': [],
    'forecast_horizon_steps': 0,
}

_BUILTIN_ARBITRAGE_REWARD_TYPES = frozenset({
    'battery_arbitrage', 'battery_lmp_arbitrage',
    'battery_lmp_arbitrage_v2', 'ev_arbitrage',
})


class TaskResourceMultiAgentEnv(MultiAgentEnv):
    """Resource-control multi-agent environment built from a Task.

    Each controllable resource (battery, EV, etc.) is an independent agent.

    Reward routing
    --------------
    When the task's reward type is a built-in arbitrage variant (e.g.
    ``battery_arbitrage``), the adapter computes per-agent arbitrage
    profit internally.  For **any other** reward type (``voltage_control``,
    ``economic_dispatch``, a custom callable, …) the adapter forwards
    the scalar reward computed by ``PowerEnv.reward_function`` and shares
    it across all agents.  This lets users switch objectives purely via
    config without writing custom code.

    Notes:
    - Battery actions: positive = discharge, negative = charge
    - Episode data recorded for analysis
    """

    def __init__(self, task: 'Task'):
        super().__init__()

        self.task = task
        self._config = task.get_config()
        self._scenario_config = self._config['scenario']
        self._agents_config = self._config['agents']
        spec = task.constraint_spec() if hasattr(task, 'constraint_spec') else None
        self._constraint_names = (
            tuple(spec.selected_names)
            if spec is not None
            else ('resource', 'thermal_overload', 'voltage_violation')
        )

        # Create the base environment directly
        self.base_env = task.create_power_env()
        self.grid = self.base_env.grid

        # Collect controllable resources
        self._resources = {}
        self._resource_info = {}
        resource_filter = self._agents_config.get('resource_filter', ['battery'])
        self._obs_config = task.get_observation_config() or _DEFAULT_RESOURCE_OBS_CONFIG
        self._obs_mode = self._obs_config['mode']
        self._forecast_horizon_steps = self._obs_config.get('forecast_horizon_steps', 0)

        if hasattr(self.base_env, 'resources'):
            for res_id, resource in self.base_env.resources.items():
                metadata = self.base_env.get_resource_metadata(res_id)
                res_type = metadata.get('type', resource.__class__.__name__.lower())
                if resource_filter is None or any(f in res_type for f in resource_filter):
                    self._resources[res_id] = resource
                    self._resource_info[res_id] = _build_resource_info(
                        res_id, res_type, metadata, resource
                    )

        # Agent definitions (RLlib expects both agents and possible_agents)
        self.possible_agents = list(self._resources.keys())
        self.agents = self.possible_agents.copy()  # currently active agents
        self._agent_ids: Set[str] = set(self.possible_agents)

        if not self.possible_agents:
            raise ValueError("No controllable resources found!")

        self.n_agents = len(self.possible_agents)

        # Reward configuration
        self._reward_type = self._agents_config.get('reward_type', 'shared')
        reward_config = self._scenario_config.get('reward', {})
        configured_reward_type = reward_config.get('type', 'battery_arbitrage')
        self._use_env_reward = configured_reward_type not in _BUILTIN_ARBITRAGE_REWARD_TYPES

        self._peak_hours = set(reward_config.get('peak_hours', [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]))
        self._off_peak_hours = set(reward_config.get('off_peak_hours', [0, 1, 2, 3, 4, 5, 6, 23]))
        self._arbitrage_weight = reward_config.get('arbitrage_weight', 1.0)
        self._soc_penalty_weight = reward_config.get('soc_penalty_weight', 0.5)
        self._target_soc = reward_config.get('target_soc', 0.5)

        self._observation_fields = self._build_observation_fields()
        self._obs_dim = len(self._observation_fields)
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True

        self.observation_space = spaces.Dict({
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        })

        # Action space: prefer the resource's own action_space (handles 2-D actions
        # for battery Q-control, PV curtailment+Q, FlexLoad curtail+shift_out).
        # Falls back to a 1-D [-power_mw, +power_mw] box for unknown types.
        self.action_space = spaces.Dict({
            agent: (
                self._resources[agent].action_space
                if hasattr(self._resources[agent], 'action_space')
                else spaces.Box(
                    low=np.array([-self._resource_info[agent]['power_mw']], dtype=np.float32),
                    high=np.array([self._resource_info[agent]['power_mw']], dtype=np.float32),
                    shape=(1,),
                    dtype=np.float32,
                )
            )
            for agent in self.possible_agents
        })

        self._step_count = 0
        self._max_steps = self._scenario_config.get('episode', {}).get('max_steps', 48)

        # Voltage limits for per-agent info enrichment (used in step())
        _grid_cfg = self._scenario_config.get('grid', {})
        self._v_min_info = float(_grid_cfg.get('v_min', 0.94))
        self._v_max_info = float(_grid_cfg.get('v_max', 1.06))

        # Episode data for analysis
        self._episode_data = {
            'powers': [],
            'socs': [],
            'rewards': [],
            'violations': [],
            'profits': [],
            'loads': [],  # total load for plotting
            'hours': [],  # hour of day for plotting
        }

    def get_agent_ids(self) -> Set[str]:
        return self._agent_ids

    def _build_observation_fields(self) -> Tuple[str, ...]:
        if self._obs_mode == 'ders_local':
            return build_ders_observation_fields(mode=self._obs_mode)
        return build_resource_observation_fields(
            mode=self._obs_mode,
            forecast_horizon_steps=self._forecast_horizon_steps,
        )

    def get_observation_fields(self) -> Tuple[str, ...]:
        return self._observation_fields

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._step_count = 0

        # Reset episode data
        self._episode_data = {
            'powers': [],
            'socs': [],
            'rewards': [],
            'violations': [],
            'profits': [],
            'loads': [],
            'hours': [],
        }

        observations = self._build_observations()
        infos = {agent: {} for agent in self.possible_agents}

        return observations, infos

    def step(self, action_dict: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        # Get current hour for price signal
        time_step = self.grid.time_step if hasattr(self.grid, 'time_step') else self._step_count
        delta_t_minutes = self._scenario_config['grid'].get('delta_t_minutes', 60)
        hour = int((time_step * delta_t_minutes) / 60) % 24

        # Convert actions to base env format.
        # Pass action as numpy array so PowerEnv._coerce_resource_action can map
        # it to the correct dict format using resource.action_names — this handles
        # all resource types (battery [P,Q], PV [curtail,q], FlexLoad [curtail,shift]).
        base_action = {}
        agent_powers = {}

        for res_id, action in action_dict.items():
            action_arr = np.atleast_1d(np.asarray(action, dtype=np.float32)).flatten()
            # First component tracks P (battery) or curtailment (PV/FlexLoad) for info.
            agent_powers[res_id] = float(action_arr[0])
            base_action[res_id] = action_arr

        # Step the base environment
        obs, base_reward, terminated, truncated, info = self.base_env.step(base_action)
        self._step_count += 1

        if self._step_count >= self._max_steps:
            truncated = True

        # Collect per-agent SOC and constraint violations
        agent_socs = {}
        agent_violations = {}
        agent_profits = {}

        for res_id in self.possible_agents:
            resource = self._resources[res_id]
            soc = float(resource.soc) if hasattr(resource, 'soc') else 0.5
            agent_socs[res_id] = soc

            soc_min = self._resource_info[res_id]['soc_min']
            soc_max = self._resource_info[res_id]['soc_max']
            violation = 0.0
            if soc < soc_min:
                violation = soc_min - soc
            elif soc > soc_max:
                violation = soc - soc_max
            agent_violations[res_id] = violation

        # Reward routing: env reward (from config) vs built-in arbitrage
        if self._use_env_reward:
            rewards = {res_id: float(base_reward) for res_id in self.possible_agents}
            agent_profits = {res_id: float(base_reward) for res_id in self.possible_agents}
        else:
            rewards = {}
            for res_id in self.possible_agents:
                power = agent_powers.get(res_id, 0.0)
                profit = 0.0
                if hour in self._peak_hours:
                    profit = self._arbitrage_weight * power * 0.1
                elif hour in self._off_peak_hours:
                    profit = -self._arbitrage_weight * power * 0.1
                agent_profits[res_id] = profit
                rewards[res_id] = profit
            rewards = maybe_share_rewards(rewards, reward_type=self._reward_type)

        # Get total load from grid for plotting
        total_load_mw = 0.0
        if hasattr(self.grid, 'current_loads_p') and self.grid.current_loads_p is not None:
            total_load_mw = float(np.sum(self.grid.current_loads_p))
        elif hasattr(self.grid, '_get_node_loads_p_current'):
            total_load_mw = float(np.sum(self.grid._get_node_loads_p_current()))

        # Record episode data
        self._episode_data['powers'].append(agent_powers.copy())
        self._episode_data['socs'].append(agent_socs.copy())
        self._episode_data['rewards'].append(rewards.copy())
        self._episode_data['violations'].append(agent_violations.copy())
        self._episode_data['profits'].append(agent_profits.copy())
        self._episode_data['loads'].append(total_load_mw)
        self._episode_data['hours'].append(hour)

        # Build outputs
        observations = self._build_observations()

        terminateds, truncateds = build_parallel_done_dicts(
            self.possible_agents,
            terminated=terminated,
            truncated=truncated,
        )

        # Grid-level cost (voltage / thermal violations from distribution grid)
        _grid_thermal = float(info.get('cost_thermal_overload', 0.0))
        _grid_voltage = float(info.get('cost_voltage_violation', 0.0))

        infos = {}
        for agent in self.possible_agents:
            resource = self._resources[agent]
            res_info = self._resource_info[agent]
            res_type = str(res_info.get('type', 'unknown'))
            bus_id = int(res_info.get('bus_id', 0))
            v_local = _read_bus_v(self.grid, bus_id)
            v_viol = max(0.0, self._v_min_info - v_local) + max(0.0, v_local - self._v_max_info)
            selected_costs = {
                name: float({
                    'resource': agent_violations.get(agent, 0.0),
                    'thermal_overload': _grid_thermal,
                    'voltage_violation': _grid_voltage,
                }.get(name, 0.0))
                for name in self._constraint_names
            }
            infos[agent] = make_agent_info(
                extra={
                    'soc': agent_socs.get(agent, 0.5),
                    'power': agent_powers.get(agent, 0.0),
                    'profit': agent_profits.get(agent, 0.0),
                    'violation': agent_violations.get(agent, 0.0),
                    'hour': hour,
                    'is_peak': hour in self._peak_hours,
                    # benchmark-aligned keys (Phase 3)
                    'current_p_mw': float(getattr(resource, 'current_p_mw', 0.0)),
                    'current_q_mvar': float(getattr(resource, 'current_q_mvar', 0.0)),
                    'type_state': res_type,
                    'voltage_violation': v_viol,
                    'cost_voltage_violation': _grid_voltage,
                },
                cost=float(sum(selected_costs.values())),
                costs=selected_costs,
                constraint_names=self._constraint_names,
            )

        return observations, rewards, terminateds, truncateds, infos

    def _build_observations(self) -> Dict[str, np.ndarray]:
        delta_t_minutes = int(self._scenario_config['grid'].get('delta_t_minutes', 60))
        if self._obs_mode == 'ders_local':
            return build_ders_observations(
                grid=self.grid,
                step_count=self._step_count,
                max_steps=self._max_steps,
                resources=self._resources,
                resource_info=self._resource_info,
                delta_t_minutes=delta_t_minutes,
                peak_hours=self._peak_hours,
                off_peak_hours=self._off_peak_hours,
            )
        return build_resource_observations(
            grid=self.grid,
            step_count=self._step_count,
            max_steps=self._max_steps,
            resources=self._resources,
            resource_info=self._resource_info,
            obs_mode=self._obs_mode,
            forecast_horizon_steps=self._forecast_horizon_steps,
            delta_t_minutes=delta_t_minutes,
            peak_hours=self._peak_hours,
            off_peak_hours=self._off_peak_hours,
            target_soc=self._target_soc,
        )

    def get_episode_data(self) -> Dict[str, List]:
        """Return episode data for analysis."""
        return self._episode_data
