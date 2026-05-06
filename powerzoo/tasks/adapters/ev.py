"""EV (Electric Vehicle) Multi-Agent Environment Adapter.

Each EV is an independent agent with commute constraints.
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
    build_ev_observation_fields,
    build_ev_observations,
)

if TYPE_CHECKING:
    from powerzoo.tasks.base import Task

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    HAS_RLLIB = True
except ImportError:
    HAS_RLLIB = False
    MultiAgentEnv = object


class TaskEVMultiAgentEnv(MultiAgentEnv):
    """EV-specific multi-agent environment built from a Task.

    Each EV is an independent agent with commute constraints.
    Designed for V2G/G2V tasks with safety constraints (penalty-based).

    Key features:
    - EVs can only charge when at home (is_home=True)
    - Departure SOC requirements enforced via penalties
    - Observation includes EV-specific features (is_home, departure_ready, etc.)
    """

    def __init__(self, task: 'Task'):
        super().__init__()

        self.task = task
        self._config = task.get_config()
        self._scenario_config = self._config['scenario']
        self._agents_config = self._config['agents']
        _DEFAULT_EV_OBS = {
            'mode': 'local',
            'supported_modes': ['global', 'local'],
            'global_features': ['total_load_mw'],
            'local_features': ['soc', 'p_mw', 'time_features',
                               'is_home', 'departure_ready'],
            'forecast_features': [],
            'forecast_horizon_steps': 0,
        }
        self._obs_config = task.get_observation_config() or _DEFAULT_EV_OBS
        self._obs_mode = self._obs_config['mode']
        self._forecast_horizon_steps = self._obs_config.get('forecast_horizon_steps', 0)

        self.base_env = task.create_power_env()
        self.grid = self.base_env.grid
        self._resources = {}
        self._resource_info = {}
        if hasattr(self.base_env, 'resources'):
            for res_id, resource in self.base_env.resources.items():
                metadata = self.base_env.get_resource_metadata(res_id)
                res_type = metadata.get('type', resource.__class__.__name__.lower())
                if 'vehicle' in res_type:
                    self._resources[res_id] = resource
                    self._resource_info[res_id] = {
                        'type': res_type,
                        'charge_power_kw': float(metadata.get('charge_power_kw', getattr(resource, 'p_charge_max_mw', 0.0) * 1000.0)),
                        'discharge_power_kw': float(metadata.get('discharge_power_kw', getattr(resource, 'p_discharge_max_mw', 0.0) * 1000.0)),
                        'capacity_kwh': float(metadata.get('capacity_kwh', getattr(resource, 'E_max', 0.0) * 1000.0)),
                        'soc_min': float(metadata.get('soc_min', getattr(resource, 'soc_min', 0.1))),
                        'soc_max': float(metadata.get('soc_max', getattr(resource, 'soc_max', 0.95))),
                        'soc_departure_min': float(metadata.get('soc_departure_min', getattr(resource, 'soc_departure_min', 0.8))),
                    }

        # Agent definitions
        self.possible_agents = list(self._resources.keys())
        self.agents = self.possible_agents.copy()
        self._agent_ids: Set[str] = set(self.possible_agents)

        if not self.possible_agents:
            raise ValueError("No EV resources found!")

        self.n_agents = len(self.possible_agents)

        # Reward configuration
        self._reward_type = self._agents_config.get('reward_type', 'shared')
        reward_config = self._scenario_config.get('reward', {})
        self._peak_hours = set(reward_config.get('peak_hours', [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]))
        self._off_peak_hours = set(reward_config.get('off_peak_hours', [0, 1, 2, 3, 4, 5, 6, 23]))
        self._arbitrage_weight = reward_config.get('arbitrage_weight', 1.0)
        self._soc_penalty_weight = reward_config.get('soc_penalty_weight', 0.3)
        self._departure_penalty_weight = reward_config.get('departure_penalty_weight', 2.0)
        self._home_violation_penalty = reward_config.get('home_violation_penalty', 1.0)

        self._observation_fields = self._build_observation_fields()
        self._obs_dim = len(self._observation_fields)
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True

        self.observation_space = spaces.Dict({
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        })

        # Action: charge/discharge power in kW, normalized to [-1, 1]
        self.action_space = spaces.Dict({
            agent: spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                shape=(1,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        })

        self._step_count = 0
        self._max_steps = self._scenario_config.get('episode', {}).get('max_steps', 168)

        # Episode data for analysis
        self._episode_data = {
            'powers': [],
            'socs': [],
            'rewards': [],
            'violations': [],
            'soc_violations': [],
            'departure_violations': [],
            'home_violations': [],
            'profits': [],
            'is_home': [],
            'departure_ready': [],
            'hours': [],
        }

    def get_agent_ids(self) -> Set[str]:
        return self._agent_ids

    def _build_observation_fields(self) -> Tuple[str, ...]:
        return build_ev_observation_fields(
            mode=self._obs_mode,
            forecast_horizon_steps=self._forecast_horizon_steps,
        )

    def get_observation_fields(self) -> Tuple[str, ...]:
        return self._observation_fields

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        self.base_env.reset(seed=seed, options=options)

        self._step_count = 0

        # Reset episode data
        self._episode_data = {
            'powers': [],
            'socs': [],
            'rewards': [],
            'violations': [],
            'soc_violations': [],
            'departure_violations': [],
            'home_violations': [],
            'profits': [],
            'is_home': [],
            'departure_ready': [],
            'hours': [],
        }

        observations = self._build_observations()
        infos = {agent: {'is_home': self._resources[agent].is_home} for agent in self.possible_agents}

        return observations, infos

    def step(self, action_dict: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        time_step = self.grid.time_step if hasattr(self.grid, 'time_step') else self._step_count
        delta_t_minutes = self._scenario_config['grid'].get('delta_t_minutes', 60)
        hour = int((time_step * delta_t_minutes) / 60) % 24

        base_action = {}
        requested_actions = {}
        agent_powers = {}
        agent_socs = {}
        agent_is_home = {}
        agent_departure_ready = {}
        agent_violations = {}
        agent_soc_violations = {}
        agent_departure_violations = {}
        agent_home_violations = {}
        agent_profits = {}
        rewards = {}

        for res_id, action in action_dict.items():
            ev = self._resources[res_id]
            info = self._resource_info[res_id]

            action_value = np.clip(coerce_scalar_action(action), -1.0, 1.0)
            requested_actions[res_id] = action_value

            power_kw = info['charge_power_kw']
            power_mw = action_value * power_kw / 1000.0  # Convert to MW for env

            # Get state before step
            was_home = ev.is_home
            soc_before = ev.soc

            agent_is_home[res_id] = 1.0 if was_home else 0.0
            agent_socs[res_id] = float(soc_before)
            base_action[res_id] = {'p_mw': power_mw}

        _, _, terminated, truncated, info = self.base_env.step(base_action)
        self._step_count += 1

        if self._step_count >= self._max_steps:
            truncated = True

        _grid_thermal = float(info.get('cost_thermal_overload', 0.0))
        _grid_voltage = float(info.get('cost_voltage_violation', 0.0))
        _grid_cost = float(info.get('cost_sum', _grid_thermal + _grid_voltage))

        for res_id in self.possible_agents:
            ev = self._resources[res_id]
            info = self._resource_info[res_id]
            action_value = requested_actions.get(res_id, 0.0)
            was_home = bool(agent_is_home.get(res_id, 1.0))
            soc_before = agent_socs.get(res_id, float(ev.soc))
            soc = float(ev.soc)

            # Record state after step
            agent_socs[res_id] = soc
            agent_is_home[res_id] = ev.is_home
            agent_departure_ready[res_id] = ev.check_departure_ready()
            actual_power_kw = ev.current_p_mw * 1000.0  # MW to kW
            agent_powers[res_id] = actual_power_kw

            # --- Constraint violations (CMDP cost, separated from reward) ---
            soc_viol = 0.0
            if soc < info['soc_min']:
                soc_viol += (info['soc_min'] - soc) * 20.0
            if soc > info['soc_max']:
                soc_viol += (soc - info['soc_max']) * 20.0

            dep_viol = 0.0
            if was_home and not ev.is_home and soc_before < info['soc_departure_min']:
                dep_viol = (info['soc_departure_min'] - soc_before) * 40.0

            home_viol = 0.0
            if not was_home and abs(action_value) > 0.1:
                home_viol = self._home_violation_penalty

            violation = soc_viol + dep_viol + home_viol
            agent_violations[res_id] = violation
            agent_soc_violations[res_id] = soc_viol
            agent_departure_violations[res_id] = dep_viol
            agent_home_violations[res_id] = home_viol

            # Calculate profit (arbitrage)
            if hour in self._peak_hours:
                price = 0.5   # High price (peak hours)
            elif hour in self._off_peak_hours:
                price = 0.1   # Low price (off-peak hours)
            else:
                price = 0.25  # Medium price (other hours)

            delta_t_hour = 1.0  # 1 hour per step
            profit = self._arbitrage_weight * price * actual_power_kw * delta_t_hour
            agent_profits[res_id] = profit

            # Reward = profit + positive shaping bonuses only.
            # Constraint violations go to cost signal, NOT reward (CMDP separation).
            reward = profit

            # Bonus for being departure-ready (positive shaping — not a penalty)
            if ev.check_departure_ready():
                reward += 20.0

            # Bonus for maintaining good SOC level (positive shaping)
            if info['soc_min'] <= soc <= info['soc_max']:
                reward += 2.0

            rewards[res_id] = reward

        # Shared reward if configured
        rewards = maybe_share_rewards(rewards, reward_type=self._reward_type)

        # Record episode data
        self._episode_data['powers'].append(agent_powers.copy())
        self._episode_data['socs'].append(agent_socs.copy())
        self._episode_data['rewards'].append(rewards.copy())
        self._episode_data['violations'].append(agent_violations.copy())
        self._episode_data['soc_violations'].append(agent_soc_violations.copy())
        self._episode_data['departure_violations'].append(agent_departure_violations.copy())
        self._episode_data['home_violations'].append(agent_home_violations.copy())
        self._episode_data['profits'].append(agent_profits.copy())
        self._episode_data['is_home'].append(agent_is_home.copy())
        self._episode_data['departure_ready'].append(agent_departure_ready.copy())
        self._episode_data['hours'].append(hour)

        # Build outputs
        observations = self._build_observations()

        terminateds, truncateds = build_parallel_done_dicts(
            self.possible_agents,
            terminated=terminated,
            truncated=truncated,
        )

        infos = {}
        for agent in self.possible_agents:
            agent_cost = agent_violations.get(agent, 0.0) + _grid_cost
            infos[agent] = make_agent_info(
                extra={
                    'soc': agent_socs.get(agent, 0.5),
                    'power': agent_powers.get(agent, 0.0),
                    'profit': agent_profits.get(agent, 0.0),
                    'violation': agent_violations.get(agent, 0.0),
                    'is_home': agent_is_home.get(agent, True),
                    'departure_ready': agent_departure_ready.get(agent, False),
                    'hour': hour,
                    'is_peak': hour in self._peak_hours,
                },
                cost=agent_cost,
                costs={
                    'soc': agent_soc_violations.get(agent, 0.0),
                    'departure': agent_departure_violations.get(agent, 0.0),
                    'home': agent_home_violations.get(agent, 0.0),
                    'thermal': _grid_thermal,
                    'voltage': _grid_voltage,
                },
            )

        return observations, rewards, terminateds, truncateds, infos

    def _build_observations(self) -> Dict[str, np.ndarray]:
        delta_t_minutes = int(self._scenario_config['grid'].get('delta_t_minutes', 60))
        return build_ev_observations(
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
        )

    def get_episode_data(self) -> Dict[str, List]:
        """Return episode data for analysis."""
        return self._episode_data
