"""PowerZoo Task Base Class

A Task is a predefined experiment setup composed from PowerEnv pieces.
It only defines the problem setup (environment + reward), not algorithm settings.

Task categories:
- simple: small grids, few resources
- middle: medium grids, multi-resource coordination
- complex: large grids, market coupling, extreme events

Each Task must specify:
1. scenario_config: scenario definition
2. agent_mode: agent mode ('single' or 'multi')
3. agents_config: agent definition (for multi-agent tasks)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Type, Union
import numpy as np

# Import type hints
try:
    from gymnasium import Env
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:
    Env = object
    MultiAgentEnv = object


@dataclass(frozen=True)
class ConstraintSpec:
    """Frozen CMDP task spec: selected channels, budgets, and fallback weights."""

    selected_names: Tuple[str, ...]
    thresholds: Tuple[float, ...]
    fallback_weights: Tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.selected_names)
        if len(self.thresholds) != n:
            raise ValueError(
                "ConstraintSpec.thresholds length must match selected_names. "
                f"Got {len(self.thresholds)} vs {n}."
            )
        if len(self.fallback_weights) != n:
            raise ValueError(
                "ConstraintSpec.fallback_weights length must match selected_names. "
                f"Got {len(self.fallback_weights)} vs {n}."
            )

    @property
    def scalar_threshold(self) -> Optional[float]:
        """Backward-compatible scalar budget alias used by legacy wrappers."""
        if not self.thresholds:
            return None
        return float(sum(self.thresholds))


class Task(ABC):
    """Task base class defining preset experiment tasks.

    A Task packages:
    1. Scenario configuration (scenario_config)
    2. Agent mode (single/multi)
    3. Agent definition (agents_config)
    4. Reward configuration

    Not included:
    - Algorithm configuration (set in training scripts)
    - Training hyperparameters

    Constraint tightness
    --------------------
    Each task accepts a ``constraint_tightness`` parameter (``'loose'``,
    ``'standard'``, ``'strict'``) that adjusts both physical parameters
    (load ratio, voltage limits, SOC bounds) and the CMDP cost threshold.
    Subclasses populate ``_TIGHTNESS_PRESETS`` with the mapping.

    Example:
        >>> from powerzoo.tasks import make_task_env
        >>>
        >>> # Create a task environment
        >>> env = make_task_env('marl_opf')
        >>>
        >>> # Strict constraints (higher load, lower cost budget)
        >>> env = make_task_env('marl_opf', constraint_tightness='strict')
        >>>
        >>> # Use directly in RLlib
        >>> from ray.rllib.algorithms.ppo import PPOConfig
        >>> config = PPOConfig().environment(env=lambda cfg: make_task_env('marl_opf'))
    """

    # Task metadata
    name: str = "base_task"
    description: str = "Base task class"
    difficulty: str = "simple"  # simple, middle, complex
    agent_mode: str = "single"  # single, multi

    # Subclasses override this with tightness → {param: value} mappings.
    # ``cost_threshold`` remains the scalar backward-compatible alias.
    _TIGHTNESS_PRESETS: Dict[str, Dict[str, Any]] = {}

    VALID_TIGHTNESS = ('loose', 'standard', 'strict')

    def __init__(self, constraint_tightness: str = 'standard', **kwargs):
        """Initialize the task.

        Args:
            constraint_tightness: One of ``'loose'``, ``'standard'``, ``'strict'``.
                Adjusts physical constraint parameters and CMDP cost_threshold.
            **kwargs: override configuration values
        """
        if constraint_tightness not in self.VALID_TIGHTNESS:
            raise ValueError(
                f"constraint_tightness must be one of {self.VALID_TIGHTNESS}, "
                f"got '{constraint_tightness}'"
            )
        self.constraint_tightness = constraint_tightness
        self._override_config = kwargs
        self._scenario_config = None
        self._agents_config = None
        self._env = None

    @property
    def effective_cost_threshold(self) -> Optional[float]:
        """Return the CMDP cost threshold for the current tightness level.

        Falls back to the class-level ``eval_protocol['cost_threshold']`` when
        no ``_TIGHTNESS_PRESETS`` are defined for this task.
        """
        spec = self.constraint_spec()
        if spec is not None and spec.scalar_threshold is not None:
            return spec.scalar_threshold
        if self._TIGHTNESS_PRESETS and self.constraint_tightness in self._TIGHTNESS_PRESETS:
            return self._TIGHTNESS_PRESETS[self.constraint_tightness].get('cost_threshold')
        # Fallback: read from class-level eval_protocol
        proto = getattr(self.__class__, 'eval_protocol', None) or {}
        return proto.get('cost_threshold')

    @property
    def effective_cost_thresholds(self) -> Optional[Tuple[float, ...]]:
        """Return vector CMDP thresholds when the task defines a ConstraintSpec."""
        spec = self.constraint_spec()
        if spec is not None:
            return spec.thresholds
        scalar = self.effective_cost_threshold
        if scalar is None:
            return None
        return (float(scalar),)

    def constraint_spec(self) -> Optional[ConstraintSpec]:
        """Return the task-level CMDP selection spec, or ``None`` for legacy tasks."""
        return None

    def _wrap_single_agent_cmdp(self, env: Env) -> Env:
        """Attach the task-level CMDP benchmark wrapper when a spec is defined."""
        spec = self.constraint_spec()
        if spec is None:
            return env
        from powerzoo.wrappers.safe_rl_wrapper import TaskCMDPWrapper
        return TaskCMDPWrapper(env, constraint_spec=spec)

    def _tightness_param(self, key: str, default: Any = None) -> Any:
        """Convenience helper: read a tightness-level param by key."""
        if self._TIGHTNESS_PRESETS and self.constraint_tightness in self._TIGHTNESS_PRESETS:
            return self._TIGHTNESS_PRESETS[self.constraint_tightness].get(key, default)
        return default
    
    @abstractmethod
    def get_scenario_config(self) -> Dict[str, Any]:
        """Return scenario configuration.

        Returns:
            Configuration dict required by PowerEnv, including:
            - name: scenario name
            - grid: grid configuration
            - resources: resource configuration list
            - reward: reward configuration
            - episode: episode configuration
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_agents_config(self) -> Dict[str, Any]:
        """Return agent configuration (for multi-agent tasks).

        Returns:
            Agent configuration dict, including:
            - agent_type: agent type ('unit', 'resource', 'custom')
            - agent_ids: list of agent IDs or generation rules
            - observation_builder: observation builder
            - action_builder: action builder
            - reward_type: reward type ('shared', 'individual')
        """
        raise NotImplementedError

    def get_reward_config(self) -> Optional[Dict[str, Any]]:
        """Return the benchmark reward configuration for this task.

        By default this reads ``scenario['reward']`` so existing tasks keep
        working. New task implementations can override this explicitly.
        """
        return self.get_scenario_config().get('reward')

    def _normalize_agents_config(self, agents_config: Dict[str, Any]) -> Dict[str, Any]:
        from powerzoo.tasks.observation import normalize_agents_observation_configs
        return normalize_agents_observation_configs(agents_config)

    def get_observation_config(self, group_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return the normalized benchmark-facing observation config.

        For grouped tasks, pass ``group_id`` to retrieve a single group's config.
        """
        agents_config = self._normalize_agents_config(self.get_agents_config())
        if 'agent_groups' in agents_config:
            groups = agents_config['agent_groups']
            if group_id is None:
                return {
                    group['group_id']: group.get('observation')
                    for group in groups
                }
            for group in groups:
                if group.get('group_id') == group_id:
                    return group.get('observation')
            raise KeyError(f"Unknown observation group_id '{group_id}'")
        return agents_config.get('observation')

    def build_reward(self):
        """Build the reward object injected into PowerEnv.

        Returns ``None`` when the task does not rely on a PowerEnv-level reward
        function and instead computes reward inside an adapter.
        """
        reward_config = self.get_reward_config()
        if reward_config is None:
            return None
        from powerzoo.tasks.rewards import get_reward_function
        return get_reward_function(reward_config)

    def create_power_env(self):
        """Create the underlying PowerEnv with task-defined reward injection."""
        from powerzoo.envs.power_env import PowerEnv
        scenario_config = self.get_scenario_config()
        reward_fn = self.build_reward()
        return PowerEnv(scenario_config, reward_fn=reward_fn)
    
    def get_config(self) -> Dict[str, Any]:
        """Return full configuration (with overrides merged).

        Returns:
            Merged configuration dict
        """
        agents_config = self._normalize_agents_config(self.get_agents_config())
        config = {
            'scenario': self.get_scenario_config(),
            'agents': agents_config,
            'meta': {
                'name': self.name,
                'description': self.description,
                'difficulty': self.difficulty,
                'agent_mode': self.agent_mode,
            }
        }
        
        # Apply overrides
        if self._override_config:
            config = self._apply_overrides(config, self._override_config)
        
        return config
    
    def _apply_overrides(self, config: Dict, overrides: Dict) -> Dict:
        """Apply overrides recursively."""
        for key, value in overrides.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key] = self._apply_overrides(config[key], value)
            else:
                config[key] = value
        return config
    
    def create_env(self) -> Union[Env, MultiAgentEnv]:
        """Create an environment instance.

        Returns:
            - Single-agent mode: Gymnasium Env
            - Multi-agent mode: RLlib MultiAgentEnv
        """
        from powerzoo.tasks.adapters import create_task_env
        return create_task_env(self)
    
    def create_single_agent_env(self) -> Env:
        """Create a single-agent environment.

        Returns:
            PowerEnv wrapped by FlattenWrapper
        """
        from powerzoo.wrappers.flatten import FlattenWrapper

        env = self.create_power_env()

        # Wrap for single-agent API
        agents_config = self.get_agents_config()
        resource_names = agents_config.get('resource_names', None)
        obs_keys = agents_config.get('obs_keys', ['grid', 'resources', 'time'])

        if resource_names:
            wrapped = FlattenWrapper(env, resource_names=resource_names, obs_keys=obs_keys)
        else:
            wrapped = FlattenWrapper(env, obs_keys=obs_keys)
        return self._wrap_single_agent_cmdp(wrapped)
    
    def create_multi_agent_env(self) -> MultiAgentEnv:
        """Create a multi-agent environment (RLlib compatible).

        Returns:
            RLlib MultiAgentEnv
        """
        from powerzoo.tasks.adapters import create_multi_agent_env
        return create_multi_agent_env(self)
    
    def info(self) -> Dict[str, Any]:
        """Return a task info summary."""
        observation_config = self.get_observation_config()
        return {
            'name': self.name,
            'description': self.description,
            'difficulty': self.difficulty,
            'agent_mode': self.agent_mode,
            'scenario_name': self.get_scenario_config().get('name', 'unknown'),
            'observation': observation_config,
            'constraint_spec': self.constraint_spec(),
        }
    
    def __repr__(self) -> str:
        return f"Task(name='{self.name}', mode='{self.agent_mode}', difficulty='{self.difficulty}')"


class SingleAgentTask(Task):
    """Base class for single-agent tasks."""
    
    agent_mode = "single"
    
    def get_agents_config(self) -> Dict[str, Any]:
        """Default single-agent configuration."""
        return {
            'agent_type': 'single',
            'obs_keys': ['grid', 'resources', 'time'],
            'resource_names': None,  # all controllable resources
        }


class MultiAgentTask(Task):
    """Base class for multi-agent tasks."""
    
    agent_mode = "multi"
    
    @abstractmethod
    def get_agents_config(self) -> Dict[str, Any]:
        """Required for multi-agent tasks."""
        raise NotImplementedError


# ── Anonymous config-driven tasks ─────────────────────────────────────────────

class ConfigTask(SingleAgentTask):
    """Anonymous single-agent task constructed from an inline config dict.

    Useful for one-off experiments or AI-generated task configs without
    requiring a registered Task subclass.

    Limitations compared to registered tasks:
    - No ``SPLIT_DATES``: the ``split`` parameter is ignored (a warning is
      emitted). Specify ``episode.start_date`` / ``episode.end_date`` in the
      config dict to control the data window.
    - No ``_TIGHTNESS_PRESETS``: ``constraint_tightness`` has no effect.
    - No ``eval_protocol``: evaluation must be configured manually.

    Example::

        config = {
            'grid': {'type': 'distribution', 'case': 'case33bw'},
            'resources': [{'type': 'battery', 'capacity_mwh': 1.0, 'charge_power_kw': 500}],
            'reward': {'type': 'lmp_arbitrage'},
            'episode': {'max_steps': 96},
        }
        task = ConfigTask(config)
        env = task.create_single_agent_env()

    """

    name = "_config_task"
    description = "Anonymous single-agent task from inline config"
    agent_mode = "single"

    def __init__(self, config: Dict[str, Any], **kwargs):
        import warnings
        if 'split' in kwargs:
            warnings.warn(
                "ConfigTask does not support 'split'; it has no SPLIT_DATES. "
                "Use episode.start_date / episode.end_date in the config dict "
                "to control the data window. The 'split' argument is ignored.",
                UserWarning,
                stacklevel=2,
            )
            kwargs.pop('split')
        # strip constraint_tightness from kwargs if present (no tightness presets)
        kwargs.pop('constraint_tightness', None)
        super().__init__(**kwargs)
        self._config = config

    @classmethod
    def from_dict(cls, config: Dict[str, Any], **kwargs) -> 'ConfigTask':
        """Convenience constructor — equivalent to ``ConfigTask(config, **kwargs)``."""
        return cls(config, **kwargs)

    def get_scenario_config(self) -> Dict[str, Any]:
        return {
            'name': self._config.get('name', 'anonymous'),
            'grid': self._config.get('grid', {}),
            'resources': self._config.get('resources', []),
            'reward': self._config.get('reward', {'type': 'zero'}),
            'episode': self._config.get('episode', {}),
        }

    def get_agents_config(self) -> Dict[str, Any]:
        base = super().get_agents_config()
        if 'agents' in self._config:
            base.update(self._config['agents'])
        return base


class ConfigMultiAgentTask(MultiAgentTask):
    """Anonymous multi-agent task constructed from an inline config dict.

    The agent type (OPF-unit, resource-based, etc.) is inferred from the
    resource list unless ``agents.agent_type`` is specified explicitly.

    Limitations:
    - No ``SPLIT_DATES``, ``_TIGHTNESS_PRESETS``, or ``eval_protocol``.
    - ``train_il()`` supports only homogeneous agent spaces by default.

    Example::

        config = {
            'grid': {'type': 'transmission', 'case': 'case5'},
            'resources': [],
            'agents': {'agent_type': 'unit', 'reward_type': 'shared'},
        }
        task = ConfigMultiAgentTask(config)
    """

    name = "_config_marl_task"
    description = "Anonymous multi-agent task from inline config"
    agent_mode = "multi"

    def __init__(self, config: Dict[str, Any], **kwargs):
        import warnings
        if 'split' in kwargs:
            warnings.warn(
                "ConfigMultiAgentTask does not support 'split'; it has no "
                "SPLIT_DATES. Use episode.start_date / episode.end_date in "
                "the config dict. The 'split' argument is ignored.",
                UserWarning,
                stacklevel=2,
            )
            kwargs.pop('split')
        kwargs.pop('constraint_tightness', None)
        super().__init__(**kwargs)
        self._config = config

    @classmethod
    def from_dict(cls, config: Dict[str, Any], **kwargs) -> 'ConfigMultiAgentTask':
        """Convenience constructor."""
        return cls(config, **kwargs)

    def _infer_agent_type(self) -> str:
        """Auto-detect the appropriate adapter from the resource list.

        Returns ``'resource'`` when controllable DER/EV resources are present,
        and ``'unit'`` (generator-per-agent OPF) otherwise.
        """
        resources = self._config.get('resources', [])
        types = {r.get('type', '') for r in resources if isinstance(r, dict)}
        controllable = types & {'battery', 'vehicle', 'flexload', 'datacenter'}
        if controllable:
            return 'resource'
        return 'unit'

    def get_scenario_config(self) -> Dict[str, Any]:
        return {
            'name': self._config.get('name', 'anonymous_marl'),
            'grid': self._config.get('grid', {}),
            'resources': self._config.get('resources', []),
            'reward': self._config.get('reward', {'type': 'zero'}),
            'episode': self._config.get('episode', {}),
        }

    def get_agents_config(self) -> Dict[str, Any]:
        agents = dict(self._config.get('agents', {}))
        if 'agent_type' not in agents:
            agents['agent_type'] = self._infer_agent_type()
        agents.setdefault('reward_type', 'shared')
        agents.setdefault('action_mode', 'score')
        # Provide a sensible default observation config when not specified,
        # so that adapters never receive None from get_observation_config().
        if 'observation' not in agents:
            agent_type = agents['agent_type']
            if agent_type == 'resource':
                agents['observation'] = {
                    'mode': 'local_plus_voltage',
                    'supported_modes': ['global', 'local', 'local_plus_voltage'],
                    'global_features': ['total_load_mw', 'voltage_summary'],
                    'local_features': ['soc', 'p_mw', 'time_features',
                                       'power_limits', 'capacity', 'local_bus_voltage'],
                }
            elif agent_type == 'unit':
                agents['observation'] = {
                    'mode': 'global',
                    'supported_modes': ['global', 'local'],
                    'global_features': ['total_load_mw', 'line_flows',
                                        'time_features'],
                    'local_features': ['bus_load', 'adjacent_line_flows',
                                       'unit_idx', 'p_min', 'p_max',
                                       'cost_coeffs'],
                }
        return agents
