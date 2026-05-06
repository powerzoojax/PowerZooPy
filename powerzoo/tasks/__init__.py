"""PowerZoo Tasks Module

Tasks are predefined experiment setups built on top of PowerEnv.

Main API:
- make_task_env(): create a task environment (most common)
- make_task(): create a Task instance
- list_tasks(): list available tasks
- register_task(): register a custom task

Task categories:
- simple: small grids, few agents
- middle: medium grids, multi-resource coordination
- complex: large grids, market coupling

Example:
    >>> from powerzoo.tasks import make_task_env, list_tasks
    >>>
    >>> # Inspect available tasks
    >>> print(list_tasks())  # ['marl_opf', ...]
    >>>
    >>> # Create a multi-agent OPF environment
    >>> env = make_task_env('marl_opf')
    >>>
    >>> # Train with RLlib
    >>> from ray.rllib.algorithms.ppo import PPOConfig
    >>> from ray.tune.registry import register_env
    >>>
    >>> register_env('opf', lambda cfg: make_task_env('marl_opf', **cfg))
    >>> config = PPOConfig().environment(env='opf', env_config={'max_steps': 48})
"""

# Base classes
from powerzoo.tasks.base import (
    ConstraintSpec,
    Task,
    SingleAgentTask,
    MultiAgentTask,
    ConfigTask,
    ConfigMultiAgentTask,
)
from powerzoo.tasks.audit import (
    audit_env,
    audit_public_tasks,
    audit_task_collection,
    audit_task_env,
)
from powerzoo.tasks.observation import OBSERVATION_MODES, make_observation_config
from powerzoo.tasks.public import (
    PUBLIC_TASKS,
    get_public_task_catalog,
    get_public_task_info,
    list_public_tasks,
)

# Registry and factories
from powerzoo.tasks.registry import (
    register_task,
    make_task_env,
    make_task,
    get_task_class,
    list_tasks,
    get_task_info,
)

# Adapters (advanced usage)
from powerzoo.tasks.adapters import (
    create_task_env,
    create_multi_agent_env,
    TaskOPFMultiAgentEnv,
    TaskResourceMultiAgentEnv,
)

# Import built-in tasks (trigger registration)
try:
    from powerzoo.tasks import simple
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to load simple tasks: {e}")

try:
    from powerzoo.tasks import middle
except ImportError:
    pass

try:
    from powerzoo.tasks import complex
except ImportError:
    pass


__all__ = [
    # Base classes
    'Task',
    'ConstraintSpec',
    'SingleAgentTask',
    'MultiAgentTask',
    'ConfigTask',
    'ConfigMultiAgentTask',
    
    # Main API
    'make_task_env',
    'make_task',
    'list_tasks',
    'register_task',
    'get_task_class',
    'get_task_info',
    'audit_env',
    'audit_task_env',
    'audit_task_collection',
    'audit_public_tasks',
    'PUBLIC_TASKS',
    'list_public_tasks',
    'get_public_task_info',
    'get_public_task_catalog',
    
    # Adapters
    'create_task_env',
    'create_multi_agent_env',
    'TaskOPFMultiAgentEnv',
    'TaskResourceMultiAgentEnv',
    'OBSERVATION_MODES',
    'make_observation_config',
]
