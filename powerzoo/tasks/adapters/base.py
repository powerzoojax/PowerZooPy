"""Factory functions for creating task environments."""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from powerzoo.tasks.base import Task

# Try to import RLlib
try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    HAS_RLLIB = True
except ImportError:
    HAS_RLLIB = False
    MultiAgentEnv = object


def create_task_env(task: 'Task', framework: str = 'auto') -> Any:
    """Create an environment from a Task configuration.

    Args:
        task: Task instance
        framework: Multi-agent framework to use.
            - ``'auto'``: use specialized task adapter (RLlib-compatible
              when ``ray`` is installed, works without it too).
            - ``'rllib'``: same as ``'auto'`` but raises if ``ray[rllib]``
              is missing.
            - ``'pettingzoo'``: task-aware PettingZoo Parallel API via
              ``TaskPettingZooWrapper``.

    Returns:
        Single-agent mode returns Gymnasium Env
        Multi-agent mode returns specialized task adapter or PettingZoo wrapper
    """
    if task.agent_mode == 'single':
        return task.create_single_agent_env()
    elif task.agent_mode == 'multi':
        return create_multi_agent_env(task, framework=framework)
    else:
        raise ValueError(f"Unknown agent_mode: {task.agent_mode}")


def create_multi_agent_env(task: 'Task', framework: str = 'auto') -> Any:
    """Create a multi-agent environment.

    Adapter selection by agents_config.agent_type:
    - 'unit': OPF scenario, each generator is an agent
    - 'resource': DER scenario, each resource is an agent
    - 'vehicle': EV V2G/G2V scenario, each EV is an agent
    - 'custom': custom agent definition

    Args:
        task: Task instance
        framework: ``'auto'``, ``'rllib'``, or ``'pettingzoo'``.
            ``'auto'`` and ``'rllib'`` both use specialized adapters;
            ``'rllib'`` additionally requires ``ray[rllib]``.
            ``'pettingzoo'`` uses a task-aware ``TaskPettingZooWrapper``.

    Returns:
        Specialized task adapter (``'auto'``/``'rllib'``) or
        ``TaskPettingZooWrapper`` (``'pettingzoo'``).
    """
    if framework == 'pettingzoo':
        return _create_pettingzoo_env(task)

    if framework == 'rllib' and not HAS_RLLIB:
        raise ImportError(
            "ray[rllib] is required for framework='rllib'. "
            "Install with: pip install 'ray[rllib]'  "
            "Or use framework='pettingzoo' for a lighter alternative."
        )

    # framework == 'rllib' or 'auto'  →  use specialized adapters.
    # The adapters gracefully fall back to `object` as base when RLlib is
    # absent, so they work in both cases; the only difference is whether
    # the returned object also satisfies the RLlib MultiAgentEnv interface.
    return _create_specialized_env(task)


def _create_specialized_env(task: 'Task') -> Any:
    """Create a task-specific multi-agent environment.

    The adapters inherit from ``MultiAgentEnv`` when RLlib is installed, and
    from ``object`` otherwise; they function correctly in both cases.
    """
    agents_config = task.get_agents_config()
    agent_type = agents_config.get('agent_type', 'unit')
    resource_filter = agents_config.get('resource_filter', [])

    if agent_type == 'unit':
        task_type = agents_config.get('task_type', 'opf')
        if task_type == 'unit_commitment':
            from powerzoo.tasks.adapters.uc import TaskUCMultiAgentEnv
            return TaskUCMultiAgentEnv(task)
        from powerzoo.tasks.adapters.opf import TaskOPFMultiAgentEnv
        return TaskOPFMultiAgentEnv(task)
    elif agent_type == 'resource':
        if 'vehicle' in resource_filter:
            from powerzoo.tasks.adapters.ev import TaskEVMultiAgentEnv
            return TaskEVMultiAgentEnv(task)
        else:
            from powerzoo.tasks.adapters.resource import TaskResourceMultiAgentEnv
            return TaskResourceMultiAgentEnv(task)
    elif agent_type == 'custom':
        env_class = agents_config.get('env_class')
        if env_class is None:
            raise ValueError("Custom agent_type requires 'env_class' in agents_config")
        return env_class(task)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")


def _create_pettingzoo_env(task: 'Task') -> Any:
    """Create a PettingZoo Parallel API wrapper over the task adapter."""
    try:
        from powerzoo.tasks.interfaces import TaskPettingZooWrapper
    except ImportError:
        raise ImportError(
            "pettingzoo is required for framework='pettingzoo'. "
            "Install with: pip install pettingzoo"
        )
    specialized_env = _create_specialized_env(task)
    return TaskPettingZooWrapper(specialized_env)
