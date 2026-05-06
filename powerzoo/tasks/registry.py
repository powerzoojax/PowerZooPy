"""PowerZoo Task Registry

Task registration and creation helpers:
- register_task(): register a custom Task
- make_task_env(): create a task environment (supports split='train'/'val'/'test')
- make_task():     create a Task object
- list_tasks():    list all available tasks
"""

from typing import Dict, Any, Optional, List, Type, Union
import importlib

from powerzoo.tasks.base import Task, ConfigTask, ConfigMultiAgentTask

# Global task registry
_TASK_REGISTRY: Dict[str, Type[Task]] = {}


def register_task(name: str, task_class: Type[Task], force: bool = False):
    """Register a Task class.

    Args:
        name:       Task name (used by make_task_env).
        task_class: Task subclass.
        force:      Overwrite if the task already exists.

    Example::

        from powerzoo.tasks import register_task, MultiAgentTask

        class MyTask(MultiAgentTask):
            name = "my_opf_task"
            def get_scenario_config(self): ...

        register_task('my_opf', MyTask)
    """
    if not isinstance(task_class, type) or not issubclass(task_class, Task):
        raise TypeError(f"task_class must be a subclass of Task, got {type(task_class)}")

    if name in _TASK_REGISTRY and not force:
        raise ValueError(f"Task '{name}' already registered. Use force=True to overwrite.")

    _TASK_REGISTRY[name] = task_class


def _resolve_task_class(name: str, split: Optional[str], kwargs: dict) -> Type[Task]:
    """Look up the task class and inject ``split`` into kwargs when appropriate.

    Shared helper for ``make_task_env`` and ``make_task``.
    """
    import logging
    logger = logging.getLogger(__name__)

    _ensure_builtin_tasks_loaded()

    if name not in _TASK_REGISTRY:
        available = list(_TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task: '{name}'. Available tasks: {available}")

    task_class = _TASK_REGISTRY[name]

    if hasattr(task_class, 'SPLIT_DATES') and 'split' not in kwargs:
        kwargs['split'] = split
    elif split is not None and not hasattr(task_class, 'SPLIT_DATES'):
        logger.debug(
            "Task '%s' does not define SPLIT_DATES; ignoring split='%s'.",
            name, split,
        )

    return task_class


def make_task_env(
    name: Union[str, Dict[str, Any]],
    split: Optional[str] = 'train',
    framework: str = 'auto',
    **kwargs,
) -> Any:
    """Create a task environment.

    Args:
        name:    Task name (e.g. ``'marl_opf'``) **or** an inline task config
                 dict.  When a dict is supplied the task is built as an
                 anonymous :class:`~powerzoo.tasks.base.ConfigTask` or
                 :class:`~powerzoo.tasks.base.ConfigMultiAgentTask`; the
                 ``split`` argument is then ignored (a warning is emitted).
        split:   Data split — ``'train'``, ``'val'``, or ``'test'``.
                 Passed to the Task constructor if it accepts ``split``.
                 Ignored (with a debug log) for tasks that do not define
                 ``SPLIT_DATES``.
        framework: Multi-agent framework — ``'auto'``, ``'rllib'``, or
                   ``'pettingzoo'``.  ``'auto'`` (default) uses specialized
                   task adapters (RLlib-compatible when ``ray`` is installed).
                   ``'pettingzoo'`` returns a task-aware PettingZoo wrapper
                   around the same specialized adapter.
        **kwargs: Additional parameters forwarded to the Task constructor,
                  overriding defaults.  ``split`` is included unless you
                  override it explicitly via kwargs.

    Returns:
        - Single-agent task: Gymnasium Env
        - Multi-agent task:  RLlib MultiAgentEnv or PettingZoo ParallelEnv

    Example::

        from powerzoo.tasks import make_task_env

        train_env = make_task_env('marl_opf', split='train')
        test_env  = make_task_env('marl_opf', split='test')

        # Anonymous config dict
        env = make_task_env({'grid': {'type': 'transmission', 'case': 'case5'},
                             'resources': []})

        # Use PettingZoo instead of RLlib
        env = make_task_env('marl_opf', split='train', framework='pettingzoo')
    """
    if isinstance(name, dict):
        task = _make_config_task(name, split, kwargs)
    else:
        task_class = _resolve_task_class(name, split, kwargs)
        task = task_class(**kwargs)
    from powerzoo.tasks.adapters import create_task_env
    return create_task_env(task, framework=framework)


def make_task(
    name: Union[str, Dict[str, Any]],
    split: Optional[str] = 'train',
    **kwargs,
) -> Task:
    """Create a Task instance (without creating an environment).

    Args:
        name:    Task name **or** inline config dict.
        split:   Data split (forwarded to Task constructor when supported).
                 Ignored for anonymous config tasks (a warning is emitted).
        **kwargs: Override configuration.

    Returns:
        Task instance.
    """
    if isinstance(name, dict):
        return _make_config_task(name, split, kwargs)
    task_class = _resolve_task_class(name, split, kwargs)
    return task_class(**kwargs)


def _make_config_task(
    config: Dict[str, Any],
    split: Optional[str],
    kwargs: dict,
) -> Task:
    """Build an anonymous ConfigTask or ConfigMultiAgentTask from a dict.

    ``split`` is forwarded but the Config*Task constructors will emit a
    warning and ignore it (they have no SPLIT_DATES).
    """
    agents = config.get('agents', {})
    agent_type = agents.get('agent_type', '')
    agent_mode = agents.get('agent_mode', '')
    is_multi = (
        agent_mode == 'multi'
        or agent_type not in ('', 'single')
    )
    merged = dict(kwargs)
    if split is not None:
        merged.setdefault('split', split)
    if is_multi:
        return ConfigMultiAgentTask(config, **merged)
    return ConfigTask(config, **merged)


def get_task_class(name: str) -> Type[Task]:
    """Get a Task class by name."""
    _ensure_builtin_tasks_loaded()

    if name not in _TASK_REGISTRY:
        available = list(_TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task: '{name}'. Available tasks: {available}")

    return _TASK_REGISTRY[name]


def list_tasks(
    difficulty: Optional[str] = None,
    agent_mode: Optional[str] = None,
) -> List[str]:
    """List available tasks.

    Args:
        difficulty: Filter by difficulty (``'simple'``, ``'middle'``, ``'complex'``).
        agent_mode: Filter by agent mode (``'single'``, ``'multi'``).

    Returns:
        Sorted list of task names.
    """
    _ensure_builtin_tasks_loaded()

    tasks = []
    for name, task_class in _TASK_REGISTRY.items():
        if difficulty is not None and getattr(task_class, 'difficulty', None) != difficulty:
            continue
        if agent_mode is not None and getattr(task_class, 'agent_mode', None) != agent_mode:
            continue
        tasks.append(name)

    return sorted(tasks)


def get_task_info(name: str) -> Dict[str, Any]:
    """Get task metadata.

    Returns a dict with keys: name, description, difficulty, agent_mode,
    has_splits, eval_protocol.
    """
    _ensure_builtin_tasks_loaded()

    if name not in _TASK_REGISTRY:
        raise ValueError(f"Unknown task: '{name}'")

    task_class = _TASK_REGISTRY[name]

    return {
        'name':          getattr(task_class, 'name', name),
        'description':   getattr(task_class, 'description', ''),
        'difficulty':    getattr(task_class, 'difficulty', 'unknown'),
        'agent_mode':    getattr(task_class, 'agent_mode', 'single'),
        'has_splits':    hasattr(task_class, 'SPLIT_DATES'),
        'split_dates':   getattr(task_class, 'SPLIT_DATES', None),
        'eval_protocol': getattr(task_class, 'eval_protocol', None),
    }


def _ensure_builtin_tasks_loaded():
    """Lazy-load built-in tasks to avoid circular imports."""
    try:
        from powerzoo.tasks import simple   # noqa: F401
    except ImportError:
        pass
    try:
        from powerzoo.tasks import middle   # noqa: F401
    except ImportError:
        pass
    try:
        from powerzoo.tasks import complex  # noqa: F401
    except ImportError:
        pass
