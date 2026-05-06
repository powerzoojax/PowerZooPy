"""PowerZoo Task Adapters

Adapters that map Tasks to RL frameworks:
- RLlib MultiAgentEnv adapter
- Gymnasium Env adapter
- Future: PettingZoo, Tianshou, etc.

Each adapter file handles one agent type:
- base.py      : factory functions (create_task_env, create_multi_agent_env)
- opf.py       : TaskOPFMultiAgentEnv      (agent_type='unit')
- resource.py  : TaskResourceMultiAgentEnv (agent_type='resource')
- ev.py        : TaskEVMultiAgentEnv       (agent_type='vehicle')
"""

from powerzoo.tasks.adapters.base import create_task_env, create_multi_agent_env
from powerzoo.tasks.adapters.opf import TaskOPFMultiAgentEnv
from powerzoo.tasks.adapters.resource import TaskResourceMultiAgentEnv
from powerzoo.tasks.adapters.ev import TaskEVMultiAgentEnv

__all__ = [
    'create_task_env',
    'create_multi_agent_env',
    'TaskOPFMultiAgentEnv',
    'TaskResourceMultiAgentEnv',
    'TaskEVMultiAgentEnv',
]
