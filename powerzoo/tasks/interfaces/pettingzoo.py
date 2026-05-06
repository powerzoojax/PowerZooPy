"""PettingZoo interface adapters for task-specific benchmark environments."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from gymnasium import spaces

try:
    from pettingzoo import ParallelEnv
    _HAS_PETTINGZOO = True
except ModuleNotFoundError:
    class ParallelEnv:  # type: ignore[no-redef]
        pass
    _HAS_PETTINGZOO = False


class TaskPettingZooWrapper(ParallelEnv):
    """PettingZoo Parallel wrapper around a task-specific multi-agent adapter."""

    metadata = {"render_modes": ["human"], "name": "powerzoo_task_parallel_v0"}

    def __init__(self, env, render_mode: Optional[str] = None):
        if not _HAS_PETTINGZOO:
            raise ImportError(
                "pettingzoo is required for TaskPettingZooWrapper.  "
                "Install it with: pip install pettingzoo"
            )
        super().__init__()
        self.env = env
        self.render_mode = render_mode
        self.possible_agents = list(getattr(env, 'possible_agents', []))
        self.agents: List[str] = []

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        observation_space = getattr(self.env, 'observation_space')
        return observation_space[agent]

    @lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        action_space = getattr(self.env, 'action_space')
        return action_space[agent]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(getattr(self.env, 'agents', self.possible_agents))
        return observations, infos

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        observations, rewards, terminateds, truncateds, infos = self.env.step(actions)

        terminations = {
            agent: terminateds.get(agent, False)
            for agent in self.agents
        }
        truncations = {
            agent: truncateds.get(agent, False)
            for agent in self.agents
        }

        done = bool(terminateds.get('__all__', False) or truncateds.get('__all__', False))
        if done:
            self.agents = []
        else:
            self.agents = list(getattr(self.env, 'agents', self.possible_agents))

        return observations, rewards, terminations, truncations, infos

    def render(self) -> Any:
        if hasattr(self.env, 'render'):
            return self.env.render()
        return None

    def close(self) -> None:
        if hasattr(self.env, 'close'):
            self.env.close()

    def state(self) -> Any:
        if hasattr(self.env, 'state'):
            return self.env.state()
        return None

    def get_observation_fields(self):
        if hasattr(self.env, 'get_observation_fields'):
            return self.env.get_observation_fields()
        raise AttributeError("Underlying environment does not expose get_observation_fields()")
