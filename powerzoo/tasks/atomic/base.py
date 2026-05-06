"""Internal atomic validation task presets.

These tasks are intentionally small and component-scoped. They are meant for
validation and smoke testing, not for the public benchmark surface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from powerzoo.tasks.base import SingleAgentTask


class BaseAtomicTask(SingleAgentTask):
    """Common base for internal atomic validation tasks."""

    difficulty = 'atomic'

    GRID_TYPE = 'distribution'
    CASE = 'Case33bw'
    DEFAULT_START_DATE = '2024-01-01'
    DEFAULT_END_DATE = '2024-01-31'
    DEFAULT_DELTA_T_MINUTES = 60
    DEFAULT_MAX_LOAD_RATIO = 0.85
    DEFAULT_MAX_STEPS = 24

    def __init__(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        delta_t_minutes: Optional[int] = None,
        max_load_ratio: Optional[float] = None,
        max_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._start_date = start_date or self.DEFAULT_START_DATE
        self._end_date = end_date or self.DEFAULT_END_DATE
        self._delta_t_minutes = (
            self.DEFAULT_DELTA_T_MINUTES if delta_t_minutes is None else delta_t_minutes
        )
        self._max_load_ratio = (
            self.DEFAULT_MAX_LOAD_RATIO if max_load_ratio is None else max_load_ratio
        )
        self._max_steps = self.DEFAULT_MAX_STEPS if max_steps is None else max_steps

    def _build_grid_config(self) -> Dict[str, Any]:
        return {
            'type': self.GRID_TYPE,
            'case': self.CASE,
            'start_date': self._start_date,
            'end_date': self._end_date,
            'delta_t_minutes': self._delta_t_minutes,
            'max_load_ratio': self._max_load_ratio,
        }

    def _build_resources(self) -> list[Dict[str, Any]]:
        return []

    def _build_reward(self) -> Dict[str, Any]:
        return {'type': 'safety_only'}

    def get_scenario_config(self) -> Dict[str, Any]:
        return {
            'name': f'{self.name}_scenario',
            'description': self.description,
            'grid': self._build_grid_config(),
            'resources': self._build_resources(),
            'reward': self._build_reward(),
            'episode': {'max_steps': self._max_steps},
        }


class AtomicResourceTask(BaseAtomicTask):
    """Base class for single-resource validation tasks."""

    RESOURCE_NAME = 'resource_0'
    DEFAULT_BUS_ID = 6

    def __init__(self, *, bus_id: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self._bus_id = self.DEFAULT_BUS_ID if bus_id is None else bus_id

    def _resource_config(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_resources(self) -> list[Dict[str, Any]]:
        config = dict(self._resource_config())
        config.setdefault('name', self.RESOURCE_NAME)
        config.setdefault('bus_id', self._bus_id)
        return [config]

    def get_agents_config(self) -> Dict[str, Any]:
        return {
            'agent_type': 'single',
            'obs_keys': ['grid', 'resources', 'time'],
            'resource_names': [self.RESOURCE_NAME],
        }
