"""Shared fixtures for powerzoo.envs.resource tests."""

import pytest
import numpy as np


@pytest.fixture
def rng():
    """Deterministic RNG for reproducible tests."""
    return np.random.default_rng(42)


class MockParentGrid:
    """Minimal mock of a parent grid for resource attachment tests.

    Provides the subset of attributes that resources read from their parent:
    - delta_t_minutes, steps_per_day
    - register_resource / unregister_resource
    - _nodes (None by default — no voltage data)
    - _time_series_data (None by default)
    """

    def __init__(self, delta_t_minutes: float = 15.0, steps_per_day: int = 96):
        self.delta_t_minutes = delta_t_minutes
        self.steps_per_day = steps_per_day
        self._nodes = None
        self._time_series_data = None
        self._resources: dict = {}
        self._next_id = 0

    def register_resource(self, resource, bus_id: int, name: str = None) -> str:
        rid = name or f"{getattr(resource, 'name', 'res')}_{self._next_id}"
        self._next_id += 1
        self._resources[rid] = resource
        return rid

    def unregister_resource(self, resource_id: str) -> None:
        self._resources.pop(resource_id, None)

    def _update_nodes_resources_map(self):
        pass


@pytest.fixture
def mock_grid():
    """A default 15-min resolution mock grid."""
    return MockParentGrid(delta_t_minutes=15.0, steps_per_day=96)


@pytest.fixture
def mock_grid_30min():
    """A 30-min resolution mock grid."""
    return MockParentGrid(delta_t_minutes=30.0, steps_per_day=48)
