"""Unit tests for powerzoo.envs.resource.base — ResourceEnv base class.

Covers:
- Initialization defaults and parameter passing
- Attach / detach lifecycle with parent grid
- Reset semantics (power zeroed, time_step reset)
- Bus ID property and setter with parent notification
- status() base fields (bus_id, local_v)
- Voltage lookup from parent nodes DataFrame
"""

import pytest
import numpy as np

from powerzoo.envs.resource.base import ResourceEnv
from .conftest import MockParentGrid


# ========================== Initialization ==========================

class TestResourceEnvInit:
    """ResourceEnv.__init__ defaults and parameter propagation."""

    def test_default_init(self):
        r = ResourceEnv()
        assert r.current_p_mw == 0.0
        assert r.current_q_mvar == 0.0
        assert r.bus_id == -1
        assert r.parent is None
        assert r.resource_id is None
        assert r.sub_resources == {}
        assert r.day_id is None

    def test_custom_bus_id(self):
        r = ResourceEnv(bus_id=5)
        assert r.bus_id == 5

    def test_delta_t_propagated(self):
        r = ResourceEnv(delta_t_minutes=30.0)
        assert r.delta_t_minutes == 30.0

    def test_auto_attach_on_init(self, mock_grid):
        """When parent is passed to __init__, resource is automatically attached."""
        r = ResourceEnv(parent=mock_grid, bus_id=3)
        assert r.parent is mock_grid
        assert r.bus_id == 3
        assert r.resource_id is not None
        assert r.resource_id in mock_grid._resources


# ========================== Attach / Detach ==========================

class TestAttachDetach:
    """Attach and detach lifecycle."""

    def test_attach_registers_resource(self, mock_grid):
        r = ResourceEnv(bus_id=2)
        rid = r.attach(mock_grid, bus_id=2)
        assert r.parent is mock_grid
        assert r.resource_id == rid
        assert rid in mock_grid._resources

    def test_attach_with_custom_name(self, mock_grid):
        r = ResourceEnv()
        rid = r.attach(mock_grid, bus_id=1, name='my_res')
        assert rid == 'my_res'

    def test_detach_clears_state(self, mock_grid):
        r = ResourceEnv(parent=mock_grid, bus_id=1)
        rid = r.resource_id
        r.detach()
        assert r.parent is None
        assert r.resource_id is None
        assert rid not in mock_grid._resources

    def test_reattach_to_different_parent(self, mock_grid):
        grid2 = MockParentGrid()
        r = ResourceEnv(parent=mock_grid, bus_id=0)
        old_rid = r.resource_id
        r.attach(grid2, bus_id=5)
        # Old parent should no longer hold resource
        assert old_rid not in mock_grid._resources
        assert r.parent is grid2

    def test_detach_without_parent_is_noop(self):
        """Detaching a resource that isn't attached should not raise."""
        r = ResourceEnv()
        r.detach()  # no error


# ========================== Bus ID setter ==========================

class TestBusIdSetter:

    def test_bus_id_setter_no_parent(self):
        r = ResourceEnv(bus_id=1)
        r.bus_id = 10
        assert r.bus_id == 10

    def test_bus_id_setter_notifies_parent(self, mock_grid):
        """Setting bus_id on an attached resource calls parent._update_nodes_resources_map."""
        r = ResourceEnv(parent=mock_grid, bus_id=1)
        call_count = 0
        original = mock_grid._update_nodes_resources_map
        def counting_update():
            nonlocal call_count
            call_count += 1
            original()
        mock_grid._update_nodes_resources_map = counting_update
        r.bus_id = 7
        assert call_count >= 1


# ========================== Reset ==========================

class TestReset:

    def test_reset_zeroes_power(self):
        r = ResourceEnv()
        r.current_p_mw = 99.0
        r.current_q_mvar = 99.0
        r.time_step = 42
        r.reset()
        assert r.current_p_mw == 0.0
        assert r.current_q_mvar == 0.0
        assert r.time_step == 0

    def test_reset_stores_day_id(self):
        r = ResourceEnv()
        r.reset(day_id=7)
        assert r.day_id == 7

    def test_reset_accepts_start_time_offset(self):
        r = ResourceEnv()
        r.reset(options={'time_step': 9, 'time_offset': 9})
        assert r.time_step == 9

    def test_reset_with_seed_sets_rng(self):
        r = ResourceEnv()
        r.reset(seed=123)
        assert r.np_random is not None
        val1 = r.np_random.random()
        r.reset(seed=123)
        val2 = r.np_random.random()
        assert val1 == val2, "Same seed should produce same first random draw"


# ========================== Step (abstract) ==========================

class TestStep:

    def test_step_raises_not_implemented(self):
        r = ResourceEnv()
        with pytest.raises(NotImplementedError):
            r.step(None)


# ========================== Status ==========================

class TestStatus:

    def test_status_keys(self):
        r = ResourceEnv()
        s = r.status()
        assert 'current_p_mw' in s
        assert 'current_q_mvar' in s
        assert 'time_step' in s


# ========================== status() local info ==========================

class TestStatusLocalInfo:
    """status() must expose bus_id and local_v (voltage lookup)."""

    def test_status_bus_id_no_parent(self):
        r = ResourceEnv(bus_id=3)
        s = r.status()
        assert s['bus_id'] == 3
        assert s['local_v'] is None  # no parent → no voltage

    def test_status_types(self):
        r = ResourceEnv(bus_id=5)
        s = r.status()
        assert isinstance(s['current_p_mw'], float)
        assert isinstance(s['current_q_mvar'], float)
        assert isinstance(s['time_step'], int)
        assert isinstance(s['bus_id'], int)

    def test_status_local_v_with_voltage(self, mock_grid):
        """If parent exposes _nodes with v_mag, status()['local_v'] is retrieved."""
        import pandas as pd
        mock_grid._nodes = pd.DataFrame(
            {'v_mag': [1.0, 0.98, 1.02]},
            index=[0, 1, 2],
        )
        r = ResourceEnv(parent=mock_grid, bus_id=1)
        s = r.status()
        assert s['local_v'] == pytest.approx(0.98, abs=1e-6)

    def test_status_local_v_missing_bus(self, mock_grid):
        """If bus_id not in parent's _nodes, local_v falls back to None."""
        import pandas as pd
        mock_grid._nodes = pd.DataFrame(
            {'v_mag': [1.0]},
            index=[0],
        )
        r = ResourceEnv(parent=mock_grid, bus_id=99)
        s = r.status()
        assert s['local_v'] is None
