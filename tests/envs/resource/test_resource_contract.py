"""Contract tests for ResourceEnv subclass conventions.

Covers:
- step() returns None for all resource subclasses (Layer A)
- cost_clipped_power appears in battery/vehicle status when action is infeasible (Layer B)
- All cost_ fields in status() are float ≥ 0 (Layer C)
- cost_ prefix convention is consistent across all resource types (Layer C)
"""

import pytest
import numpy as np

from powerzoo.envs.resource.battery import BatteryEnv
from powerzoo.envs.resource.renewable import RenewableEnv, SolarEnv, WindEnv
from powerzoo.envs.resource.vehicle import VehicleEnv
from powerzoo.envs.resource.datacenter import DataCenterEnv
from powerzoo.envs.resource.flexload import FlexLoad


# ========================== Fixtures ==========================

@pytest.fixture
def battery():
    b = BatteryEnv(normalize_actions=False, capacity_mwh=50.0, power_mw=20.0,
                   soc_min=0.1, soc_max=0.9, initial_soc=0.5)
    b.reset(seed=42)
    return b


@pytest.fixture
def vehicle():
    v = VehicleEnv(normalize_actions=False, E_max_kWh=60.0,
                   p_charge_max_kW=7.0, p_discharge_max_kW=7.0,
                   soc_init=0.5, delta_t_minutes=60.0)
    v.reset(seed=42)
    return v


@pytest.fixture
def datacenter():
    dc = DataCenterEnv(normalize_actions=False, n_gpus=100)
    dc.reset(seed=42)
    return dc


@pytest.fixture
def flexload():
    fl = FlexLoad(action_scale='physical')
    fl.reset(seed=42)
    return fl


@pytest.fixture
def renewable():
    """Standalone renewable (no parent, no time series — exercises fallback)."""
    r = RenewableEnv(normalize_actions=False, capacity_mw=100.0)
    r.reset(seed=42)
    return r


# ========================== A. step() → None ==========================

class TestStepReturnsNone:
    """step() must return None for all ResourceEnv subclasses."""

    def test_battery_step_returns_none(self, battery):
        result = battery.step(5.0)
        assert result is None

    def test_vehicle_step_returns_none(self, vehicle):
        result = vehicle.step(0.001)
        assert result is None

    def test_datacenter_step_returns_none(self, datacenter):
        result = datacenter.step(np.array([0.5, 0.5, 0.5]))
        assert result is None

    def test_flexload_step_returns_none(self, flexload):
        result = flexload.step(np.array([1.0, 0.0]))
        assert result is None

    def test_renewable_step_returns_none(self, renewable):
        result = renewable.step(None)
        assert result is None

    def test_renewable_step_returns_none_no_data(self, renewable):
        """Early-exit path (no time series) must also return None."""
        renewable._available_cf = None
        result = renewable.step(0.5)
        assert result is None


# ========================== B. cost_clipped_power ==========================

class TestCostClippedPower:
    """Battery and vehicle expose cost_clipped_power in status()."""

    def test_battery_no_clip_when_feasible(self, battery):
        """Small action within SOC limits → zero clip."""
        battery.step(1.0)  # 1 MW discharge, well within 20 MW / SOC capacity
        s = battery.status()
        assert 'cost_clipped_power' in s
        assert s['cost_clipped_power'] == pytest.approx(0.0, abs=1e-9)

    def test_battery_clip_when_infeasible(self):
        """Action far exceeds available energy → non-zero clip."""
        b = BatteryEnv(normalize_actions=False, capacity_mwh=1.0, power_mw=100.0,
                       soc_min=0.0, soc_max=1.0, initial_soc=0.01)
        b.reset(seed=0)
        b.step(100.0)  # Request 100 MW discharge but almost no energy
        s = b.status()
        assert s['cost_clipped_power'] > 0.0

    def test_vehicle_no_clip_at_home(self, vehicle):
        """Small charge at home → zero clip."""
        vehicle.step(-0.001)
        s = vehicle.status()
        assert 'cost_clipped_power' in s
        assert s['cost_clipped_power'] == pytest.approx(0.0, abs=1e-9)

    def test_vehicle_clip_when_away(self):
        """Vehicle away from home, any non-zero action → full clip."""
        v = VehicleEnv(normalize_actions=False, E_max_kWh=60.0,
                       p_charge_max_kW=7.0, delta_t_minutes=60.0)
        v.reset(seed=0)
        v.is_home = False
        v.step(0.005)
        s = v.status()
        assert s['cost_clipped_power'] == pytest.approx(0.005, abs=1e-9)


# ========================== C. cost_ prefix convention ==========================

class TestCostPrefixConvention:
    """All cost_ fields in status() must be float ≥ 0."""

    @pytest.mark.parametrize("resource_fixture", [
        "battery", "vehicle", "datacenter", "flexload", "renewable",
    ])
    def test_cost_fields_are_nonneg_floats(self, resource_fixture, request):
        resource = request.getfixturevalue(resource_fixture)
        resource.step(None)
        for key, val in resource.status().items():
            if key.startswith('cost_'):
                assert isinstance(val, (int, float)), \
                    f"{resource.__class__.__name__}.status()['{key}'] is {type(val)}, expected float"
                assert val >= 0.0, \
                    f"{resource.__class__.__name__}.status()['{key}'] = {val} < 0"

    def test_battery_has_cost_clipped_power(self, battery):
        battery.step(None)
        assert 'cost_clipped_power' in battery.status()

    def test_vehicle_has_cost_clipped_power(self, vehicle):
        vehicle.step(None)
        assert 'cost_clipped_power' in vehicle.status()

    def test_vehicle_has_cost_unmet_energy(self, vehicle):
        vehicle.step(None)
        assert 'cost_unmet_energy' in vehicle.status()

    def test_datacenter_has_cost_overtemp(self, datacenter):
        datacenter.step(None)
        assert 'cost_overtemp' in datacenter.status()

    def test_flexload_has_cost_buffer_overflow(self, flexload):
        flexload.step(None)
        assert 'cost_buffer_overflow' in flexload.status()


# ========================== D. grid_obs / grid_action interface ==========================

class TestGridInterface:
    """grid_obs / grid_obs_names / grid_action_bounds / grid_action_from_normalized contract."""

    def test_battery_grid_obs_shape_and_dtype(self, battery):
        feats = battery.grid_obs()
        assert feats.shape == (4,)
        assert feats.dtype == np.float32

    def test_battery_grid_obs_names_length_matches(self, battery):
        names = battery.grid_obs_names('bat')
        assert len(names) == len(battery.grid_obs())
        assert all('bat' in n for n in names)

    def test_renewable_grid_obs_strips_time_encoding(self, renewable):
        feats = renewable.grid_obs()
        assert feats.shape == (2,)                      # no time_sin/cos
        assert len(renewable.grid_obs_names('r')) == 2

    def test_grid_obs_names_length_equals_grid_obs_dim(self, battery, renewable):
        for res, rid in [(battery, 'b'), (renewable, 'r')]:
            assert len(res.grid_obs()) == len(res.grid_obs_names(rid))

    def test_battery_grid_action_bounds_match_power_mw(self, battery):
        lo, hi = battery.grid_action_bounds()
        assert lo == pytest.approx(-battery.power_mw)
        assert hi == pytest.approx(battery.power_mw)

    def test_renewable_grid_action_bounds(self, renewable):
        lo, hi = renewable.grid_action_bounds()
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(1.0)

    def test_renewable_grid_action_inverted_semantics(self, renewable):
        # +1 → no curtailment → physical 0.0; -1 → full curtailment → physical 1.0
        assert renewable.grid_action_from_normalized(1.0) == pytest.approx(0.0)
        assert renewable.grid_action_from_normalized(-1.0) == pytest.approx(1.0)
        assert renewable.grid_action_from_normalized(0.0) == pytest.approx(0.5)

    def test_battery_grid_action_normalized_midpoint_is_zero(self, battery):
        assert battery.grid_action_from_normalized(0.0) == pytest.approx(0.0)

    def test_battery_grid_action_endpoints(self, battery):
        lo, hi = battery.grid_action_bounds()
        assert battery.grid_action_from_normalized(-1.0) == pytest.approx(lo)
        assert battery.grid_action_from_normalized(1.0) == pytest.approx(hi)
