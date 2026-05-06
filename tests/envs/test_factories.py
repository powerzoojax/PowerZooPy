"""Tests for powerzoo.envs.factories — factory helpers for environment composition.

Factory module provides:
  - GRID_TYPES / RESOURCE_TYPES: registry dictionaries
  - create_grid(config): instantiate grid env from config dict
  - create_resource(config, default_delta_t_minutes): instantiate resource
  - build_resource_metadata(id, config): normalised metadata extraction
  - attach_resources(grid, configs): bulk create + attach resources

Domain notes:
  - Resources use different unit systems: battery (MWh/MW), vehicle (kWh/kW)
  - Config aliases handled: 'capacity_mw' ↔ 'capacity_mwh', etc.
  - delta_t_minutes cascades from grid to resources for consistency
"""
import numpy as np
import pytest

from powerzoo.envs.factories import (
    GRID_TYPES,
    RESOURCE_TYPES,
    create_grid,
    create_resource,
    build_resource_metadata,
    attach_resources,
)


# ── Registry Dictionaries ────────────────────────────────────────────

class TestRegistries:
    """GRID_TYPES and RESOURCE_TYPES completeness."""

    def test_grid_types_has_transmission(self):
        assert 'transmission' in GRID_TYPES

    def test_grid_types_has_distribution(self):
        assert 'distribution' in GRID_TYPES

    def test_resource_types_complete(self):
        expected = {'solar', 'wind', 'battery', 'vehicle', 'flexload', 'datacenter'}
        assert expected == set(RESOURCE_TYPES.keys())


# ── create_grid ──────────────────────────────────────────────────────

class TestCreateGrid:
    """Grid creation from config dict."""

    def test_create_transmission_default(self):
        grid = create_grid({'type': 'transmission', 'case': 'Case5'})
        from powerzoo.envs.grid.trans import TransGridEnv
        assert isinstance(grid, TransGridEnv)

    def test_create_distribution(self):
        grid = create_grid({'type': 'distribution', 'case': 'Case33bw'})
        from powerzoo.envs.grid.dist import DistGridEnv
        assert isinstance(grid, DistGridEnv)

    def test_unknown_grid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown grid type"):
            create_grid({'type': 'nonexistent'})

    def test_passes_delta_t(self):
        grid = create_grid({'type': 'transmission', 'case': 'Case5',
                            'delta_t_minutes': 15.0})
        assert grid.delta_t_minutes == 15.0

    def test_passes_max_load_ratio(self):
        grid = create_grid({'type': 'transmission', 'case': 'Case5',
                            'max_load_ratio': 0.7})
        assert grid.max_load_ratio == 0.7


# ── create_resource ──────────────────────────────────────────────────

class TestCreateResource:
    """Resource creation with type dispatch and alias handling."""

    def test_create_battery(self):
        from powerzoo.envs.resource.battery import BatteryEnv
        res = create_resource({'type': 'battery', 'capacity_mwh': 100.0},
                              default_delta_t_minutes=30.0)
        assert isinstance(res, BatteryEnv)
        assert res.capacity_mwh == 100.0

    def test_create_battery_default_one_way_095(self):
        """No efficiency keys → BatteryEnv one-way defaults (0.95 / 0.95)."""
        res = create_resource({'type': 'battery'}, default_delta_t_minutes=30.0)
        assert res.eta_charge == pytest.approx(0.95)
        assert res.eta_discharge == pytest.approx(0.95)

    def test_create_battery_legacy_efficiency_key(self):
        """YAML ``efficiency`` maps to ``eta_roundtrip`` (sqrt decomposition)."""
        res = create_resource(
            {'type': 'battery', 'efficiency': 0.81}, default_delta_t_minutes=30.0
        )
        assert res.eta_charge == pytest.approx(0.9)
        assert res.eta_discharge == pytest.approx(0.9)

    def test_create_solar(self):
        from powerzoo.envs.resource.renewable import SolarEnv
        res = create_resource({'type': 'solar', 'capacity_mw': 50.0},
                              default_delta_t_minutes=30.0)
        assert isinstance(res, SolarEnv)

    def test_create_wind(self):
        from powerzoo.envs.resource.renewable import WindEnv
        res = create_resource({'type': 'wind', 'capacity_mw': 80.0},
                              default_delta_t_minutes=30.0)
        assert isinstance(res, WindEnv)

    def test_create_vehicle(self):
        from powerzoo.envs.resource.vehicle import VehicleEnv
        res = create_resource({'type': 'vehicle', 'E_max_kWh': 75.0},
                              default_delta_t_minutes=15.0)
        assert isinstance(res, VehicleEnv)

    def test_create_flexload(self):
        from powerzoo.envs.resource.flexload import FlexLoad
        res = create_resource({'type': 'flexload', 'curtail_cap_mw': 5.0},
                              default_delta_t_minutes=30.0)
        assert isinstance(res, FlexLoad)

    def test_create_datacenter(self):
        from powerzoo.envs.resource.datacenter import DataCenterEnv
        res = create_resource({'type': 'datacenter', 'n_gpus': 500},
                              default_delta_t_minutes=30.0)
        assert isinstance(res, DataCenterEnv)

    def test_unknown_resource_raises(self):
        with pytest.raises(ValueError, match="Unknown resource type"):
            create_resource({'type': 'fusion_reactor'}, default_delta_t_minutes=30.0)

    def test_battery_alias_capacity_mw(self):
        """'capacity_mw' should be accepted as alias for battery capacity."""
        res = create_resource({'type': 'battery', 'capacity_mw': 75.0},
                              default_delta_t_minutes=30.0)
        assert res.capacity_mwh == 75.0

    def test_vehicle_delta_t_inherited(self):
        """Vehicle should inherit delta_t from grid if not specified."""
        res = create_resource({'type': 'vehicle'}, default_delta_t_minutes=15.0)
        assert res.delta_t_minutes == 15.0


# ── build_resource_metadata ──────────────────────────────────────────

class TestBuildResourceMetadata:
    """Metadata normalisation for attached resources.

    Physical parameters are read from the instantiated resource object so that
    metadata always reflects actual construction values (including defaults),
    not a re-parse of the config dict.
    """

    def test_battery_metadata(self):
        from powerzoo.envs.resource.battery import BatteryEnv
        resource = BatteryEnv(capacity_mwh=100.0, power_mw=25.0)
        meta = build_resource_metadata(
            'bess_0',
            {'type': 'battery', 'bus_id': 3},
            resource,
        )
        assert meta['type'] == 'battery'
        assert meta['name'] == 'bess_0'
        assert meta['bus_id'] == 3
        assert meta['capacity_mwh'] == 100.0
        assert meta['power_mw'] == 25.0

    def test_battery_metadata_defaults(self):
        """Metadata reflects BatteryEnv defaults, not re-parsed config defaults."""
        from powerzoo.envs.resource.battery import BatteryEnv
        resource = BatteryEnv()  # all defaults
        meta = build_resource_metadata(
            'bess_1',
            {'type': 'battery', 'bus_id': 0},
            resource,
        )
        assert meta['capacity_mwh'] == resource.capacity_mwh
        assert meta['initial_soc'] == resource.initial_soc
        assert meta['soc_min'] == resource.soc_min
        assert meta['soc_max'] == resource.soc_max

    def test_vehicle_metadata(self):
        from powerzoo.envs.resource.vehicle import VehicleEnv
        resource = VehicleEnv(E_max_kWh=60.0, p_charge_max_kW=7.0, p_discharge_max_kW=7.0)
        meta = build_resource_metadata(
            'ev_0',
            {'type': 'vehicle', 'bus_id': 5},
            resource,
        )
        assert meta['type'] == 'vehicle'
        assert meta['capacity_kwh'] == pytest.approx(60.0)
        assert meta['charge_power_kw'] == pytest.approx(7.0)
        assert meta['discharge_power_kw'] == pytest.approx(7.0)
        assert meta['initial_soc'] == resource.soc_init

    def test_datacenter_metadata(self):
        from powerzoo.envs.resource.datacenter import DataCenterEnv
        resource = DataCenterEnv(n_gpus=1000)
        meta = build_resource_metadata(
            'dc_0',
            {'type': 'datacenter', 'bus_id': 2},
            resource,
        )
        assert meta['type'] == 'datacenter'
        assert meta['n_gpus'] == 1000

    def test_solar_metadata(self):
        from powerzoo.envs.resource.renewable import SolarEnv
        resource = SolarEnv(capacity_mw=50.0)
        meta = build_resource_metadata(
            'solar_0',
            {'type': 'solar', 'bus_id': 1},
            resource,
        )
        assert meta['type'] == 'solar'
        assert meta['capacity_mw'] == 50.0
        assert meta['normalize_actions'] is True

    def test_solar_metadata_normalize_actions_false(self):
        from powerzoo.envs.resource.renewable import SolarEnv
        resource = SolarEnv(capacity_mw=50.0, normalize_actions=False)
        meta = build_resource_metadata(
            'solar_1',
            {'type': 'solar', 'bus_id': 1},
            resource,
        )
        assert meta['normalize_actions'] is False

    def test_wind_metadata(self):
        from powerzoo.envs.resource.renewable import WindEnv
        resource = WindEnv(capacity_mw=80.0)
        meta = build_resource_metadata(
            'wind_0',
            {'type': 'wind', 'bus_id': 2},
            resource,
        )
        assert meta['type'] == 'wind'
        assert meta['capacity_mw'] == 80.0
        assert meta['normalize_actions'] is True

    def test_flexload_metadata(self):
        from powerzoo.envs.resource.flexload import FlexLoad
        resource = FlexLoad(curtail_cap_mw=5.0, shift_cap_mw=8.0, baseline_mw=40.0,
                            shift_horizon=4)
        meta = build_resource_metadata(
            'fl_0',
            {'type': 'flexload', 'bus_id': 3},
            resource,
        )
        assert meta['type'] == 'flexload'
        assert meta['curtail_cap_mw'] == 5.0
        assert meta['shift_cap_mw'] == 8.0
        assert meta['baseline_mw'] == 40.0
        assert meta['shift_horizon'] == 4

    def test_unknown_type_raises(self):
        """Unregistered resource types must raise rather than silently skip."""
        from powerzoo.envs.resource.battery import BatteryEnv
        resource = BatteryEnv()
        with pytest.raises(ValueError, match="Unhandled resource type"):
            build_resource_metadata(
                'x_0',
                {'type': 'fusion_reactor', 'bus_id': 0},
                resource,
            )


# ── attach_resources ─────────────────────────────────────────────────

class TestAttachResources:
    """Bulk resource creation and attachment to a grid."""

    def test_attach_returns_resources_and_metadata(self):
        from powerzoo.envs.grid.trans import TransGridEnv
        grid = TransGridEnv(time_series=np.ones(48) * 100)
        resources_config = [
            {'type': 'battery', 'bus_id': 2, 'capacity_mwh': 50.0, 'power_mw': 10.0},
        ]
        resources, metadata = attach_resources(grid, resources_config)
        assert len(resources) == 1
        assert len(metadata) == 1
        rid = list(resources.keys())[0]
        assert rid in metadata
        assert metadata[rid]['type'] == 'battery'

    def test_attach_multiple_resources(self):
        from powerzoo.envs.grid.trans import TransGridEnv
        grid = TransGridEnv(time_series=np.ones(48) * 100)
        resources_config = [
            {'type': 'battery', 'bus_id': 1, 'capacity_mwh': 50.0},
            {'type': 'battery', 'bus_id': 3, 'capacity_mwh': 30.0},
        ]
        resources, metadata = attach_resources(grid, resources_config)
        assert len(resources) == 2
        assert len(metadata) == 2

    def test_attach_empty_config(self):
        from powerzoo.envs.grid.trans import TransGridEnv
        grid = TransGridEnv(time_series=np.ones(48) * 100)
        resources, metadata = attach_resources(grid, [])
        assert len(resources) == 0
        assert len(metadata) == 0
