"""Atomic single-resource validation tasks."""

from __future__ import annotations

from powerzoo.tasks.atomic.base import AtomicResourceTask


class AtomicBatteryTask(AtomicResourceTask):
    name = 'atomic_battery'
    description = 'Atomic battery validation preset'
    GRID_TYPE = 'distribution'
    CASE = 'Case33bw'
    RESOURCE_NAME = 'battery_0'
    DEFAULT_BUS_ID = 6

    def _resource_config(self):
        return {
            'type': 'battery',
            'capacity_mwh': 0.5,
            'power_mw': 0.2,
            'initial_soc': 0.5,
            'soc_min': 0.1,
            'soc_max': 0.9,
        }


class AtomicSolarTask(AtomicResourceTask):
    name = 'atomic_solar'
    description = 'Atomic solar validation preset'
    GRID_TYPE = 'transmission'
    CASE = 'Case5'
    RESOURCE_NAME = 'solar_0'
    DEFAULT_BUS_ID = 2
    DEFAULT_DELTA_T_MINUTES = 30
    DEFAULT_MAX_STEPS = 48

    def _resource_config(self):
        return {
            'type': 'solar',
            'capacity_mw': 10.0,
        }


class AtomicWindTask(AtomicResourceTask):
    name = 'atomic_wind'
    description = 'Atomic wind validation preset'
    GRID_TYPE = 'transmission'
    CASE = 'Case5'
    RESOURCE_NAME = 'wind_0'
    DEFAULT_BUS_ID = 3
    DEFAULT_DELTA_T_MINUTES = 30
    DEFAULT_MAX_STEPS = 48

    def _resource_config(self):
        return {
            'type': 'wind',
            'capacity_mw': 12.0,
        }


class AtomicVehicleTask(AtomicResourceTask):
    name = 'atomic_vehicle'
    description = 'Atomic vehicle validation preset'
    GRID_TYPE = 'distribution'
    CASE = 'Case33bw'
    RESOURCE_NAME = 'vehicle_0'
    DEFAULT_BUS_ID = 10

    def _resource_config(self):
        return {
            'type': 'vehicle',
            'capacity_kwh': 60.0,
            'charge_power_kw': 7.0,
            'discharge_power_kw': 7.0,
            'initial_soc': 0.6,
            'soc_min': 0.1,
            'soc_max': 0.95,
            'soc_departure_min': 0.8,
            'commute_schedule': [
                {'departure': 8.0, 'arrival': 18.0, 'energy_kWh': 15.0},
            ],
        }


class AtomicFlexLoadTask(AtomicResourceTask):
    name = 'atomic_flexload'
    description = 'Atomic flex-load validation preset'
    GRID_TYPE = 'transmission'
    CASE = 'Case5'
    RESOURCE_NAME = 'flexload_0'
    DEFAULT_BUS_ID = 4
    DEFAULT_DELTA_T_MINUTES = 30
    DEFAULT_MAX_STEPS = 48

    def _resource_config(self):
        return {
            'type': 'flexload',
            'curtail_cap_mw': 5.0,
            'shift_cap_mw': 5.0,
            'shift_horizon': 4,
            'baseline_mw': 20.0,
        }


class AtomicDataCenterTask(AtomicResourceTask):
    name = 'atomic_datacenter'
    description = 'Atomic datacenter validation preset'
    GRID_TYPE = 'distribution'
    CASE = 'Case33bw'
    RESOURCE_NAME = 'dc_0'
    DEFAULT_BUS_ID = 6
    DEFAULT_MAX_STEPS = 48

    def _resource_config(self):
        return {
            'type': 'datacenter',
            'n_gpus': 256,
            'infer_gpu_peak': 96,
            'p_base_mw': 0.2,
            't_critical': 35.0,
        }
