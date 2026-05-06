"""Factory helpers for benchmark-facing environment composition."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Tuple, Type

from powerzoo.case import load_case
from powerzoo.envs.grid import DistGridEnv, TransGridEnv
from powerzoo.envs.grid.trans import ACConfig
from powerzoo.envs.resource import (
    BatteryEnv, DataCenterEnv, FlexLoad, SolarEnv, VehicleEnv, WindEnv,
)


GRID_TYPES: Dict[str, Type] = {
    'transmission': TransGridEnv,
    'distribution': DistGridEnv,
    # NOTE: DistGrid3PhaseEnv is experimental and not registered here.
    # Use direct import: from powerzoo.envs.grid import DistGrid3PhaseEnv
}

GRID_COMMON_KEYS = (
    'start_date',
    'end_date',
    'delta_t_minutes',
    'data_loader',
    'load_columns',
    'max_load_ratio',
    'min_load_ratio',
    'time_series',
    'max_episode_steps',
    'randomize_start_time',
    'time_alignment',
)

# Maintenance note: when adding a new parameter to TransGridEnv or
# DistGridEnv, also add the key name here so that config dicts are
# forwarded correctly.  Missing keys are silently ignored.
GRID_SPECIFIC_KEYS = {
    'transmission': ('physics', 'solver_mode', 'solver_type', 'difficulty'),
    'distribution': ('max_iter', 'tol', 'v_slack', 'v_min', 'v_max', 'difficulty'),
}

RESOURCE_TYPES: Dict[str, Type] = {
    'solar': SolarEnv,
    'wind': WindEnv,
    'battery': BatteryEnv,
    'vehicle': VehicleEnv,
    'flexload': FlexLoad,
    'datacenter': DataCenterEnv,
}


def _pick(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def create_grid(grid_config: Dict[str, Any]):
    """Create a grid environment from config."""
    grid_type = grid_config.get('type', 'transmission')
    grid_class = GRID_TYPES.get(grid_type)
    if grid_class is None:
        raise ValueError(
            f"Unknown grid type: {grid_type}. "
            f"Expected one of {sorted(GRID_TYPES)}."
        )

    case_spec = grid_config.get('case', 'Case5')
    case = load_case(case_spec) if isinstance(case_spec, str) else case_spec

    grid_params: Dict[str, Any] = {
        'case': case,
        **{k: grid_config[k]
           for k in (*GRID_COMMON_KEYS, *GRID_SPECIFIC_KEYS[grid_type])
           if k in grid_config},
    }

    # Build ACConfig from flat config-dict keys when targeting TransGridEnv
    if grid_type == 'transmission':
        _AC_KEYS = {'ac_v_min': 'v_min', 'ac_v_max': 'v_max',
                    'ac_q_factor': 'q_factor', 'ac_backend': 'backend',
                    'acopf_solver': 'solver'}
        ac_kwargs = {field: grid_config[key]
                     for key, field in _AC_KEYS.items()
                     if key in grid_config}
        if ac_kwargs:
            grid_params['ac_config'] = ACConfig(**ac_kwargs)

    return grid_class(**grid_params)


def create_resource(res_config: Dict[str, Any], *, default_delta_t_minutes: float):
    """Create a resource from config, honoring legacy and current key aliases."""
    res_type = res_config['type']
    resource_class = RESOURCE_TYPES.get(res_type)
    if resource_class is None:
        raise ValueError(
            f"Unknown resource type: {res_type}. "
            f"Expected one of {sorted(RESOURCE_TYPES)}."
        )

    if res_type in ('solar', 'wind'):
        solar_kwargs: Dict[str, Any] = {
            'capacity_mw': _pick(res_config, 'capacity_mw', default=50.0),
            'profile_column': res_config.get('profile_column'),
            'custom_data_loader': res_config.get('custom_data_loader'),
            'normalize_actions': _pick(res_config, 'normalize_actions', default=True),
            'delta_t_minutes': _pick(res_config, 'delta_t_minutes',
                                     default=default_delta_t_minutes),
            'enable_q_control': res_config.get('enable_q_control', False),
        }
        if res_config.get('s_rated_mva') is not None:
            solar_kwargs['s_rated_mva'] = res_config['s_rated_mva']
        return resource_class(**solar_kwargs)

    if res_type == 'battery':
        capacity_mwh = res_config.get('capacity_mwh')
        if capacity_mwh is None and 'capacity_mw' in res_config:
            warnings.warn(
                "battery config: 'capacity_mw' is deprecated as alias for "
                "'capacity_mwh'.  'capacity_mw' is a power unit (MW) while "
                "battery capacity is energy (MWh).  Please use 'capacity_mwh'.",
                FutureWarning,
                stacklevel=2,
            )
            capacity_mwh = res_config['capacity_mw']
        if capacity_mwh is None:
            capacity_mwh = 50.0
        bat_kwargs: Dict[str, Any] = {
            'capacity_mwh': capacity_mwh,
            'power_mw': _pick(res_config, 'power_mw', default=20.0),
            'eta_charge': res_config.get('eta_charge'),
            'eta_discharge': res_config.get('eta_discharge'),
            'soc_min': _pick(res_config, 'soc_min', default=0.1),
            'soc_max': _pick(res_config, 'soc_max', default=0.9),
            'initial_soc': _pick(res_config, 'initial_soc', default=0.5),
            'delta_t_minutes': _pick(
                res_config, 'delta_t_minutes', default=default_delta_t_minutes
            ),
        }
        # Round-trip shorthand (sqrt decomposition).  Prefer ``eta_roundtrip``;
        # legacy ``efficiency`` key maps to the same parameter.  Omitted keys
        # leave BatteryEnv at one-way defaults (0.95 / 0.95).
        eta_rt = _pick(res_config, 'eta_roundtrip', 'efficiency')
        if eta_rt is not None:
            bat_kwargs['eta_roundtrip'] = float(eta_rt)
        bat_kwargs['enable_q_control'] = res_config.get('enable_q_control', False)
        if res_config.get('s_rated_mva') is not None:
            bat_kwargs['s_rated_mva'] = res_config['s_rated_mva']
        bat_kwargs['bus_id'] = res_config.get('bus_id', -1)
        return resource_class(**bat_kwargs)

    if res_type == 'vehicle':
        return resource_class(
            E_max_kWh=_pick(res_config, 'E_max_kWh', 'capacity_kwh', default=60.0),
            soc_init=_pick(res_config, 'soc_init', 'initial_soc', default=0.8),
            soc_min=_pick(res_config, 'soc_min', default=0.1),
            soc_max=_pick(res_config, 'soc_max', default=0.95),
            soc_departure_min=_pick(res_config, 'soc_departure_min', default=0.8),
            p_charge_max_kW=_pick(res_config, 'p_charge_max_kW', 'charge_power_kw', default=7.0),
            p_discharge_max_kW=_pick(res_config, 'p_discharge_max_kW',
                                     'discharge_power_kw', default=7.0),
            eta_charge=_pick(res_config, 'eta_charge', default=0.95),
            eta_discharge=_pick(res_config, 'eta_discharge', default=0.95),
            commute_schedule=res_config.get('commute_schedule'),
            delta_t_minutes=_pick(res_config, 'delta_t_minutes',
                                  default=default_delta_t_minutes),
        )

    if res_type == 'flexload':
        return resource_class(
            curtail_cap_mw=_pick(res_config, 'curtail_cap_mw', default=10.0),
            shift_cap_mw=_pick(res_config, 'shift_cap_mw', default=10.0),
            shift_horizon=_pick(res_config, 'shift_horizon', default=4),
            baseline_mw=_pick(res_config, 'baseline_mw', default=50.0),
            curtail_cost_per_mwh=_pick(res_config, 'curtail_cost_per_mwh', default=50.0),
            shift_cost_per_mwh=_pick(res_config, 'shift_cost_per_mwh', default=10.0),
            complementarity_penalty=_pick(res_config, 'complementarity_penalty', default=100.0),
            price_ref=_pick(res_config, 'price_ref', default=100.0),
            action_scale=_pick(res_config, 'action_scale', default='unit'),
            delta_t_minutes=_pick(res_config, 'delta_t_minutes',
                                  default=default_delta_t_minutes),
        )

    # res_type == 'datacenter'
    return resource_class(
        n_gpus=_pick(res_config, 'n_gpus', default=1000),
        gpu_idle_w=_pick(res_config, 'gpu_idle_w', default=55.0),
        gpu_active_w=_pick(res_config, 'gpu_active_w', default=1100.0),
        p_base_mw=_pick(res_config, 'p_base_mw', default=0.5),
        infer_gpu_peak=_pick(res_config, 'infer_gpu_peak', default=400),
        cop_ref=_pick(res_config, 'cop_ref', default=5.0),
        cop_decay=_pick(res_config, 'cop_decay', default=0.04),
        t_ref=_pick(res_config, 't_ref', default=20.0),
        c_thermal=_pick(res_config, 'c_thermal', default=500.0),
        ua_cooling=_pick(res_config, 'ua_cooling', default=200.0),
        h_wall=_pick(res_config, 'h_wall', default=5.0),
        t_set_min=_pick(res_config, 't_set_min', default=18.0),
        t_set_max=_pick(res_config, 't_set_max', default=27.0),
        t_critical=_pick(res_config, 't_critical', default=35.0),
        p_aux_frac=_pick(res_config, 'p_aux_frac', default=0.05),
        train_cfg=res_config.get('train_cfg'),
        finetune_cfg=res_config.get('finetune_cfg'),
        delta_t_minutes=_pick(res_config, 'delta_t_minutes',
                              default=default_delta_t_minutes),
    )


def build_resource_metadata(
    resource_id: str,
    res_config: Dict[str, Any],
    resource: Any,
) -> Dict[str, Any]:
    """Build normalized metadata for an attached resource.

    Physical parameters are read directly from the instantiated *resource*
    object so that metadata always reflects the values the environment was
    actually constructed with (including any defaults applied during
    construction), rather than re-parsing the config dict.
    """
    metadata: Dict[str, Any] = {
        'type': res_config['type'],
        'name': resource_id,
        'bus_id': res_config['bus_id'],
    }
    if res_config['type'] == 'battery':
        metadata.update({
            'capacity_mwh': resource.capacity_mwh,
            'power_mw': resource.power_mw,
            'initial_soc': resource.initial_soc,
            'soc_min': resource.soc_min,
            'soc_max': resource.soc_max,
        })
    elif res_config['type'] == 'vehicle':
        # VehicleEnv stores capacity in MWh and power in MW internally;
        # metadata uses kWh / kW to match the config convention.
        metadata.update({
            'capacity_kwh': resource.capacity_mwh * 1000.0,
            'charge_power_kw': resource.p_charge_max_mw * 1000.0,
            'discharge_power_kw': resource.p_discharge_max_mw * 1000.0,
            'initial_soc': resource.soc_init,
            'soc_min': resource.soc_min,
            'soc_max': resource.soc_max,
            'soc_departure_min': resource.soc_departure_min,
        })
    elif res_config['type'] == 'datacenter':
        metadata.update({
            'n_gpus': resource.n_gpus,
            'gpu_idle_w': resource.gpu_idle_w,
            'gpu_active_w': resource.gpu_active_w,
            'p_base_mw': resource.p_base_mw,
            'infer_gpu_peak': resource.infer_gpu_peak,
            'cop_ref': resource.cop_ref,
            't_critical': resource.t_critical,
        })
    elif res_config['type'] in ('solar', 'wind'):
        metadata['capacity_mw'] = resource.capacity_mw
        metadata['normalize_actions'] = getattr(resource, 'normalize_actions', True)
        metadata['enable_q_control'] = getattr(resource, 'enable_q_control', False)
    elif res_config['type'] == 'flexload':
        metadata.update({
            'curtail_cap_mw': resource.curtail_cap_mw,
            'shift_cap_mw': resource.shift_cap_mw,
            'baseline_mw': resource.baseline_mw,
            'shift_horizon': resource.shift_horizon,
        })
    else:
        raise ValueError(
            f"Unhandled resource type in metadata extraction: {res_config['type']!r}"
        )
    return metadata


def attach_resources(
    grid: Any,
    resources_config: Iterable[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Create and attach resources to a grid, returning resources and metadata."""
    resources: Dict[str, Any] = {}
    metadata: Dict[str, Dict[str, Any]] = {}

    for res_config in resources_config:
        resource = create_resource(
            res_config,
            default_delta_t_minutes=grid.delta_t_minutes,
        )
        res_dt = getattr(resource, 'delta_t_minutes', None)
        if res_dt is not None and res_dt != grid.delta_t_minutes:
            warnings.warn(
                f"Resource delta_t_minutes={res_dt} differs from grid "
                f"delta_t_minutes={grid.delta_t_minutes}.  Energy calculations "
                f"(e.g. SOC updates) will use the resource's own timestep, "
                f"which may cause modelling inconsistencies.",
                UserWarning,
                stacklevel=2,
            )
        resource_id = resource.attach(
            grid,
            bus_id=res_config['bus_id'],
            name=res_config.get('name'),
        )
        resources[resource_id] = resource
        metadata[resource_id] = build_resource_metadata(resource_id, res_config, resource)

    return resources, metadata
