"""Resource-centric observation builders for benchmark tasks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np

from .base import concat_observation_parts, get_time_context
from .forecast import (
    build_ev_home_forecast,
    build_load_forecast,
    build_price_forecast,
    get_price_signal,
)
from .grid import RESOURCE_GLOBAL_FIELDS, grid_voltage_summary


def _resource_local_fields() -> Tuple[str, ...]:
    return (
        'soc',
        'p_mw_norm',
        'time_of_day',
        'hour_norm',
        'price_signal',
        'is_peak',
        'is_offpeak',
        'power_limit_norm',
        'capacity_norm',
        'episode_progress',
        'target_soc',
    )


def _read_local_bus_voltage(grid: Any, bus_id: int) -> float:
    """Return the v_mag at *bus_id* (1-indexed) from the current power-flow result.

    Falls back to 1.0 p.u. when the grid has not run a power flow yet or when
    the bus index is out of range.
    """
    nodes = getattr(grid, '_nodes', None)
    if nodes is None or 'v_mag' not in nodes.columns:
        return 1.0
    values = nodes['v_mag'].values
    idx = int(bus_id) - 1
    if idx < 0 or idx >= len(values):
        return 1.0
    v = float(values[idx])
    return v if np.isfinite(v) else 1.0


def build_resource_observation_fields(
    *,
    mode: str,
    forecast_horizon_steps: int,
) -> Tuple[str, ...]:
    """Return standardized DER observation field names."""
    local_fields = _resource_local_fields()
    if mode == 'global':
        return RESOURCE_GLOBAL_FIELDS + local_fields
    if mode == 'local':
        return local_fields
    if mode == 'local_plus_voltage':
        return local_fields + ('local_bus_voltage',)
    return local_fields + tuple(
        [f'load_forecast_t+{step}' for step in range(1, forecast_horizon_steps + 1)]
        + [f'price_forecast_t+{step}' for step in range(1, forecast_horizon_steps + 1)]
    )


def build_resource_observations(
    *,
    grid: Any,
    resources: Mapping[str, Any],
    resource_info: Mapping[str, Mapping[str, float]],
    step_count: int,
    max_steps: int,
    obs_mode: str,
    forecast_horizon_steps: int,
    delta_t_minutes: int,
    peak_hours: Iterable[int],
    off_peak_hours: Iterable[int],
    target_soc: float,
) -> Dict[str, np.ndarray]:
    """Build standardized per-resource observations for DER tasks."""
    context = get_time_context(
        grid=grid,
        step_count=step_count,
        max_steps=max_steps,
        delta_t_minutes=delta_t_minutes,
    )
    hour = int(context['hour'])
    price_signal, is_peak, is_offpeak = get_price_signal(hour, peak_hours, off_peak_hours)
    global_summary = grid_voltage_summary(grid)
    load_forecast = build_load_forecast(grid, forecast_horizon_steps)
    price_forecast = build_price_forecast(
        time_step=int(context['time_step']),
        horizon_steps=forecast_horizon_steps,
        delta_t_minutes=delta_t_minutes,
        peak_hours=peak_hours,
        off_peak_hours=off_peak_hours,
    )

    observations: Dict[str, np.ndarray] = {}
    for res_id, resource in resources.items():
        info = resource_info[res_id]
        power_mw = max(float(info['power_mw']), 1e-6)
        capacity_mwh = max(float(info['capacity_mwh']), 1e-6)
        local_obs = np.array([
            float(getattr(resource, 'soc', 0.5)),
            float(getattr(resource, 'current_p_mw', 0.0)) / power_mw,
            context['time_of_day'],
            hour / 24.0,
            price_signal,
            is_peak,
            is_offpeak,
            power_mw / 100.0,
            capacity_mwh / 100.0,
            context['episode_progress'],
            float(target_soc),
        ], dtype=np.float32)

        if obs_mode == 'global':
            observations[res_id] = concat_observation_parts(global_summary, local_obs)
        elif obs_mode == 'local':
            observations[res_id] = local_obs
        elif obs_mode == 'local_plus_voltage':
            bus_id = int(info.get('bus_id', 0))
            v_local = np.array([_read_local_bus_voltage(grid, bus_id)], dtype=np.float32)
            observations[res_id] = concat_observation_parts(local_obs, v_local)
        else:
            observations[res_id] = concat_observation_parts(local_obs, load_forecast, price_forecast)

    return observations


def _ev_local_fields() -> Tuple[str, ...]:
    return (
        'soc',
        'is_home',
        'departure_ready',
        'time_to_departure_norm',
        'time_of_day',
        'hour_norm',
        'price_signal',
        'is_peak',
        'is_offpeak',
        'charge_max_norm',
        'discharge_max_norm',
        'episode_progress',
        'soc_departure_min',
    )


def build_ev_observation_fields(
    *,
    mode: str,
    forecast_horizon_steps: int,
) -> Tuple[str, ...]:
    """Return standardized EV observation field names."""
    local_fields = _ev_local_fields()
    if mode == 'global':
        return RESOURCE_GLOBAL_FIELDS + local_fields
    if mode == 'local':
        return local_fields
    return local_fields + tuple(
        [f'load_forecast_t+{step}' for step in range(1, forecast_horizon_steps + 1)]
        + [f'price_forecast_t+{step}' for step in range(1, forecast_horizon_steps + 1)]
        + [f'home_forecast_t+{step}' for step in range(1, forecast_horizon_steps + 1)]
    )


def _time_to_departure(ev: Any) -> float:
    time_to_departure = 24.0
    current_time = float(getattr(ev, 'time_of_day', 0.0))
    for trip in getattr(ev, 'commute_schedule', []):
        dep_time = float(trip['departure'])
        if dep_time > current_time:
            time_to_departure = min(time_to_departure, dep_time - current_time)
        else:
            time_to_departure = min(time_to_departure, 24.0 - current_time + dep_time)
    return time_to_departure


# ─────────────────────────────────────────────────────────────────────────────
# DERs local observation builder (12-dim, heterogeneous-agent benchmark)
# ─────────────────────────────────────────────────────────────────────────────

DERS_OBS_DIM = 12


def _ders_local_fields() -> Tuple[str, ...]:
    """12 field names for the ``'ders_local'`` observation mode.

    Layout::

        Shared context (7):
          time_of_day, hour_norm, price_signal, is_peak,
          is_offpeak, episode_progress, local_bus_voltage
        Device state (5):
          ders_state_[0-4]  — type-specific, same slot count for all agents
    """
    return (
        'time_of_day',
        'hour_norm',
        'price_signal',
        'is_peak',
        'is_offpeak',
        'episode_progress',
        'local_bus_voltage',
        'ders_state_0',
        'ders_state_1',
        'ders_state_2',
        'ders_state_3',
        'ders_state_4',
    )


def build_ders_observation_fields(*, mode: str = 'ders_local') -> Tuple[str, ...]:
    """Return the 12 field names for the DERs benchmark observation."""
    return _ders_local_fields()


def _ders_device_obs(
    resource: Any,
    res_type: str,
    resource_info: Mapping[str, float],
) -> np.ndarray:
    """Return a 5-dim device-specific state vector for one DER resource.

    Slot semantics by type::

        Battery   [soc, p_mw_norm, q_mvar_norm, soc_headroom, soc_floor]
        Solar/PV  [available_cf, p_mw_norm, q_mvar_norm, curtailment_norm, 0.0]
        FlexLoad  [curtail_norm, shift_out_norm, shift_in_norm,
                   buffer_fill_ratio, buffer_energy_norm]
    """
    if 'battery' in res_type:
        soc = float(getattr(resource, 'soc', 0.5))
        power_mw = max(float(resource_info.get('power_mw', getattr(resource, 'power_mw', 20.0))), 1e-6)
        s_rated = max(float(getattr(resource, 's_rated_mva', power_mw)), 1e-6)
        soc_min = float(resource_info.get('soc_min', getattr(resource, 'soc_min', 0.1)))
        soc_max = float(resource_info.get('soc_max', getattr(resource, 'soc_max', 0.9)))
        p_norm = float(getattr(resource, 'current_p_mw', 0.0)) / power_mw
        q_norm = float(getattr(resource, 'current_q_mvar', 0.0)) / s_rated
        headroom = float(np.clip(soc_max - soc, 0.0, 1.0))
        floor = float(np.clip(soc - soc_min, 0.0, 1.0))
        return np.array([soc, p_norm, q_norm, headroom, floor], dtype=np.float32)

    if res_type in ('solar', 'wind') or 'renewable' in res_type:
        cap_mw = max(float(resource_info.get('power_mw', getattr(resource, 'capacity_mw', 100.0))), 1e-6)
        s_rated = max(float(getattr(resource, 's_rated_mva', cap_mw)), 1e-6)
        avail_cf = float(getattr(resource, 'available_cf', 0.0))
        p_mw = float(getattr(resource, 'current_p_mw', 0.0))
        p_norm = p_mw / cap_mw
        q_norm = float(getattr(resource, 'current_q_mvar', 0.0)) / s_rated
        avail_p = avail_cf * cap_mw
        curt_norm = float(np.clip((avail_p - p_mw) / max(avail_p, 1e-6), 0.0, 1.0))
        return np.array([avail_cf, p_norm, q_norm, curt_norm, 0.0], dtype=np.float32)

    if 'flex' in res_type or res_type == 'flexload':
        curtail_cap = max(float(getattr(resource, 'curtail_cap_mw', 10.0)), 1e-6)
        shift_cap = max(float(getattr(resource, 'shift_cap_mw', 10.0)), 1e-6)
        horizon = max(int(getattr(resource, 'shift_horizon', 4)), 1)
        buf = getattr(resource, '_buffer', None)
        curt_norm = float(np.clip(getattr(resource, '_curtailed_mw', 0.0) / curtail_cap, 0.0, 1.0))
        sout_norm = float(np.clip(getattr(resource, '_shift_out_mw', 0.0) / shift_cap, 0.0, 1.0))
        sin_norm = float(np.clip(getattr(resource, '_shift_in_mw', 0.0) / shift_cap, 0.0, 1.0))
        fill_ratio = float(np.clip(buf.fill_ratio, 0.0, 1.0)) if buf is not None else 0.0
        buf_cap_mw = shift_cap * horizon
        en_norm = float(np.clip(buf.total_mw / buf_cap_mw, 0.0, 1.0)) if buf is not None else 0.0
        return np.array([curt_norm, sout_norm, sin_norm, fill_ratio, en_norm], dtype=np.float32)

    # Generic fallback for unknown resource types
    soc = float(getattr(resource, 'soc', 0.5))
    power_mw = max(float(resource_info.get('power_mw', getattr(resource, 'power_mw', 20.0))), 1e-6)
    p_norm = float(getattr(resource, 'current_p_mw', 0.0)) / power_mw
    return np.array([soc, p_norm, 0.0, 0.0, 0.0], dtype=np.float32)


def build_ders_observations(
    *,
    grid: Any,
    resources: Mapping[str, Any],
    resource_info: Mapping[str, Mapping[str, float]],
    step_count: int,
    max_steps: int,
    delta_t_minutes: int,
    peak_hours: Iterable[int],
    off_peak_hours: Iterable[int],
) -> Dict[str, np.ndarray]:
    """Build 12-dim per-resource observations for the DERs benchmark.

    Observation layout (12 dims, index into vector)::

        [0]   time_of_day       — normalised daily time [0, 1]
        [1]   hour_norm         — hour / 24
        [2]   price_signal      — 0 = off-peak, 1 = normal, 2 = peak
        [3]   is_peak           — binary {0, 1}
        [4]   is_offpeak        — binary {0, 1}
        [5]   episode_progress  — step_count / max_steps  [0, 1]
        [6]   local_bus_voltage — v_pu at resource's bus   (≈ 0.94 – 1.06)
        [7]   ders_state_0      \\ type-specific 5-dim device state
        [8]   ders_state_1      |   Battery:  [soc, p_norm, q_norm, headroom, floor]
        [9]   ders_state_2      |   PV:       [avail_cf, p_norm, q_norm, curt_norm, 0]
        [10]  ders_state_3      |   FlexLoad: [curt_n, sout_n, sin_n, fill, en_n]
        [11]  ders_state_4      /
    """
    context = get_time_context(
        grid=grid,
        step_count=step_count,
        max_steps=max_steps,
        delta_t_minutes=delta_t_minutes,
    )
    hour = int(context['hour'])
    price_signal, is_peak, is_offpeak = get_price_signal(hour, peak_hours, off_peak_hours)

    shared_context = np.array([
        context['time_of_day'],
        hour / 24.0,
        price_signal,
        is_peak,
        is_offpeak,
        context['episode_progress'],
    ], dtype=np.float32)

    observations: Dict[str, np.ndarray] = {}
    for res_id, resource in resources.items():
        info = resource_info[res_id]
        res_type = str(info.get('type', resource.__class__.__name__.lower()))
        bus_id = int(info.get('bus_id', 0))
        v_local = np.array([_read_local_bus_voltage(grid, bus_id)], dtype=np.float32)
        device_obs = _ders_device_obs(resource, res_type, info)
        observations[res_id] = np.concatenate([shared_context, v_local, device_obs])

    return observations


def build_ev_observations(
    *,
    grid: Any,
    resources: Mapping[str, Any],
    resource_info: Mapping[str, Mapping[str, float]],
    step_count: int,
    max_steps: int,
    obs_mode: str,
    forecast_horizon_steps: int,
    delta_t_minutes: int,
    peak_hours: Iterable[int],
    off_peak_hours: Iterable[int],
) -> Dict[str, np.ndarray]:
    """Build standardized per-EV observations."""
    context = get_time_context(
        grid=grid,
        step_count=step_count,
        max_steps=max_steps,
        delta_t_minutes=delta_t_minutes,
    )
    hour = int(context['hour'])
    price_signal, is_peak, is_offpeak = get_price_signal(hour, peak_hours, off_peak_hours)
    global_summary = grid_voltage_summary(grid)
    load_forecast = build_load_forecast(grid, forecast_horizon_steps)
    price_forecast = build_price_forecast(
        time_step=int(context['time_step']),
        horizon_steps=forecast_horizon_steps,
        delta_t_minutes=delta_t_minutes,
        peak_hours=peak_hours,
        off_peak_hours=off_peak_hours,
    )

    observations: Dict[str, np.ndarray] = {}
    for res_id, ev in resources.items():
        info = resource_info[res_id]
        avail = ev.available_power()
        charge_scale = max(float(info['charge_power_kw']), 1e-6)
        discharge_scale = max(float(info['discharge_power_kw']), 1e-6)

        local_obs = np.array([
            float(ev.soc),
            1.0 if ev.is_home else 0.0,
            1.0 if ev.check_departure_ready() else 0.0,
            _time_to_departure(ev) / 24.0,
            context['time_of_day'],
            hour / 24.0,
            price_signal,
            is_peak,
            is_offpeak,
            float(avail['p_charge_max_mw']) * 1000.0 / charge_scale,
            float(avail['p_discharge_max_mw']) * 1000.0 / discharge_scale,
            context['episode_progress'],
            float(info['soc_departure_min']),
        ], dtype=np.float32)

        if obs_mode == 'global':
            observations[res_id] = concat_observation_parts(global_summary, local_obs)
        elif obs_mode == 'local':
            observations[res_id] = local_obs
        else:
            home_forecast = build_ev_home_forecast(ev, forecast_horizon_steps)
            observations[res_id] = concat_observation_parts(
                local_obs,
                load_forecast,
                price_forecast,
                home_forecast,
            )

    return observations
