"""Grid-centric observation builders for benchmark tasks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np

from .base import concat_observation_parts, get_time_context
from .forecast import build_load_forecast, current_total_load_norm


RESOURCE_GLOBAL_FIELDS: Tuple[str, str, str] = (
    'total_load_norm',
    'mean_voltage_dev',
    'grid_stress',
)


def grid_voltage_summary(grid: Any) -> np.ndarray:
    """Return a compact normalized global grid summary for distribution tasks."""
    if hasattr(grid, '_get_state'):
        state = grid._get_state()
    else:
        state = {}

    nodes = state.get('nodes')
    if nodes is None or 'v_mag' not in nodes.columns:
        return np.zeros(3, dtype=np.float32)

    v_mag = nodes['v_mag'].values.astype(np.float32)
    mean_v_dev = float(np.mean(v_mag) - 1.0)
    max_v_dev = float(np.max(np.abs(v_mag - 1.0)))
    loss_mw = float(state.get('p_loss_MW', 0.0))
    loss_ratio = loss_mw / max(float(np.max(np.abs(v_mag))), 1.0)
    return np.array(
        [current_total_load_norm(grid), mean_v_dev, max(max_v_dev, loss_ratio)],
        dtype=np.float32,
    )


def _opf_global_fields(n_lines: int) -> Tuple[str, ...]:
    return tuple(
        ['total_load_norm']
        + [f'line_flow_util_{i}' for i in range(n_lines)]
        + ['time_of_day', 'episode_progress']
    )


def _opf_local_fields(n_lines: int) -> Tuple[str, ...]:
    return tuple(
        ['bus_load_norm']
        + [f'adjacent_line_flow_util_{i}' for i in range(n_lines)]
        + [
            'time_of_day',
            'episode_progress',
            'unit_idx_norm',
            'p_min_norm',
            'p_max_norm',
            'mc_a',
            'mc_b_norm',
            'mc_c_norm',
        ]
    )


def build_opf_observation_fields(
    *,
    mode: str,
    n_lines: int,
    forecast_horizon_steps: int,
) -> Tuple[str, ...]:
    """Return standardized OPF field names for the selected observation mode."""
    if mode == 'global':
        return _opf_global_fields(n_lines) + (
            'unit_idx_norm',
            'p_min_norm',
            'p_max_norm',
            'mc_a',
            'mc_b_norm',
            'mc_c_norm',
        )
    if mode == 'local':
        return _opf_local_fields(n_lines)
    return _opf_local_fields(n_lines) + tuple(
        f'load_forecast_t+{step}'
        for step in range(1, forecast_horizon_steps + 1)
    )


def build_opf_observations(
    *,
    grid: Any,
    case: Any,
    possible_agents: Iterable[str],
    n_units: int,
    n_lines: int,
    p_min: np.ndarray,
    p_max: np.ndarray,
    mc_a: np.ndarray,
    mc_b: np.ndarray,
    mc_c: np.ndarray,
    step_count: int,
    max_steps: int,
    obs_mode: str,
    forecast_horizon_steps: int,
    total_load_mw: float,
    line_flows: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build per-agent OPF observation vectors."""
    line_caps = case.lines['cap'].values.astype(np.float32)
    normalized_flows = np.where(line_caps > 0, line_flows / line_caps, 0.0).astype(np.float32)

    context = get_time_context(
        grid=grid,
        step_count=step_count,
        max_steps=max_steps,
        delta_t_minutes=int(grid.delta_t_minutes),
    )
    time_feats = np.array(
        [context['time_of_day'], context['episode_progress']],
        dtype=np.float32,
    )

    unit_bus_ids = (
        case.units['bus_id'].values
        if 'bus_id' in case.units.columns
        else np.zeros(n_units, dtype=int)
    )
    node_load_arr = (
        grid._get_node_loads_p_current()
        if hasattr(grid, '_get_node_loads_p_current')
        else np.array([total_load_mw / max(n_units, 1)])
    )
    global_obs = concat_observation_parts([total_load_mw / 1000.0], normalized_flows, time_feats)
    load_forecast = build_load_forecast(grid, forecast_horizon_steps)

    observations: Dict[str, np.ndarray] = {}
    lines = case.lines
    has_line_bus_cols = '#from' in lines.columns and '#to' in lines.columns
    from_ids = lines['#from'].values.astype(int) if has_line_bus_cols else None
    to_ids = lines['#to'].values.astype(int) if has_line_bus_cols else None

    for i, agent in enumerate(possible_agents):
        bus_id = int(unit_bus_ids[i])
        connected_mask = np.zeros(n_lines, dtype=np.float32)
        if has_line_bus_cols:
            connected_mask = ((from_ids == bus_id) | (to_ids == bus_id)).astype(np.float32)

        local_flows = normalized_flows * connected_mask
        bus_load = float(node_load_arr[min(bus_id, len(node_load_arr) - 1)]) / 1000.0
        local_unit = np.array([
            float(i) / max(n_units, 1),
            p_min[i] / 100.0,
            p_max[i] / 100.0,
            mc_a[i],
            mc_b[i] / 100.0,
            mc_c[i] / 1000.0,
        ], dtype=np.float32)
        local_obs = concat_observation_parts([bus_load], local_flows, time_feats, local_unit)

        if obs_mode == 'global':
            observations[agent] = concat_observation_parts(global_obs, local_unit)
        elif obs_mode == 'local':
            observations[agent] = local_obs
        else:
            observations[agent] = concat_observation_parts(local_obs, load_forecast)

    return observations
