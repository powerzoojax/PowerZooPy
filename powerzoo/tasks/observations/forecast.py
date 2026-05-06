"""Forecast-related helpers for benchmark observations."""

from __future__ import annotations

from typing import Any, Iterable, Tuple

import numpy as np

from powerzoo.data import signals as S


def get_price_signal(
    hour: int,
    peak_hours: Iterable[int],
    off_peak_hours: Iterable[int],
) -> Tuple[float, float, float]:
    """Return normalized current price, peak flag, and off-peak flag."""
    peak_hours = set(peak_hours)
    off_peak_hours = set(off_peak_hours)

    if hour in peak_hours:
        return 1.0, 1.0, 0.0
    if hour in off_peak_hours:
        return 0.0, 0.0, 1.0
    return 0.5, 0.0, 0.0


def build_price_forecast(
    *,
    time_step: int,
    horizon_steps: int,
    delta_t_minutes: int,
    peak_hours: Iterable[int],
    off_peak_hours: Iterable[int],
) -> np.ndarray:
    """Build a normalized future price-signal vector."""
    if horizon_steps <= 0:
        return np.zeros(0, dtype=np.float32)

    values = []
    for offset in range(1, horizon_steps + 1):
        future_hour = int(((time_step + offset) * delta_t_minutes) / 60) % 24
        price_signal, _, _ = get_price_signal(future_hour, peak_hours, off_peak_hours)
        values.append(price_signal)
    return np.asarray(values, dtype=np.float32)


def _load_series(grid: Any) -> np.ndarray:
    _LOAD = S.LOAD_ACTUAL_MW
    time_series = getattr(grid, '_time_series_data', None)
    if time_series is None:
        return np.zeros(0, dtype=np.float32)
    if _LOAD in time_series.columns:
        return time_series[_LOAD].values.astype(np.float32)
    # Legacy fallback
    if 'ActualDemand' in time_series.columns:
        return time_series['ActualDemand'].values.astype(np.float32)
    return np.zeros(0, dtype=np.float32)


def load_normalizer(grid: Any) -> float:
    """Return a stable normalization constant for current/future total load."""
    series = _load_series(grid)
    if series.size:
        return max(float(np.max(series)), 1.0)

    case = getattr(grid, 'case', None)
    if case is not None and hasattr(case, 'units') and 'p_max' in case.units.columns:
        return max(float(case.units['p_max'].sum()), 1.0)
    return 1.0


def current_total_load_norm(grid: Any) -> float:
    """Return normalized current total load."""
    if hasattr(grid, '_get_node_loads_p_current'):
        total_load_mw = float(np.sum(grid._get_node_loads_p_current()))
    else:
        total_load_mw = 0.0
    return total_load_mw / load_normalizer(grid)


def build_load_forecast(grid: Any, horizon_steps: int) -> np.ndarray:
    """Build a normalized future total-load forecast vector."""
    if horizon_steps <= 0:
        return np.zeros(0, dtype=np.float32)

    series = _load_series(grid)
    if series.size == 0 or not hasattr(grid, '_get_current_time_index'):
        return np.zeros(horizon_steps, dtype=np.float32)

    current_idx = grid._get_current_time_index()
    if current_idx < 0:
        return np.zeros(horizon_steps, dtype=np.float32)

    norm = load_normalizer(grid)
    values = []
    last_idx = len(series) - 1
    for offset in range(1, horizon_steps + 1):
        values.append(series[min(current_idx + offset, last_idx)] / norm)
    return np.asarray(values, dtype=np.float32)


def build_ev_home_forecast(ev: Any, horizon_steps: int) -> np.ndarray:
    """Predict future EV at-home availability over the next horizon."""
    if horizon_steps <= 0:
        return np.zeros(0, dtype=np.float32)

    step_hours = float(getattr(ev, 'delta_t_minutes', 60.0)) / 60.0
    current_time = float(getattr(ev, 'time_of_day', 0.0))
    commute_schedule = list(getattr(ev, 'commute_schedule', []))
    if not commute_schedule:
        return np.ones(horizon_steps, dtype=np.float32)

    values = []
    for offset in range(1, horizon_steps + 1):
        future_time = (current_time + offset * step_hours) % 24.0
        is_home = 1.0
        for trip in commute_schedule:
            departure = float(trip['departure'])
            arrival = float(trip['arrival'])
            if departure <= arrival:
                in_trip = departure <= future_time < arrival
            else:
                in_trip = future_time >= departure or future_time < arrival
            if in_trip:
                is_home = 0.0
                break
        values.append(is_home)
    return np.asarray(values, dtype=np.float32)
