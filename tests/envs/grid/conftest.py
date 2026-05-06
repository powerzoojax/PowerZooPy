"""Shared fixtures for powerzoo.envs.grid tests.

Provides reusable grids, mock cases, and time-series data for unit testing.
Power system test fixtures follow standard IEEE test feeder conventions.
"""
import pytest
import numpy as np
import pandas as pd

from powerzoo.data import signals as S


@pytest.fixture
def rng():
    """Deterministic RNG for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_time_series():
    """A 2-day, 30-min resolution time series (96 steps) with demand profile.

    Demand follows a simplified daily pattern:
      - Daytime (8h-20h): high demand
      - Night (20h-8h): low demand

    Uses semantic signal names as column headers.
    """
    n_steps = 96  # 2 days × 48 steps/day
    idx = pd.date_range('2024-01-01', periods=n_steps, freq='30min', tz='UTC')
    hours = idx.hour + idx.minute / 60.0
    demand = 400 + 200 * np.sin(2 * np.pi * (hours - 6) / 24.0)
    solar = np.maximum(0, np.sin(2 * np.pi * (hours - 6) / 24.0)) * 150
    wind = 50 + 30 * np.sin(2 * np.pi * hours / 12.0)
    return pd.DataFrame(
        {S.LOAD_ACTUAL_MW: demand, S.SOLAR_AVAILABLE_MW: solar, S.WIND_AVAILABLE_MW: wind},
        index=idx,
    )


@pytest.fixture
def simple_numpy_demand():
    """1-day demand as flat numpy array (48 steps at 30-min)."""
    return np.linspace(300, 600, 48)  # monotonic ramp


@pytest.fixture
def case5():
    """IEEE 5-bus transmission test case."""
    from powerzoo.case import load_case
    return load_case(5)


@pytest.fixture
def case33():
    """IEEE 33-bus distribution test case (Case33bw)."""
    from powerzoo.case.distribution import Case33bw
    return Case33bw()
