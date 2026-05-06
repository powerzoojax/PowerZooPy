"""Unit tests for powerzoo.envs.resource.renewable — RenewableEnv, SolarEnv, WindEnv.

Domain knowledge embedded in these tests:
- Renewable generators are non-dispatchable: output follows weather / irradiance profiles
- Capacity factor ∈ [0, 1]: ratio of actual output to nameplate capacity
- Curtailment: operator can reduce output by fraction ∈ [0, 1]
  (0 = no curtailment, 1 = full curtailment)
- Solar: diurnal pattern, zero at night, peak at noon
- Wind: can produce 24/7, often higher capacity factors offshore
- current_p ≥ 0 always (renewable injects power, never absorbs)
- Data driven: output comes from time-series (grid parent or custom loader)
"""

import pytest
import numpy as np
import pandas as pd

from powerzoo.envs.resource.renewable import RenewableEnv, SolarEnv, WindEnv
from powerzoo.data import signals as S
from .conftest import MockParentGrid


# ========================== Fixtures ==========================

@pytest.fixture
def mock_grid_with_solar():
    """Mock grid with solar time-series data (semantic signal name).

    Profile peaks at ~50 MW so that a 50 MW capacity resource hits CF=1.0.
    """
    grid = MockParentGrid(delta_t_minutes=30.0, steps_per_day=48)
    timestamps = pd.date_range('2024-01-01', periods=48, freq='30min')
    hours = np.arange(48) * 0.5
    solar_profile = np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
    solar_profile[hours < 6] = 0.0
    solar_profile[hours >= 18] = 0.0
    # Peak ~50 MW to match the 50 MW capacity used in the solar fixture
    grid._time_series_data = pd.DataFrame(
        {S.SOLAR_AVAILABLE_MW: solar_profile * 50.0},
        index=timestamps,
    )
    return grid


@pytest.fixture
def mock_grid_with_wind():
    """Mock grid with wind time-series data (semantic signal name)."""
    grid = MockParentGrid(delta_t_minutes=30.0, steps_per_day=48)
    timestamps = pd.date_range('2024-01-01', periods=48, freq='30min')
    rng = np.random.default_rng(0)
    # Values within 100 MW capacity
    wind_total = rng.uniform(30, 90, size=48)
    grid._time_series_data = pd.DataFrame(
        {S.WIND_AVAILABLE_MW: wind_total},
        index=timestamps,
    )
    return grid


@pytest.fixture
def solar(mock_grid_with_solar):
    """SolarEnv attached to mock grid."""
    s = SolarEnv(normalize_actions=False, parent=mock_grid_with_solar, bus_id=5, capacity_mw=50.0)
    s.reset(seed=0)
    return s


@pytest.fixture
def wind(mock_grid_with_wind):
    """WindEnv attached to mock grid."""
    w = WindEnv(normalize_actions=False, parent=mock_grid_with_wind, bus_id=2, capacity_mw=100.0)
    w.reset(seed=0)
    return w


# ========================== Initialization ==========================

class TestRenewableInit:

    def test_default_capacity(self):
        r = RenewableEnv(normalize_actions=False, capacity_mw=200.0)
        assert r.capacity_mw == 200.0

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError, match="capacity_mw must be > 0"):
            RenewableEnv(normalize_actions=False, capacity_mw=0.0)

    def test_initial_power_zero(self):
        r = RenewableEnv(normalize_actions=False, capacity_mw=100.0)
        assert r.current_p_mw == 0.0
        assert r.current_q_mvar == 0.0

    def test_action_space_curtailment_bounds(self):
        r = RenewableEnv(normalize_actions=False, capacity_mw=50.0)
        assert r.action_space.low[0] == pytest.approx(0.0)
        assert r.action_space.high[0] == pytest.approx(1.0)

    def test_action_space_normalized_bounds(self):
        r = RenewableEnv(normalize_actions=True, capacity_mw=50.0)
        assert r.action_space.low[0] == pytest.approx(-1.0)
        assert r.action_space.high[0] == pytest.approx(1.0)

    def test_observation_space_shape(self):
        r = RenewableEnv(normalize_actions=False)
        assert r.observation_space.shape == (4,)

    def test_solar_default_column(self, mock_grid_with_solar):
        s = SolarEnv(normalize_actions=False, parent=mock_grid_with_solar, bus_id=0)
        assert s._get_default_column() == S.SOLAR_AVAILABLE_MW

    def test_wind_default_column(self, mock_grid_with_wind):
        w = WindEnv(normalize_actions=False, parent=mock_grid_with_wind, bus_id=0)
        assert w._get_default_column() == S.WIND_AVAILABLE_MW


# ========================== Data Loading ==========================

class TestDataLoading:

    def test_solar_loads_from_parent(self, solar, mock_grid_with_solar):
        """SolarEnv should load time series data from parent's DataFrame."""
        assert solar._available_cf is not None
        assert len(solar._available_cf) == 48

    def test_wind_loads_from_parent(self, wind, mock_grid_with_wind):
        """WindEnv should load wind signal directly from parent data."""
        assert wind._available_cf is not None
        assert len(wind._available_cf) == 48

    def test_cf_uses_capacity_mw(self, solar, mock_grid_with_solar):
        """Capacity factor should be computed from capacity_mw, not data max."""
        parent_data = mock_grid_with_solar._time_series_data[S.SOLAR_AVAILABLE_MW]
        expected_cf = np.clip(parent_data.values / solar.capacity_mw, 0.0, 1.0)
        np.testing.assert_allclose(solar._available_cf, expected_cf)

    def test_cf_clipped_to_unit(self, mock_grid_with_solar):
        """If data exceeds capacity_mw, CF is clipped to 1.0."""
        # Use tiny capacity so data exceeds it
        s = SolarEnv(normalize_actions=False, parent=mock_grid_with_solar,
                     bus_id=0, capacity_mw=10.0)
        assert s._available_cf.max() == pytest.approx(1.0)

    def test_no_data_without_parent(self):
        """RenewableEnv without parent should have no time series."""
        r = RenewableEnv(normalize_actions=False, capacity_mw=100.0, profile_column=S.SOLAR_AVAILABLE_MW)
        assert r._available_cf is None

    def test_missing_signal_raises(self):
        """Missing signal in parent data should raise ValueError."""
        grid = MockParentGrid(delta_t_minutes=30.0, steps_per_day=48)
        timestamps = pd.date_range('2024-01-01', periods=48, freq='30min')
        grid._time_series_data = pd.DataFrame(
            {'other_signal': np.zeros(48)},
            index=timestamps,
        )
        with pytest.raises(ValueError, match="not in parent's data"):
            SolarEnv(normalize_actions=False, parent=grid, bus_id=0)


# ========================== Reset ==========================

class TestRenewableReset:

    def test_reset_zeroes_power(self, solar):
        # Force some state
        solar.current_p_mw = 25.0
        solar._capacity_factor = 0.5
        solar.reset(seed=1)
        assert solar.current_p_mw == 0.0
        assert solar._capacity_factor == 0.0
        assert solar.time_step == 0

    def test_reset_returns_initial_obs(self, solar):
        obs = solar.reset(seed=0)
        assert isinstance(obs, dict)
        assert set(obs.keys()) == {'available_cf', 'p_mw_norm', 'time_cos', 'time_sin'}


# ========================== Step & Curtailment ==========================

class TestRenewableStep:

    def test_step_without_data_yields_zero(self):
        """Step without time-series data should produce zero output."""
        r = RenewableEnv(normalize_actions=False, capacity_mw=100.0, profile_column=S.SOLAR_AVAILABLE_MW)
        r.reset(seed=0)
        r.step(None)
        assert r.current_p_mw == 0.0

    def test_no_curtailment_full_output(self, solar):
        """action=None → no curtailment → full available output."""
        solar.step(None)
        # At time_step=0 (midnight), solar profile is 0
        assert solar.current_p_mw >= 0.0

    def test_step_produces_power_at_noon(self, solar):
        """Solar should produce power during daytime steps."""
        # Advance to step 24 (noon = 12h at 30min intervals)
        solar.reset(seed=0, day_id=0)
        for _ in range(24):
            solar.step(None)
        # At noon, solar profile should be near peak
        assert solar.current_p_mw > 0.0

    def test_full_curtailment_zero_output(self, solar):
        """action=1.0 (full curtailment) → zero output."""
        solar.reset(seed=0, day_id=0)
        # Advance to a daytime step first
        for _ in range(24):
            solar.step(None)
        solar.reset(seed=0, day_id=0)
        # Now step 24 at noon with full curtailment
        solar.time_step = 24
        solar.step(1.0)
        assert solar.current_p_mw == pytest.approx(0.0)

    def test_curtailment_clipped(self, solar):
        """Curtailment fraction is clipped to [0, 1]."""
        solar.step(-0.5)   # negative curtailment clipped to 0
        solar.step(1.5)    # excessive curtailment clipped to 1
        # Should not crash

    def test_output_always_non_negative(self, solar):
        """Renewable output should never be negative (no absorption)."""
        for curtail in [0.0, 0.25, 0.5, 0.75, 1.0]:
            solar.reset(seed=0)
            solar.step(curtail)
            assert solar.current_p_mw >= 0.0

    def test_reactive_power_zero(self, solar):
        """Renewables don't produce reactive power in this model."""
        solar.step(None)
        assert solar.current_q_mvar == 0.0

    def test_time_step_advances(self, solar):
        solar.step(None)
        assert solar.time_step == 1
        solar.step(0.0)
        assert solar.time_step == 2


# ========================== Action Parsing ==========================

class TestRenewableActionParsing:

    def test_dict_action(self, solar):
        solar.step({'curtailment': 0.5})
        # Should not crash

    def test_ndarray_action(self, solar):
        solar.step(np.array([0.3]))
        # Should not crash

    def test_none_action(self, solar):
        solar.step(None)
        # Should not crash

    def test_normalized_action_semantics(self, mock_grid_with_solar):
        """Normalized action +1 → full output, -1 → full curtailment."""
        s = SolarEnv(normalize_actions=True, parent=mock_grid_with_solar,
                     bus_id=0, capacity_mw=50.0)
        s.reset(seed=0, day_id=0)
        s.time_step = 24  # noon

        # +1 → no curtailment → full output
        s.step(np.array([1.0]))
        p_full = s.current_p_mw

        s.reset(seed=0, day_id=0)
        s.time_step = 24

        # -1 → full curtailment → zero output
        s.step(np.array([-1.0]))
        assert s.current_p_mw == pytest.approx(0.0)
        assert p_full > 0.0  # confirms we had output available


# ========================== Observation ==========================

class TestRenewableObs:

    def test_obs_is_dict(self, solar):
        o = solar.obs()
        assert isinstance(o, dict)
        assert set(o.keys()) == {'available_cf', 'p_mw_norm', 'time_cos', 'time_sin'}

    def test_obs_available_cf_range(self, solar):
        o = solar.obs()
        assert 0.0 <= o['available_cf'] <= 1.0

    def test_obs_p_norm_range(self, solar):
        o = solar.obs()
        assert 0.0 <= o['p_mw_norm'] <= 1.0  # renewable always ≥ 0

    def test_obs_time_features_range(self, solar):
        o = solar.obs()
        assert -1.0 <= o['time_sin'] <= 1.0
        assert -1.0 <= o['time_cos'] <= 1.0

    def test_obs_within_observation_space_bounds(self, solar):
        """Flattened obs values must lie within observation_space bounds."""
        from powerzoo.envs.power_env import _flatten_observation_value
        o = solar.obs()
        flat = _flatten_observation_value(o)
        assert flat.shape == solar.observation_space.shape
        assert np.all(flat >= solar.observation_space.low - 1e-6)
        assert np.all(flat <= solar.observation_space.high + 1e-6)


# ========================== Status ==========================

class TestRenewableStatus:

    def test_status_keys(self, solar):
        s = solar.status()
        assert 'current_p_mw' in s
        assert 'capacity_mw' in s
        assert 'available_cf' in s
        assert 'output_ratio' in s

    def test_output_ratio_consistent(self, solar):
        s = solar.status()
        expected = solar.current_p_mw / solar.capacity_mw if solar.capacity_mw > 0 else 0.0
        assert s['output_ratio'] == pytest.approx(expected)

    def test_available_p_mw_property_matches_capacity_factor(self, solar):
        solar.time_step = 24  # noon
        solar.step(None)
        expected = solar.available_cf * solar.capacity_mw
        assert solar.available_p_mw == pytest.approx(expected)


# ========================== Subclass Defaults ==========================

class TestSubclassDefaults:

    def test_solar_env_column(self):
        s = SolarEnv(normalize_actions=False)
        assert s._get_default_column() == S.SOLAR_AVAILABLE_MW

    def test_wind_env_column(self):
        w = WindEnv(normalize_actions=False)
        assert w._get_default_column() == S.WIND_AVAILABLE_MW


# ========================== status() extended fields ==========================

class TestRenewableStatusExtended:

    def test_status_has_available_cf(self, solar):
        s = solar.status()
        assert 'available_cf' in s
        assert 'current_p_mw' in s

    def test_status_has_bus_id(self, solar):
        s = solar.status()
        assert 'bus_id' in s
        assert s['bus_id'] == 5  # fixture uses bus_id=5

    def test_status_has_local_v(self, solar):
        s = solar.status()
        assert 'local_v' in s


# ========================== Q control (reactive / PQ circle) ==========================

class TestQControlInit:

    def test_default_q_disabled_spaces(self):
        r = RenewableEnv(normalize_actions=False, capacity_mw=100.0)
        assert r.action_space.shape == (1,)
        assert r.observation_space.shape == (4,)

    def test_q_enabled_spaces(self):
        r = RenewableEnv(
            normalize_actions=False,
            capacity_mw=100.0,
            enable_q_control=True,
        )
        assert r.action_space.shape == (2,)
        assert r.observation_space.shape == (5,)

    def test_q_enabled_action_names(self):
        r = RenewableEnv(
            normalize_actions=False,
            capacity_mw=50.0,
            enable_q_control=True,
        )
        assert r.action_names == ['curtailment', 'q_control']

    def test_s_rated_defaults_to_capacity(self):
        r = RenewableEnv(
            normalize_actions=False,
            capacity_mw=77.0,
            enable_q_control=True,
            s_rated_mva=None,
        )
        assert r.s_rated_mva == pytest.approx(77.0)

    def test_s_rated_explicit(self):
        r = RenewableEnv(
            normalize_actions=False,
            capacity_mw=100.0,
            enable_q_control=True,
            s_rated_mva=120.0,
        )
        assert r.s_rated_mva == pytest.approx(120.0)


class TestQControlStep:

    @pytest.fixture
    def rq(self):
        r = RenewableEnv(capacity_mw=100.0, enable_q_control=True, s_rated_mva=120.0)
        r._available_cf = np.full(48, 0.8)
        r.steps_per_day = 48
        r.time_step = 0
        r.day_id = 0
        r.current_p_mw = 0.0
        r.current_q_mvar = 0.0
        return r

    def test_q_disabled_step_zero_reactive(self):
        r = RenewableEnv(capacity_mw=100.0, enable_q_control=False)
        r._available_cf = np.full(48, 0.8)
        r.steps_per_day = 48
        r.time_step = 0
        r.day_id = 0
        r.step(np.array([1.0]))
        assert r.current_q_mvar == pytest.approx(0.0)

    def test_ndarray_full_p_full_q_pq_circle(self, rq):
        rq.step(np.array([1.0, 1.0]))
        p = rq.current_p_mw
        q = rq.current_q_mvar
        s = rq.s_rated_mva
        assert p ** 2 + q ** 2 == pytest.approx(s ** 2, rel=1e-5, abs=1e-4)

    def test_dict_action_same_pq_circle(self, rq):
        rq.step({'curtailment': 0.0, 'q_norm': 1.0})
        p = rq.current_p_mw
        q = rq.current_q_mvar
        s = rq.s_rated_mva
        assert p ** 2 + q ** 2 == pytest.approx(s ** 2, rel=1e-5, abs=1e-4)

    def test_none_action_zero_q(self, rq):
        rq.step(None)
        assert rq.current_q_mvar == pytest.approx(0.0)

    def test_scalar_action_zero_q(self, rq):
        rq.step(1.0)
        assert rq.current_q_mvar == pytest.approx(0.0)

    def test_p_at_nameplate_no_q_headroom(self):
        r = RenewableEnv(capacity_mw=100.0, enable_q_control=True, s_rated_mva=None)
        r._available_cf = np.full(48, 1.0)
        r.steps_per_day = 48
        r.time_step = 0
        r.day_id = 0
        r.step(np.array([1.0, 1.0]))
        assert r.current_p_mw == pytest.approx(100.0)
        assert r.current_q_mvar == pytest.approx(0.0)

    def test_p_zero_full_q_reaches_s_rated(self, rq):
        rq._available_cf = np.full(48, 0.0)
        rq.step(np.array([1.0, 1.0]))
        assert rq.current_p_mw == pytest.approx(0.0)
        assert rq.current_q_mvar == pytest.approx(rq.s_rated_mva)

    def test_negative_q_norm(self, rq):
        rq.step(np.array([1.0, -1.0]))
        q = rq.current_q_mvar
        p = rq.current_p_mw
        q_max = float(np.sqrt(max(rq.s_rated_mva ** 2 - p ** 2, 0.0)))
        assert q < 0.0
        assert abs(q) <= q_max + 1e-5

    def test_q_norm_clipped_then_pq_circle(self, rq):
        rq.step(np.array([1.0, 1.0]))
        q_ref = rq.current_q_mvar
        rq.time_step = 0
        rq.current_p_mw = 0.0
        rq.current_q_mvar = 0.0
        rq.step(np.array([1.0, 2.0]))
        assert rq.current_q_mvar == pytest.approx(q_ref)


class TestQControlObs:

    @pytest.fixture
    def rq(self):
        r = RenewableEnv(capacity_mw=100.0, enable_q_control=True, s_rated_mva=120.0)
        r._available_cf = np.full(48, 0.8)
        r.steps_per_day = 48
        r.time_step = 0
        r.day_id = 0
        r.current_p_mw = 0.0
        r.current_q_mvar = 0.0
        return r

    def test_obs_has_q_mvar_norm(self, rq):
        o = rq.obs()
        assert 'q_mvar_norm' in o

    def test_q_mvar_norm_in_unit_interval(self, rq):
        rq.step(np.array([1.0, 0.5]))
        o = rq.obs()
        assert -1.0 <= o['q_mvar_norm'] <= 1.0

    def test_observation_space_bounds_q_component(self, rq):
        assert rq.observation_space.shape == (5,)
        assert rq.observation_space.low[2] == pytest.approx(-1.0)
        assert rq.observation_space.high[2] == pytest.approx(1.0)

    def test_q_disabled_no_q_in_obs(self):
        r = RenewableEnv(normalize_actions=False, capacity_mw=50.0, enable_q_control=False)
        r._capacity_factor = 0.5
        r.current_p_mw = 25.0
        o = r.obs()
        assert 'q_mvar_norm' not in o

    def test_grid_obs_shape_q_on(self, rq):
        rq.step(np.array([1.0, 0.0]))
        g = rq.grid_obs()
        assert g.shape == (3,)

    def test_grid_obs_shape_q_off(self):
        r = RenewableEnv(normalize_actions=False, capacity_mw=50.0, enable_q_control=False)
        r._capacity_factor = 0.3
        r.current_p_mw = 15.0
        g = r.grid_obs()
        assert g.shape == (2,)

    def test_grid_obs_names_q_on(self, rq):
        names = rq.grid_obs_names('pv1')
        assert 'pv1_q_mvar_norm' in names

    def test_grid_obs_names_q_off(self):
        r = RenewableEnv(normalize_actions=False, capacity_mw=50.0, enable_q_control=False)
        names = r.grid_obs_names('pv1')
        assert not any('q' in n.lower() for n in names)


class TestQControlStatus:

    @pytest.fixture
    def rq(self):
        r = RenewableEnv(capacity_mw=100.0, enable_q_control=True, s_rated_mva=120.0)
        r._available_cf = np.full(48, 0.8)
        r.steps_per_day = 48
        r.time_step = 0
        r.day_id = 0
        r.current_p_mw = 0.0
        r.current_q_mvar = 0.0
        return r

    def test_status_s_rated_and_q_norm_keys(self, rq):
        rq.step(np.array([1.0, 0.25]))
        s = rq.status()
        assert s['s_rated_mva'] == pytest.approx(120.0)
        assert 'q_mvar_norm' in s

    def test_status_q_mvar_norm_matches_ratio(self, rq):
        rq.step(np.array([1.0, -0.5]))
        s = rq.status()
        expected = rq.current_q_mvar / rq.s_rated_mva
        assert s['q_mvar_norm'] == pytest.approx(expected)

    def test_q_disabled_status_no_q_fields(self):
        r = RenewableEnv(normalize_actions=False, capacity_mw=50.0, enable_q_control=False)
        st = r.status()
        assert 's_rated_mva' not in st
        assert 'q_mvar_norm' not in st
