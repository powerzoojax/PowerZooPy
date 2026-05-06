"""Unit tests for powerzoo.envs.resource.vehicle — VehicleEnv (EV).

Domain knowledge embedded in these tests:
- EV battery SOC dynamics with separate charge (G2V) and discharge (V2G) efficiencies
- Multi-trip commute schedule with departure/arrival times and energy consumption
- Vehicle availability: can only charge/discharge when at home (connected to charger)
- Departure SOC requirement: SOC must meet threshold before leaving
- V2G (vehicle-to-grid): positive current_p = discharge, negative = charge (G2V)
- Typical EV parameters: 40-80 kWh battery, 3.3-7.7 kW Level 2 charger
- Trip energy consumption deducted from SOC at departure
- Power limits: separate p_charge_max and p_discharge_max
- Piecewise linear action denormalization: RL action 0 → physical 0 (idle)
- Unmet energy tracking: cost_unmet_energy exposed when SOC < commute need
"""

import pytest
import numpy as np

from powerzoo.envs.resource.vehicle import VehicleEnv
from .conftest import MockParentGrid


# ========================== Fixtures ==========================

@pytest.fixture
def ev():
    """Standard EV: 60 kWh battery, 7 kW charger, single trip 8am-6pm, 60-min steps."""
    v = VehicleEnv(
        normalize_actions=False,
        E_max_kWh=60.0,
        soc_init=0.8,
        soc_min=0.1,
        soc_max=0.95,
        soc_departure_min=0.8,
        p_charge_max_kW=7.0,
        p_discharge_max_kW=7.0,
        eta_charge=0.95,
        eta_discharge=0.95,
        delta_t_minutes=60.0,
    )
    v.reset(seed=42)
    return v


@pytest.fixture
def ev_multi_trip():
    """EV with 3 daily trips, 60-min resolution."""
    v = VehicleEnv(
        normalize_actions=False,
        E_max_kWh=60.0,
        soc_init=0.8,
        p_charge_max_kW=7.0,
        p_discharge_max_kW=7.0,
        commute_schedule=[
            {'departure': 8.0, 'arrival': 9.0, 'energy_kWh': 10.0},
            {'departure': 12.0, 'arrival': 13.0, 'energy_kWh': 5.0},
            {'departure': 18.0, 'arrival': 19.0, 'energy_kWh': 10.0},
        ],
        delta_t_minutes=60.0,
    )
    v.reset(seed=0)
    return v


@pytest.fixture
def ev_15min():
    """EV with 15-min resolution for finer-grained tests."""
    v = VehicleEnv(
        normalize_actions=False,
        E_max_kWh=60.0,
        soc_init=0.8,
        p_charge_max_kW=7.0,
        p_discharge_max_kW=7.0,
        commute_schedule=[
            {'departure': 8.0, 'arrival': 18.0, 'energy_kWh': 15.0},
        ],
        delta_t_minutes=15.0,
    )
    v.reset(seed=1)
    return v


@pytest.fixture
def ev_normalized():
    """EV with normalized actions and asymmetric charge/discharge power."""
    v = VehicleEnv(
        normalize_actions=True,
        E_max_kWh=60.0,
        soc_init=0.5,
        p_charge_max_kW=7.0,
        p_discharge_max_kW=5.0,
        delta_t_minutes=60.0,
    )
    v.reset(seed=0)
    return v


# ========================== Initialization ==========================

class TestVehicleInit:

    def test_default_single_trip(self):
        v = VehicleEnv(normalize_actions=False)
        assert len(v.commute_schedule) == 1
        assert v.commute_schedule[0]['departure'] == 8.0
        assert v.commute_schedule[0]['arrival'] == 18.0

    def test_unit_conversion_kwh_to_mwh(self):
        v = VehicleEnv(normalize_actions=False, E_max_kWh=60.0)
        assert v.capacity_mwh == pytest.approx(0.060, abs=1e-6)

    def test_unit_conversion_kw_to_mw(self):
        v = VehicleEnv(normalize_actions=False, p_charge_max_kW=7.0, p_discharge_max_kW=11.0)
        assert v.p_charge_max_mw == pytest.approx(0.007, abs=1e-6)
        assert v.p_discharge_max_mw == pytest.approx(0.011, abs=1e-6)

    def test_commute_schedule_sorted(self):
        v = VehicleEnv(normalize_actions=False, commute_schedule=[
            {'departure': 18.0, 'arrival': 19.0, 'energy_kWh': 10.0},
            {'departure': 8.0, 'arrival': 9.0, 'energy_kWh': 10.0},
        ], delta_t_minutes=60.0)
        assert v.commute_schedule[0]['departure'] < v.commute_schedule[1]['departure']

    def test_trip_too_short_raises(self):
        """Trip duration shorter than Δt should raise ValueError."""
        with pytest.raises(ValueError, match="shorter than step size"):
            VehicleEnv(
                normalize_actions=False,
                commute_schedule=[
                    {'departure': 8.0, 'arrival': 8.1, 'energy_kWh': 5.0},
                ],
                delta_t_minutes=60.0,  # 1 hour > 0.1h trip
            )

    def test_name_attribute(self):
        assert VehicleEnv.name == 'vehicle'

    def test_action_space_bounds(self):
        v = VehicleEnv(normalize_actions=False, p_charge_max_kW=7.0, p_discharge_max_kW=11.0)
        assert v.action_space.low[0] == pytest.approx(-0.007, abs=1e-5)
        assert v.action_space.high[0] == pytest.approx(0.011, abs=1e-5)

    def test_observation_space_shape(self):
        v = VehicleEnv(normalize_actions=False)
        assert v.observation_space.shape == (9,)


# ========================== Reset ==========================

class TestVehicleReset:

    def test_reset_initial_state(self, ev):
        ev.step(None)
        ev.reset(seed=0)
        assert ev.soc == ev.soc_init
        assert ev.is_home is True
        assert ev.time_of_day == 0.0
        assert ev.time_step == 0

    def test_reset_returns_obs(self, ev):
        """reset() returns obs() dict (9 keys matching observation_space)."""
        o = ev.reset(seed=0)
        assert isinstance(o, dict)
        assert 'soc' in o
        assert 'is_home' in o
        assert 'time_to_arrival_norm' in o
        assert len(o) == 9

    def test_reset_clears_unmet_energy(self, ev):
        ev.unmet_energy_mwh = 99.0
        ev.reset(seed=0)
        assert ev.unmet_energy_mwh == 0.0

    def test_reset_clears_trip_idx(self, ev_multi_trip):
        ev_multi_trip._schedule_cursor = 2
        ev_multi_trip.reset(seed=0)
        assert ev_multi_trip._schedule_cursor == 0


# ========================== Sign Convention ==========================

class TestEVSignConvention:
    """V2G: positive = discharge (injection), G2V: negative = charge (absorption)."""

    def test_charge_negative(self, ev):
        ev.step(-0.005)  # charge at 5 kW (= -0.005 MW)
        assert ev.current_p_mw < 0

    def test_discharge_positive(self, ev):
        ev.step(0.005)  # V2G at 5 kW
        assert ev.current_p_mw > 0

    def test_idle(self, ev):
        ev.step(None)
        assert ev.current_p_mw == 0.0


# ========================== SOC Dynamics (G2V / V2G) ==========================

class TestEVSOCDynamics:

    def test_charge_increases_soc(self, ev):
        soc0 = ev.soc
        ev.step(-0.005)
        assert ev.soc > soc0

    def test_discharge_decreases_soc(self, ev):
        soc0 = ev.soc
        ev.step(0.005)
        assert ev.soc < soc0

    def test_charge_energy_accounting(self, ev):
        """G2V: energy stored = P_grid × Δt × η_c."""
        soc0 = ev.soc
        p_charge = -0.005  # MW (5 kW charging)
        ev.step(p_charge)
        dt_h = ev.delta_t_minutes / 60.0
        expected_energy_to_cell = abs(p_charge) * dt_h * ev.eta_charge
        expected_delta_soc = expected_energy_to_cell / ev.capacity_mwh
        actual_delta_soc = ev.soc - soc0
        assert actual_delta_soc == pytest.approx(expected_delta_soc, rel=1e-4)

    def test_discharge_energy_accounting(self, ev):
        """V2G: energy from cell = P_grid × Δt / η_d."""
        soc0 = ev.soc
        p_v2g = 0.005  # MW
        ev.step(p_v2g)
        dt_h = ev.delta_t_minutes / 60.0
        expected_energy_from_cell = p_v2g * dt_h / ev.eta_discharge
        expected_delta_soc = expected_energy_from_cell / ev.capacity_mwh
        actual_delta_soc = soc0 - ev.soc
        assert actual_delta_soc == pytest.approx(expected_delta_soc, rel=1e-4)


# ========================== Availability / Commute ==========================

class TestAvailability:

    def test_starts_at_home(self, ev):
        assert ev.is_home is True
        assert ev.time_of_day == 0.0

    def test_vehicle_departs(self, ev):
        """Advance to departure time → vehicle should leave."""
        # Default: depart at 8.0, Δt=60 min.
        # _update_availability() runs at the START of step() using current time_of_day.
        # After 8 steps, time_of_day = 8.0. At the 9th step, availability check
        # sees 8.0 ∈ [8.0, 9.0) → departure triggers.
        for _ in range(9):
            ev.step(None)
        assert ev.is_home is False, "Vehicle should have departed at 8:00"

    def test_departure_consumes_energy(self, ev):
        """Commute energy is deducted from SOC at departure."""
        soc_before_depart = ev.soc
        # Advance past departure: 9 steps to trigger departure at time_of_day=8.0
        for _ in range(9):
            ev.step(None)
        # Default trip: 15 kWh = 0.015 MWh
        expected_loss = 0.015 / ev.capacity_mwh
        assert ev.soc < soc_before_depart
        # SOC drop should be at least the trip energy
        assert (soc_before_depart - ev.soc) >= expected_loss - 0.01

    def test_vehicle_arrives(self, ev):
        """After departure, vehicle arrives at arrival time."""
        # Depart at 8 (triggered at step 9), arrive at 18 (triggered at step 19).
        # After 9 steps: time_of_day=9.0, departed.
        for _ in range(9):
            ev.step(None)
        assert ev.is_home is False
        # Continue until arrival triggers (time_of_day reaches 18.0)
        for _ in range(10):
            ev.step(None)
        assert ev.is_home is True, "Vehicle should be home after arrival"

    def test_no_charge_while_away(self, ev):
        """Vehicle cannot charge when not at home."""
        # Depart: 9 steps triggers departure at time_of_day=8.0
        for _ in range(9):
            ev.step(None)
        assert ev.is_home is False
        soc_before = ev.soc
        ev.step(-0.007)  # try to charge
        assert ev.current_p_mw == 0.0
        assert ev.soc == soc_before

    def test_multi_trip_schedule(self, ev_multi_trip):
        """Vehicle departs and arrives multiple times per day."""
        departures_seen = 0
        arrivals_seen = 0
        was_home = True
        for _ in range(24):
            ev_multi_trip.step(None)
            if was_home and not ev_multi_trip.is_home:
                departures_seen += 1
            if not was_home and ev_multi_trip.is_home:
                arrivals_seen += 1
            was_home = ev_multi_trip.is_home
        assert departures_seen >= 2, "Should see multiple departures"
        assert arrivals_seen >= 2, "Should see multiple arrivals"


# ========================== SOC Constraints ==========================

class TestEVSOCConstraints:

    def test_soc_never_below_min(self, ev):
        """Under heavy V2G, SOC should not go below soc_min."""
        for _ in range(100):
            ev.step(ev.p_discharge_max_mw)
        assert ev.soc >= ev.soc_min - 1e-9

    def test_soc_never_above_max(self, ev):
        """Under continuous charging, SOC should not exceed soc_max."""
        for _ in range(100):
            ev.step(-ev.p_charge_max_mw)
        assert ev.soc <= ev.soc_max + 1e-9

    def test_available_power_at_soc_min(self):
        v = VehicleEnv(normalize_actions=False, E_max_kWh=60.0, soc_init=0.1, soc_min=0.1,
                       p_discharge_max_kW=7.0, delta_t_minutes=60.0)
        v.reset(seed=0)
        avail = v.available_power()
        assert avail.discharge_max_mw == pytest.approx(0.0, abs=1e-9)

    def test_available_power_at_soc_max(self):
        v = VehicleEnv(normalize_actions=False, E_max_kWh=60.0, soc_init=0.95, soc_max=0.95,
                       p_charge_max_kW=7.0, delta_t_minutes=60.0)
        v.reset(seed=0)
        avail = v.available_power()
        assert avail.charge_max_mw == pytest.approx(0.0, abs=1e-9)

    def test_available_power_not_home(self, ev):
        """When away, available power should be zero."""
        # Force away
        ev.is_home = False
        avail = ev.available_power()
        assert avail.discharge_max_mw == 0.0
        assert avail.charge_max_mw == 0.0


# ========================== Departure Readiness ==========================

class TestDepartureReady:

    def test_ready_when_soc_above_threshold(self, ev):
        ev.soc = 0.85
        assert ev.check_departure_ready() is True

    def test_not_ready_when_soc_below_threshold(self, ev):
        ev.soc = 0.3
        assert ev.check_departure_ready() is False

    def test_exact_threshold(self, ev):
        ev.soc = ev.soc_departure_min
        assert ev.check_departure_ready() is True


# ========================== Time Tracking ==========================

class TestTimeTracking:

    def test_time_advances(self, ev):
        ev.step(None)
        assert ev.time_of_day == pytest.approx(1.0)  # 60-min step
        ev.step(None)
        assert ev.time_of_day == pytest.approx(2.0)

    def test_time_wraps_at_24(self, ev):
        for _ in range(25):
            ev.step(None)
        assert 0.0 <= ev.time_of_day < 24.0

    def test_time_to_next_departure(self, ev):
        """At midnight, next departure at 8:00 → 8 hours."""
        ttd = ev._time_to_next_departure()
        assert ttd == pytest.approx(8.0)


# ========================== Action Parsing ==========================

class TestEVActionParsing:

    def test_dict_action(self, ev):
        ev.step({'p_mw': -0.003})
        assert ev.current_p_mw < 0

    def test_ndarray_action(self, ev):
        ev.step(np.array([-0.003]))
        assert ev.current_p_mw < 0

    def test_none_action(self, ev):
        ev.step(None)
        assert ev.current_p_mw == 0.0


# ========================== Piecewise Linear Action Denormalization ==========================

class TestPiecewiseActionMapping:
    """RL action 0 must map to physical 0 (idle), even with asymmetric power limits."""

    def test_zero_action_maps_to_zero(self, ev_normalized):
        """Action=0 → physical power 0, regardless of charge/discharge asymmetry."""
        ev_normalized.step(np.array([0.0]))
        assert ev_normalized.current_p_mw == pytest.approx(0.0, abs=1e-9)

    def test_positive_one_maps_to_discharge_max(self, ev_normalized):
        """Action=+1 → p_discharge_max (5 kW = 0.005 MW)."""
        soc0 = ev_normalized.soc
        ev_normalized.step(np.array([1.0]))
        assert ev_normalized.current_p_mw == pytest.approx(0.005, abs=1e-6)

    def test_negative_one_maps_to_charge_max(self, ev_normalized):
        """Action=-1 → -p_charge_max (7 kW = -0.007 MW)."""
        ev_normalized.step(np.array([-1.0]))
        assert ev_normalized.current_p_mw == pytest.approx(-0.007, abs=1e-6)

    def test_half_positive_maps_correctly(self, ev_normalized):
        """Action=+0.5 → 0.5 * p_discharge_max."""
        ev_normalized.step(np.array([0.5]))
        assert ev_normalized.current_p_mw == pytest.approx(0.0025, abs=1e-6)

    def test_half_negative_maps_correctly(self, ev_normalized):
        """Action=-0.5 → -0.5 * p_charge_max."""
        ev_normalized.step(np.array([-0.5]))
        assert ev_normalized.current_p_mw == pytest.approx(-0.0035, abs=1e-6)


# ========================== Unmet Energy (Free Energy Exploit Fix) ==========================

class TestUnmetEnergy:
    """cost_unmet_energy tracks energy shortfall at departure."""

    def test_no_unmet_energy_when_fully_charged(self, ev):
        """Sufficient SOC → unmet_energy_mwh stays 0."""
        # SOC starts at 0.8, trip needs 15 kWh = 0.015 MWh → 0.25 SOC drop.
        # 0.8 - 0.25 = 0.55 > soc_min=0.1 → no shortfall.
        for _ in range(9):
            ev.step(None)
        assert ev.unmet_energy_mwh == pytest.approx(0.0, abs=1e-9)

    def test_unmet_energy_when_soc_too_low(self):
        """Very low SOC at departure → non-zero unmet_energy_mwh."""
        v = VehicleEnv(
            normalize_actions=False,
            E_max_kWh=60.0,
            soc_init=0.15,  # barely above soc_min
            soc_min=0.1,
            commute_schedule=[
                {'departure': 8.0, 'arrival': 18.0, 'energy_kWh': 15.0},
            ],
            delta_t_minutes=60.0,
        )
        v.reset(seed=0)
        # Advance to departure (9 steps)
        for _ in range(9):
            v.step(None)
        # SOC=0.15, trip needs 0.015/0.060 = 0.25 SOC drop.
        # new_soc = 0.15 - 0.25 = -0.10 < soc_min=0.1
        # unmet_soc = 0.1 - (-0.10) = 0.20
        # unmet_energy = 0.20 * 0.060 = 0.012 MWh
        assert v.unmet_energy_mwh == pytest.approx(0.012, abs=1e-6)

    def test_cost_unmet_energy_in_status(self):
        """cost_unmet_energy appears in status() and follows cost_ convention."""
        v = VehicleEnv(
            normalize_actions=False,
            E_max_kWh=60.0,
            soc_init=0.15,
            soc_min=0.1,
            commute_schedule=[
                {'departure': 8.0, 'arrival': 18.0, 'energy_kWh': 15.0},
            ],
            delta_t_minutes=60.0,
        )
        v.reset(seed=0)
        for _ in range(9):
            v.step(None)
        s = v.status()
        assert 'cost_unmet_energy' in s
        assert s['cost_unmet_energy'] > 0.0

    def test_unmet_energy_resets_each_step(self, ev):
        """unmet_energy_mwh is reset to 0 at the start of each step."""
        ev.unmet_energy_mwh = 99.0
        ev.step(None)
        # No departure at time_of_day=0 → should be 0
        assert ev.unmet_energy_mwh == pytest.approx(0.0, abs=1e-9)


# ========================== Time to Arrival ==========================

class TestTimeToArrival:

    def test_time_to_arrival_at_home(self, ev):
        """When at home, time_to_arrival = 0."""
        assert ev._time_to_next_arrival() == pytest.approx(0.0)

    def test_time_to_arrival_while_away(self, ev):
        """When away, time_to_arrival reflects scheduled arrival."""
        # Depart at step 9 (time_of_day=8.0), arrival scheduled at 18.0
        for _ in range(9):
            ev.step(None)
        assert ev.is_home is False
        # time_of_day is now 9.0 (after step), arrival at 18.0 → 9 hours
        tta = ev._time_to_next_arrival()
        assert tta == pytest.approx(9.0, abs=0.1)

    def test_time_to_arrival_in_obs(self, ev):
        o = ev.obs()
        assert 'time_to_arrival_norm' in o

    def test_time_to_arrival_in_status(self, ev):
        s = ev.status()
        assert 'time_to_arrival' in s


# ========================== Observation ==========================

class TestEVObservation:

    def test_obs_is_dict(self, ev):
        o = ev.obs()
        assert isinstance(o, dict)
        assert len(o) == 9

    def test_obs_is_home_flag(self, ev):
        o = ev.obs()
        assert o['is_home'] == 1.0  # is_home = True

    def test_obs_departure_ready(self, ev):
        o = ev.obs()
        assert o['departure_ready'] == 1.0  # soc=0.8 >= soc_departure_min=0.8


# ========================== Status ==========================

class TestEVStatus:

    def test_status_keys(self, ev):
        s = ev.status()
        expected = {'soc', 'current_p_mw', 'current_q_mvar', 'capacity_mwh', 'is_home',
                    'time_of_day', 'soc_min', 'soc_max', 'soc_departure_min',
                    'bus_id', 'departure_ready', 'time_to_departure', 'time_to_arrival',
                    'cost_clipped_power', 'cost_unmet_energy'}
        assert expected.issubset(s.keys())


# ========================== status() extended fields ==========================

class TestEVStatusExtended:

    def test_status_has_time_to_departure(self, ev):
        s = ev.status()
        assert 'time_to_departure' in s
        assert isinstance(s['time_to_departure'], float)
        assert s['time_to_departure'] >= 0.0

    def test_status_has_local_v(self, ev):
        s = ev.status()
        assert 'local_v' in s
        assert s['local_v'] is None  # no parent with nodes


# ========================== Back-to-back Trip Handling ==========================

class TestBackToBackTrips:
    """_current_trip_idx prevents overlap when arrival == next departure."""

    def test_back_to_back_trips(self):
        """Trip A arrives at 9.0, Trip B departs at 9.0 → both handled in same step."""
        v = VehicleEnv(
            normalize_actions=False,
            E_max_kWh=60.0,
            soc_init=0.8,
            commute_schedule=[
                {'departure': 8.0, 'arrival': 9.0, 'energy_kWh': 5.0},
                {'departure': 9.0, 'arrival': 10.0, 'energy_kWh': 5.0},
            ],
            delta_t_minutes=60.0,
        )
        v.reset(seed=0)
        # Advance to time_of_day=8.0 (9 steps idle)
        for _ in range(9):
            v.step(None)
        assert v.is_home is False  # departed on trip 0

        # Now time_of_day=9.0 → trip 0 arrives AND trip 1 departs
        v.step(None)
        # Should have arrived from trip 0 then immediately departed on trip 1
        assert v.is_home is False
        assert v._schedule_cursor == 1  # on trip 1 now

    def test_cross_day_back_to_back(self):
        """Last trip's arrival and first trip's departure in same window across day boundary."""
        v = VehicleEnv(
            normalize_actions=False,
            E_max_kWh=60.0,
            soc_init=0.8,
            commute_schedule=[
                {'departure': 8.0, 'arrival': 9.0, 'energy_kWh': 5.0},
                {'departure': 18.0, 'arrival': 19.0, 'energy_kWh': 5.0},
            ],
            delta_t_minutes=60.0,
        )
        v.reset(seed=0)
        # Run full day to complete both trips
        for _ in range(24):
            v.step(None)
        # _schedule_cursor should wrap back to 0 for next day
        assert v._schedule_cursor == 0
        assert v.is_home is True


# ========================== Edge Cases ==========================

class TestEVEdgeCases:

    def test_full_day_simulation(self, ev):
        """Run 24 hours without crashing."""
        for _ in range(24):
            ev.step(None)
        assert ev.time_step == 24

    def test_multiple_days(self, ev):
        """Run 3 full days — vehicle should cycle home/away correctly."""
        for _ in range(72):  # 3 × 24
            ev.step(None)
        assert ev.time_step == 72
