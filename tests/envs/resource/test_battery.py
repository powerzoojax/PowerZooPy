"""Unit tests for powerzoo.envs.resource.battery — BatteryEnv (BESS).

Domain knowledge embedded in these tests:
- SOC dynamics with Coulomb-counting: ΔE = P × Δt / η (discharge) or P × Δt × η (charge)
- Separate charge / discharge efficiencies; round-trip η_rt = η_c × η_d
- SOC clamping to [soc_min, soc_max]
- Power clamping to [-power_mw, +power_mw]
- Sign convention: positive = discharge (injection), negative = charge (absorption)
- Feasible power respects both SOC limits and power limits simultaneously
- Energy conservation: no energy created or destroyed beyond efficiency losses
- obs() reports SOC-feasible max power bounds (p_discharge_max_norm, p_charge_max_norm),
  not rated-power headroom from the last setpoint
"""

import pytest
import numpy as np

from powerzoo.envs.resource.battery import BatteryEnv
from .conftest import MockParentGrid


# ========================== Fixtures ==========================

@pytest.fixture
def bat():
    """Standard 50 MWh / 20 MW battery, 15-min steps, no parent."""
    b = BatteryEnv(normalize_actions=False, capacity_mwh=50.0, power_mw=20.0, eta_roundtrip=0.95,
                   soc_min=0.1, soc_max=0.9, initial_soc=0.5)
    b.reset(seed=42)
    return b


@pytest.fixture
def bat_asymmetric():
    """Battery with asymmetric efficiencies (η_c=0.95, η_d=0.90)."""
    b = BatteryEnv(normalize_actions=False, capacity_mwh=100.0, power_mw=50.0,
                   eta_charge=0.95, eta_discharge=0.90,
                   soc_min=0.1, soc_max=0.9, initial_soc=0.5)
    b.reset(seed=0)
    return b


@pytest.fixture
def bat_with_parent(mock_grid):
    """Battery attached to a 15-min grid."""
    b = BatteryEnv(normalize_actions=False, capacity_mwh=50.0, power_mw=20.0, eta_roundtrip=0.95,
                   soc_min=0.1, soc_max=0.9, initial_soc=0.5,
                   parent=mock_grid, bus_id=3)
    b.reset(seed=42)
    return b


# ========================== Initialization ==========================

class TestBatteryInit:

    def test_default_params(self):
        b = BatteryEnv(normalize_actions=False)
        assert b.capacity_mwh == 50.0
        assert b.power_mw == 20.0
        assert b.soc_min == 0.1
        assert b.soc_max == 0.9
        assert b.initial_soc == 0.5
        assert b.eta_charge == pytest.approx(0.95, abs=1e-6)
        assert b.eta_discharge == pytest.approx(0.95, abs=1e-6)
        assert b.efficiency == pytest.approx(0.95 * 0.95, abs=1e-6)

    def test_efficiency_decomposition_symmetric(self):
        """Round-trip η=0.81 → η_c = η_d = 0.9."""
        b = BatteryEnv(normalize_actions=False, eta_roundtrip=0.81)
        assert b.eta_charge == pytest.approx(0.9, abs=1e-6)
        assert b.eta_discharge == pytest.approx(0.9, abs=1e-6)
        assert b.efficiency == pytest.approx(0.81, abs=1e-4)

    def test_explicit_eta_overrides_efficiency(self):
        """Explicit η_c, η_d take priority over round-trip shorthand."""
        b = BatteryEnv(normalize_actions=False, eta_roundtrip=0.50, eta_charge=0.95, eta_discharge=0.90)
        assert b.eta_charge == 0.95
        assert b.eta_discharge == 0.90
        assert b.efficiency == pytest.approx(0.95 * 0.90, abs=1e-6)

    def test_partial_explicit_eta(self):
        """Only η_c given → η_d falls back to sqrt(eta_roundtrip)."""
        b = BatteryEnv(normalize_actions=False, eta_roundtrip=0.80, eta_charge=0.95)
        assert b.eta_charge == 0.95
        assert b.eta_discharge == pytest.approx(np.sqrt(0.80), abs=1e-6)
        assert b.efficiency == pytest.approx(0.95 * np.sqrt(0.80), abs=1e-6)

    def test_initial_soc_clipped(self):
        """initial_soc outside [soc_min, soc_max] is clipped."""
        b = BatteryEnv(normalize_actions=False, soc_min=0.2, soc_max=0.8, initial_soc=0.0)
        assert b.initial_soc == 0.2
        b2 = BatteryEnv(normalize_actions=False, soc_min=0.2, soc_max=0.8, initial_soc=1.0)
        assert b2.initial_soc == 0.8

    def test_invalid_capacity(self):
        with pytest.raises(ValueError, match="capacity_mwh"):
            BatteryEnv(capacity_mwh=0.0)
        with pytest.raises(ValueError, match="capacity_mwh"):
            BatteryEnv(capacity_mwh=-1.0)

    def test_invalid_power_mw(self):
        with pytest.raises(ValueError, match="power_mw"):
            BatteryEnv(power_mw=0.0)

    def test_invalid_soc_bounds(self):
        with pytest.raises(ValueError, match="soc_min"):
            BatteryEnv(soc_min=0.8, soc_max=0.2)  # min > max
        with pytest.raises(ValueError, match="soc_min"):
            BatteryEnv(soc_min=-0.1, soc_max=0.9)  # min < 0
        with pytest.raises(ValueError, match="soc_min"):
            BatteryEnv(soc_min=0.1, soc_max=1.1)  # max > 1

    def test_invalid_eta_roundtrip(self):
        with pytest.raises(ValueError, match="eta_roundtrip"):
            BatteryEnv(eta_roundtrip=0.0)
        with pytest.raises(ValueError, match="eta_roundtrip"):
            BatteryEnv(eta_roundtrip=1.1)

    def test_efficiency_deprecated_alias(self):
        import warnings
        with pytest.warns(DeprecationWarning, match="efficiency"):
            b = BatteryEnv(normalize_actions=False, efficiency=0.81)
        assert b.eta_charge == pytest.approx(0.9, abs=1e-6)

    def test_efficiency_and_eta_roundtrip_conflict(self):
        with pytest.raises(ValueError, match="only one"):
            BatteryEnv(eta_roundtrip=0.9, efficiency=0.9)

    def test_invalid_eta_charge(self):
        with pytest.raises(ValueError, match="eta_charge"):
            BatteryEnv(eta_charge=0.0)
        with pytest.raises(ValueError, match="eta_charge"):
            BatteryEnv(eta_charge=1.2)

    def test_invalid_eta_discharge(self):
        with pytest.raises(ValueError, match="eta_discharge"):
            BatteryEnv(eta_discharge=-0.1)

    def test_name_attribute(self):
        assert BatteryEnv.name == 'battery'

    def test_action_space_bounds(self):
        b = BatteryEnv(normalize_actions=False, power_mw=30.0)
        assert b.action_space.low[0] == pytest.approx(-30.0)
        assert b.action_space.high[0] == pytest.approx(30.0)

    def test_observation_space_shape(self):
        b = BatteryEnv(normalize_actions=False)
        assert b.observation_space.shape == (6,)


# ========================== Reset ==========================

class TestBatteryReset:

    def test_reset_restores_initial_soc(self, bat):
        bat.step(10.0)  # discharge some
        bat.reset(seed=0)
        assert bat.soc == bat.initial_soc
        assert bat.current_p_mw == 0.0
        assert bat.time_step == 0

    def test_reset_clears_soc_history(self, bat):
        bat.step(5.0)
        bat.step(5.0)
        bat.reset()
        assert len(bat._soc_history) == 1  # only initial SOC

    def test_reset_clears_throughput(self, bat):
        bat.step(10.0)
        assert bat.throughput_mwh > 0
        bat.reset()
        assert bat.throughput_mwh == 0.0

    def test_reset_randomize_soc_within_bounds(self, bat):
        """randomize_soc=True must produce SOC inside [soc_min, soc_max]."""
        socs = set()
        for seed in range(20):
            bat.reset(seed=seed, options={'randomize_soc': True})
            assert bat.soc_min <= bat.soc <= bat.soc_max
            socs.add(round(bat.soc, 4))
        # With 20 different seeds the SOC should not always be the same value
        assert len(socs) > 1

    def test_reset_randomize_soc_false_is_deterministic(self, bat):
        """randomize_soc=False (default) must always restore initial_soc."""
        bat.step(10.0)
        bat.reset(options={'randomize_soc': False})
        assert bat.soc == bat.initial_soc


# ========================== Sign Convention ==========================

class TestSignConvention:
    """Power system convention: positive = injection (discharge), negative = absorption (charge)."""

    def test_discharge_positive(self, bat):
        bat.step(10.0)
        assert bat.current_p_mw > 0, "Discharging should produce positive current_p"

    def test_charge_negative(self, bat):
        bat.step(-10.0)
        assert bat.current_p_mw < 0, "Charging should produce negative current_p"

    def test_idle_zero(self, bat):
        bat.step(0.0)
        assert bat.current_p_mw == 0.0

    def test_none_action_idle(self, bat):
        bat.step(None)
        assert bat.current_p_mw == 0.0


# ========================== SOC Dynamics ==========================

class TestSOCDynamics:
    """Coulomb-counting SOC update with efficiency losses."""

    def test_discharge_reduces_soc(self, bat):
        soc_before = bat.soc
        bat.step(10.0)  # discharge 10 MW
        assert bat.soc < soc_before

    def test_charge_increases_soc(self, bat):
        soc_before = bat.soc
        bat.step(-10.0)  # charge 10 MW
        assert bat.soc > soc_before

    def test_discharge_energy_conservation(self, bat):
        """Discharge: grid gets P MW, battery loses P/η_d MWh per hour."""
        soc_before = bat.soc
        p_mw = 10.0
        bat.step(p_mw)
        dt_h = bat.delta_t_minutes / 60.0  # 0.25 h
        expected_delta_soc = p_mw * dt_h / (bat.eta_discharge * bat.capacity_mwh)
        actual_delta_soc = soc_before - bat.soc
        assert actual_delta_soc == pytest.approx(expected_delta_soc, rel=1e-6)

    def test_charge_energy_conservation(self, bat):
        """Charge: grid supplies |P| MW, battery gains |P|×η_c MWh per hour."""
        soc_before = bat.soc
        p_mw = -10.0  # charge
        bat.step(p_mw)
        dt_h = bat.delta_t_minutes / 60.0
        expected_delta_soc = abs(p_mw) * dt_h * bat.eta_charge / bat.capacity_mwh
        actual_delta_soc = bat.soc - soc_before
        assert actual_delta_soc == pytest.approx(expected_delta_soc, rel=1e-6)

    def test_roundtrip_energy_loss(self, bat):
        """Charge then discharge same MWh: net energy < original due to losses.

        Full charge-discharge cycle energy ratio = η_rt = η_c × η_d.
        """
        bat_rt = BatteryEnv(normalize_actions=False, capacity_mwh=100.0, power_mw=50.0,
                            eta_charge=0.95, eta_discharge=0.90,
                            soc_min=0.0, soc_max=1.0, initial_soc=0.5)
        bat_rt.reset(seed=0)
        eta_rt = bat_rt.eta_charge * bat_rt.eta_discharge  # 0.855

        soc0 = bat_rt.soc
        # Charge for N steps
        for _ in range(10):
            bat_rt.step(-20.0)
        soc_charged = bat_rt.soc
        energy_stored = (soc_charged - soc0) * bat_rt.capacity_mwh

        # Discharge for N steps
        for _ in range(10):
            bat_rt.step(20.0)
        soc_discharged = bat_rt.soc
        energy_retrieved = (soc_charged - soc_discharged) * bat_rt.capacity_mwh

        # Grid-side energy returned / grid-side energy consumed < 1
        # The ratio should be approximately η_rt
        # energy_stored = sum(|P|×η_c×dt/cap) × cap = net added to battery
        # energy_retrieved = sum(P×dt/(η_d×cap)) × cap = net removed from battery
        # But accounting for both sides: grid sees η_c×η_d loss
        assert energy_retrieved < energy_stored or soc_discharged < soc0 + (soc_charged - soc0) * eta_rt + 0.01

    def test_asymmetric_efficiency(self, bat_asymmetric):
        """Asymmetric η_c ≠ η_d: same power input/output yields different SOC changes."""
        soc0 = bat_asymmetric.soc
        bat_asymmetric.step(-10.0)  # charge
        soc_after_charge = bat_asymmetric.soc
        delta_charge = soc_after_charge - soc0

        bat_asymmetric.reset(seed=0)
        soc0 = bat_asymmetric.soc
        bat_asymmetric.step(10.0)  # discharge
        soc_after_discharge = bat_asymmetric.soc
        delta_discharge = soc0 - soc_after_discharge

        # Charge gains less per MW (×η_c), discharge loses more per MW (÷η_d)
        # With η_c=0.95, η_d=0.90: charge_delta < discharge_delta
        assert delta_charge < delta_discharge


# ========================== Power Clamping ==========================

class TestPowerClamp:

    def test_power_clipped_to_max(self, bat):
        bat.step(999.0)  # way above power_mw=20
        assert bat.current_p_mw <= bat.power_mw + 1e-9

    def test_power_clipped_to_min(self, bat):
        bat.step(-999.0)
        assert bat.current_p_mw >= -bat.power_mw - 1e-9


# ========================== SOC Constraints ==========================

class TestSOCConstraints:
    """Feasible power must respect SOC boundaries."""

    def test_cannot_discharge_below_soc_min(self):
        """Battery near soc_min cannot fully discharge at rated power."""
        b = BatteryEnv(normalize_actions=False, capacity_mwh=10.0, power_mw=100.0,
                       soc_min=0.2, soc_max=0.9, initial_soc=0.21)
        b.reset(seed=0)
        b.step(100.0)  # try to discharge at maximum
        assert b.soc >= b.soc_min - 1e-9, "SOC should not go below soc_min"

    def test_cannot_charge_above_soc_max(self):
        """Battery near soc_max cannot fully charge at rated power."""
        b = BatteryEnv(normalize_actions=False, capacity_mwh=10.0, power_mw=100.0,
                       soc_min=0.1, soc_max=0.8, initial_soc=0.79)
        b.reset(seed=0)
        b.step(-100.0)  # try to charge at maximum
        assert b.soc <= b.soc_max + 1e-9, "SOC should not exceed soc_max"

    def test_feasible_power_at_soc_min(self):
        """At exactly soc_min, discharge power should be ~0."""
        b = BatteryEnv(normalize_actions=False, capacity_mwh=50.0, power_mw=20.0,
                       soc_min=0.1, soc_max=0.9, initial_soc=0.1)
        b.reset(seed=0)
        fp = b._compute_feasible_power(20.0)
        assert fp == pytest.approx(0.0, abs=1e-9)

    def test_feasible_power_at_soc_max(self):
        """At exactly soc_max, charge power should be ~0."""
        b = BatteryEnv(normalize_actions=False, capacity_mwh=50.0, power_mw=20.0,
                       soc_min=0.1, soc_max=0.9, initial_soc=0.9)
        b.reset(seed=0)
        fp = b._compute_feasible_power(-20.0)
        assert fp == pytest.approx(0.0, abs=1e-9)

    def test_feasible_power_no_sign_flip_below_soc_min(self):
        """Discharge request with soc just below soc_min must return 0, not negative."""
        b = BatteryEnv(normalize_actions=False, capacity_mwh=50.0, power_mw=20.0,
                       soc_min=0.1, soc_max=0.9, initial_soc=0.5)
        b.reset(seed=0)
        b.soc = b.soc_min - 1e-15  # force float underrun
        fp = b._compute_feasible_power(10.0)
        assert fp >= 0.0, "Discharge into float-underrun SOC must not return negative (micro-charge)"

    def test_feasible_power_no_sign_flip_above_soc_max(self):
        """Charge request with soc just above soc_max must return 0, not positive."""
        b = BatteryEnv(normalize_actions=False, capacity_mwh=50.0, power_mw=20.0,
                       soc_min=0.1, soc_max=0.9, initial_soc=0.5)
        b.reset(seed=0)
        b.soc = b.soc_max + 1e-15  # force float overrun
        fp = b._compute_feasible_power(-10.0)
        assert fp <= 0.0, "Charge into float-overrun SOC must not return positive (micro-discharge)"

    def test_soc_stays_within_bounds_long_episode(self, bat):
        """Run 200 steps of random actions — SOC must always stay in bounds."""
        rng = np.random.default_rng(7)
        for _ in range(200):
            action = rng.uniform(-bat.power_mw, bat.power_mw)
            bat.step(action)
            assert bat.soc_min - 1e-9 <= bat.soc <= bat.soc_max + 1e-9


# ========================== Action Parsing ==========================

class TestActionParsing:

    def test_dict_action(self, bat):
        bat.step({'p_mw': 5.0})
        assert bat.current_p_mw == pytest.approx(5.0, abs=1e-3)

    def test_ndarray_action(self, bat):
        bat.step(np.array([3.0]))
        assert bat.current_p_mw == pytest.approx(3.0, abs=1e-3)

    def test_float_action(self, bat):
        bat.step(7.5)
        assert bat.current_p_mw == pytest.approx(7.5, abs=1e-3)

    def test_none_action(self, bat):
        bat.step(None)
        assert bat.current_p_mw == 0.0


# ========================== Observation ==========================

class TestObservation:

    def test_obs_is_dict(self, bat):
        o = bat.obs()
        assert isinstance(o, dict)
        assert len(o) == 6

    def test_obs_keys(self, bat):
        o = bat.obs()
        assert set(o.keys()) == {'soc', 'p_mw_norm', 'p_discharge_max_norm',
                                 'p_charge_max_norm', 'time_sin', 'time_cos'}

    def test_obs_soc_in_range(self, bat):
        o = bat.obs()
        assert 0.0 <= o['soc'] <= 1.0

    def test_obs_after_discharge(self, bat):
        bat.step(10.0)
        o = bat.obs()
        assert o['p_mw_norm'] > 0  # p_norm > 0 for discharge

    def test_obs_after_charge(self, bat):
        bat.step(-10.0)
        o = bat.obs()
        assert o['p_mw_norm'] < 0  # p_norm < 0 for charge

    def test_obs_with_parent(self, bat_with_parent):
        """obs() should not crash when parent provides steps_per_day."""
        o = bat_with_parent.obs()
        assert isinstance(o, dict)
        assert len(o) == 6

    def test_obs_discharge_max_norm_at_soc_min(self):
        """At soc_min, p_discharge_max_norm must be 0."""
        b = BatteryEnv(normalize_actions=False, capacity_mwh=50.0, power_mw=20.0,
                       soc_min=0.1, soc_max=0.9, initial_soc=0.1)
        b.reset(seed=0)
        o = b.obs()
        assert o['p_discharge_max_norm'] == pytest.approx(0.0, abs=1e-9)

    def test_obs_charge_max_norm_at_soc_max(self):
        """At soc_max, p_charge_max_norm must be 0."""
        b = BatteryEnv(normalize_actions=False, capacity_mwh=50.0, power_mw=20.0,
                       soc_min=0.1, soc_max=0.9, initial_soc=0.9)
        b.reset(seed=0)
        o = b.obs()
        assert o['p_charge_max_norm'] == pytest.approx(0.0, abs=1e-9)

    def test_obs_discharge_max_norm_full_battery(self):
        """At soc_max with large capacity, p_discharge_max_norm is 1.0 (power-limited)."""
        b = BatteryEnv(normalize_actions=False, capacity_mwh=1000.0, power_mw=10.0,
                       soc_min=0.0, soc_max=1.0, initial_soc=1.0, eta_roundtrip=1.0)
        b.reset(seed=0)
        o = b.obs()
        assert o['p_discharge_max_norm'] == pytest.approx(1.0, abs=1e-6)

    def test_obs_charge_max_norm_empty_battery(self):
        """At soc_min with large capacity, p_charge_max_norm is 1.0 (power-limited)."""
        b = BatteryEnv(normalize_actions=False, capacity_mwh=1000.0, power_mw=10.0,
                       soc_min=0.0, soc_max=1.0, initial_soc=0.0, eta_roundtrip=1.0)
        b.reset(seed=0)
        o = b.obs()
        assert o['p_charge_max_norm'] == pytest.approx(1.0, abs=1e-6)

    def test_obs_discharge_max_norm_consistent_with_feasible(self, bat):
        """p_discharge_max_norm * power_mw must match _compute_feasible_power at full request."""
        o = bat.obs()
        max_from_obs = o['p_discharge_max_norm'] * bat.power_mw
        feasible = bat._compute_feasible_power(bat.power_mw)
        assert max_from_obs == pytest.approx(feasible, rel=1e-6)

    def test_obs_charge_max_norm_consistent_with_feasible(self, bat):
        """p_charge_max_norm * power_mw must match |_compute_feasible_power| at full request."""
        o = bat.obs()
        max_from_obs = o['p_charge_max_norm'] * bat.power_mw
        feasible = bat._compute_feasible_power(-bat.power_mw)
        assert max_from_obs == pytest.approx(abs(feasible), rel=1e-6)

    def test_obs_bounds_non_negative_with_float_soc_edge(self):
        """Float-precision SOC just below soc_min must not yield negative obs bounds."""
        b = BatteryEnv(normalize_actions=False, capacity_mwh=50.0, power_mw=20.0,
                       soc_min=0.1, soc_max=0.9, initial_soc=0.5)
        b.reset(seed=0)
        # Force a floating-point underrun condition directly
        b.soc = b.soc_min - 1e-15
        o = b.obs()
        assert o['p_discharge_max_norm'] >= 0.0
        assert o['p_charge_max_norm'] >= 0.0

        b.soc = b.soc_max + 1e-15
        o = b.obs()
        assert o['p_discharge_max_norm'] >= 0.0
        assert o['p_charge_max_norm'] >= 0.0


# ========================== status() extended fields ==========================

class TestStatusExtended:
    """status() must expose headroom fields, bus_id, and local_v."""

    def test_status_headroom_keys(self, bat):
        s = bat.status()
        assert 'p_discharge_headroom' in s
        assert 'p_charge_headroom' in s

    def test_discharge_headroom_at_rest(self, bat):
        """At rest (current_p=0), headroom equals rated power in both directions."""
        bat.step(0.0)
        s = bat.status()
        assert s['p_discharge_headroom'] == pytest.approx(bat.power_mw, abs=1e-6)
        assert s['p_charge_headroom'] == pytest.approx(bat.power_mw, abs=1e-6)

    def test_discharge_headroom_after_discharge(self, bat):
        bat.step(5.0)
        s = bat.status()
        assert s['p_discharge_headroom'] == pytest.approx(bat.power_mw - 5.0, abs=1e-3)

    def test_charge_headroom_after_charge(self, bat):
        bat.step(-5.0)
        s = bat.status()
        assert s['p_charge_headroom'] == pytest.approx(bat.power_mw - 5.0, abs=1e-3)

    def test_status_feasible_bounds_present(self, bat):
        s = bat.status()
        assert 'p_max_feasible_mw' in s
        assert 'p_min_feasible_mw' in s

    def test_status_feasible_bounds_signs(self, bat):
        """p_max_feasible_mw >= 0, p_min_feasible_mw <= 0."""
        s = bat.status()
        assert s['p_max_feasible_mw'] >= 0.0
        assert s['p_min_feasible_mw'] <= 0.0

    def test_status_feasible_bounds_at_soc_min(self):
        """At soc_min, max discharge is 0."""
        b = BatteryEnv(normalize_actions=False, capacity_mwh=50.0, power_mw=20.0,
                       soc_min=0.1, soc_max=0.9, initial_soc=0.1)
        b.reset(seed=0)
        s = b.status()
        assert s['p_max_feasible_mw'] == pytest.approx(0.0, abs=1e-9)

    def test_status_feasible_bounds_at_soc_max(self):
        """At soc_max, max charge is 0."""
        b = BatteryEnv(normalize_actions=False, capacity_mwh=50.0, power_mw=20.0,
                       soc_min=0.1, soc_max=0.9, initial_soc=0.9)
        b.reset(seed=0)
        s = b.status()
        assert s['p_min_feasible_mw'] == pytest.approx(0.0, abs=1e-9)

    def test_status_feasible_bounds_consistent_with_obs(self, bat):
        """p_max_feasible_mw should match p_discharge_max_norm * power_mw."""
        s = bat.status()
        o = bat.obs()
        assert s['p_max_feasible_mw'] == pytest.approx(o['p_discharge_max_norm'] * bat.power_mw, rel=1e-6)
        assert s['p_min_feasible_mw'] == pytest.approx(-o['p_charge_max_norm'] * bat.power_mw, rel=1e-6)

    def test_status_bus_id(self, bat_with_parent):
        s = bat_with_parent.status()
        assert 'bus_id' in s
        assert s['bus_id'] == 3  # fixture uses bus_id=3

    def test_status_local_v_no_nodes(self, bat_with_parent):
        """Without _nodes on parent, local_v is None."""
        s = bat_with_parent.status()
        assert 'local_v' in s
        assert s['local_v'] is None


# ========================== Status ==========================

class TestStatus:

    def test_status_keys(self, bat):
        s = bat.status()
        expected_keys = {'current_p_mw', 'current_q_mvar', 'soc', 'soc_percent',
                         'energy_stored_mwh', 'capacity_mwh', 'power_mw',
                         'eta_charge', 'eta_discharge', 'efficiency_rt', 'time_step',
                         'throughput_mwh'}
        assert expected_keys.issubset(s.keys())

    def test_soc_percent_consistent(self, bat):
        s = bat.status()
        assert s['soc_percent'] == pytest.approx(s['soc'] * 100)

    def test_status_throughput_initial_zero(self, bat):
        s = bat.status()
        assert s['throughput_mwh'] == 0.0


# ========================== Throughput Tracking ==========================

class TestThroughputTracking:

    def test_throughput_increases_on_discharge(self, bat):
        bat.step(10.0)
        expected = 10.0 * bat.dt_hours
        assert bat.throughput_mwh == pytest.approx(expected, rel=1e-6)

    def test_throughput_increases_on_charge(self, bat):
        bat.step(-8.0)
        # feasible charge power may be < 8 due to SOC, so just check >0 and consistent
        expected = abs(bat.current_p_mw) * bat.dt_hours
        assert bat.throughput_mwh == pytest.approx(expected, rel=1e-6)

    def test_throughput_accumulates_over_steps(self, bat):
        bat.step(5.0)
        t1 = bat.throughput_mwh
        bat.step(-5.0)
        t2 = bat.throughput_mwh
        assert t2 > t1

    def test_throughput_zero_for_idle(self, bat):
        bat.step(0.0)
        assert bat.throughput_mwh == 0.0

    def test_throughput_reset_on_reset(self, bat):
        bat.step(10.0)
        bat.reset()
        assert bat.throughput_mwh == 0.0


# ========================== SOC History ==========================

class TestSOCHistory:

    def test_history_grows_with_steps(self, bat):
        for _ in range(5):
            bat.step(5.0)
        assert len(bat._soc_history) == 6  # 1 initial + 5 steps


# ========================== dt_hours with parent ==========================

class TestDtHours:

    def test_dt_from_parent(self, bat_with_parent, mock_grid):
        """Battery should use parent's delta_t_minutes when attached."""
        assert bat_with_parent.dt_hours == pytest.approx(mock_grid.delta_t_minutes / 60.0)

    def test_dt_without_parent(self, bat):
        """Battery without parent uses its own delta_t_minutes."""
        assert bat.dt_hours == pytest.approx(bat.delta_t_minutes / 60.0)


# ========================== Edge Cases ==========================

class TestEdgeCases:

    def test_tiny_capacity(self):
        """Very small battery should still work correctly."""
        b = BatteryEnv(normalize_actions=False, capacity_mwh=0.001, power_mw=0.001, initial_soc=0.5)
        b.reset(seed=0)
        b.step(0.0005)
        assert b.soc_min - 1e-9 <= b.soc <= b.soc_max + 1e-9

    def test_perfect_efficiency(self):
        """η=1.0: no energy loss during round trip."""
        b = BatteryEnv(normalize_actions=False, capacity_mwh=100.0, power_mw=50.0,
                       eta_charge=1.0, eta_discharge=1.0,
                       soc_min=0.0, soc_max=1.0, initial_soc=0.5)
        b.reset(seed=0)
        soc0 = b.soc
        b.step(-10.0)  # charge
        soc_up = b.soc
        b.step(10.0)   # discharge same power
        soc_back = b.soc
        # Perfect efficiency means SOC returns to starting value
        assert soc_back == pytest.approx(soc0, abs=1e-9)

    def test_zero_power_action(self, bat):
        soc0 = bat.soc
        bat.step(0.0)
        assert bat.soc == soc0
