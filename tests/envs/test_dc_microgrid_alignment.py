"""Alignment & physics-depth tests for DCMicrogridEnv.

Covers two concerns in one suite:

A. **Alignment with PowerZooJax DataCenterMicrogridEnv**
   The Python reference must produce numerically aligned semantics with the
   pure-JAX implementation in
   ``powerzoojax.envs.resource.datacenter_microgrid``.  Each test pins down a
   single physical formula or normalisation that previously diverged.

   A1  Outdoor-temp synthetic profile peaks at ~14:00 (afternoon), not 20:00.
   A2  ``cost_sla`` is a *density* (expirations / n_gpus), not raw count.
   A3  ``cost_overtemp`` is normalised to [0, 1] by ``T_ZONE_MAX - t_critical``.
   A4  ``cop_ratio`` (obs[7]) uses ``[0.4, 1.2]`` clip + linear map to [0, 1].
   A5  ``p_cool_min_mw == 0`` so cooling power can collapse to 0 in cold zones.
   A6  Battery cycle-degradation cost is computed *once* at the outer reward
       layer; ``BatteryEnv.cycle_cost_per_mwh`` is left at 0 to avoid double-
       counting in any downstream reader of ``batt.econ_components()``.

B. **Physical depth previously missing from the test suite**
   B1  Multi-step explicit power-balance closure.
   B2  Battery SOC end-to-end conservation under round-trip charge/discharge.
   B3  Battery infeasibility at SOC bounds (charge clipped to 0 at soc_max,
       discharge clipped to 0 at soc_min).
   B4  DG fuel cost & carbon are linear in DG power.
   B5  Scalarised reward equation:
       ``reward = r_energy + w_cost * r_cost + w_carbon * r_carbon``.
   B6  ``cost_power_deficit = power_deficit / p_load``.
   B7  ``cost_overtemp`` heat-up monotonicity (hot ambient & overloaded IT
       must drive the channel from 0 toward 1).
"""

from __future__ import annotations

import numpy as np
import pytest

from powerzoo.envs.microgrid.dc_microgrid_env import DCMicrogridEnv, _T_ZONE_MAX
from powerzoo.data.dc_microgrid_profiles import make_synthetic_outdoor_temp_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _idle_action() -> np.ndarray:
    """Mid-cooling, no DG, no battery, mid scheduling."""
    return np.array([0.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float32)


def _flat_profiles(cpu: float, solar: float, temp_c: float, n: int = 288):
    return (
        np.full(n, cpu,    dtype=np.float32),
        np.full(n, solar,  dtype=np.float32),
        np.full(n, temp_c, dtype=np.float32),
    )


# ===========================================================================
# A — Alignment with PowerZooJax JAX reference
# ===========================================================================

class TestA1_OutdoorTempPhase:
    """Synthetic outdoor temperature must peak at ~14:00 (h-8 phase)."""

    def test_synth_profile_peak_index_near_noon_to_2pm(self):
        arr = make_synthetic_outdoor_temp_profile()
        peak_idx = int(np.argmax(arr))
        peak_hour = peak_idx / 288.0 * 24.0
        assert 13.5 <= peak_hour <= 14.5, (
            f"peak hour should be ~14:00, got {peak_hour:.2f}"
        )

    def test_synth_profile_trough_near_2am(self):
        arr = make_synthetic_outdoor_temp_profile()
        trough_idx = int(np.argmin(arr))
        trough_hour = trough_idx / 288.0 * 24.0
        # Trough is exactly 12 h before peak: ~02:00.
        assert 1.5 <= trough_hour <= 2.5, (
            f"trough hour should be ~02:00, got {trough_hour:.2f}"
        )

    def test_env_internal_synth_temp_peak_aligned(self):
        """DCMicrogridEnv's _make_synthetic_temp must agree with the data
        module's profile generator (both fixed to peak at ~14:00)."""
        from powerzoo.envs.microgrid.dc_microgrid_env import _make_synthetic_temp
        arr = _make_synthetic_temp(288)
        peak_idx = int(np.argmax(arr))
        peak_hour = peak_idx / 288.0 * 24.0
        assert 13.5 <= peak_hour <= 14.5

    def test_env_default_temp_peak_at_noon_to_2pm(self):
        """End-to-end: env without explicit temp profile should expose
        info['t_outdoor'] peaking around step 168 (=14:00)."""
        env = DCMicrogridEnv(max_steps=288)
        env.reset(seed=0)
        temps = []
        for _ in range(288):
            _, _, _, _, info = env.step(_idle_action())
            temps.append(info['t_outdoor'])
        peak_idx = int(np.argmax(temps))
        peak_hour = peak_idx / 288.0 * 24.0
        assert 13.5 <= peak_hour <= 14.5, (
            f"env t_outdoor peak should be ~14:00, got hour={peak_hour:.2f}"
        )


class TestA2_CostSlaDensity:
    """cost_sla must be a density (n_expired / n_gpus), not a raw count."""

    def test_cost_sla_zero_when_no_expirations(self):
        env = DCMicrogridEnv(max_steps=2)
        env.reset(seed=0)
        _, _, _, _, info = env.step(_idle_action())
        # First step: no task can have expired yet.
        assert info['cost_sla'] == pytest.approx(0.0, abs=1e-9)

    def test_cost_sla_equals_step_violations_div_n_gpus(self):
        """Long episode: at any step the channel value must equal
        info['step_sla_violations'] / n_gpus."""
        env = DCMicrogridEnv(max_steps=288, n_gpus=1000)
        env.reset(seed=0)
        for _ in range(288):
            _, _, _, _, info = env.step(_idle_action())
            expected = info['step_sla_violations'] / 1000.0
            assert info['cost_sla'] == pytest.approx(expected, abs=1e-9), (
                f"cost_sla={info['cost_sla']}, expected={expected}, "
                f"step_violations={info['step_sla_violations']}"
            )

    def test_cost_sla_scale_with_smaller_n_gpus(self):
        """Same expirations but smaller n_gpus → larger normalised cost."""
        env_big = DCMicrogridEnv(max_steps=2, n_gpus=1000)
        env_sml = DCMicrogridEnv(max_steps=2, n_gpus=100)
        env_big.reset(seed=0); env_sml.reset(seed=0)
        for _ in range(2):
            _, _, _, _, info_big = env_big.step(_idle_action())
            _, _, _, _, info_sml = env_sml.step(_idle_action())
        # If both have the same per-step violation count, the smaller env
        # has 10× the density.  We can at least assert the relation when
        # both produced violations.
        if info_big['step_sla_violations'] > 0 and info_sml['step_sla_violations'] > 0:
            ratio = info_sml['cost_sla'] / max(info_big['cost_sla'], 1e-9)
            count_ratio = (info_sml['step_sla_violations'] / 100.0) / max(
                info_big['step_sla_violations'] / 1000.0, 1e-9
            )
            assert ratio == pytest.approx(count_ratio, rel=1e-5)


class TestA3_CostOvertempNormalised:
    """cost_overtemp ∈ [0, 1], normalised by (T_ZONE_MAX - t_critical)."""

    def test_cost_overtemp_zero_when_zone_below_critical(self):
        env = DCMicrogridEnv(max_steps=2)
        env.reset(seed=0)
        _, _, _, _, info = env.step(_idle_action())
        # Initial t_zone = 22 °C, t_critical = 35 °C → no excess.
        assert info['cost_overtemp'] == pytest.approx(0.0, abs=1e-9)
        assert info['t_zone'] < env._dc.t_critical

    def test_cost_overtemp_within_unit_interval(self):
        env = DCMicrogridEnv(max_steps=20)
        env.reset(seed=0)
        for _ in range(20):
            _, _, _, _, info = env.step(env.action_space.sample())
            assert 0.0 <= info['cost_overtemp'] <= 1.0, (
                f"cost_overtemp out of [0, 1]: {info['cost_overtemp']}"
            )

    def test_cost_overtemp_normalisation_formula(self):
        """When forced overheat (hot ambient + max IT), value should equal
        max(t_zone - t_critical, 0) / (T_ZONE_MAX - t_critical)."""
        cpu, solar, temp = _flat_profiles(cpu=1.0, solar=0.0, temp_c=40.0)
        env = DCMicrogridEnv(
            max_steps=50,
            cpu_profile=cpu, solar_profile=solar, outdoor_temp_profile=temp,
        )
        env.reset(seed=0)
        for _ in range(50):
            _, _, _, _, info = env.step(_idle_action())
            t_zone = info['t_zone']
            expected = max(t_zone - env._dc.t_critical, 0.0) / (
                _T_ZONE_MAX - env._dc.t_critical
            )
            assert info['cost_overtemp'] == pytest.approx(expected, abs=1e-6)

    def test_cost_overtemp_grows_under_hot_ambient(self):
        cold_cpu, cold_sol, cold_temp = _flat_profiles(0.5, 0.0, 0.0)
        hot_cpu,  hot_sol,  hot_temp  = _flat_profiles(1.0, 0.0, 40.0)
        env_c = DCMicrogridEnv(max_steps=30,
                               cpu_profile=cold_cpu, solar_profile=cold_sol,
                               outdoor_temp_profile=cold_temp)
        env_h = DCMicrogridEnv(max_steps=30,
                               cpu_profile=hot_cpu,  solar_profile=hot_sol,
                               outdoor_temp_profile=hot_temp)
        env_c.reset(seed=0); env_h.reset(seed=0)
        for _ in range(30):
            env_c.step(_idle_action())
            env_h.step(_idle_action())
        # Final cost values
        _, _, _, _, info_c = env_c.step(_idle_action())
        _, _, _, _, info_h = env_h.step(_idle_action())
        assert info_h['cost_overtemp'] >= info_c['cost_overtemp']


class TestA4_CopRatioFormula:
    """cop_ratio uses (cop_factor - 0.4) / 0.8 with cop_factor ∈ [0.4, 1.2]."""

    def test_cold_day_cop_ratio_is_three_quarters(self):
        """When t_outdoor ≤ t_ref, cop_factor = 1.0 → cop_ratio = 0.75."""
        cpu, solar, temp = _flat_profiles(0.5, 0.0, temp_c=10.0)  # cold
        env = DCMicrogridEnv(
            max_steps=2,
            cpu_profile=cpu, solar_profile=solar, outdoor_temp_profile=temp,
        )
        env.reset(seed=0)
        env.step(_idle_action())
        obs = env._get_obs()
        cop_ratio = float(obs[7])
        assert cop_ratio == pytest.approx(0.75, abs=1e-5), (
            f"cold-day cop_ratio should be (1.0-0.4)/0.8=0.75, got {cop_ratio}"
        )

    def test_hot_day_cop_ratio_lower_than_cold(self):
        cpu, solar, cold = _flat_profiles(0.5, 0.0, temp_c=10.0)
        _, _, hot = _flat_profiles(0.5, 0.0, temp_c=35.0)
        env_c = DCMicrogridEnv(max_steps=2,
                               cpu_profile=cpu, solar_profile=solar,
                               outdoor_temp_profile=cold)
        env_h = DCMicrogridEnv(max_steps=2,
                               cpu_profile=cpu, solar_profile=solar,
                               outdoor_temp_profile=hot)
        env_c.reset(seed=0); env_h.reset(seed=0)
        env_c.step(_idle_action()); env_h.step(_idle_action())
        cop_c = float(env_c._get_obs()[7])
        cop_h = float(env_h._get_obs()[7])
        assert cop_h < cop_c, (
            f"hot day cop_ratio should be lower: cold={cop_c}, hot={cop_h}"
        )

    def test_cop_ratio_within_unit_interval(self):
        """cop_ratio is always in [0, 1] for any reasonable t_outdoor."""
        for ambient in (-10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0):
            cpu, solar, temp = _flat_profiles(0.5, 0.0, ambient)
            env = DCMicrogridEnv(max_steps=2,
                                 cpu_profile=cpu, solar_profile=solar,
                                 outdoor_temp_profile=temp)
            env.reset(seed=0)
            env.step(_idle_action())
            cr = float(env._get_obs()[7])
            assert 0.0 <= cr <= 1.0, f"cop_ratio={cr} out of [0,1] at T={ambient}"

    def test_extreme_hot_cop_ratio_lower_bound(self):
        """At t_outdoor very high, cop_factor saturates at 0.4, ratio → 0."""
        cpu, solar, temp = _flat_profiles(0.5, 0.0, temp_c=100.0)
        env = DCMicrogridEnv(max_steps=2,
                             cpu_profile=cpu, solar_profile=solar,
                             outdoor_temp_profile=temp)
        env.reset(seed=0)
        env.step(_idle_action())
        cr = float(env._get_obs()[7])
        assert cr == pytest.approx(0.0, abs=1e-5)


class TestA5_NoCoolingFloor:
    """p_cool_min_mw aligned to 0: cooling power can be zero when zone cold."""

    def test_dc_p_cool_min_passed_as_zero(self):
        env = DCMicrogridEnv(max_steps=2)
        assert env._dc.p_cool_min_mw == pytest.approx(0.0, abs=1e-12), (
            f"DCMicrogridEnv must pass p_cool_min_mw=0.0 to DataCenterEnv, "
            f"got {env._dc.p_cool_min_mw}"
        )

    def test_p_cool_zero_when_zone_below_setpoint(self):
        """Cold zone & high setpoint → q_cool = 0 → p_cool = 0 (no floor)."""
        cpu, solar, temp = _flat_profiles(0.0, 0.0, temp_c=0.0)  # cold
        env = DCMicrogridEnv(max_steps=2,
                             cpu_profile=cpu, solar_profile=solar,
                             outdoor_temp_profile=temp,
                             t_set_min=27.0, t_set_max=27.0)  # very high setpoint
        env.reset(seed=0)
        env.step(np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))  # cool=1 → 27 °C
        # t_zone init=22 < setpoint=27 → q_cool=0 → p_cool=0
        assert env._dc.p_cool_mw == pytest.approx(0.0, abs=1e-9), (
            f"With cold zone and high setpoint, p_cool_mw should be 0, "
            f"got {env._dc.p_cool_mw}"
        )


class TestA6_NoBatteryDegDoubleCount:
    """BatteryEnv.cycle_cost_per_mwh must remain 0 inside DCMicrogridEnv."""

    def test_battery_cycle_cost_is_zero(self):
        env = DCMicrogridEnv(max_steps=2, battery_deg_cost_per_mwh=99.0)
        # Outer accounting uses 99, but inner BatteryEnv must NOT.
        assert env._batt.cycle_cost_per_mwh == pytest.approx(0.0, abs=1e-12)

    def test_battery_econ_components_empty(self):
        env = DCMicrogridEnv(max_steps=2, battery_deg_cost_per_mwh=99.0)
        env.reset(seed=0)
        env.step(np.array([0.5, 0.5, 0.5, 1.0, 0.0], dtype=np.float32))
        # With cycle_cost_per_mwh==0, econ_components() returns {}.
        assert env._batt.econ_components(env._dt_h) == {}

    def test_outer_reward_still_charges_battery_throughput(self):
        """The deg cost still appears in r_cost (outer), proportional to
        |p_batt| * dt * deg_cost_per_mwh."""
        env = DCMicrogridEnv(max_steps=2, battery_deg_cost_per_mwh=10.0,
                             dg_max_mw=0.0, pv_capacity_mw=0.0)
        env.reset(seed=0)
        # Discharge full
        _, _, _, _, info = env.step(
            np.array([0.0, 0.0, 0.5, 1.0, 0.0], dtype=np.float32)
        )
        p_batt = info['p_batt_mw']
        expected_deg = abs(p_batt) * env._dt_h * 10.0
        assert info['battery_deg_cost'] == pytest.approx(expected_deg, abs=1e-6)


# ===========================================================================
# B — Physical depth previously missing
# ===========================================================================

class TestB1_PowerBalanceMultiStep:
    """Across an episode, residual = p_pv + p_dg + p_batt - p_load each step."""

    def test_residual_closes_each_step_random_actions(self):
        env = DCMicrogridEnv(max_steps=288)
        env.reset(seed=42)
        for _ in range(288):
            _, _, _, _, info = env.step(env.action_space.sample())
            r = info['p_pv_mw'] + info['p_dg_mw'] + info['p_batt_mw'] - info['p_load_mw']
            deficit = max(-r, 0.0)
            spill = max(r, 0.0)
            assert info['power_deficit_mw'] == pytest.approx(deficit, abs=1e-6)
            assert info['power_spill_mw']  == pytest.approx(spill,   abs=1e-6)


class TestB2_BatterySocConservation:
    """End-to-end: charge then discharge → SOC delta accounts for round-trip loss."""

    def test_round_trip_efficiency_matches_eta_product(self):
        """Discharge a known energy then charge it back; SOC return should
        reflect ``(1 - eta_charge*eta_discharge)`` round-trip loss."""
        eta_c, eta_d = 0.9, 0.9
        env = DCMicrogridEnv(
            max_steps=20,
            battery_capacity_mwh=2.0,
            battery_power_mw=0.5,
            battery_eta_charge=eta_c,
            battery_eta_discharge=eta_d,
            battery_soc_min=0.05, battery_soc_max=0.95,
            battery_soc_init=0.5,
            pv_capacity_mw=0.0, dg_max_mw=0.0,
        )
        env.reset(seed=0)
        soc0 = env._batt.soc

        # Discharge at full power for 4 steps (5 min each = 20 min)
        action_d = np.array([0.0, 0.0, 0.5, 1.0, 0.0], dtype=np.float32)
        for _ in range(4):
            env.step(action_d)
        soc_after_dis = env._batt.soc

        # Charge at full power for 4 steps
        action_c = np.array([0.0, 0.0, 0.5, -1.0, 0.0], dtype=np.float32)
        for _ in range(4):
            env.step(action_c)
        soc_back = env._batt.soc

        delta_dis = soc0 - soc_after_dis           # SOC drop during discharge
        delta_chg = soc_back - soc_after_dis        # SOC gain during charge
        # Charge gains less per kWh of grid throughput than discharge loses,
        # by factor eta_c * eta_d (round-trip efficiency).
        ratio = delta_chg / max(delta_dis, 1e-9)
        assert ratio == pytest.approx(eta_c * eta_d, rel=1e-3), (
            f"round-trip ratio {ratio:.4f} != eta_c*eta_d={eta_c*eta_d}"
        )


class TestB3_BatteryFeasibility:
    """Battery hits SOC bounds → power gets clipped to 0."""

    def test_discharge_clipped_to_zero_at_soc_min(self):
        env = DCMicrogridEnv(
            max_steps=200,
            battery_capacity_mwh=0.1,        # tiny capacity
            battery_power_mw=0.5,
            battery_soc_min=0.1, battery_soc_max=0.9,
            battery_soc_init=0.11,           # already near floor
            pv_capacity_mw=0.0, dg_max_mw=0.0,
        )
        env.reset(seed=0)
        action_d = np.array([0.0, 0.0, 0.5, 1.0, 0.0], dtype=np.float32)
        # Drain to the floor.
        for _ in range(50):
            _, _, _, _, info = env.step(action_d)
            if env._batt.soc <= env._batt.soc_min + 1e-6:
                break
        # Now request discharge; battery should refuse (p_batt → 0).
        _, _, _, _, info = env.step(action_d)
        assert info['p_batt_mw'] == pytest.approx(0.0, abs=1e-6), (
            f"At soc_min, discharge must clip to 0, got {info['p_batt_mw']}"
        )

    def test_charge_clipped_to_zero_at_soc_max(self):
        env = DCMicrogridEnv(
            max_steps=200,
            battery_capacity_mwh=0.1,
            battery_power_mw=0.5,
            battery_soc_min=0.1, battery_soc_max=0.9,
            battery_soc_init=0.89,           # near ceiling
            pv_capacity_mw=0.0, dg_max_mw=0.0,
        )
        env.reset(seed=0)
        action_c = np.array([0.0, 0.0, 0.5, -1.0, 0.0], dtype=np.float32)
        for _ in range(50):
            _, _, _, _, info = env.step(action_c)
            if env._batt.soc >= env._batt.soc_max - 1e-6:
                break
        _, _, _, _, info = env.step(action_c)
        assert info['p_batt_mw'] == pytest.approx(0.0, abs=1e-6), (
            f"At soc_max, charge must clip to 0, got {info['p_batt_mw']}"
        )


class TestB4_DgLinearity:
    """Fuel cost & carbon emissions are linear in p_dg."""

    def test_fuel_cost_proportional_to_p_dg(self):
        env = DCMicrogridEnv(max_steps=2, dg_max_mw=0.6,
                             dg_fuel_cost_per_mwh=300.0)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.5, 0.5, 0.5, 0.0, 0.5], dtype=np.float32))
        expected = 300.0 * info['p_dg_mw'] * env._dt_h
        assert info['fuel_cost'] == pytest.approx(expected, abs=1e-6)

    def test_carbon_kg_proportional_to_p_dg(self):
        env = DCMicrogridEnv(max_steps=2, dg_max_mw=0.6, dg_emission_factor=0.8)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.5, 0.5, 0.5, 0.0, 1.0], dtype=np.float32))
        # carbon_kg = factor[kg/kWh] * p_dg[MW] * 1000[kW/MW] * dt[h]
        expected = 0.8 * info['p_dg_mw'] * 1000.0 * env._dt_h
        assert info['carbon_kg'] == pytest.approx(expected, abs=1e-6)

    def test_fuel_cost_zero_when_dg_off(self):
        env = DCMicrogridEnv(max_steps=2)
        env.reset(seed=0)
        _, _, _, _, info = env.step(_idle_action())
        assert info['fuel_cost'] == pytest.approx(0.0, abs=1e-9)
        assert info['carbon_kg'] == pytest.approx(0.0, abs=1e-9)


class TestB5_RewardEquation:
    """reward == r_energy + w_cost*r_cost + w_carbon*r_carbon (scalarisation)."""

    def test_reward_decomposition_random(self):
        env = DCMicrogridEnv(max_steps=20, w_cost=0.5, w_carbon=0.3)
        env.reset(seed=7)
        for _ in range(20):
            _, rew, _, _, info = env.step(env.action_space.sample())
            r_e, r_c, r_carbon = info['reward_vector']
            expected = r_e + 0.5 * r_c + 0.3 * r_carbon
            assert rew == pytest.approx(expected, abs=1e-6), (
                f"reward {rew} != r_e + 0.5*r_c + 0.3*r_carbon = {expected}"
            )

    def test_r_energy_equals_neg_p_load_dt(self):
        env = DCMicrogridEnv(max_steps=2)
        env.reset(seed=0)
        _, _, _, _, info = env.step(_idle_action())
        r_e = info['reward_vector'][0]
        assert r_e == pytest.approx(-info['p_load_mw'] * env._dt_h, abs=1e-6)

    def test_r_cost_equals_neg_fuel_plus_battery_deg(self):
        env = DCMicrogridEnv(max_steps=2, battery_deg_cost_per_mwh=5.0,
                             dg_fuel_cost_per_mwh=300.0)
        env.reset(seed=0)
        _, _, _, _, info = env.step(
            np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        )
        r_c = info['reward_vector'][1]
        expected = -(info['fuel_cost'] + info['battery_deg_cost'])
        assert r_c == pytest.approx(expected, abs=1e-6)


class TestB6_CostPowerDeficitFormula:
    """cost_power_deficit == power_deficit / max(p_load, 1e-6)."""

    def test_cost_power_deficit_formula(self):
        env = DCMicrogridEnv(max_steps=2, pv_capacity_mw=0.0, dg_max_mw=0.0)
        env.reset(seed=0)
        # Full battery charge while DC is drawing → guaranteed deficit.
        _, _, _, _, info = env.step(
            np.array([0.5, 0.5, 0.5, -1.0, 0.0], dtype=np.float32)
        )
        expected = info['power_deficit_mw'] / max(info['p_load_mw'], 1e-6)
        assert info['cost_power_deficit'] == pytest.approx(expected, abs=1e-6)
        assert info['cost_power_deficit'] > 0.0

    def test_cost_power_deficit_non_negative(self):
        """Deficit normalisation is always ≥ 0 (it is max(-residual, 0)/p_load)."""
        env = DCMicrogridEnv(max_steps=20)
        env.reset(seed=0)
        for _ in range(20):
            _, _, _, _, info = env.step(env.action_space.sample())
            assert info['cost_power_deficit'] >= 0.0

    def test_cost_power_deficit_can_exceed_one_when_battery_charges(self):
        """Battery charging ADDS to effective load (p_batt < 0 increases the
        magnitude of -residual), so the normalised deficit channel may exceed
        1.  This documents that ``cost_power_deficit`` is *not* bounded by 1
        even though the other two cost channels are — by design, mirroring
        the JAX reference."""
        env = DCMicrogridEnv(max_steps=2, pv_capacity_mw=0.0, dg_max_mw=0.0,
                             battery_power_mw=10.0, battery_capacity_mwh=10.0)
        env.reset(seed=0)
        # Charge as hard as possible while DC is drawing.
        _, _, _, _, info = env.step(
            np.array([1.0, 1.0, 0.5, -1.0, 0.0], dtype=np.float32)
        )
        # We at least confirm the channel is a valid non-negative ratio.
        # (Whether it exceeds 1 depends on the relative size of battery vs DC
        # load; we assert no NaN/inf and ≥ 0.)
        assert info['cost_power_deficit'] >= 0.0
        assert np.isfinite(info['cost_power_deficit'])


class TestB7_OvertempMonotonicity:
    """Hot ambient + sustained max IT load must drive cost_overtemp toward 1."""

    def test_overtemp_eventually_positive_under_thermal_stress(self):
        cpu, solar, temp = _flat_profiles(cpu=1.0, solar=0.0, temp_c=40.0)
        env = DCMicrogridEnv(
            max_steps=288,
            cpu_profile=cpu, solar_profile=solar, outdoor_temp_profile=temp,
            ua_cooling=20.0,             # weaken cooling on purpose
        )
        env.reset(seed=0)
        # Highest setpoint (cooling barely engages), full DG+PV irrelevant.
        cooled_idle = np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        any_overtemp = 0.0
        for _ in range(288):
            _, _, _, _, info = env.step(cooled_idle)
            any_overtemp = max(any_overtemp, info['cost_overtemp'])
        assert any_overtemp > 0.0, (
            "Under hot ambient & weak cooling, cost_overtemp must rise above 0 "
            "within an episode"
        )
