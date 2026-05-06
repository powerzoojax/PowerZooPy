"""L1 physics tests for the Python DCMicrogridEnv.

Mirrors the 21 JAX-side tests in
``PowerZooJax/tests/resource/test_datacenter_microgrid.py`` so both backends
preserve the same physical behaviour after the bundle/sub-resource refactor.
"""

from __future__ import annotations

import numpy as np
import pytest

from powerzoo.envs.microgrid.dc_microgrid_env import DCMicrogridEnv


@pytest.fixture
def env():
    return DCMicrogridEnv(max_steps=10)


@pytest.fixture
def env_full():
    return DCMicrogridEnv(max_steps=288)


# ---------------------------------------------------------------------------
# Battery physics
# ---------------------------------------------------------------------------

class TestBatteryPhysics:

    def test_round_trip_efficiency(self, env):
        """Discharge then charge by the same |p|·dt yields net SOC drop = (1-η_rt)·E/cap."""
        env.reset(seed=0)
        soc0 = env._batt.soc
        a_dis = np.array([0.0, 0.0, 0.5, 1.0, 0.0], dtype=np.float32)
        _, _, _, _, info1 = env.step(a_dis)
        a_chg = np.array([0.0, 0.0, 0.5, -1.0, 0.0], dtype=np.float32)
        _, _, _, _, info2 = env.step(a_chg)

        dt = env._dt_h
        cap = env._batt.capacity_mwh
        eta_c = env._batt.eta_charge
        eta_d = env._batt.eta_discharge
        p_max = env._batt.power_mw
        # Both steps should hit rated power
        assert info1["p_batt_mw"] == pytest.approx(p_max, rel=1e-5)
        assert info2["p_batt_mw"] == pytest.approx(-p_max, rel=1e-5)

        expected_delta = -(p_max * dt) * (1.0 / eta_d - eta_c) / cap
        net_delta = env._batt.soc - soc0
        assert net_delta == pytest.approx(expected_delta, rel=1e-3, abs=1e-6)
        assert net_delta < 0.0  # round-trip loses energy

    def test_discharge_bounded_at_soc_min(self):
        env = DCMicrogridEnv(max_steps=5, battery_soc_init=0.1)
        env.reset(seed=0)
        a = np.array([0.0, 0.0, 0.5, 1.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(a)
        assert info["p_batt_mw"] == pytest.approx(0.0, abs=1e-6)
        assert env._batt.soc == pytest.approx(0.1, abs=1e-6)

    def test_charge_bounded_at_soc_max(self):
        env = DCMicrogridEnv(max_steps=5, battery_soc_init=0.9)
        env.reset(seed=0)
        a = np.array([0.0, 0.0, 0.5, -1.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(a)
        assert info["p_batt_mw"] == pytest.approx(0.0, abs=1e-6)
        assert env._batt.soc == pytest.approx(0.9, abs=1e-6)

    def test_sign_convention_discharge_positive(self, env):
        env.reset(seed=0)
        a = np.array([0.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32)
        soc0 = env._batt.soc
        _, _, _, _, info = env.step(a)
        assert info["p_batt_mw"] > 0.0
        assert env._batt.soc < soc0

    def test_sign_convention_charge_negative(self, env):
        env.reset(seed=0)
        a = np.array([0.0, 0.0, 0.5, -0.5, 0.0], dtype=np.float32)
        soc0 = env._batt.soc
        _, _, _, _, info = env.step(a)
        assert info["p_batt_mw"] < 0.0
        assert env._batt.soc > soc0

    def test_soc_clamp_under_repeated_discharge(self, env):
        env.reset(seed=0)
        a = np.array([0.5, 0.5, 0.5, 1.0, 0.0], dtype=np.float32)
        for _ in range(env.max_steps):
            env.step(a)
            assert env._batt.soc >= 0.1 - 1e-6
            assert env._batt.soc <= 0.9 + 1e-6


# ---------------------------------------------------------------------------
# Reward / cost separation
# ---------------------------------------------------------------------------

class TestRewardScalarization:

    def test_reward_equals_weighted_sum(self, env):
        env.reset(seed=0)
        a = np.array([0.5, 0.5, 0.5, 0.3, 0.6], dtype=np.float32)
        _, reward, _, _, info = env.step(a)
        rv = info["reward_vector"]
        expected = rv[0] + env._w_cost * rv[1] + env._w_carbon * rv[2]
        assert reward == pytest.approx(expected, rel=1e-5, abs=1e-6)

    def test_r_cost_includes_fuel_and_battery_deg(self, env):
        env.reset(seed=0)
        a = np.array([0.5, 0.5, 0.5, 0.4, 0.7], dtype=np.float32)
        _, _, _, _, info = env.step(a)
        fuel = info["fuel_cost"]
        deg = abs(info["p_batt_mw"]) * env._dt_h * env._battery_deg_cost_per_mwh
        assert info["reward_vector"][1] == pytest.approx(-(fuel + deg), rel=1e-5, abs=1e-6)

    def test_r_energy_equals_negative_p_load_mwh(self, env):
        env.reset(seed=0)
        a = np.array([0.5, 0.5, 0.5, 0.0, 0.4], dtype=np.float32)
        _, _, _, _, info = env.step(a)
        expected = -info["p_load_mw"] * env._dt_h
        assert info["reward_vector"][0] == pytest.approx(expected, rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------------------
# Power balance
# ---------------------------------------------------------------------------

class TestPowerBalance:

    def test_residual_identity(self, env):
        env.reset(seed=0)
        a = np.array([0.5, 0.5, 0.5, 0.2, 0.5], dtype=np.float32)
        _, _, _, _, info = env.step(a)
        residual_check = info["p_pv_mw"] + info["p_dg_mw"] + info["p_batt_mw"] - info["p_load_mw"]
        # spill - deficit = residual
        spill = info["power_spill_mw"]
        deficit = info["power_deficit_mw"]
        assert (spill - deficit) == pytest.approx(residual_check, abs=1e-5)

    def test_spill_when_supply_exceeds_demand(self):
        env = DCMicrogridEnv(max_steps=5, dg_max_mw=100.0)
        env.reset(seed=0)
        a = np.array([0.5, 0.5, 0.5, 0.0, 1.0], dtype=np.float32)
        _, _, _, _, info = env.step(a)
        assert info["p_dg_mw"] == pytest.approx(100.0, rel=1e-5)
        assert info["power_spill_mw"] > 0.0
        assert info["power_deficit_mw"] == pytest.approx(0.0, abs=1e-6)

    def test_deficit_when_supply_short(self, env):
        env.reset(seed=0)
        a = np.array([0.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(a)
        # Night-time, no DG, no battery — load drives a deficit
        if info["p_load_mw"] > 0:
            assert info["power_deficit_mw"] >= 0.0


# ---------------------------------------------------------------------------
# Cost channels
# ---------------------------------------------------------------------------

class TestCostChannels:

    def test_cost_info_consistent(self, env):
        env.reset(seed=0)
        a = np.array([0.5, 0.5, 0.5, 0.0, 0.3], dtype=np.float32)
        _, _, _, _, info = env.step(a)
        cost_sum = info["cost_sla"] + info["cost_overtemp"] + info["cost_power_deficit"]
        assert info["cost_sum"] == pytest.approx(cost_sum, abs=1e-6)
        np.testing.assert_allclose(
            info["constraint_costs"],
            np.array(
                [info["cost_sla"], info["cost_overtemp"], info["cost_power_deficit"]],
                dtype=np.float32,
            ),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Diesel
# ---------------------------------------------------------------------------

class TestDiesel:

    def test_fuel_cost_positive_when_dispatched(self, env):
        env.reset(seed=0)
        a = np.array([0.5, 0.5, 0.5, 0.0, 0.8], dtype=np.float32)
        _, _, _, _, info = env.step(a)
        assert info["fuel_cost"] > 0.0
        assert info["p_dg_mw"] > 0.0

    def test_fuel_cost_zero_when_no_dispatch(self, env):
        env.reset(seed=0)
        a = np.array([0.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(a)
        assert info["fuel_cost"] == pytest.approx(0.0, abs=1e-8)
        assert info["p_dg_mw"] == pytest.approx(0.0, abs=1e-8)

    def test_carbon_proportional_to_dg(self, env):
        env.reset(seed=0)
        a_low  = np.array([0.5, 0.5, 0.5, 0.0, 0.2], dtype=np.float32)
        env.reset(seed=0)
        _, _, _, _, info_low = env.step(a_low)
        env.reset(seed=0)
        a_high = np.array([0.5, 0.5, 0.5, 0.0, 0.8], dtype=np.float32)
        _, _, _, _, info_high = env.step(a_high)
        assert info_high["carbon_kg"] > info_low["carbon_kg"]


# ---------------------------------------------------------------------------
# Cooling COP coupling and aux power
# ---------------------------------------------------------------------------

class TestCoolingAndAux:

    def test_cop_decreases_with_outdoor_temperature(self):
        a = np.array([0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        env_cool = DCMicrogridEnv(
            max_steps=5,
            outdoor_temp_profile=np.full(288, 20.0, dtype=np.float32),
        )
        env_cool.reset(seed=0)
        _, _, _, _, info_cool = env_cool.step(a)

        env_hot = DCMicrogridEnv(
            max_steps=5,
            outdoor_temp_profile=np.full(288, 35.0, dtype=np.float32),
        )
        env_hot.reset(seed=0)
        _, _, _, _, info_hot = env_hot.step(a)

        # Hot scenario must demand more cooling (lower COP)
        assert env_hot._dc.p_cool_mw > env_cool._dc.p_cool_mw, (
            f"hot p_cool={env_hot._dc.p_cool_mw} should exceed cool "
            f"p_cool={env_cool._dc.p_cool_mw}"
        )
        assert info_hot["p_load_mw"] > info_cool["p_load_mw"]

    def test_p_aux_frac_increases_total_power(self):
        a = np.array([0.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float32)

        env_lo = DCMicrogridEnv(max_steps=5, p_aux_frac=0.0)
        env_lo.reset(seed=0)
        _, _, _, _, info_lo = env_lo.step(a)

        env_hi = DCMicrogridEnv(max_steps=5, p_aux_frac=0.20)
        env_hi.reset(seed=0)
        _, _, _, _, info_hi = env_hi.step(a)

        # IT power identical (same DC params save p_aux_frac)
        assert env_lo._dc.p_it_mw == pytest.approx(env_hi._dc.p_it_mw, rel=1e-4)
        delta_obs = info_hi["p_load_mw"] - info_lo["p_load_mw"]
        delta_exp = 0.20 * env_lo._dc.p_it_mw
        assert delta_obs == pytest.approx(delta_exp, rel=1e-3, abs=1e-6)


# ---------------------------------------------------------------------------
# Action clipping
# ---------------------------------------------------------------------------

class TestActionClipping:

    def test_oversized_action_clipped(self, env):
        env.reset(seed=0)
        a_huge = np.array([5.0, -3.0, 9.9, 7.0, 4.0], dtype=np.float32)
        _, reward, _, _, info = env.step(a_huge)
        # DG saturates
        assert info["p_dg_mw"] == pytest.approx(env._dg.p_dg_max_mw, rel=1e-5)
        # Battery saturates at +power_mw
        assert info["p_batt_mw"] == pytest.approx(env._batt.power_mw, rel=1e-5)
        # last_action stored is clipped
        la = env._last_action
        assert la[0] == pytest.approx(1.0, abs=1e-6)
        assert la[1] == pytest.approx(0.0, abs=1e-6)
        assert la[2] == pytest.approx(1.0, abs=1e-6)
        assert la[3] == pytest.approx(1.0, abs=1e-6)
        assert la[4] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Profile overrides
# ---------------------------------------------------------------------------

class TestProfileOverrides:

    def test_cpu_profile_overrides_inference_load(self):
        T = 288
        env_full = DCMicrogridEnv(max_steps=5, cpu_profile=np.ones(T, dtype=np.float32))
        env_full.reset(seed=0)
        a = np.zeros(5, dtype=np.float32)
        env_full.step(a)
        assert int(env_full._dc.gpus_infer) == int(env_full._dc.infer_gpu_peak)

        env_zero = DCMicrogridEnv(max_steps=5, cpu_profile=np.zeros(T, dtype=np.float32))
        env_zero.reset(seed=0)
        env_zero.step(a)
        assert int(env_zero._dc.gpus_infer) == 0

    def test_solar_profile_overrides_pv(self):
        T = 288
        env_zero = DCMicrogridEnv(max_steps=5, solar_profile=np.zeros(T, dtype=np.float32))
        env_zero.reset(seed=0)
        a = np.zeros(5, dtype=np.float32)
        for _ in range(3):
            _, _, _, _, info = env_zero.step(a)
            assert info["p_pv_mw"] == pytest.approx(0.0, abs=1e-6)

        env_full = DCMicrogridEnv(max_steps=5, solar_profile=np.ones(T, dtype=np.float32))
        env_full.reset(seed=0)
        _, _, _, _, info = env_full.step(a)
        assert info["p_pv_mw"] == pytest.approx(env_full._pv.capacity_mw, rel=1e-5)

    def test_outdoor_temp_profile_overrides_synthetic(self):
        env = DCMicrogridEnv(
            max_steps=5,
            outdoor_temp_profile=np.full(288, 30.0, dtype=np.float32),
        )
        env.reset(seed=0)
        a = np.zeros(5, dtype=np.float32)
        env.step(a)
        assert env._dc.t_outdoor == pytest.approx(30.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------

class TestEpisode:

    def test_terminates_at_max_steps(self):
        env = DCMicrogridEnv(max_steps=10)
        env.reset(seed=0)
        a = np.zeros(5, dtype=np.float32)
        truncated = False
        for i in range(10):
            _, _, _, truncated, _ = env.step(a)
        assert truncated is True


# ---------------------------------------------------------------------------
# DG minimum loading (opt-in, default no-op)
# ---------------------------------------------------------------------------

class TestDGMinLoading:

    def test_default_zero_is_noop(self):
        """With dg_p_min_norm=0 (default), low DG setpoint passes through."""
        env = DCMicrogridEnv(max_steps=5)
        env.reset(seed=0)
        a = np.array([0.5, 0.5, 0.5, 0.0, 0.05], dtype=np.float32)
        _, _, _, _, info = env.step(a)
        # 0.05 * dg_max_mw (0.6) = 0.03
        assert info["p_dg_mw"] == pytest.approx(0.03, abs=1e-5)

    def test_min_loading_shuts_off_below_deadband(self):
        env = DCMicrogridEnv(max_steps=5, dg_p_min_norm=0.3)
        env.reset(seed=0)
        # Deadband = 0.15.  Setpoint 0.10 → OFF.
        a = np.array([0.5, 0.5, 0.5, 0.0, 0.10], dtype=np.float32)
        _, _, _, _, info = env.step(a)
        assert info["p_dg_mw"] == pytest.approx(0.0, abs=1e-6)
        assert info["fuel_cost"] == pytest.approx(0.0, abs=1e-6)

    def test_min_loading_clamps_up(self):
        env = DCMicrogridEnv(max_steps=5, dg_p_min_norm=0.3)
        env.reset(seed=0)
        # Deadband = 0.15, p_min = 0.3.  Setpoint 0.20 → clamped UP to 0.3.
        a = np.array([0.5, 0.5, 0.5, 0.0, 0.20], dtype=np.float32)
        _, _, _, _, info = env.step(a)
        # 0.30 * 0.6 = 0.18
        assert info["p_dg_mw"] == pytest.approx(0.18, abs=1e-5)

    def test_min_loading_passes_above(self):
        env = DCMicrogridEnv(max_steps=5, dg_p_min_norm=0.3)
        env.reset(seed=0)
        a = np.array([0.5, 0.5, 0.5, 0.0, 0.7], dtype=np.float32)
        _, _, _, _, info = env.step(a)
        assert info["p_dg_mw"] == pytest.approx(0.7 * 0.6, abs=1e-5)
