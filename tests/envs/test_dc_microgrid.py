"""DC Microgrid env tests (tests/envs/test_dc_microgrid.py).

Covers the benchmark acceptance criteria and blocking-issue regressions
for the Python dc_microgrid task surface.

Test groups
-----------
T1  Registry & instantiation      — make_task / make_task_env round-trip
T2  Dimensions & defaults         — obs 18-D, action 5-D, 288 steps × 5 min
T3  Smoke step                    — reset + step works, required info keys present
T4  Cost/reward channels          — cost_sla, cost_overtemp, cost_power_deficit,
                                    constraint_costs, cost_sum, reward_vector
T5  cpu_profile drives load       — regression: was no-op before override hook fix
T6  outdoor_temp_profile drives   — regression: was no-op before override hook fix
    thermal
T7  dg_margin_norm is dynamic     — regression: was hardcoded 1.0
T8  workload_swap/shock semantics — not in-place reverse/spike; source-based OOD
T9  workload OOD strict raises    — strict=True + synthetic data → ValueError
"""

from __future__ import annotations

import numpy as np
import pytest

from powerzoo.envs.microgrid.dc_microgrid_env import DCMicrogridEnv
from powerzoo.data.dc_microgrid_profiles import (
    make_all_synthetic_profiles,
    apply_ood_transform,
    VALID_OOD_SCENARIOS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(**kw) -> DCMicrogridEnv:
    return DCMicrogridEnv(**kw)


def _zero_action() -> np.ndarray:
    return np.array([0.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float32)


# ===========================================================================
# T1 — Registry & instantiation
# ===========================================================================

class TestT1_RegistryInstantiation:
    def test_make_task_dc_microgrid(self):
        from powerzoo.tasks.registry import make_task
        task = make_task('dc_microgrid')
        env = task.create_single_agent_env()
        obs, info = env.reset(seed=0)
        assert obs.shape == (18,)

    def test_make_task_dc_microgrid_safe(self):
        from powerzoo.tasks.registry import make_task
        task = make_task('dc_microgrid_safe')
        env = task.create_single_agent_env()
        obs, _ = env.reset(seed=0)
        assert obs.shape == (18,)

    def test_make_task_env_dc_microgrid(self):
        from powerzoo.tasks.registry import make_task_env
        env = make_task_env('dc_microgrid')
        obs, _ = env.reset()
        assert obs is not None

    def test_dc_microgrid_in_task_list(self):
        from powerzoo.tasks.registry import list_tasks
        tasks = list_tasks()
        assert 'dc_microgrid' in tasks
        assert 'dc_microgrid_safe' in tasks

    def test_dc_scheduling_unaffected(self):
        """Old dc_scheduling task must still work and keep its original config."""
        from powerzoo.tasks.registry import make_task
        task = make_task('dc_scheduling')
        cfg = task.get_scenario_config()
        assert cfg['episode']['max_steps'] == 48


# ===========================================================================
# T2 — Dimensions & defaults
# ===========================================================================

class TestT2_DimensionsDefaults:
    def test_obs_dim_18(self):
        env = _make_env()
        obs, _ = env.reset(seed=0)
        assert obs.shape == (18,), f"Expected (18,), got {obs.shape}"

    def test_action_dim_5(self):
        env = _make_env()
        assert env.action_space.shape == (5,)

    def test_default_max_steps_288(self):
        env = _make_env()
        assert env.max_steps == 288

    def test_default_delta_t_5min(self):
        env = _make_env()
        assert env.delta_t_minutes == pytest.approx(5.0)

    def test_action_space_bounds(self):
        env = _make_env()
        lo, hi = env.action_space.low, env.action_space.high
        # train, ft, cooling ∈ [0, 1]; battery ∈ [-1, 1]; dg ∈ [0, 1]
        assert lo[0] == pytest.approx(0.0) and hi[0] == pytest.approx(1.0)
        assert lo[3] == pytest.approx(-1.0) and hi[3] == pytest.approx(1.0)
        assert lo[4] == pytest.approx(0.0) and hi[4] == pytest.approx(1.0)

    def test_obs_in_valid_range_after_reset(self):
        env = _make_env()
        obs, _ = env.reset(seed=42)
        assert obs.shape == (18,)
        # Obs should be finite
        assert np.all(np.isfinite(obs)), f"Non-finite obs values: {obs}"

    def test_episode_terminates_at_288(self):
        env = _make_env()
        env.reset(seed=0)
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            steps += 1
        assert steps == 288, f"Episode should last 288 steps, got {steps}"


# ===========================================================================
# T3 — Smoke step
# ===========================================================================

class TestT3_SmokeStep:
    def test_reset_step_shapes(self):
        env = _make_env()
        obs, info = env.reset(seed=0)
        obs2, rew, terminated, truncated, info2 = env.step(_zero_action())
        assert obs.shape == (18,)
        assert obs2.shape == (18,)
        assert isinstance(rew, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_info_keys_present(self):
        env = _make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(_zero_action())
        for key in ('constraint_costs', 'cost_sla', 'cost_overtemp', 'cost_power_deficit',
                    'cost_sum', 'reward_vector', 'p_load_mw', 'gpu_util',
                    't_outdoor'):
            assert key in info, f"Missing info key: {key}"

    def test_reward_is_negative(self):
        """Reward should be non-positive (energy + costs are penalties)."""
        env = _make_env()
        env.reset(seed=0)
        rewards = []
        for _ in range(10):
            _, rew, _, _, _ = env.step(_zero_action())
            rewards.append(rew)
        assert all(r <= 0.0 for r in rewards), (
            f"All rewards should be ≤ 0, got max={max(rewards):.4f}"
        )


# ===========================================================================
# T4 — Cost / reward channels
# ===========================================================================

class TestT4_CostRewardChannels:
    def test_constraint_costs_equal_sum_of_channels(self):
        env = _make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(_zero_action())
        expected = info['cost_sla'] + info['cost_overtemp'] + info['cost_power_deficit']
        assert info['cost_sum'] == pytest.approx(expected, abs=1e-6)
        np.testing.assert_allclose(
            info['constraint_costs'],
            np.array(
                [info['cost_sla'], info['cost_overtemp'], info['cost_power_deficit']],
                dtype=np.float32,
            ),
            atol=1e-6,
        )

    def test_cost_sum_alias(self):
        env = _make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(_zero_action())
        assert info['cost_sum'] == pytest.approx(float(np.sum(info['constraint_costs'])), abs=1e-9)

    def test_reward_vector_length_3(self):
        env = _make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(_zero_action())
        assert len(info['reward_vector']) == 3

    def test_cost_channels_non_negative(self):
        env = _make_env()
        env.reset(seed=0)
        for _ in range(5):
            _, _, _, _, info = env.step(env.action_space.sample())
            for ch in ('cost_sla', 'cost_overtemp', 'cost_power_deficit'):
                assert info[ch] >= 0.0, f"{ch} must be non-negative, got {info[ch]}"

    def test_power_deficit_appears_without_generation(self):
        """With no PV, no DG, no battery, there should be power deficit."""
        env = DCMicrogridEnv(max_steps=2, pv_capacity_mw=0.0, dg_max_mw=0.0)
        env.reset(seed=0)
        # Battery fully discharged action = charge (negative), DG/PV = 0
        action = np.array([0.5, 0.5, 0.5, -1.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info['cost_power_deficit'] > 0.0, (
            "With no generation and battery charging, power_deficit must be > 0"
        )


# ===========================================================================
# T5 — cpu_profile drives load (regression: was no-op)
# ===========================================================================

class TestT5_CpuProfileDrivesLoad:
    def test_cpu_zero_gives_zero_gpu_util(self):
        env = DCMicrogridEnv(max_steps=2, cpu_profile=np.zeros(288, np.float32))
        env.reset(seed=0)
        _, _, _, _, info = env.step(_zero_action())
        assert info['gpu_util'] == pytest.approx(0.0, abs=1e-4)

    def test_cpu_one_gives_positive_gpu_util(self):
        env = DCMicrogridEnv(max_steps=2, cpu_profile=np.ones(288, np.float32))
        env.reset(seed=0)
        _, _, _, _, info = env.step(_zero_action())
        assert info['gpu_util'] > 0.0

    def test_cpu_zero_vs_one_p_load_differ(self):
        env_lo = DCMicrogridEnv(max_steps=2, cpu_profile=np.zeros(288, np.float32))
        env_hi = DCMicrogridEnv(max_steps=2, cpu_profile=np.ones(288, np.float32))
        for env in (env_lo, env_hi):
            env.reset(seed=0)
        _, _, _, _, info_lo = env_lo.step(_zero_action())
        _, _, _, _, info_hi = env_hi.step(_zero_action())
        diff = abs(info_hi['p_load_mw'] - info_lo['p_load_mw'])
        assert diff > 0.01, (
            f"p_load_mw should differ: lo={info_lo['p_load_mw']:.4f}, "
            f"hi={info_hi['p_load_mw']:.4f}"
        )

    def test_set_profiles_cpu_takes_effect(self):
        """set_profiles(cpu=...) before reset must change observed load."""
        env = DCMicrogridEnv(max_steps=2)
        env.reset(seed=0)
        _, _, _, _, info_synth = env.step(_zero_action())

        env.set_profiles(cpu=np.zeros(288, np.float32))
        env.reset(seed=0)
        _, _, _, _, info_zero = env.step(_zero_action())

        assert info_zero['gpu_util'] == pytest.approx(0.0, abs=1e-4)
        assert info_zero['p_load_mw'] < info_synth['p_load_mw']


# ===========================================================================
# T6 — outdoor_temp_profile drives thermal (regression: was no-op)
# ===========================================================================

class TestT6_OutdoorTempProfileDrivesThermal:
    def test_t_outdoor_matches_profile_each_step(self):
        """info['t_outdoor'] must track the injected profile step-by-step."""
        profile = np.linspace(5.0, 35.0, 288, dtype=np.float32)
        env = DCMicrogridEnv(max_steps=5, outdoor_temp_profile=profile)
        env.reset(seed=0)
        for t in range(5):
            _, _, _, _, info = env.step(_zero_action())
            assert info['t_outdoor'] == pytest.approx(float(profile[t]), abs=0.1), (
                f"Step {t}: expected t_outdoor={profile[t]:.1f}, got {info['t_outdoor']:.1f}"
            )

    def test_cold_vs_hot_profile_t_outdoor(self):
        cold = np.full(288, 0.0, np.float32)
        hot  = np.full(288, 40.0, np.float32)
        env_c = DCMicrogridEnv(max_steps=2, outdoor_temp_profile=cold)
        env_h = DCMicrogridEnv(max_steps=2, outdoor_temp_profile=hot)
        env_c.reset(seed=0); env_h.reset(seed=0)
        _, _, _, _, ic = env_c.step(_zero_action())
        _, _, _, _, ih = env_h.step(_zero_action())
        assert ic['t_outdoor'] == pytest.approx(0.0, abs=0.5)
        assert ih['t_outdoor'] == pytest.approx(40.0, abs=0.5)

    def test_hot_ambient_raises_t_zone(self):
        """After several steps, hot ambient should produce higher zone temperature."""
        cold = np.full(288, 0.0, np.float32)
        hot  = np.full(288, 40.0, np.float32)
        env_c = DCMicrogridEnv(max_steps=5, outdoor_temp_profile=cold)
        env_h = DCMicrogridEnv(max_steps=5, outdoor_temp_profile=hot)
        for env in (env_c, env_h):
            env.reset(seed=0)
            for _ in range(5):
                env.step(_zero_action())
        assert env_h._dc.t_zone > env_c._dc.t_zone, (
            f"Hot ambient should give higher t_zone: "
            f"cold={env_c._dc.t_zone:.2f}, hot={env_h._dc.t_zone:.2f}"
        )


# ===========================================================================
# T7 — dg_margin_norm is dynamic (regression: was hardcoded 1.0)
# ===========================================================================

class TestT7_DgMarginNormDynamic:
    def test_dg_off_margin_is_one(self):
        env = _make_env(max_steps=2)
        env.reset(seed=0)
        env.step(np.array([0.5, 0.5, 0.5, 0.0, 0.0], np.float32))  # dg=0
        assert env._get_obs()[10] == pytest.approx(1.0, abs=1e-5)

    def test_dg_full_margin_is_zero(self):
        env = _make_env(max_steps=2)
        env.reset(seed=0)
        env.step(np.array([0.5, 0.5, 0.5, 0.0, 1.0], np.float32))  # dg=1
        assert env._get_obs()[10] == pytest.approx(0.0, abs=1e-5)

    def test_dg_half_margin_between(self):
        env = _make_env(max_steps=2)
        env.reset(seed=0)
        env.step(np.array([0.5, 0.5, 0.5, 0.0, 0.5], np.float32))  # dg=0.5
        margin = env._get_obs()[10]
        assert 0.0 < margin < 1.0, f"Half DG margin should be in (0,1), got {margin}"

    def test_dg_margin_resets_to_one(self):
        env = _make_env(max_steps=2)
        env.reset(seed=0)
        env.step(np.array([0.5, 0.5, 0.5, 0.0, 1.0], np.float32))  # full DG
        env.reset(seed=0)  # reset
        assert env._get_obs()[10] == pytest.approx(1.0, abs=1e-5)


# ===========================================================================
# T8 — workload OOD: source-based, not in-place transform
# ===========================================================================

class TestT8_WorkloadOODSourceBased:
    def _real_profiles(self) -> dict:
        """Synthetic profiles but with real_data=True sentinel."""
        p = make_all_synthetic_profiles()
        p['real_data'] = True
        return p

    def test_workload_swap_not_time_reversal(self):
        """workload_swap must NOT return cpu[::-1] (the old wrong implementation)."""
        p = self._real_profiles()
        orig = p['cpu'].copy()
        out = apply_ood_transform(p, 'workload_swap', strict=False)
        assert not np.allclose(out['cpu'], orig[::-1]), (
            "workload_swap must not be time-reversal of input cpu profile"
        )

    def test_workload_shock_not_peak_spike(self):
        """workload_shock must NOT be in-place peak*scale (old wrong impl)."""
        p = self._real_profiles()
        orig = p['cpu'].copy()
        out = apply_ood_transform(p, 'workload_shock', strict=False)
        peak_idx = int(np.argmax(orig))
        n = len(orig)
        shock_w = max(1, n // 48)
        lo, hi = max(0, peak_idx - shock_w // 2), min(n, peak_idx + shock_w // 2)
        old_spike = orig[lo:hi] * 2.0  # old workload_shock_scale=2.0
        # New output must NOT match old spike pattern
        assert not np.allclose(out['cpu'][lo:hi], np.clip(old_spike, 0, 1)), (
            "workload_shock must not be in-place peak spike of the current cpu"
        )

    def test_workload_swap_references_azure_in_source(self):
        """Implementation must reference 'azure' as the swap target."""
        import inspect
        from powerzoo.data import dc_microgrid_profiles
        src = inspect.getsource(dc_microgrid_profiles)
        assert 'azure' in src.lower()

    def test_workload_shock_references_alibaba_in_source(self):
        """Implementation must reference 'alibaba' as the shock target."""
        import inspect
        from powerzoo.data import dc_microgrid_profiles
        src = inspect.getsource(dc_microgrid_profiles)
        assert 'alibaba' in src.lower()

    def test_non_workload_ood_does_not_require_real_data(self):
        """Non-workload OOD scenarios should not check real_data flag."""
        p = make_all_synthetic_profiles()  # real_data=False
        for scenario in ('renewable_drought', 'cooling_stress'):
            out = apply_ood_transform(p, scenario, strict=True)
            assert 'cpu' in out


# ===========================================================================
# T9 — workload OOD strict raises
# ===========================================================================

class TestT9_WorkloadOODStrictRaises:
    def test_workload_swap_strict_raises_on_synthetic(self):
        p = make_all_synthetic_profiles()  # real_data=False
        with pytest.raises(ValueError, match="real workload data"):
            apply_ood_transform(p, 'workload_swap', strict=True)

    def test_workload_shock_strict_raises_on_synthetic(self):
        p = make_all_synthetic_profiles()
        with pytest.raises(ValueError, match="real workload data"):
            apply_ood_transform(p, 'workload_shock', strict=True)

    def test_workload_swap_strict_false_does_not_raise(self):
        """strict=False with real_data=False must not raise."""
        p = make_all_synthetic_profiles()
        out = apply_ood_transform(p, 'workload_swap', strict=False)
        assert 'cpu' in out

    def test_workload_shock_strict_false_does_not_raise(self):
        p = make_all_synthetic_profiles()
        out = apply_ood_transform(p, 'workload_shock', strict=False)
        assert 'cpu' in out

    def test_unknown_scenario_raises_value_error(self):
        p = make_all_synthetic_profiles()
        with pytest.raises(ValueError, match="Unknown OOD scenario"):
            apply_ood_transform(p, 'nonexistent_scenario', strict=False)
