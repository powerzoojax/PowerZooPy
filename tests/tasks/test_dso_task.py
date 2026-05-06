"""Regression tests for powerzoo.tasks.dso_task.

Mirrors the coverage of PowerZooJax tests/grid/test_dso_d3d4.py (D4 section)
so that the two libraries' DSO task surfaces can be verified to be aligned.

Test groups
-----------
TestDSO_Config       — constants, make_dso_env, make_dso_1flex_env
TestDSO_CostWrapper  — task-selected CMDP metadata keeps voltage-only benchmark semantics
TestDSO_Baselines    — no_control / tou / droop, horizon follows env
TestDSO_Metrics      — compute_dso_metrics key names + definitions
TestDSO_1Flex        — 1-device variant end-to-end
"""

import numpy as np
import pytest
from gymnasium import spaces

from powerzoo.tasks.dso_task import (
    DSO_FLEXLOAD_CONFIG,
    DSO_V_MIN,
    DSO_V_MAX,
    DSO_CONSTRAINT_SPEC,
    compute_dso_metrics,
    dso_droop_heuristic_rollout,
    dso_no_control_rollout,
    dso_tou_heuristic_rollout,
    make_dso_1flex_env,
    make_dso_env,
    rollout_dso,
    _env_max_steps,
)
from powerzoo.wrappers.safe_rl_wrapper import GymnasiumSafeWrapper, TaskCMDPWrapper


# ---------------------------------------------------------------------------
# Module-scoped fixtures — one env creation per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env():
    """6-device DSO env (default 48-step episode, 30-min dt)."""
    return make_dso_env()


@pytest.fixture(scope="module")
def env_1flex():
    """1-device DSO variant for action-dim tests."""
    return make_dso_1flex_env()


@pytest.fixture(scope="module")
def baseline_results(env):
    """No-control rollout used as baseline in multiple metric tests."""
    return dso_no_control_rollout(env, seed=0)


@pytest.fixture(scope="module")
def tou_results(env):
    return dso_tou_heuristic_rollout(env, seed=0)


@pytest.fixture(scope="module")
def droop_results(env):
    return dso_droop_heuristic_rollout(env, seed=0)


# ===========================================================================
# TestDSO_Config — constants + env factory
# ===========================================================================

class TestDSO_Config:
    """Constants and environment factory surface."""

    def test_flexload_config_6_devices(self):
        """DSO spec: exactly 6 FlexLoad devices."""
        assert len(DSO_FLEXLOAD_CONFIG) == 6

    def test_flexload_bus_ids(self):
        """FlexLoad buses must be [6, 14, 18, 22, 28, 33] — matches JAX DSO_FLEXLOAD_CONFIG."""
        bus_ids = [cfg["bus_id"] for cfg in DSO_FLEXLOAD_CONFIG]
        assert bus_ids == [6, 14, 18, 22, 28, 33]

    def test_voltage_limits(self):
        """DSO voltage window: v_min=0.94, v_max=1.06."""
        assert DSO_V_MIN == pytest.approx(0.94)
        assert DSO_V_MAX == pytest.approx(1.06)

    def test_make_dso_env_action_space(self, env):
        """action_space = Box(12,) with [0, 1] bounds (unit scale)."""
        sp = env.action_space
        assert isinstance(sp, spaces.Box)
        assert sp.shape == (12,), f"expected (12,), got {sp.shape}"
        assert float(sp.low.min()) >= 0.0
        assert float(sp.high.max()) <= 1.0 + 1e-6

    def test_make_dso_env_obs_space_is_flat_box(self, env):
        """Observation space is a flat Box after FlattenWrapper."""
        assert isinstance(env.observation_space, spaces.Box)
        assert len(env.observation_space.shape) == 1
        assert env.observation_space.shape[0] > 0

    def test_make_dso_env_is_task_cmdp_wrapper(self, env):
        """Outermost wrapper must be the generic TaskCMDPWrapper."""
        assert isinstance(env, TaskCMDPWrapper)

    def test_make_dso_env_max_steps(self, env):
        """Default episode length is 48 steps."""
        assert _env_max_steps(env) == 48

    def test_reward_is_negative(self, env):
        """Reward = -loss_penalty * loss_MW ≤ 0."""
        obs, _ = env.reset(seed=0)
        _, reward, _, _, _ = env.step(env.action_space.sample() * 0)
        assert reward <= 0.0, f"reward should be ≤ 0, got {reward}"

    def test_reward_tracks_loss(self, env):
        """Reward must equal -0.1 * p_loss_MW (default loss_penalty_weight)."""
        obs, _ = env.reset(seed=0)
        _, reward, _, _, info = env.step(np.zeros(12, dtype=np.float32))
        expected = -0.1 * float(info["p_loss_MW"])
        assert abs(reward - expected) < 1e-5, (
            f"reward {reward} != -0.1 * p_loss_MW {expected}")


# ===========================================================================
# TestDSO_CostWrapper — task-selected CMDP alignment
# ===========================================================================

class TestDSO_CostWrapper:
    """The DSO task selects voltage violation without mutating core env semantics."""

    def test_selected_constraint_names_present_after_reset(self, env):
        _, info = env.reset(seed=0)
        assert tuple(info["selected_constraint_names"]) == ("voltage_violation",)
        assert tuple(info["constraint_names"]) == (
            "voltage_violation",
            "thermal_overload",
            "resource",
        )

    def test_selected_cost_equals_voltage_violation_after_step(self, env):
        """Task selection must point to the voltage channel only."""
        env.reset(seed=0)
        for _ in range(5):
            _, _, _, _, info = env.step(np.zeros(12, dtype=np.float32))
            assert info["selected_cost_sum"] == info["cost_voltage_violation"], (
                f"selected_cost_sum {info['selected_cost_sum']} != "
                f"cost_voltage_violation {info['cost_voltage_violation']}"
            )

    def test_selected_cost_vector_shape(self, env):
        """The selected DSO CMDP vector is 1-D while full vector keeps all channels."""
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(12, dtype=np.float32))
        assert info["selected_constraint_costs"].shape == (1,)
        assert info["constraint_costs"].shape == (3,)

    def test_cost_sum_still_present(self, env):
        """Full env aggregate cost survives task-level CMDP selection."""
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(12, dtype=np.float32))
        assert "cost_sum" in info
        assert "cost_voltage_violation" in info
        assert "cost_thermal_overload" in info

    def test_cost_nonnegative(self, env):
        """Voltage violation count is always ≥ 0."""
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(12, dtype=np.float32))
        assert info["selected_cost_sum"] >= 0.0

    def test_scalar_safe_wrapper_still_exposes_info_cost(self, env):
        """Compatibility wrappers project the selected vector back to scalar info['cost']."""
        safe_env = GymnasiumSafeWrapper(env, cost_threshold=DSO_CONSTRAINT_SPEC.scalar_threshold)
        safe_env.reset(seed=0)
        _, _, _, _, info = safe_env.step(np.zeros(12, dtype=np.float32))
        assert info["cost"] == info["cost_voltage_violation"]


# ===========================================================================
# TestDSO_Baselines — no_control / tou / droop
# ===========================================================================

class TestDSO_Baselines:
    """Baseline rollout helpers: completeness, physics, horizon generality.

    Mirrors PowerZooJax TestDSO_D4_Baselines.
    """

    def test_no_control_completes_full_episode(self, baseline_results):
        """No-control rollout must produce exactly max_steps steps."""
        assert len(baseline_results["rewards"]) == 48
        assert len(baseline_results["losses"]) == 48

    def test_no_control_zero_curtailment(self, baseline_results):
        """No-control: all curtailed and shift-out values must be zero."""
        assert all(abs(c) < 1e-6 for c in baseline_results["curtailed"])
        assert all(abs(s) < 1e-6 for s in baseline_results["shifted"])

    def test_no_control_positive_losses(self, baseline_results):
        """Network losses must be positive even with no control."""
        assert all(l > 0 for l in baseline_results["losses"])

    def test_tou_completes_full_episode(self, tou_results):
        """TOU rollout must complete 48 steps."""
        assert len(tou_results["rewards"]) == 48

    def test_tou_has_nonzero_curtailment(self, tou_results):
        """TOU should curtail during peak hours → total_curtail > 0."""
        assert sum(tou_results["curtailed"]) > 0, (
            "TOU heuristic should curtail during peak hours")

    def test_tou_has_nonzero_shift(self, tou_results):
        """TOU should shift out during peak hours → total_shift > 0."""
        assert sum(tou_results["shifted"]) > 0, (
            "TOU heuristic should shift out during peak hours")

    def test_droop_completes_full_episode(self, droop_results):
        """Droop rollout must complete 48 steps."""
        assert len(droop_results["rewards"]) == 48
        assert len(droop_results["losses"]) == 48

    def test_horizon_follows_env_max_steps(self):
        """rollout_dso n_steps=None → episode length from env.max_steps_per_episode.

        Finding 3 fix: horizon is not hardcoded to 48.
        """
        env24 = make_dso_env(max_steps=24)
        assert _env_max_steps(env24) == 24

        r_no = dso_no_control_rollout(env24, seed=0)
        r_tou = dso_tou_heuristic_rollout(env24, seed=0)
        r_droop = dso_droop_heuristic_rollout(env24, seed=0)

        assert len(r_no["rewards"]) == 24, (
            f"no_control: expected 24 steps, got {len(r_no['rewards'])}")
        assert len(r_tou["rewards"]) == 24, (
            f"tou: expected 24 steps, got {len(r_tou['rewards'])}")
        assert len(r_droop["rewards"]) == 24, (
            f"droop: expected 24 steps, got {len(r_droop['rewards'])}")

    def test_tou_steps_per_day_not_hardcoded(self):
        """TOU % operator uses steps_per_day from env, not literal 48.

        Finding 3 fix: create a 15-min-step env (96 steps/day, 24-step episode).
        TOU should still complete without IndexError or wrong modulo.
        """
        env15 = make_dso_env(delta_t_minutes=15.0, max_steps=24)
        r = dso_tou_heuristic_rollout(env15, seed=0)
        assert len(r["rewards"]) == 24

    def test_rollout_explicit_n_steps_override(self, env):
        """Explicit n_steps= argument overrides env max_steps."""
        r = rollout_dso(env, lambda obs: np.zeros(12, dtype=np.float32),
                        n_steps=10, seed=0)
        assert len(r["rewards"]) == 10


# ===========================================================================
# TestDSO_Metrics — compute_dso_metrics
# ===========================================================================

class TestDSO_Metrics:
    """Scalar metric computation.

    Mirrors PowerZooJax TestDSO_D4_Metrics.
    Key alignment: same 10 key names, served_flex_ratio = shift_in / shift_out.
    """

    _EXPECTED_KEYS = {
        "total_reward", "total_loss_mwh", "mean_loss_mw",
        "total_violations", "total_curtailed_mwh",
        "total_shifted_mwh", "total_shift_in_mwh",
        "served_flex_ratio", "network_loss_reduction_pct",
        "peak_shaving_pct",
    }

    def test_metrics_structure(self, baseline_results):
        """compute_dso_metrics returns exactly the 10 expected keys."""
        metrics = compute_dso_metrics(baseline_results)
        assert set(metrics.keys()) == self._EXPECTED_KEYS

    def test_no_control_values(self, baseline_results):
        """No-control: positive network loss, zero curtailment/shift energy."""
        metrics = compute_dso_metrics(baseline_results)
        assert metrics["total_loss_mwh"] > 0
        assert metrics["mean_loss_mw"] > 0
        np.testing.assert_allclose(metrics["total_curtailed_mwh"], 0.0, atol=1e-6)
        np.testing.assert_allclose(metrics["total_shifted_mwh"], 0.0, atol=1e-6)

    def test_served_flex_ratio_is_shift_in_over_shift_out(self):
        """served_flex_ratio = shift_in / shift_out (buffer clearance).

        Finding 2 fix: old Python code used curtailed/(curtailed+shifted).
        JAX uses shift_in.sum() / max(shifted.sum(), 1e-8).
        """
        fake = {
            "rewards": [0.0] * 5,
            "losses": [0.1] * 5,
            "violations": [0] * 5,
            "curtailed": [1.0] * 5,   # 5 MW total curtailed
            "shifted": [2.0] * 5,     # 10 MW total shifted out
            "shift_in": [1.6] * 5,    # 8 MW came back in → ratio = 8/10 = 0.8
        }
        metrics = compute_dso_metrics(fake)
        np.testing.assert_allclose(
            metrics["served_flex_ratio"], 0.8, atol=1e-9,
            err_msg="served_flex_ratio must be shift_in / shift_out = 8/10 = 0.8")

    def test_served_flex_ratio_zero_when_no_shift(self):
        """served_flex_ratio = 0 when shift_out = 0 (no division by zero)."""
        fake = {
            "rewards": [0.0],
            "losses": [0.1],
            "violations": [0],
            "curtailed": [1.0],
            "shifted": [0.0],
            "shift_in": [0.0],
        }
        metrics = compute_dso_metrics(fake)
        assert metrics["served_flex_ratio"] == 0.0

    def test_relative_metrics_present_with_baseline(self, baseline_results, tou_results):
        """network_loss_reduction_pct and peak_shaving_pct require baseline."""
        # Without baseline: both None
        metrics_no_base = compute_dso_metrics(tou_results)
        assert metrics_no_base["network_loss_reduction_pct"] is None
        assert metrics_no_base["peak_shaving_pct"] is None

        # With baseline: both numeric
        metrics = compute_dso_metrics(tou_results, baseline_results=baseline_results)
        assert metrics["network_loss_reduction_pct"] is not None
        assert metrics["peak_shaving_pct"] is not None
        assert isinstance(metrics["network_loss_reduction_pct"], float)

    def test_self_comparison_zero_reduction(self, baseline_results):
        """Comparing no-control to itself → 0% network loss reduction.

        Mirrors JAX TestDSO_D4_Metrics.test_no_control_self_comparison.
        """
        metrics = compute_dso_metrics(
            baseline_results, baseline_results=baseline_results)
        np.testing.assert_allclose(
            metrics["network_loss_reduction_pct"], 0.0, atol=1e-4)

    def test_tou_curtailment_mwh_positive(self, tou_results):
        """TOU metrics must show non-zero curtailed energy."""
        metrics = compute_dso_metrics(tou_results)
        assert metrics["total_curtailed_mwh"] > 0

    def test_dt_hours_scales_energy(self):
        """dt_hours parameter must correctly scale MWh outputs."""
        fake = {
            "rewards": [0.0],
            "losses": [1.0],
            "violations": [0],
            "curtailed": [2.0],
            "shifted": [0.0],
            "shift_in": [0.0],
        }
        m_30min = compute_dso_metrics(fake, dt_hours=0.5)
        m_15min = compute_dso_metrics(fake, dt_hours=0.25)
        np.testing.assert_allclose(
            m_30min["total_loss_mwh"], 0.5, atol=1e-9)
        np.testing.assert_allclose(
            m_15min["total_loss_mwh"], 0.25, atol=1e-9)


# ===========================================================================
# TestDSO_1Flex — 1-device variant
# ===========================================================================

class TestDSO_1Flex:
    """End-to-end tests for the 1-flex DSO variant.

    Mirrors PowerZooJax TestDSO_1Flex.
    Action space must be Box(2,); all baselines must work.
    """

    def test_1flex_action_space(self, env_1flex):
        """1-flex: action_space = Box(2,)."""
        assert env_1flex.action_space.shape == (2,), (
            f"expected (2,), got {env_1flex.action_space.shape}")

    def test_1flex_is_task_cmdp_wrapper(self, env_1flex):
        """1-flex env must also be wrapped with TaskCMDPWrapper."""
        assert isinstance(env_1flex, TaskCMDPWrapper)

    def test_1flex_selected_cost_equals_voltage_violation(self, env_1flex):
        """1-flex env keeps the same voltage-only benchmark selection."""
        env_1flex.reset(seed=0)
        _, _, _, _, info = env_1flex.step(np.zeros(2, dtype=np.float32))
        assert info["selected_cost_sum"] == info["cost_voltage_violation"]

    def test_1flex_no_control_rollout(self, env_1flex):
        """no_control baseline works with 1-flex (action_dim=2)."""
        r = dso_no_control_rollout(env_1flex, seed=0)
        assert len(r["rewards"]) == 48
        assert all(abs(c) < 1e-6 for c in r["curtailed"])

    def test_1flex_tou_rollout(self, env_1flex):
        """TOU baseline works with 1-flex and curtails during peak."""
        r = dso_tou_heuristic_rollout(env_1flex, seed=0)
        assert len(r["rewards"]) == 48
        assert sum(r["curtailed"]) > 0

    def test_1flex_droop_rollout(self, env_1flex):
        """Droop baseline works with 1-flex without error."""
        r = dso_droop_heuristic_rollout(env_1flex, seed=0)
        assert len(r["rewards"]) == 48

    def test_1flex_metrics(self, env_1flex):
        """compute_dso_metrics works for 1-flex results."""
        baseline = dso_no_control_rollout(env_1flex, seed=0)
        tou = dso_tou_heuristic_rollout(env_1flex, seed=0)
        metrics = compute_dso_metrics(tou, baseline_results=baseline)
        assert set(metrics.keys()) == {
            "total_reward", "total_loss_mwh", "mean_loss_mw",
            "total_violations", "total_curtailed_mwh",
            "total_shifted_mwh", "total_shift_in_mwh",
            "served_flex_ratio", "network_loss_reduction_pct",
            "peak_shaving_pct",
        }

    def test_1flex_vs_6flex_independent(self, env, env_1flex):
        """1-flex and 6-flex envs are independent; both produce full episodes."""
        r6 = dso_no_control_rollout(env, seed=0)
        r1 = dso_no_control_rollout(env_1flex, seed=0)
        assert len(r6["rewards"]) == 48
        assert len(r1["rewards"]) == 48
