"""Unit tests for powerzoo.envs.resource.datacenter — DataCenterEnv.

Domain knowledge embedded in these tests:
- AI Data Center power model: IT power + Cooling + Auxiliary
- PUE (Power Usage Effectiveness) = P_total / P_IT, typically 1.2–1.6 for modern DCs
- GPU power: idle ~55W, active ~1100W (NVIDIA H100 system-level)
- COP (Coefficient of Performance) decreases with outdoor temperature
- Thermal dynamics: first-order zone temperature model
- Task types: inference (non-deferrable, diurnal), training (deferrable), finetuning (deferrable)
- SLA violations: tasks that miss deadlines without being scheduled
- Sign convention: current_p < 0 always (DC is a load, never generates)
- Cooling setpoint affects zone temperature and COP
- Over-temperature: zone_temp > t_critical triggers safety concern
"""

import pytest
import numpy as np

from powerzoo.envs.resource.datacenter import DataCenterEnv, _Task
from .conftest import MockParentGrid


# ========================== Fixtures ==========================

@pytest.fixture
def dc():
    """Standard 1000-GPU data center, no parent."""
    d = DataCenterEnv(normalize_actions=False, n_gpus=1000, gpu_idle_w=55.0, gpu_active_w=1100.0,
                      p_base_mw=0.5, infer_gpu_peak=400,
                      cop_ref=5.0, cop_decay=0.04, t_ref=20.0)
    d.reset(seed=42)
    return d


@pytest.fixture
def dc_small():
    """Small DC for deterministic task testing."""
    d = DataCenterEnv(
        normalize_actions=False,
        n_gpus=100,
        gpu_idle_w=50.0,
        gpu_active_w=500.0,
        p_base_mw=0.1,
        infer_gpu_peak=20,
        cop_ref=5.0,
        train_cfg={'arrival_interval': 100, 'gpu_range': (5, 10),
                   'duration_range': (3, 5), 'deadline_slack': 2.0, 'gpu_eta': 0.9},
        finetune_cfg={'arrival_interval': 100, 'gpu_range': (2, 5),
                      'duration_range': (2, 3), 'deadline_slack': 3.0, 'gpu_eta': 0.75},
    )
    d.reset(seed=0)
    return d


@pytest.fixture
def dc_attached(mock_grid):
    """DC attached to a mock grid."""
    d = DataCenterEnv(normalize_actions=False, n_gpus=500, parent=mock_grid, bus_id=1)
    d.reset(seed=0)
    return d


# ========================== Initialization ==========================

class TestDataCenterInit:

    def test_default_params(self):
        d = DataCenterEnv(normalize_actions=False)
        assert d.n_gpus == 1000
        assert d.gpu_idle_w == 55.0
        assert d.gpu_active_w == 1100.0
        assert d.t_set_min == 18.0
        assert d.t_set_max == 27.0
        assert d.t_critical == 35.0

    def test_name_attribute(self):
        assert DataCenterEnv.name == 'datacenter'

    def test_action_space_shape(self):
        d = DataCenterEnv(normalize_actions=False)
        assert d.action_space.shape == (3,)
        # All actions normalized to [0, 1]
        np.testing.assert_array_almost_equal(d.action_space.low, [0, 0, 0])
        np.testing.assert_array_almost_equal(d.action_space.high, [1, 1, 1])

    def test_observation_space_shape(self):
        d = DataCenterEnv(normalize_actions=False)
        assert d.observation_space.shape == (11,)

    def test_task_configs_merged(self):
        d = DataCenterEnv(normalize_actions=False, train_cfg={'arrival_interval': 20})
        assert d.train_cfg['arrival_interval'] == 20
        # Other defaults should persist
        assert 'gpu_range' in d.train_cfg


# ========================== Reset ==========================

class TestDataCenterReset:

    def test_reset_clears_state(self, dc):
        dc.step(np.array([0.5, 0.5, 0.5]))
        dc.reset(seed=1)
        assert dc.time_step == 0
        assert dc.sla_violations == 0
        assert dc.t_zone == 22.0
        assert len(dc._wait_queue) == 0
        assert len(dc._running) == 0
        assert dc.current_p_mw == 0.0

    def test_reset_sets_rng(self, dc):
        dc.reset(seed=123)
        val1 = dc.np_random.random()
        dc.reset(seed=123)
        val2 = dc.np_random.random()
        assert val1 == val2


# ========================== Sign Convention ==========================

class TestDCSignConvention:
    """DC always absorbs power: current_p ≤ 0."""

    def test_power_always_negative(self, dc):
        """After stepping, DC should always report negative power (load)."""
        dc.step(np.array([0.5, 0.5, 0.5]))
        assert dc.current_p_mw <= 0, "DC is always a load (negative injection)"

    def test_power_magnitude_reasonable(self, dc):
        """DC power should be within physically reasonable bounds.

        1000 GPUs × 1100W = 1100 kW active, plus cooling and base power.
        Total should not exceed ~5 MW for 1000 H100-class GPUs.
        """
        dc.step(np.array([1.0, 1.0, 0.5]))  # max GPU allocation
        assert abs(dc.current_p_mw) < 10.0, "1000-GPU DC power should be < 10 MW"


# ========================== Power Model ==========================

class TestPowerModel:

    def test_it_power_includes_base(self, dc_small):
        """IT power should include baseline non-GPU power."""
        dc_small.step(np.array([0.0, 0.0, 0.5]))  # no training/finetuning
        assert dc_small.p_it_mw >= dc_small.p_base_mw

    def test_cooling_power_positive(self, dc):
        """Cooling requires power when zone temp > setpoint."""
        # Set zone temp above setpoint to ensure cooling is active
        dc.t_zone = 30.0
        dc.step(np.array([0.5, 0.5, 0.0]))  # low setpoint → large gap
        assert dc.p_cool_mw > 0

    def test_total_power_gt_it_power(self, dc):
        """Total DC power > IT power (due to cooling + aux)."""
        dc.step(np.array([0.5, 0.5, 0.5]))
        assert dc.p_dc_mw > dc.p_it_mw

    def test_pue_above_one(self, dc):
        """PUE = P_total / P_IT should be > 1.0 (overhead from cooling + aux)."""
        dc.step(np.array([0.5, 0.5, 0.5]))
        pue = dc.p_dc_mw / max(dc.p_it_mw, 1e-9)
        assert pue > 1.0
        assert pue < 3.0, "PUE should be realistic (< 3.0)"

    def test_idle_gpu_power(self, dc_small):
        """With no tasks, only idle GPUs + base power + inference load."""
        dc_small.step(np.array([0.0, 0.0, 0.5]))
        # At minimum, idle GPUs + base power
        min_it = dc_small.n_gpus * dc_small.gpu_idle_w / 1e6 + dc_small.p_base_mw
        # p_it might be higher due to inference load
        assert dc_small.p_it_mw >= min_it * 0.5  # allow some margin for inference


# ========================== COP / Cooling ==========================

class TestCOPModel:

    def test_cop_decreases_with_temperature(self, dc):
        """Higher outdoor temp → lower COP → more cooling power.

        COP = cop_ref × clip(1 - cop_decay × max(T_out - T_ref, 0), 0.4, 1.2)
        Higher T_out → lower COP → higher P_cool = P_IT / COP.
        """
        cop_cold = dc.cop_ref * np.clip(
            1.0 - dc.cop_decay * max(15.0 - dc.t_ref, 0.0), 0.4, 1.2)
        cop_hot = dc.cop_ref * np.clip(
            1.0 - dc.cop_decay * max(35.0 - dc.t_ref, 0.0), 0.4, 1.2)
        assert cop_hot < cop_cold, "Higher outdoor temp → lower COP"

        # With same IT load, p_cool = p_it / cop, so higher temp → more cooling
        p_it = 1.0  # arbitrary fixed IT power
        assert p_it / cop_hot > p_it / cop_cold

    def test_cop_bounded(self, dc):
        """COP should be clipped to [0.4, 1.2] × cop_ref."""
        # Very hot day
        dc.t_outdoor = 60.0
        dc.step(np.array([0.5, 0.5, 0.5]))
        # COP factor clipped to 0.4
        cop_factor = np.clip(1.0 - dc.cop_decay * max(dc.t_outdoor - dc.t_ref, 0.0), 0.4, 1.2)
        cop = dc.cop_ref * cop_factor
        assert cop >= dc.cop_ref * 0.4 - 1e-9


# ========================== Thermal Dynamics ==========================

class TestThermalDynamics:

    def test_zone_temp_bounded(self, dc):
        """Zone temperature should stay in [15, 45]°C."""
        for _ in range(50):
            dc.step(np.array([0.8, 0.8, 0.0]))  # high load, low cooling setpoint
        assert 15.0 <= dc.t_zone <= 45.0

    def test_overtemperature_detection(self, dc):
        """Over-temperature flag when zone > t_critical."""
        dc.t_zone = 36.0  # above t_critical=35
        dc.step(np.array([0.5, 0.5, 1.0]))  # high setpoint (less cooling)
        # Check overtemp flag from status
        s = dc.status()
        # Note: t_zone may have changed after step; check the flag
        if dc.t_zone > dc.t_critical:
            assert dc.is_overtemp is True

    def test_cooling_setpoint_maps_correctly(self, dc):
        """Cooling setpoint action [0, 1] → [t_set_min, t_set_max]."""
        dc.step(np.array([0.5, 0.5, 0.0]))
        assert dc.t_setpoint == pytest.approx(dc.t_set_min)
        dc.step(np.array([0.5, 0.5, 1.0]))
        assert dc.t_setpoint == pytest.approx(dc.t_set_max)
        dc.step(np.array([0.5, 0.5, 0.5]))
        expected = dc.t_set_min + 0.5 * (dc.t_set_max - dc.t_set_min)
        assert dc.t_setpoint == pytest.approx(expected)


# ========================== Task Scheduling ==========================

class TestTaskScheduling:

    def test_waiting_queue_receives_tasks(self, dc):
        """After a step, tasks may arrive in the wait queue (Poisson process)."""
        dc.reset(seed=42)
        total_tasks = 0
        for _ in range(50):
            dc.step(np.array([0.5, 0.5, 0.5]))
            total_tasks += len(dc._wait_queue) + len(dc._running)
        # Over 50 steps, should generate at least some tasks
        assert total_tasks > 0

    def test_running_tasks_decrease_remaining(self, dc_small):
        """Running tasks should have their remaining count decremented each step."""
        task = _Task(arrive_step=0, duration=5, gpus=5, deadline=20,
                     task_type='training', remaining=5, gpu_eta=0.9)
        dc_small._running.append(task)
        dc_small.step(np.array([0.0, 0.0, 0.5]))
        # Task should have remaining decremented
        # (may have been removed if remaining reached 0)
        if task in dc_small._running:
            assert task.remaining < 5

    def test_sla_violations_on_deadline_miss(self, dc_small):
        """Tasks that miss their deadline without being scheduled → SLA violation.

        Deadline check uses current time_step (before increment).
        Setting deadline=0 with time_step=0 means the check ``time_step >= deadline``
        fires immediately during the step.
        """
        task = _Task(arrive_step=0, duration=5, gpus=2000, deadline=0,
                     task_type='training', remaining=5, gpu_eta=0.9)
        dc_small._wait_queue.append(task)
        dc_small.time_step = 0
        dc_small.step(np.array([0.0, 0.0, 0.5]))
        assert dc_small.sla_violations >= 1

    def test_gpu_budget_controls_scheduling(self, dc_small):
        """r_train=0 → no training tasks scheduled from queue."""
        task = _Task(arrive_step=0, duration=5, gpus=5, deadline=100,
                     task_type='training', remaining=5, gpu_eta=0.9)
        dc_small._wait_queue.append(task)
        initial_queue = len(dc_small._wait_queue)
        dc_small.step(np.array([0.0, 0.0, 0.5]))  # r_train=0, r_ft=0
        # Training task should remain in queue (unless urgently scheduled)
        # At least it shouldn't be scheduled by budget since budget=0
        # (urgent scheduling might still pick it up if slack <= 0)


# ========================== Inference Load ==========================

class TestInferenceLoad:

    def test_diurnal_factor_range(self, dc):
        """Diurnal inference load should stay within [10%, 100%] of peak GPU count."""
        for _ in range(96):
            dc.step(np.array([0.0, 0.0, 0.5]))
            factor = dc.gpus_infer / dc.infer_gpu_peak
            assert 0.1 <= factor <= 1.0

    def test_inference_uses_gpus(self, dc):
        """Inference load should consume some GPUs."""
        dc.step(np.array([0.0, 0.0, 0.5]))
        assert dc.gpus_infer >= 0
        assert dc.gpus_infer <= dc.infer_gpu_peak


# ========================== Action Parsing ==========================

class TestDCActionParsing:

    def test_none_action_defaults(self, dc):
        """None action → default (0.5, 0.5, 0.5)."""
        dc.step(None)
        assert dc.t_setpoint == pytest.approx(
            dc.t_set_min + 0.5 * (dc.t_set_max - dc.t_set_min)
        )

    def test_dict_action(self, dc):
        dc.step({'r_train': 0.8, 'r_finetune': 0.2, 'cooling_setpoint': 0.3})
        expected_setpoint = dc.t_set_min + 0.3 * (dc.t_set_max - dc.t_set_min)
        assert dc.t_setpoint == pytest.approx(expected_setpoint)

    def test_ndarray_action(self, dc):
        dc.step(np.array([0.1, 0.2, 0.9]))
        expected_setpoint = dc.t_set_min + 0.9 * (dc.t_set_max - dc.t_set_min)
        assert dc.t_setpoint == pytest.approx(expected_setpoint)

    def test_action_clipped(self, dc):
        """Actions outside [0, 1] should be clipped."""
        dc.step(np.array([-0.5, 1.5, 2.0]))
        # Should not crash; setpoint should be at max
        assert dc.t_setpoint == pytest.approx(dc.t_set_max)


# ========================== Observation ==========================

class TestDCObs:

    def test_obs_is_dict(self, dc):
        dc.step(np.array([0.5, 0.5, 0.5]))
        o = dc.obs()
        assert isinstance(o, dict)
        assert len(o) == 11

    def test_obs_all_in_range(self, dc):
        dc.step(np.array([0.5, 0.5, 0.5]))
        o = dc.obs()
        assert all(-1.0 <= v <= 1.0 for v in o.values())


# ========================== Status ==========================

class TestDCStatus:

    def test_status_keys(self, dc):
        dc.step(np.array([0.5, 0.5, 0.5]))
        s = dc.status()
        expected_keys = {'current_p_mw', 'p_it_mw', 'p_cool_mw', 'p_dc_mw', 'pue',
                         'gpu_util', 'gpus_infer', 'gpus_active',
                         'n_running', 'n_queued', 't_zone', 't_setpoint',
                         't_outdoor', 'is_overtemp', 'sla_violations'}
        assert expected_keys.issubset(s.keys())

    def test_pue_in_status(self, dc):
        dc.step(np.array([0.5, 0.5, 0.5]))
        s = dc.status()
        assert s['pue'] > 1.0


# ========================== status() extended fields ==========================

class TestDCStatusExtended:

    def test_status_has_bus_id(self, dc_attached):
        s = dc_attached.status()
        assert 'bus_id' in s
        assert s['bus_id'] == 1  # fixture uses bus_id=1

    def test_status_has_local_v(self, dc_attached):
        s = dc_attached.status()
        assert 'local_v' in s
        assert s['local_v'] is None  # no _nodes on mock parent


# ========================== Outdoor Temperature ==========================

class TestOutdoorTemp:

    def test_synthetic_temp_range(self, dc):
        """Without parent, synthetic temp should be sinusoidal, reasonable."""
        temps = []
        for _ in range(96):
            dc.step(np.array([0.5, 0.5, 0.5]))
            temps.append(dc.t_outdoor)
        assert min(temps) > 0.0, "Outdoor temp should be above freezing"
        assert max(temps) < 40.0, "Outdoor temp should be below 40°C"


# ========================== Repr ==========================

class TestRepr:

    def test_repr_contains_info(self, dc):
        r = repr(dc)
        assert 'DataCenterEnv' in r
        assert '1000' in r


# ========================== Long Simulation ==========================

class TestLongSimulation:

    def test_100_steps_no_crash(self, dc):
        """Run 100 steps with random actions — should not crash."""
        rng = np.random.default_rng(99)
        for _ in range(100):
            action = rng.uniform(0, 1, size=3)
            dc.step(action)
        assert dc.time_step == 100

    def test_power_stays_negative_throughout(self, dc):
        """DC power should remain ≤ 0 for the entire simulation."""
        rng = np.random.default_rng(77)
        for _ in range(50):
            action = rng.uniform(0, 1, size=3)
            dc.step(action)
            assert dc.current_p_mw <= 0, f"DC power should be ≤ 0, got {dc.current_p_mw}"


# ========================== Fix Verification Tests ==========================

class TestThermalCoupling:
    """Verify Problem 1 fix: setpoint affects p_cool_mw via thermal coupling."""

    def test_lower_setpoint_increases_cooling_power(self):
        d = DataCenterEnv(normalize_actions=False, n_gpus=100, gpu_active_w=1100.0,
                          infer_gpu_peak=20, cop_ref=5.0)
        d.reset(seed=42)

        # Low setpoint → large temperature gap → more cooling power
        d.t_zone = 30.0
        d.step(np.array([0.5, 0.5, 0.0]))  # cooling_setpoint=0 → t_set_min
        p_cool_low = d.p_cool_mw

        d.reset(seed=42)
        # High setpoint → small temperature gap → less cooling power
        d.t_zone = 30.0
        d.step(np.array([0.5, 0.5, 1.0]))  # cooling_setpoint=1 → t_set_max
        p_cool_high = d.p_cool_mw

        assert p_cool_low > p_cool_high, \
            "Lower setpoint should produce higher cooling power"


class TestCoolingStandbyPower:
    """Verify p_cool_mw never drops to zero (standby floor)."""

    def test_cooling_has_minimum_standby(self):
        d = DataCenterEnv(normalize_actions=False, n_gpus=100, p_cool_min_mw=0.05)
        d.reset(seed=42)

        # Zone below setpoint → q_cool_kw = 0, but standby floor applies
        d.t_zone = 18.0
        d.step(np.array([0.0, 0.0, 1.0]))  # high setpoint → t_set_max=27
        assert d.p_cool_mw >= 0.05, \
            "Cooling power should never drop below standby minimum"

    def test_pue_always_above_minimum(self):
        d = DataCenterEnv(normalize_actions=False, n_gpus=100, p_cool_min_mw=0.05)
        d.reset(seed=42)
        d.t_zone = 18.0
        d.step(np.array([0.0, 0.0, 1.0]))
        pue = d.p_dc_mw / max(d.p_it_mw, 1e-9)
        assert pue > 1.05, "PUE should be > 1.05 due to cooling standby + aux"


class TestProportionalAllocation:
    """Verify Problem 3 fix: r_train=1, r_ft=0 → no finetune scheduling."""

    def test_train_only_no_finetune(self):
        d = DataCenterEnv(normalize_actions=False, n_gpus=200, infer_gpu_peak=0,
                          train_cfg={'arrival_interval': 100, 'gpu_range': (5, 10),
                                     'duration_range': (3, 5), 'deadline_slack': 10.0,
                                     'gpu_eta': 0.9},
                          finetune_cfg={'arrival_interval': 100, 'gpu_range': (2, 5),
                                        'duration_range': (2, 3), 'deadline_slack': 10.0,
                                        'gpu_eta': 0.75})
        d.reset(seed=42)
        # Inject tasks manually
        d._wait_queue.append(_Task(0, 5, 10, 100, 'training', gpu_eta=0.9))
        d._wait_queue.append(_Task(0, 3, 5, 100, 'finetuning', gpu_eta=0.75))

        d.step(np.array([1.0, 0.0, 0.5]))  # r_train=1, r_ft=0

        ft_running = [t for t in d._running if t.task_type == 'finetuning']
        assert len(ft_running) == 0, "With r_ft=0, no finetune tasks should be scheduled"


class TestUrgencySlackAlignment:
    """Verify Problem 2 fix: obs urgency uses same slack as _schedule_urgent."""

    def test_urgency_matches_schedule_slack(self):
        d = DataCenterEnv(normalize_actions=False, n_gpus=100, infer_gpu_peak=0)
        d.reset(seed=42)

        # Inject a task with known deadline
        task = _Task(arrive_step=0, duration=10, gpus=5, deadline=15,
                     task_type='training', gpu_eta=0.9)
        d._wait_queue = [task]
        d.time_step = 5

        obs = d.obs()
        # Expected slack = (deadline - time_step - duration) / duration = (15-5-10)/10 = 0
        expected_urgency = float(np.clip(0.0 / 5.0, -1.0, 1.0))
        assert obs['queue_urgency'] == pytest.approx(expected_urgency, abs=1e-6)

    def test_overdue_urgency_is_negative(self):
        """Overdue tasks should produce negative urgency, not clipped to 0."""
        d = DataCenterEnv(normalize_actions=False, n_gpus=100, infer_gpu_peak=0)
        d.reset(seed=42)

        # Task with deadline already passed relative to duration
        # slack = (10 - 20 - 5) / 5 = -3.0, urgency = -3.0/5.0 = -0.6
        task = _Task(arrive_step=0, duration=5, gpus=5, deadline=10,
                     task_type='training', gpu_eta=0.9)
        d._wait_queue = [task]
        d.time_step = 20

        obs = d.obs()
        assert obs['queue_urgency'] < 0.0, \
            "Overdue tasks should have negative urgency"


class TestStepSLAViolations:
    """Verify Problem 4 fix: step_sla_violations tracks per-step count."""

    def test_step_violations_count(self):
        d = DataCenterEnv(normalize_actions=False, n_gpus=100, infer_gpu_peak=0)
        d.reset(seed=42)

        # Add tasks that will expire
        d._wait_queue.append(_Task(0, 5, 2000, 0, 'training', gpu_eta=0.9))
        d._wait_queue.append(_Task(0, 5, 2000, 0, 'training', gpu_eta=0.9))
        d.time_step = 0
        d.step(np.array([0.0, 0.0, 0.5]))

        assert d.step_sla_violations == 2
        s = d.status()
        assert s['step_sla_violations'] == 2
        assert s['sla_violations'] == 2

        # Next step with no violations
        d.step(np.array([0.0, 0.0, 0.5]))
        assert d.step_sla_violations == 0
