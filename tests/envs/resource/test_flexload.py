"""Unit tests for powerzoo.envs.resource.flexload — FlexLoad (Demand Response).

Domain knowledge embedded in these tests:
- Demand response: controllable loads that can curtail or shift consumption
- Curtailment (action[0]): permanent load reduction (interruptible industrial load)
- Demand shifting (action[1]): defer consumption to future steps (smart appliances, HVAC)
- Two independent actions: curtailment and shift-out (2D action space)
- Deferred demand is released uniformly over shift_horizon future steps
- current_p > 0 = injection convention (reduces net load at bus)
- Energy conservation: shifted energy must be consumed later (not lost)
- Buffer overflow: consecutive shifts accumulate honestly, tracked for CMDP cost
- Complementarity: simultaneous curtail + shift penalised via cost_simultaneous
"""

import pytest
import numpy as np

from powerzoo.envs.resource.flexload import FlexLoad


# ========================== Fixtures ==========================

@pytest.fixture
def fl():
    """Standard FlexLoad: 10 MW curtail, 10 MW shift, 4-step horizon, physical actions."""
    f = FlexLoad(action_scale='physical', curtail_cap_mw=10.0, shift_cap_mw=10.0,
                 shift_horizon=4, baseline_mw=50.0)
    f.reset(seed=42)
    return f


@pytest.fixture
def fl_unit():
    """FlexLoad with unit-scaled actions [0, 1]."""
    f = FlexLoad(action_scale='unit', curtail_cap_mw=10.0, shift_cap_mw=10.0,
                 shift_horizon=4, baseline_mw=50.0)
    f.reset(seed=42)
    return f


@pytest.fixture
def fl_tanh():
    """FlexLoad with tanh-scaled actions [-1, 1]."""
    f = FlexLoad(action_scale='tanh', curtail_cap_mw=10.0, shift_cap_mw=10.0,
                 shift_horizon=4, baseline_mw=50.0)
    f.reset(seed=42)
    return f


# ========================== Initialization ==========================

class TestFlexLoadInit:

    def test_default_params(self):
        f = FlexLoad(action_scale='physical')
        assert f.curtail_cap_mw == 10.0
        assert f.shift_cap_mw == 10.0
        assert f.shift_horizon == 4
        assert f.baseline_mw == 50.0
        assert f.curtail_cost_per_mwh == 50.0
        assert f.shift_cost_per_mwh == 10.0
        assert f.complementarity_penalty == 100.0

    def test_custom_params(self):
        f = FlexLoad(action_scale='physical', curtail_cap_mw=20.0, shift_cap_mw=15.0,
                     shift_horizon=8, baseline_mw=100.0)
        assert f.curtail_cap_mw == 20.0
        assert f.shift_cap_mw == 15.0
        assert f.shift_horizon == 8

    def test_baseline_mw_zero_clipped(self):
        """baseline_mw=0 should be handled (set to 1.0)."""
        f = FlexLoad(action_scale='physical', baseline_mw=0.0)
        assert f.baseline_mw == 1.0

    def test_name_attribute(self):
        assert FlexLoad.name == 'flexload'

    def test_action_space_physical(self):
        f = FlexLoad(action_scale='physical', curtail_cap_mw=10.0, shift_cap_mw=15.0)
        assert f.action_space.shape == (2,)
        assert f.action_space.low[0] == pytest.approx(0.0)
        assert f.action_space.high[0] == pytest.approx(10.0)
        assert f.action_space.high[1] == pytest.approx(15.0)

    def test_action_space_unit(self):
        f = FlexLoad(action_scale='unit')
        assert f.action_space.shape == (2,)
        np.testing.assert_array_equal(f.action_space.low, [0.0, 0.0])
        np.testing.assert_array_equal(f.action_space.high, [1.0, 1.0])

    def test_action_space_tanh(self):
        f = FlexLoad(action_scale='tanh')
        assert f.action_space.shape == (2,)
        np.testing.assert_array_equal(f.action_space.low, [-1.0, -1.0])
        np.testing.assert_array_equal(f.action_space.high, [1.0, 1.0])

    def test_observation_space_shape(self):
        f = FlexLoad(action_scale='physical')
        assert f.observation_space.shape == (8,)

    def test_invalid_action_scale(self):
        with pytest.raises(ValueError, match="action_scale"):
            FlexLoad(action_scale='invalid')

    def test_normalize_actions_compat_true(self):
        """normalize_actions=True should map to action_scale='unit'."""
        f = FlexLoad(normalize_actions=True)
        assert f.action_scale == 'unit'
        assert f.action_space.shape == (2,)

    def test_normalize_actions_compat_false(self):
        """normalize_actions=False should map to action_scale='physical'."""
        f = FlexLoad(normalize_actions=False, curtail_cap_mw=10.0, shift_cap_mw=15.0)
        assert f.action_scale == 'physical'
        assert f.action_space.high[0] == pytest.approx(10.0)

    def test_repr(self):
        f = FlexLoad(action_scale='physical', curtail_cap_mw=10, shift_cap_mw=5, shift_horizon=4)
        r = repr(f)
        assert 'FlexLoad' in r
        assert '10' in r


# ========================== Reset ==========================

class TestFlexLoadReset:

    def test_reset_clears_state(self, fl):
        fl.step(np.array([5.0, 3.0]))
        fl.step(np.array([0.0, 8.0]))
        fl.reset(seed=0)
        assert fl._curtailed_mw == 0.0
        assert fl._shift_out_mw == 0.0
        assert fl._shift_in_mw == 0.0
        assert len(fl._deferred_buffer) == 0
        assert fl.current_p_mw == 0.0
        assert fl.time_step == 0


# ========================== Curtailment ==========================

class TestCurtailment:
    """Curtailment: permanent load reduction (action[0] > 0)."""

    def test_curtail_positive_injection(self, fl):
        """Curtailment reduces net load -> positive current_p (injection convention)."""
        fl.step(np.array([5.0, 0.0]))
        assert fl.current_p_mw > 0

    def test_curtail_clipped_to_cap(self, fl):
        """Curtailment exceeding capacity is clipped."""
        fl.step(np.array([999.0, 0.0]))
        assert fl._curtailed_mw <= fl.curtail_cap_mw + 1e-9

    def test_curtail_is_permanent(self, fl):
        """Curtailed energy is not recovered later -- no buffer entries."""
        fl.step(np.array([5.0, 0.0]))
        assert fl._curtailed_mw == 5.0
        assert fl._shift_out_mw == 0.0
        assert len(fl._deferred_buffer) == 0

    def test_curtail_current_p_value(self, fl):
        """current_p after curtailment = curtail_mw (no shift, no buffer)."""
        fl.step(np.array([8.0, 0.0]))
        assert fl.current_p_mw == pytest.approx(8.0)


# ========================== Demand Shifting ==========================

class TestDemandShifting:
    """Demand shifting: defer consumption to future (action[1] > 0)."""

    def test_shift_adds_to_buffer(self, fl):
        """Shifting defers demand into the buffer."""
        fl.step(np.array([0.0, 6.0]))
        assert fl._shift_out_mw == 6.0
        assert len(fl._deferred_buffer) == fl.shift_horizon

    def test_shift_distributes_evenly(self, fl):
        """Shifted demand is spread evenly over shift_horizon steps."""
        shift_mw = 8.0
        fl.step(np.array([0.0, shift_mw]))
        per_step = shift_mw / fl.shift_horizon
        for entry in fl._deferred_buffer:
            assert entry == pytest.approx(per_step)

    def test_shift_clipped_to_cap(self, fl):
        """Shift exceeding capacity is clipped."""
        fl.step(np.array([0.0, 999.0]))
        assert fl._shift_out_mw <= fl.shift_cap_mw + 1e-9

    def test_shift_positive_injection(self, fl):
        """Shifting demand out -> positive injection (load reduced now)."""
        fl.step(np.array([0.0, 6.0]))
        assert fl.current_p_mw == pytest.approx(6.0)

    def test_deferred_released_in_subsequent_steps(self, fl):
        """Deferred energy trickles back over shift_horizon steps."""
        fl.step(np.array([0.0, 8.0]))  # shift 8 MW into 4 steps
        per_step = 8.0 / 4.0
        released_total = 0.0
        for _ in range(fl.shift_horizon):
            fl.step(np.array([0.0, 0.0]))  # idle
            # shift_in reduces injection
            released_total += fl._shift_in_mw
        assert released_total == pytest.approx(8.0, abs=0.1)


# ========================== Energy Conservation ==========================

class TestFlexEnergyConservation:

    def test_shift_then_release_balances(self, fl):
        """Sum of current_p over shift + release steps should net to ~0.

        Shift defers demand (positive injection now), then releases it back
        (negative injection via shift_in). Net energy impact over full cycle = 0.
        """
        fl.step(np.array([0.0, 10.0]))  # shift 10 MW
        total_p = fl.current_p_mw
        for _ in range(fl.shift_horizon + 2):
            fl.step(np.array([0.0, 0.0]))
            total_p += fl.current_p_mw
        assert total_p == pytest.approx(0.0, abs=0.1)

    def test_curtailment_is_net_positive(self, fl):
        """Curtailment permanently reduces load -> net positive injection over time."""
        fl.step(np.array([5.0, 0.0]))
        total_p = fl.current_p_mw
        for _ in range(10):
            fl.step(np.array([0.0, 0.0]))
            total_p += fl.current_p_mw
        assert total_p > 0, "Curtailment should produce net positive injection"


# ========================== Buffer Accumulation ==========================

class TestBufferAccumulation:

    def test_consecutive_shifts_accumulate(self, fl):
        """Multiple shifts before release -> buffer grows beyond shift_horizon."""
        fl.step(np.array([0.0, 5.0]))
        fl.step(np.array([0.0, 5.0]))
        assert len(fl._deferred_buffer) >= 4

    def test_overflow_tracked(self, fl):
        """Buffer overflow is tracked in status for CMDP cost computation."""
        for _ in range(10):
            fl.step(np.array([0.0, fl.shift_cap_mw]))
        s = fl.status()
        assert s['buffer_size'] > 0


# ========================== Idle Action ==========================

class TestIdle:

    def test_idle_action(self, fl):
        fl.step(np.array([0.0, 0.0]))
        assert fl._curtailed_mw == 0.0
        assert fl._shift_out_mw == 0.0

    def test_none_action(self, fl):
        fl.step(None)
        assert fl._curtailed_mw == 0.0
        assert fl._shift_out_mw == 0.0

    def test_idle_only_releases_deferred(self, fl):
        """Idle after shift -> current_p = -shift_in (negative because deferred returns)."""
        fl.step(np.array([0.0, 8.0]))  # shift -> buffer filled
        fl.step(None)  # idle -> should release one chunk
        assert fl._shift_in_mw > 0
        # current_p = 0 + 0 - shift_in < 0
        assert fl.current_p_mw < 0


# ========================== Action Parsing ==========================

class TestFlexActionParsing:

    def test_dict_action(self, fl):
        fl.step({'curtail_mw': 5.0, 'shift_out_mw': 0.0})
        assert fl._curtailed_mw == 5.0

    def test_ndarray_action(self, fl):
        fl.step(np.array([0.0, 3.0]))
        assert fl._shift_out_mw == 3.0

    def test_scalar_action_warning(self, fl):
        """Scalar action should warn and interpret as curtailment only."""
        with pytest.warns(UserWarning, match="scalar action"):
            fl.step(7.0)
        assert fl._curtailed_mw == 7.0
        assert fl._shift_out_mw == 0.0


# ========================== Action Scaling ==========================

class TestActionScaling:

    def test_unit_scale_full_capacity(self, fl_unit):
        """action=[1, 1] in unit mode -> full capacity."""
        fl_unit.step(np.array([1.0, 1.0]))
        assert fl_unit._curtailed_mw == pytest.approx(10.0)
        assert fl_unit._shift_out_mw == pytest.approx(10.0)

    def test_unit_scale_zero(self, fl_unit):
        """action=[0, 0] in unit mode -> no action."""
        fl_unit.step(np.array([0.0, 0.0]))
        assert fl_unit._curtailed_mw == pytest.approx(0.0)
        assert fl_unit._shift_out_mw == pytest.approx(0.0)

    def test_unit_scale_half(self, fl_unit):
        """action=[0.5, 0.5] in unit mode -> half capacity."""
        fl_unit.step(np.array([0.5, 0.5]))
        assert fl_unit._curtailed_mw == pytest.approx(5.0)
        assert fl_unit._shift_out_mw == pytest.approx(5.0)

    def test_tanh_scale_full(self, fl_tanh):
        """action=[1, 1] in tanh mode -> full capacity."""
        fl_tanh.step(np.array([1.0, 1.0]))
        assert fl_tanh._curtailed_mw == pytest.approx(10.0)
        assert fl_tanh._shift_out_mw == pytest.approx(10.0)

    def test_tanh_scale_minus_one(self, fl_tanh):
        """action=[-1, -1] in tanh mode -> zero (no action)."""
        fl_tanh.step(np.array([-1.0, -1.0]))
        assert fl_tanh._curtailed_mw == pytest.approx(0.0)
        assert fl_tanh._shift_out_mw == pytest.approx(0.0)

    def test_tanh_scale_zero(self, fl_tanh):
        """action=[0, 0] in tanh mode -> half capacity."""
        fl_tanh.step(np.array([0.0, 0.0]))
        assert fl_tanh._curtailed_mw == pytest.approx(5.0)
        assert fl_tanh._shift_out_mw == pytest.approx(5.0)


# ========================== Observation ==========================

class TestFlexObs:

    def test_obs_is_dict(self, fl):
        o = fl.obs()
        assert isinstance(o, dict)
        assert len(o) == 8

    def test_obs_keys(self, fl):
        o = fl.obs()
        expected = {'curtail_norm', 'shift_out_norm', 'shift_in_norm',
                    'buffer_fill_ratio', 'buffer_energy_norm',
                    'time_sin', 'time_cos', 'price_norm'}
        assert set(o.keys()) == expected

    def test_obs_after_curtail(self, fl):
        fl.step(np.array([5.0, 0.0]))
        o = fl.obs()
        assert o['curtail_norm'] > 0
        assert o['shift_out_norm'] == pytest.approx(0.0)

    def test_obs_after_shift(self, fl):
        fl.step(np.array([0.0, 5.0]))
        o = fl.obs()
        assert o['curtail_norm'] == pytest.approx(0.0)
        assert o['shift_out_norm'] > 0

    def test_obs_price_after_set_lmp(self, fl):
        fl.set_lmp(50.0)
        fl.step(np.array([0.0, 0.0]))
        o = fl.obs()
        assert o['price_norm'] == pytest.approx(0.5)  # 50 / 100

    def test_obs_clipped(self, fl):
        """All obs values should be within their defined range."""
        fl.step(np.array([fl.curtail_cap_mw, 0.0]))
        o = fl.obs()
        for k, v in o.items():
            if k in ('time_sin', 'time_cos'):
                assert -1.0 <= v <= 1.0
            elif k == 'price_norm':
                assert 0.0 <= v <= 2.0
            else:
                assert 0.0 <= v <= 1.0


# ========================== Status ==========================

class TestFlexStatus:

    def test_status_keys(self, fl):
        fl.step(np.array([0.0, 0.0]))
        s = fl.status()
        expected_keys = {'current_p_mw', 'current_q_mvar', 'curtailed_mw',
                         'shift_out_mw', 'shift_in_mw', 'buffer_size',
                         'buffer_total_mw', 'buffer_overflow_mwh',
                         'current_lmp', 'time_step', 'bus_id', 'local_v',
                         'cost_curtailment', 'cost_shift_discomfort',
                         'cost_buffer_overflow', 'cost_simultaneous'}
        assert expected_keys.issubset(s.keys())

    def test_cost_curtailment(self, fl):
        """Curtailment cost = c(t) * dt_h * curtail_cost_per_mwh."""
        fl.step(np.array([5.0, 0.0]))
        s = fl.status()
        expected = 5.0 * (15.0 / 60.0) * 50.0
        assert s['cost_curtailment'] == pytest.approx(expected)

    def test_cost_simultaneous(self, fl):
        """Simultaneous curtail + shift -> penalty on min(c, s)."""
        fl.step(np.array([3.0, 5.0]))
        s = fl.status()
        expected = min(3.0, 5.0) * 100.0
        assert s['cost_simultaneous'] == pytest.approx(expected)

    def test_no_simultaneous_cost_when_exclusive(self, fl):
        """Only curtailing -> no complementarity penalty."""
        fl.step(np.array([5.0, 0.0]))
        s = fl.status()
        assert s['cost_simultaneous'] == pytest.approx(0.0)

    def test_overflow_detection(self, fl):
        """Consecutive max shifts create buffer overflow."""
        for _ in range(10):
            fl.step(np.array([0.0, fl.shift_cap_mw]))
        s = fl.status()
        assert s['buffer_size'] > fl.shift_horizon
        assert s['cost_buffer_overflow'] > 0


# ========================== LMP and Bid ==========================

class TestLMPAndBid:

    def test_set_lmp(self, fl):
        fl.set_lmp(75.0)
        assert fl._current_lmp == 75.0

    def test_get_bid_structure(self, fl):
        bid = fl.get_bid()
        assert 'curtail_cap_mw' in bid
        assert 'shift_cap_mw' in bid
        assert 'shift_horizon' in bid
        assert bid['curtail_cap_mw'] == 10.0

    def test_bid_shift_zero_when_overflow(self, fl):
        """When buffer overflows, shift capacity should be reported as 0."""
        # Overflow the buffer beyond shift_horizon
        for _ in range(fl.shift_horizon + 2):
            fl.step(np.array([0.0, fl.shift_cap_mw]))
        bid = fl.get_bid()
        assert bid['shift_cap_mw'] == 0.0


# ========================== Edge Cases ==========================

class TestFlexEdgeCases:

    def test_zero_capacities(self):
        """FlexLoad with zero capacities should handle gracefully."""
        f = FlexLoad(action_scale='physical', curtail_cap_mw=0.0, shift_cap_mw=0.0)
        f.reset(seed=0)
        f.step(np.array([10.0, 10.0]))
        assert f._curtailed_mw == 0.0
        assert f._shift_out_mw == 0.0

    def test_shift_horizon_one(self):
        """shift_horizon=1: all deferred demand released next step."""
        f = FlexLoad(action_scale='physical', shift_cap_mw=10.0, shift_horizon=1)
        f.reset(seed=0)
        f.step(np.array([0.0, 8.0]))
        assert len(f._deferred_buffer) == 1
        f.step(np.array([0.0, 0.0]))
        assert f._shift_in_mw == pytest.approx(8.0)
        assert len(f._deferred_buffer) == 0

    def test_long_simulation(self, fl):
        """100 steps of random actions without crashing."""
        rng = np.random.default_rng(99)
        for _ in range(100):
            action = rng.uniform(0, 1, size=2) * np.array([fl.curtail_cap_mw, fl.shift_cap_mw])
            fl.step(action)
        assert fl.time_step == 100
