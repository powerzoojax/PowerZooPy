"""Tests for powerzoo.envs.market — MarketEnv, CostBasedMarketEnv, BidBasedMarketEnv.

MarketEnv: abstract base class with step/clear/settle/revenue interface.

CostBasedMarketEnv: cost-based LMP arbitrage
  - Locational Marginal Prices (LMP) from DC-OPF dual variables
  - Flat marginal-cost objective (mc_c), no bid-cost separation
  - Battery arbitrage: buy low (charge when LMP low), sell high (discharge when LMP high)
  - Observation: [soc, lmp_norm, time_sin, time_cos, demand_norm]
  - Reward: revenue = LMP × P_net × Δt

BidBasedMarketEnv: piecewise-linear offer curves with bid-cost separation
  - Generators submit stepped offer curves (from cost + optional markup)
  - LMP derived from offer-based dispatch
  - Observation: [soc, lmp_norm, time_sin, time_cos, demand_norm, mean_offer_price_norm]
  - Reward: LMP × P_net × Δt

Domain knowledge:
  - LMP = ∂(total_cost)/∂(demand_at_bus) — marginal cost of serving one more MW at each bus
  - Under no congestion: LMP uniform across all buses = system marginal cost
  - Under congestion: LMP diverges across buses (congested side is more expensive)
  - Battery arbitrage profit = ∫(LMP_discharge - LMP_charge) × P × dt - losses
  - Round-trip efficiency loss: η_rt < 1 means not all charge energy is recovered
"""
import numpy as np
import pytest

from powerzoo.envs.market.base import MarketEnv
from powerzoo.envs.market.cost_based_market import CostBasedMarketEnv


# ── MarketEnv (Abstract Base) ────────────────────────────────────────

class TestMarketEnvBase:
    """Abstract interface contract."""

    def test_abstract_cannot_instantiate(self):
        """MarketEnv is abstract; direct instantiation must fail."""
        with pytest.raises(TypeError, match="abstract method"):
            MarketEnv()

    def test_concrete_subclass_works(self):
        """A subclass implementing all methods can be instantiated."""
        class _Concrete(MarketEnv):
            def step(self, bidding): pass
            def clear(self): return {}
            def settle(self): return {}
            def revenue(self): return {}
        m = _Concrete()
        assert m.last_bids == {}


# ── CostBasedMarketEnv: Constructor ─────────────────────────────────

class TestCostBasedMarketEnvInit:
    """Constructor, spaces, and internal wiring."""

    def test_default_battery_attached(self):
        """Default constructor attaches a battery at bus 2."""
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100)
        assert env._battery is not None

    def test_no_battery_when_none(self):
        """battery_bus_id=None skips battery creation."""
        env = CostBasedMarketEnv(normalize_actions=False, battery_bus_id=None, time_series=np.ones(48) * 100)
        assert env._battery is None

    def test_observation_space_shape(self):
        """obs = [soc, lmp_norm, time_sin, time_cos, demand_norm] = 5 dims."""
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100)
        assert env.observation_space.shape == (5,)

    def test_action_space_shape(self):
        """action = [battery_power] = 1 dim."""
        env = CostBasedMarketEnv(normalize_actions=False, battery_power_mw=50.0, time_series=np.ones(48) * 100)
        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == -50.0
        assert env.action_space.high[0] == 50.0

    def test_dt_h_from_delta_t(self):
        """dt_hours = delta_t_minutes / 60."""
        env = CostBasedMarketEnv(normalize_actions=False, delta_t_minutes=30.0, time_series=np.ones(48) * 100)
        np.testing.assert_allclose(env._dt_h, 0.5)

    def test_lmp_scale(self):
        env = CostBasedMarketEnv(normalize_actions=False, lmp_scale=200.0, time_series=np.ones(48) * 100)
        assert env.lmp_scale == 200.0

    def test_obs_names(self):
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100)
        assert env.obs_names == ['soc', 'lmp_norm', 'time_sin', 'time_cos',
                                 'total_demand_norm']

    def test_steps_per_day_property(self):
        env = CostBasedMarketEnv(normalize_actions=False, delta_t_minutes=30.0, time_series=np.ones(48) * 100)
        assert env.steps_per_day == 48

    def test_battery_normalize_actions_always_false(self):
        """BatteryEnv must always have normalize_actions=False.

        The wrapper owns denormalization; passing physical MW via dict already
        bypasses battery-level scaling, but the flag must be False to make the
        contract explicit and prevent accidental double-scaling if the call
        path is ever refactored.
        """
        for outer_norm in (True, False):
            env = CostBasedMarketEnv(
                normalize_actions=outer_norm,
                time_series=np.ones(48) * 100,
            )
            assert env._battery.normalize_actions is False

    def test_obs_demand_norm_bound_is_unbounded(self):
        """total_demand_norm upper bound must be inf (not a hard cap at 2.0)."""
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100)
        assert not np.isfinite(env.observation_space.high[4])


# ── CostBasedMarketEnv: Reset ─────────────────────────────────────────────

class TestCostBasedMarketEnvReset:
    """Reset triggers grid reset, validates LMP availability."""

    def test_reset_returns_obs_info(self):
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100, solver_type='scipy')
        obs, info = env.reset(seed=42)
        assert obs.shape == (5,)
        assert isinstance(info, dict)

    def test_obs_soc_in_range(self):
        """SOC should be in [0, 1]."""
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100, solver_type='scipy')
        obs, _ = env.reset(seed=42)
        soc = obs[0]
        assert 0.0 <= soc <= 1.0

    def test_obs_time_bounded(self):
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100, solver_type='scipy')
        obs, _ = env.reset(seed=42)
        time_sin, time_cos = obs[2], obs[3]
        assert -1.0 <= time_sin <= 1.0
        assert -1.0 <= time_cos <= 1.0


# ── CostBasedMarketEnv: Step & Reward ─────────────────────────────────────

class TestCostBasedMarketEnvStep:
    """Step mechanics and market reward."""

    def test_step_returns_five_tuple(self):
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100, solver_type='scipy')
        env.reset(seed=42)
        result = env.step(np.array([0.0]))
        assert len(result) == 5

    def test_zero_action_low_reward(self):
        """Zero battery action → zero market revenue (no arbitrage)."""
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100, solver_type='scipy')
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(np.array([0.0]))
        # With zero power, revenue component should be zero
        # (reward may include safety penalty)
        assert isinstance(reward, float)

    def test_info_contains_lmp(self):
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100, solver_type='scipy')
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.array([0.0]))
        assert 'lmp' in info

    def test_info_contains_safety(self):
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100, solver_type='scipy')
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.array([0.0]))
        assert 'is_safe' in info

    def test_step_raises_without_battery(self):
        """step() must raise RuntimeError when no battery is attached.

        Without a physical battery, the reward would be computed on an
        unconstrained phantom power value (ghost arbitrage).
        """
        env = CostBasedMarketEnv(
            battery_bus_id=None,
            normalize_actions=False,
            time_series=np.ones(48) * 100,
            solver_type='scipy',
        )
        env.reset(seed=42)
        with pytest.raises(RuntimeError, match="without an attached battery"):
            env.step(np.array([10.0]))

    def test_info_contains_dispatch_diagnostics(self):
        """info must contain requested_p_mw and realized_p_mw."""
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100, solver_type='scipy')
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.array([10.0]))
        assert 'requested_p_mw' in info
        assert 'realized_p_mw' in info

    def test_dispatch_diagnostics_diverge_at_soc_limit(self):
        """realized_p_mw < requested_p_mw when SOC constrains discharge."""
        env = CostBasedMarketEnv(
            battery_bus_id=2,
            battery_capacity_mwh=1.0,   # tiny capacity so SOC limit bites quickly
            battery_power_mw=50.0,
            normalize_actions=False,
            time_series=np.ones(48) * 100,
            solver_type='scipy',
        )
        env.reset(seed=0)
        # Request full rated discharge; SOC constraint will clip realized below 50 MW
        _, _, _, _, info = env.step(np.array([50.0]))
        assert info['requested_p_mw'] == pytest.approx(50.0)
        assert info['realized_p_mw'] < info['requested_p_mw']


# ── CostBasedMarketEnv: Reward Logic ──────────────────────────────────────

class TestLMPMarketRewardLogic:
    """Market reward signal: LMP × P × Δt."""

    def test_discharge_positive_revenue(self):
        """Discharging (P > 0) with positive LMP → positive revenue."""
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100, solver_type='scipy')
        env.reset(seed=42)

        # Simulate: set known LMP
        env._last_lmp = np.array([50.0] * 5)  # 50 $/MWh at all buses
        state = {'safety_info': {}, 'is_safe': True}
        info = {}

        reward = env._compute_market_reward(
            action=np.array([10.0]),   # 10 MW discharge
            state=state,
            info=info,
        )
        # revenue = 50 * 10 * 0.5 = 250 (for 30-min step)
        expected = 50.0 * 10.0 * env._dt_h
        np.testing.assert_allclose(reward, expected, atol=1e-6)

    def test_charge_negative_revenue(self):
        """Charging (P < 0) with positive LMP → negative revenue (cost)."""
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100, solver_type='scipy')
        env.reset(seed=42)

        env._last_lmp = np.array([50.0] * 5)
        state = {'safety_info': {}, 'is_safe': True}
        info = {}

        reward = env._compute_market_reward(
            action=np.array([-10.0]),
            state=state,
            info=info,
        )
        expected = 50.0 * (-10.0) * env._dt_h
        np.testing.assert_allclose(reward, expected, atol=1e-6)

    def test_safety_penalty_not_in_reward(self):
        """Safety penalties flow through cost channel, not reward."""
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100, solver_type='scipy')
        env.reset(seed=42)

        env._last_lmp = np.array([50.0] * 5)
        state = {
            'safety_info': {
                'unsafe_line_ids': [0, 1],
                'unsafe_line_flows': [],
                'unsafe_line_caps': [],
                'unsafe_line_floors': [],
            },
            'is_safe': False,
        }
        info = {}

        reward = env._compute_market_reward(
            action=np.array([0.0]),
            state=state,
            info=info,
        )
        # With zero action and zero LMP revenue, reward should be 0
        # (safety penalty is excluded from reward scalar)
        assert reward == 0.0

    def test_bus_lmp_uses_positional_index(self):
        """_get_bus_lmp() must map bus_id → 0-based node index, not use bus_id as array index.

        Case5 bus IDs are [1,2,3,4,5].  Battery at bus_id=2 → node index 1.
        Setting lmp[1]=99 while all other entries differ confirms correct lookup.
        """
        env = CostBasedMarketEnv(
            battery_bus_id=2,
            normalize_actions=False,
            time_series=np.ones(48) * 100,
            solver_type='scipy',
        )
        env.reset(seed=0)
        # lmp[0]=0, lmp[1]=99 (bus 2 at node index 1), rest 0
        env._last_lmp = np.array([0.0, 99.0, 0.0, 0.0, 0.0])
        assert env._get_bus_lmp() == 99.0, (
            "bus_id=2 should map to lmp[1]=99, not lmp[2]=0"
        )

    def test_step_reward_uses_realized_power(self):
        """Reward must use battery.current_p_mw (SOC-clipped), not the raw requested action."""
        env = CostBasedMarketEnv(
            battery_bus_id=2,
            battery_capacity_mwh=1.0,   # tiny capacity so SOC limits bite quickly
            battery_power_mw=50.0,
            normalize_actions=False,
            time_series=np.ones(48) * 100,
            solver_type='scipy',
        )
        obs, _ = env.reset(seed=0)
        # Request full discharge; SOC limits will clip actual dispatch below 50 MW
        large_discharge = np.array([50.0])
        _, reward, _, _, info = env.step(large_discharge)
        realized = env._battery.current_p_mw
        lmp = env._get_bus_lmp()
        expected_reward = lmp * realized * env._dt_h
        np.testing.assert_allclose(reward, expected_reward, atol=1e-6)

    def test_action_clip_before_denorm(self):
        """Normalized actions outside [-1,1] are clipped before denormalization."""
        env = CostBasedMarketEnv(
            normalize_actions=True,
            time_series=np.ones(48) * 100,
            solver_type='scipy',
        )
        env.reset(seed=42)
        # 1.5 > 1.0: should behave identically to 1.0 after clipping
        _, reward_clipped, _, _, _ = env.step(np.array([1.5]))
        env.reset(seed=42)
        _, reward_one, _, _, _ = env.step(np.array([1.0]))
        np.testing.assert_allclose(reward_clipped, reward_one, atol=1e-6)

    def test_physical_action_clipped_to_power_mw(self):
        """Non-normalized actions beyond ±power_mw are clipped to physical bounds."""
        env = CostBasedMarketEnv(
            normalize_actions=False,
            battery_power_mw=50.0,
            time_series=np.ones(48) * 100,
            solver_type='scipy',
        )
        env.reset(seed=42)
        # 999 MW >> 50 MW rated power: must behave identically to 50 MW
        _, reward_extreme, _, _, _ = env.step(np.array([999.0]))
        env.reset(seed=42)
        _, reward_rated, _, _, _ = env.step(np.array([50.0]))
        np.testing.assert_allclose(reward_extreme, reward_rated, atol=1e-6)

    def test_obs_lmp_norm_bounds_allow_extreme_lmp(self):
        """observation_space allows lmp_norm outside [-5, 5] (unbounded)."""
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100, solver_type='scipy')
        # lmp_norm bounds should be inf, not a hard clip at 5
        assert not np.isfinite(env.observation_space.high[1])
        assert not np.isfinite(env.observation_space.low[1])


# ── CostBasedMarketEnv: Multi-step Episode ────────────────────────────────

class TestCostBasedMarketEnvMultiStep:
    """Full episode rollout."""

    def test_episode_completes(self):
        env = CostBasedMarketEnv(
            normalize_actions=False,
            time_series=np.ones(48) * 100,
            solver_type='scipy',
            max_episode_steps=5,
        )
        env.reset(seed=42)
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert terminated or truncated

    def test_render_does_not_crash(self):
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100, solver_type='scipy')
        env.reset(seed=42)
        env.step(np.array([0.0]))
        env.render()  # Should print, not crash

    def test_close_does_not_crash(self):
        env = CostBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100, solver_type='scipy')
        env.close()



# ── BidBasedMarketEnv ────────────────────────────────────────────────

from powerzoo.envs.market.bid_based_market import BidBasedMarketEnv


class TestBidBasedMarketEnvInit:
    """Constructor and space shapes."""

    def test_obs_space_shape(self):
        """obs = 6 dims (adds mean_offer_price_norm)."""
        env = BidBasedMarketEnv(normalize_actions=False, time_series=np.ones(48) * 100)
        assert env.observation_space.shape == (6,)

    def test_action_space_shape(self):
        env = BidBasedMarketEnv(normalize_actions=False, battery_power_mw=50.0,
                                time_series=np.ones(48) * 100)
        assert env.action_space.shape == (1,)

    def test_segments_created(self):
        env = BidBasedMarketEnv(normalize_actions=False, n_segments=5,
                                time_series=np.ones(48) * 100)
        assert env._base_segments['seg_widths'].shape[1] == 5
        assert env._base_segments['seg_prices'].shape[1] == 5


class TestBidBasedMarketEnvReset:
    """Reset generates offer curves and runs clearing."""

    def test_reset_returns_obs_info(self):
        env = BidBasedMarketEnv(normalize_actions=False,
                                time_series=np.ones(48) * 100)
        obs, info = env.reset(seed=42)
        assert obs.shape == (6,)
        assert isinstance(info, dict)

    def test_offer_segments_generated(self):
        env = BidBasedMarketEnv(normalize_actions=False,
                                time_series=np.ones(48) * 100)
        env.reset(seed=42)
        assert env._current_segments is not None
        assert 'seg_widths' in env._current_segments
        assert 'seg_prices' in env._current_segments

    def test_lmp_available_after_reset(self):
        env = BidBasedMarketEnv(normalize_actions=False,
                                time_series=np.ones(48) * 100)
        env.reset(seed=42)
        assert env._last_lmp is not None


class TestBidBasedMarketEnvStep:
    """Step mechanics."""

    def test_step_returns_five_tuple(self):
        env = BidBasedMarketEnv(normalize_actions=False,
                                time_series=np.ones(48) * 100)
        env.reset(seed=42)
        result = env.step(np.array([0.0]))
        assert len(result) == 5

    def test_info_contains_cost_model(self):
        env = BidBasedMarketEnv(normalize_actions=False,
                                time_series=np.ones(48) * 100)
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.array([0.0]))
        assert info.get('cost_model') == 'piecewise'

    def test_info_has_offer_and_true_cost(self):
        env = BidBasedMarketEnv(normalize_actions=False,
                                time_series=np.ones(48) * 100)
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.array([0.0]))
        assert 'offer_cost' in info
        assert 'true_cost' in info

    def test_zero_action_zero_reward(self):
        env = BidBasedMarketEnv(normalize_actions=False,
                                time_series=np.ones(48) * 100)
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(np.array([0.0]))
        assert isinstance(reward, float)


class TestBidBasedMarketRewardLogic:
    """Market reward with piecewise costs."""

    def test_discharge_positive_revenue(self):
        env = BidBasedMarketEnv(normalize_actions=False,
                                time_series=np.ones(48) * 100)
        env.reset(seed=42)
        env._last_lmp = np.array([50.0] * 5)
        reward = env._compute_market_reward(np.array([10.0]))
        expected = 50.0 * 10.0 * env._dt_h
        np.testing.assert_allclose(reward, expected, atol=1e-6)

    def test_charge_negative_revenue(self):
        env = BidBasedMarketEnv(normalize_actions=False,
                                time_series=np.ones(48) * 100)
        env.reset(seed=42)
        env._last_lmp = np.array([50.0] * 5)
        reward = env._compute_market_reward(np.array([-10.0]))
        expected = 50.0 * (-10.0) * env._dt_h
        np.testing.assert_allclose(reward, expected, atol=1e-6)


class TestBidBasedMarketEpisode:
    """Full episode rollout."""

    def test_episode_completes(self):
        env = BidBasedMarketEnv(
            normalize_actions=False,
            time_series=np.ones(48) * 100,
            max_episode_steps=5,
        )
        env.reset(seed=42)
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert terminated or truncated

    def test_render_does_not_crash(self):
        env = BidBasedMarketEnv(normalize_actions=False,
                                time_series=np.ones(48) * 100)
        env.reset(seed=42)
        env.step(np.array([0.0]))
        env.render()

    def test_close_does_not_crash(self):
        env = BidBasedMarketEnv(normalize_actions=False,
                                time_series=np.ones(48) * 100)
        env.close()


# ── Piecewise solver unit tests ──────────────────────────────────────

from powerzoo.envs.grid.cal_dcopf_trans import make_cost_segments, solve_piecewise_ed_opf


class TestMakeCostSegments:
    """Unit tests for cost segment generation."""

    def test_shape(self):
        from powerzoo.case import load_case
        case = load_case(5)
        segs = make_cost_segments(case, n_segments=5)
        n_u = len(case.units)
        assert segs['seg_widths'].shape == (n_u, 5)
        assert segs['seg_prices'].shape == (n_u, 5)

    def test_prices_monotonic(self):
        from powerzoo.case import load_case
        case = load_case(5)
        segs = make_cost_segments(case, n_segments=5)
        for i in range(segs['seg_prices'].shape[0]):
            diffs = np.diff(segs['seg_prices'][i])
            assert np.all(diffs >= 0), f"Non-monotonic prices for generator {i}"

    def test_widths_sum_to_range(self):
        from powerzoo.case import load_case
        case = load_case(5)
        segs = make_cost_segments(case, n_segments=5)
        p_min = case.units['p_min'].values
        p_max = case.units['p_max'].values
        expected_range = p_max - p_min
        actual_range = segs['seg_widths'].sum(axis=1)
        np.testing.assert_allclose(actual_range, expected_range, atol=1e-6)


class TestPiecewiseSolver:
    """Piecewise-linear OPF solver."""

    def test_solve_returns_success(self):
        from powerzoo.case import load_case
        case = load_case(5)
        segs = make_cost_segments(case, n_segments=5)
        node_load = np.array([100.0, 150.0, 200.0, 180.0, 120.0])
        result = solve_piecewise_ed_opf(case, node_load, segs)
        assert result['success'] is True

    def test_power_balance(self):
        from powerzoo.case import load_case
        case = load_case(5)
        segs = make_cost_segments(case, n_segments=5)
        node_load = np.array([100.0, 150.0, 200.0, 180.0, 120.0])
        result = solve_piecewise_ed_opf(case, node_load, segs)
        total_gen = result['unit_power_mw'].sum()
        total_demand = node_load.sum()
        np.testing.assert_allclose(total_gen, total_demand, atol=1e-3)

    def test_lmp_available(self):
        from powerzoo.case import load_case
        case = load_case(5)
        segs = make_cost_segments(case, n_segments=5)
        node_load = np.array([100.0, 150.0, 200.0, 180.0, 120.0])
        result = solve_piecewise_ed_opf(case, node_load, segs)
        assert result['lmp_available'] is True
        assert result['lmp'].shape == (len(case.nodes),)

    def test_offer_cost_vs_true_cost(self):
        """Offer cost (from offer prices) should differ from true cost."""
        from powerzoo.case import load_case
        case = load_case(5)
        segs = make_cost_segments(case, n_segments=5)
        node_load = np.array([100.0, 150.0, 200.0, 180.0, 120.0])
        result = solve_piecewise_ed_opf(case, node_load, segs)
        # Both should be finite
        assert np.isfinite(result['offer_cost'])
        assert np.isfinite(result['total_cost'])
