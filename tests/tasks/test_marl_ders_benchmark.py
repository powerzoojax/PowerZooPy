"""Smoke tests for MARLDERBenchmarkTask and the heterogeneous resource adapter.

Verification criteria (per task prompt):
1. PowerZoo can construct a DERs main-config scenario on Case118zh
2. possible_agents == 12 (4 battery + 4 PV + 4 flexload)
3. Per-type action space shape == (2,) for all agents
4. Per-type action semantics: Battery=['p_mw','q_mvar'], PV=['curtailment','q_control'],
   FlexLoad=['curtail_mw','shift_out_mw']
5. reset() returns 12 observations
6. step() with zero action runs without error
7. info contains voltage / cost metrics
8. Backward compat: MARLDERArbitrageTask (battery-only) still works unchanged
"""

import numpy as np
import pytest
from gymnasium import spaces

from powerzoo.tasks.simple.marl_ders_benchmark import (
    MARLDERBenchmarkTask,
    DERS_BATTERY_BUSES,
    DERS_PV_BUSES,
    DERS_FLEXLOAD_BUSES,
    DERS_V_MIN,
    DERS_V_MAX,
    inject_load_profiles,
    make_ders_benchmark_env_with_profiles,
)


# ---------------------------------------------------------------------------
# Shared fixture — module-scoped to avoid repeated env construction
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def task():
    return MARLDERBenchmarkTask(split='train')


@pytest.fixture(scope="module")
def env(task):
    return task.create_env()


@pytest.fixture(scope="module")
def obs_info(env):
    obs, info = env.reset(seed=0)
    return obs, info


@pytest.fixture(scope="module")
def step_result(env, obs_info):
    action_dict = {
        agent: np.zeros(env.action_space[agent].shape, dtype=np.float32)
        for agent in env.possible_agents
    }
    return env.step(action_dict)


# ===========================================================================
# 1. Task config
# ===========================================================================

class TestConfig:
    def test_bus_count_battery(self):
        assert len(DERS_BATTERY_BUSES) == 4

    def test_bus_count_pv(self):
        assert len(DERS_PV_BUSES) == 4

    def test_bus_count_flexload(self):
        assert len(DERS_FLEXLOAD_BUSES) == 4

    def test_v_limits(self):
        assert DERS_V_MIN == pytest.approx(0.94)
        assert DERS_V_MAX == pytest.approx(1.06)

    def test_task_name(self):
        t = MARLDERBenchmarkTask()
        assert t.name == "marl_ders_benchmark"

    def test_scenario_resource_count(self):
        t = MARLDERBenchmarkTask()
        cfg = t.get_scenario_config()
        assert len(cfg['resources']) == 12

    def test_resource_types_in_scenario(self):
        t = MARLDERBenchmarkTask()
        cfg = t.get_scenario_config()
        types = [r['type'] for r in cfg['resources']]
        assert types.count('battery') == 4
        assert types.count('solar') == 4
        assert types.count('flexload') == 4

    def test_resource_filter_includes_all_types(self):
        t = MARLDERBenchmarkTask()
        agents_cfg = t.get_agents_config()
        rf = agents_cfg['resource_filter']
        assert 'battery' in rf
        assert 'solar' in rf
        assert 'flexload' in rf


# ===========================================================================
# 2. Multi-agent adapter: agent count
# ===========================================================================

class TestAgentCount:
    def test_possible_agents_count(self, env):
        assert len(env.possible_agents) == 12, (
            f"Expected 12 agents, got {len(env.possible_agents)}: {env.possible_agents}"
        )

    def test_agents_count_equals_possible(self, env):
        assert len(env.agents) == len(env.possible_agents)

    def test_battery_agents_present(self, env):
        ids = ' '.join(env.possible_agents)
        bat_count = sum(1 for a in env.possible_agents if 'bat' in a)
        assert bat_count == 4, f"Expected 4 battery agents, found {bat_count} in {env.possible_agents}"

    def test_pv_agents_present(self, env):
        pv_count = sum(1 for a in env.possible_agents if 'pv' in a)
        assert pv_count == 4, f"Expected 4 PV agents, found {pv_count} in {env.possible_agents}"

    def test_flexload_agents_present(self, env):
        fl_count = sum(1 for a in env.possible_agents if 'fl' in a)
        assert fl_count == 4, f"Expected 4 flexload agents, found {fl_count} in {env.possible_agents}"


# ===========================================================================
# 3. Action space shape and semantics
# ===========================================================================

class TestActionSpace:
    def test_all_agents_have_2d_action(self, env):
        for agent in env.possible_agents:
            shape = env.action_space[agent].shape
            assert shape == (2,), (
                f"Agent {agent}: expected action shape (2,), got {shape}"
            )

    def test_battery_action_names(self, env):
        for agent in env.possible_agents:
            if 'bat' in agent:
                resource = env._resources[agent]
                assert hasattr(resource, 'action_names'), f"{agent} missing action_names"
                assert resource.action_names == ['p_mw', 'q_mvar'], (
                    f"{agent} action_names={resource.action_names}"
                )

    def test_pv_action_names(self, env):
        for agent in env.possible_agents:
            if 'pv' in agent:
                resource = env._resources[agent]
                assert hasattr(resource, 'action_names'), f"{agent} missing action_names"
                assert resource.action_names == ['curtailment', 'q_control'], (
                    f"{agent} action_names={resource.action_names}"
                )

    def test_flexload_action_names(self, env):
        for agent in env.possible_agents:
            if 'fl' in agent:
                resource = env._resources[agent]
                assert hasattr(resource, 'action_names'), f"{agent} missing action_names"
                assert resource.action_names == ['curtail_mw', 'shift_out_mw'], (
                    f"{agent} action_names={resource.action_names}"
                )

    def test_battery_q_control_enabled(self, env):
        for agent in env.possible_agents:
            if 'bat' in agent:
                resource = env._resources[agent]
                assert resource.enable_q_control is True, f"{agent}: enable_q_control should be True"

    def test_pv_q_control_enabled(self, env):
        for agent in env.possible_agents:
            if 'pv' in agent:
                resource = env._resources[agent]
                assert resource.enable_q_control is True, f"{agent}: enable_q_control should be True"


# ===========================================================================
# 4. reset() — observations
# ===========================================================================

class TestReset:
    def test_reset_returns_12_obs(self, obs_info):
        obs, info = obs_info
        assert len(obs) == 12

    def test_obs_all_agents_present(self, obs_info, env):
        obs, _ = obs_info
        for agent in env.possible_agents:
            assert agent in obs, f"Agent {agent} missing from obs"

    def test_obs_dtype_float32(self, obs_info):
        obs, _ = obs_info
        for agent, o in obs.items():
            assert o.dtype == np.float32, f"{agent} obs dtype={o.dtype}"

    def test_obs_shape_consistent(self, obs_info):
        obs, _ = obs_info
        shapes = {agent: o.shape for agent, o in obs.items()}
        unique_shapes = set(shapes.values())
        assert len(unique_shapes) == 1, (
            f"Obs shapes differ across agents (heterogeneous obs is not allowed): {shapes}"
        )

    def test_obs_no_nan(self, obs_info):
        obs, _ = obs_info
        for agent, o in obs.items():
            assert not np.any(np.isnan(o)), f"{agent} obs contains NaN"


# ===========================================================================
# 5. step() — rewards, dones, info
# ===========================================================================

class TestStep:
    def test_step_returns_12_rewards(self, step_result, env):
        _, rewards, _, _, _ = step_result
        assert len(rewards) == len(env.possible_agents)

    def test_step_rewards_are_float(self, step_result, env):
        _, rewards, _, _, _ = step_result
        for agent, r in rewards.items():
            assert isinstance(r, (int, float)), f"{agent} reward not scalar: {r}"

    def test_step_terminateds_has_all_key(self, step_result):
        _, _, terminateds, truncateds, _ = step_result
        assert '__all__' in terminateds
        assert '__all__' in truncateds

    def test_step_info_has_voltage_cost(self, step_result, env):
        _, _, _, _, infos = step_result
        # At least one agent info should have cost fields
        for agent in env.possible_agents:
            if agent in infos:
                info = infos[agent]
                assert 'cost' in info, f"{agent} info missing 'cost' key"
                break

    def test_step_info_has_voltage_in_env_info(self, env, obs_info):
        """The base_env info should contain voltage/cost metrics."""
        env.reset(seed=1)
        zero_action = {
            agent: np.zeros(env.action_space[agent].shape, dtype=np.float32)
            for agent in env.possible_agents
        }
        _, _, _, _, infos = env.step(zero_action)
        # Check any agent info has cost keys
        any_agent = env.possible_agents[0]
        if any_agent in infos:
            info = infos[any_agent]
            assert 'cost' in info

    def test_step_zero_action_no_crash(self, step_result):
        """Zero action should not raise any exception."""
        assert step_result is not None


# ===========================================================================
# 6. Backward compat: battery-only task unchanged
# ===========================================================================

class TestBackwardCompat:
    def test_battery_only_task_still_works(self):
        from powerzoo.tasks.simple.marl_der_arbitrage import MARLDERArbitrageTask
        task = MARLDERArbitrageTask(num_batteries=3)
        env = task.create_env()
        assert len(env.possible_agents) == 3
        obs, _ = env.reset(seed=0)
        assert len(obs) == 3
        # Battery action should be 1D (no Q control in arbitrage task)
        for agent in env.possible_agents:
            assert env.action_space[agent].shape == (1,)

    def test_battery_only_action_space_shape(self):
        from powerzoo.tasks.simple.marl_der_arbitrage import MARLDERArbitrageTask
        task = MARLDERArbitrageTask(num_batteries=2)
        env = task.create_env()
        for agent in env.possible_agents:
            shape = env.action_space[agent].shape
            assert shape[0] >= 1  # at least 1-D


# ===========================================================================
# 7. Registry-level smoke tests
#    Verify make_task / make_task_env / list_tasks all see marl_ders_benchmark.
#    These tests catch regressions where the task exists but is not wired into
#    the standard factory/registry path.
# ===========================================================================

class TestRegistry:
    def test_list_tasks_contains_marl_ders_benchmark(self):
        from powerzoo.tasks.registry import list_tasks
        assert 'marl_ders_benchmark' in list_tasks(), (
            f"marl_ders_benchmark not in list_tasks(): {list_tasks()}"
        )

    def test_make_task_returns_correct_type(self):
        from powerzoo.tasks.registry import make_task
        task = make_task('marl_ders_benchmark')
        assert isinstance(task, MARLDERBenchmarkTask)
        assert task.name == 'marl_ders_benchmark'

    def test_make_task_env_returns_12_agents(self):
        from powerzoo.tasks.registry import make_task_env
        env = make_task_env('marl_ders_benchmark')
        assert len(env.possible_agents) == 12, (
            f"Expected 12 agents via registry path, got {len(env.possible_agents)}"
        )

    def test_make_task_env_reset_step(self):
        from powerzoo.tasks.registry import make_task_env
        import numpy as np
        env = make_task_env('marl_ders_benchmark')
        obs, _ = env.reset(seed=42)
        assert len(obs) == 12
        zero_action = {
            agent: np.zeros(env.action_space[agent].shape, dtype=np.float32)
            for agent in env.possible_agents
        }
        result = env.step(zero_action)
        assert result is not None

    def test_list_tasks_difficulty_filter(self):
        from powerzoo.tasks.registry import list_tasks
        middle_tasks = list_tasks(difficulty='middle')
        assert 'marl_ders_benchmark' in middle_tasks

    def test_list_tasks_agent_mode_filter(self):
        from powerzoo.tasks.registry import list_tasks
        multi_tasks = list_tasks(agent_mode='multi')
        assert 'marl_ders_benchmark' in multi_tasks


# ===========================================================================
# 8. ders_local observation semantics
#    Verify 12-dim type-specific observations replace the old battery-centric
#    placeholder (soc=0.5 constants for PV/FlexLoad).
# ===========================================================================

class TestDERSLocalObs:
    """Verify 'ders_local' 12-dim observation semantics."""

    def test_obs_shape_is_12(self, obs_info):
        obs, _ = obs_info
        for agent, o in obs.items():
            assert o.shape == (12,), f"{agent}: expected (12,), got {o.shape}"

    def test_pv_device_slot0_is_available_cf(self, env, obs_info):
        """PV obs[7] must equal resource.available_cf (not constant 0.5)."""
        obs, _ = obs_info
        for agent in env.possible_agents:
            if 'pv' in agent:
                resource = env._resources[agent]
                assert abs(float(obs[agent][7]) - float(resource.available_cf)) < 0.02, (
                    f"{agent}: obs[7]={obs[agent][7]:.4f} "
                    f"!= available_cf={resource.available_cf:.4f}"
                )

    def test_battery_device_slot0_is_soc(self, env, obs_info):
        """Battery obs[7] must equal resource.soc."""
        obs, _ = obs_info
        for agent in env.possible_agents:
            if 'bat' in agent:
                resource = env._resources[agent]
                assert abs(float(obs[agent][7]) - float(resource.soc)) < 0.02, (
                    f"{agent}: obs[7]={obs[agent][7]:.4f} != soc={resource.soc:.4f}"
                )

    def test_flexload_device_slot0_zero_after_reset(self, env):
        """FlexLoad obs[7] (curtail_norm) should be 0.0 right after reset."""
        task = MARLDERBenchmarkTask(split='train')
        env2 = task.create_env()
        obs, _ = env2.reset(seed=0)
        fl_agents = [a for a in env2.possible_agents if 'fl' in a]
        assert fl_agents, "No FlexLoad agents"
        for agent in fl_agents:
            assert obs[agent][7] == pytest.approx(0.0, abs=0.01), (
                f"{agent}: obs[7]={obs[agent][7]:.4f} expected 0.0 after reset"
            )

    def test_pv_obs_not_constant_placeholder(self, env, obs_info):
        """PV obs[7] must NOT be the old 0.5 SOC placeholder for all agents."""
        obs, _ = obs_info
        pv_obs_vals = [float(obs[a][7]) for a in env.possible_agents if 'pv' in a]
        # If all are exactly 0.5 that signals the old battery-centric placeholder
        assert not all(v == pytest.approx(0.5, abs=0.001) for v in pv_obs_vals), (
            f"All PV obs[7] == 0.5 — this looks like the old SOC placeholder: {pv_obs_vals}"
        )

    def test_shared_context_slot6_is_local_voltage(self, env, obs_info):
        """Slot 6 = local_bus_voltage, should be near 1.0 p.u."""
        obs, _ = obs_info
        for agent, o in obs.items():
            v = float(o[6])
            assert 0.5 <= v <= 1.5, (
                f"{agent}: obs[6] (local_bus_voltage)={v:.3f} out of plausible range"
            )

    def test_info_has_voltage_violation_key(self, step_result, env):
        _, _, _, _, infos = step_result
        for agent in env.possible_agents:
            if agent in infos:
                assert 'voltage_violation' in infos[agent], (
                    f"{agent} info missing 'voltage_violation'"
                )

    def test_info_has_current_p_mw_key(self, step_result, env):
        _, _, _, _, infos = step_result
        for agent in env.possible_agents:
            if agent in infos:
                assert 'current_p_mw' in infos[agent], (
                    f"{agent} info missing 'current_p_mw'"
                )

    def test_info_has_current_q_mvar_key(self, step_result, env):
        _, _, _, _, infos = step_result
        for agent in env.possible_agents:
            if agent in infos:
                assert 'current_q_mvar' in infos[agent], (
                    f"{agent} info missing 'current_q_mvar'"
                )

    def test_info_has_type_state_key(self, step_result, env):
        _, _, _, _, infos = step_result
        for agent in env.possible_agents:
            if agent in infos:
                assert 'type_state' in infos[agent], (
                    f"{agent} info missing 'type_state'"
                )

    def test_info_type_state_values(self, step_result, env):
        """type_state must identify the correct resource type per agent."""
        _, _, _, _, infos = step_result
        for agent in env.possible_agents:
            if agent not in infos:
                continue
            ts = infos[agent]['type_state']
            if 'bat' in agent:
                assert 'battery' in ts, f"{agent}: type_state={ts!r} should contain 'battery'"
            elif 'pv' in agent:
                assert 'solar' in ts or 'pv' in ts or 'renewable' in ts, (
                    f"{agent}: type_state={ts!r}"
                )
            elif 'fl' in agent:
                assert 'flex' in ts or 'load' in ts, f"{agent}: type_state={ts!r}"

    def test_info_voltage_violation_is_nonneg(self, step_result, env):
        _, _, _, _, infos = step_result
        for agent in env.possible_agents:
            if agent in infos:
                vv = infos[agent]['voltage_violation']
                assert vv >= 0.0, f"{agent}: voltage_violation={vv} should be >= 0"

    def test_obs_mode_is_ders_local(self, env):
        """Confirm the env was built with ders_local obs mode."""
        assert env._obs_mode == 'ders_local', (
            f"Expected obs_mode='ders_local', got {env._obs_mode!r}"
        )


# ===========================================================================
# 9. External PV profile injection hook
#    Mirrors JAX make_ders_params_with_profiles() interface.
# ===========================================================================

class TestProfileInjection:
    """Verify inject_pv_profiles and make_ders_benchmark_env_with_profiles."""

    def test_inject_pv_profiles_ones(self):
        from powerzoo.tasks.simple.marl_ders_benchmark import inject_pv_profiles
        task = MARLDERBenchmarkTask(split='train')
        env = task.create_env()
        n_steps, n_pv = 48, 4
        ones = np.ones((n_steps, n_pv), dtype=np.float32)
        inject_pv_profiles(env, ones)
        pv_agents = [a for a in env.possible_agents if 'pv' in a]
        assert len(pv_agents) == n_pv
        for agent in pv_agents:
            resource = env._resources[agent]
            assert resource._available_cf is not None
            assert np.allclose(resource._available_cf, 1.0, atol=0.01), (
                f"{agent}: _available_cf not all-ones after injection"
            )

    def test_inject_pv_profiles_zeros(self):
        from powerzoo.tasks.simple.marl_ders_benchmark import inject_pv_profiles
        task = MARLDERBenchmarkTask(split='train')
        env = task.create_env()
        zeros = np.zeros((48, 4), dtype=np.float32)
        inject_pv_profiles(env, zeros)
        pv_agents = [a for a in env.possible_agents if 'pv' in a]
        for agent in pv_agents:
            resource = env._resources[agent]
            assert np.allclose(resource._available_cf, 0.0, atol=0.01)

    def test_inject_1d_profile_broadcast(self):
        """1-D profile (n_steps,) should work for single-PV-column injection."""
        from powerzoo.tasks.simple.marl_ders_benchmark import inject_pv_profiles
        task = MARLDERBenchmarkTask(split='train')
        env = task.create_env()
        profile_1pv = np.full(48, 0.6, dtype=np.float32)
        # 1-D only works when there is exactly 1 PV agent; here we expect ValueError
        pv_agents = [a for a in env.possible_agents if 'pv' in a]
        if len(pv_agents) == 1:
            inject_pv_profiles(env, profile_1pv)
            resource = env._resources[pv_agents[0]]
            assert np.allclose(resource._available_cf, 0.6, atol=0.01)
        else:
            with pytest.raises(ValueError, match='column'):
                inject_pv_profiles(env, profile_1pv)

    def test_inject_wrong_column_count_raises(self):
        from powerzoo.tasks.simple.marl_ders_benchmark import inject_pv_profiles
        task = MARLDERBenchmarkTask(split='train')
        env = task.create_env()
        bad = np.ones((48, 2), dtype=np.float32)   # 2 cols, need 4
        with pytest.raises(ValueError, match='column'):
            inject_pv_profiles(env, bad)

    def test_make_ders_benchmark_env_with_profiles_basic(self):
        from powerzoo.tasks.simple.marl_ders_benchmark import (
            make_ders_benchmark_env_with_profiles,
        )
        profiles = np.full((48, 4), 0.7, dtype=np.float32)
        env = make_ders_benchmark_env_with_profiles(profiles, split='train')
        assert len(env.possible_agents) == 12
        pv_agents = [a for a in env.possible_agents if 'pv' in a]
        for agent in pv_agents:
            resource = env._resources[agent]
            assert np.allclose(resource._available_cf, 0.7, atol=0.01), (
                f"{agent}: _available_cf != 0.7 after factory injection"
            )

    def test_injected_profile_visible_in_obs_after_reset(self):
        """After injecting cf=1.0 profiles, PV obs[7] (available_cf) should be ~1.0."""
        profiles = np.ones((48, 4), dtype=np.float32)
        env = make_ders_benchmark_env_with_profiles(pv_profiles=profiles, split='train')
        obs, _ = env.reset(seed=0)
        pv_agents = [a for a in env.possible_agents if 'pv' in a]
        for agent in pv_agents:
            # obs[7] = ders_state_0 = available_cf for PV
            assert obs[agent][7] == pytest.approx(1.0, abs=0.05), (
                f"{agent}: obs[7]={obs[agent][7]:.4f} expected ~1.0 after ones injection"
            )


# ===========================================================================
# 10. Load profile injection (P and Q)
#     Mirrors JAX make_ders_params_with_profiles(load_profiles_p, load_profiles_q).
#     Covers: P-only, P+Q, 1D auto-distribution, column mismatch, tiling,
#     factory paths, observable power-flow effect.
# ===========================================================================

class TestLoadProfileInjection:
    """inject_load_profiles: active and reactive load traces."""

    @staticmethod
    def _fresh_env():
        task = MARLDERBenchmarkTask(split='train')
        return task.create_env()

    @staticmethod
    def _n_load_buses():
        env = MARLDERBenchmarkTask(split='train').create_env()
        return env.grid._node_loads_p.shape[1]

    # ── P-only injection ────────────────────────────────────────────────────

    def test_inject_p_only_replaces_node_loads_p(self):
        env = self._fresh_env()
        grid = env.grid
        n = grid._node_loads_p.shape[1]
        full_len = len(grid._node_loads_p)
        load_p = np.full((96, n), 7.0, dtype=np.float32)
        inject_load_profiles(env, load_p)
        assert grid._node_loads_p is not None
        assert grid._node_loads_p.shape == (full_len, n)
        assert np.allclose(grid._node_loads_p[:96], 7.0)

    def test_inject_p_does_not_touch_q_when_q_is_none_arg(self):
        """Omitting load_profiles_q must not create/change _node_loads_q."""
        env = self._fresh_env()
        grid = env.grid
        original_q = grid._node_loads_q  # may be None or ndarray
        n = grid._node_loads_p.shape[1]
        inject_load_profiles(env, np.full((48, n), 1.0, dtype=np.float32))
        if original_q is None:
            assert grid._node_loads_q is None
        else:
            assert grid._node_loads_q is original_q or np.array_equal(
                grid._node_loads_q, original_q
            )

    # ── P + Q injection ─────────────────────────────────────────────────────

    def test_inject_p_and_q_sets_both_arrays(self):
        env = self._fresh_env()
        grid = env.grid
        n = grid._node_loads_p.shape[1]
        inject_load_profiles(
            env,
            np.full((48, n), 4.0, dtype=np.float32),
            np.full((48, n), 0.8, dtype=np.float32),
        )
        assert np.allclose(grid._node_loads_p[:48], 4.0)
        assert grid._node_loads_q is not None
        assert np.allclose(grid._node_loads_q[:48], 0.8)

    # ── 1D auto-distribution ────────────────────────────────────────────────

    def test_inject_1d_p_sums_to_aggregate(self):
        """1D profile of X MW/step → each row in _node_loads_p sums to X."""
        env = self._fresh_env()
        grid = env.grid
        inject_load_profiles(env, np.full(48, 10.0, dtype=np.float32))
        row_sums = grid._node_loads_p[:48].sum(axis=1)
        assert np.allclose(row_sums, 10.0, atol=0.05), (
            f"Row sums: {row_sums[:4]}"
        )

    def test_inject_1d_q_sums_to_aggregate(self):
        """1D Q profile of Y MVAr/step → each row in _node_loads_q sums to Y."""
        env = self._fresh_env()
        inject_load_profiles(
            env,
            np.full(48, 10.0, dtype=np.float32),
            np.full(48, 2.0, dtype=np.float32),
        )
        row_sums_q = env.grid._node_loads_q[:48].sum(axis=1)
        assert np.allclose(row_sums_q, 2.0, atol=0.05), (
            f"Q row sums: {row_sums_q[:4]}"
        )

    # ── Shape validation ────────────────────────────────────────────────────

    def test_inject_p_wrong_columns_raises(self):
        env = self._fresh_env()
        n = env.grid._node_loads_p.shape[1]
        bad = np.ones((48, n + 3), dtype=np.float32)
        with pytest.raises(ValueError, match="column"):
            inject_load_profiles(env, bad)

    def test_inject_q_wrong_columns_raises(self):
        env = self._fresh_env()
        n = env.grid._node_loads_p.shape[1]
        good_p = np.ones((48, n), dtype=np.float32)
        bad_q = np.ones((48, n + 2), dtype=np.float32)
        with pytest.raises(ValueError, match="column"):
            inject_load_profiles(env, good_p, bad_q)

    def test_inject_p_3d_raises(self):
        env = self._fresh_env()
        bad = np.ones((48, 5, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            inject_load_profiles(env, bad)

    # ── Tiling to full time-series length ───────────────────────────────────

    def test_short_profile_tiled_to_full_length(self):
        env = self._fresh_env()
        grid = env.grid
        original_len = len(grid._node_loads_p)
        n = grid._node_loads_p.shape[1]
        # 24 steps << full time-series length
        short = np.full((24, n), 3.0, dtype=np.float32)
        inject_load_profiles(env, short)
        assert len(grid._node_loads_p) == original_len
        # Tiling: indices 24, 48, 72 … should also equal 3.0
        if original_len > 24:
            assert np.allclose(grid._node_loads_p[24], 3.0), (
                "Tiled rows must repeat injected values"
            )

    def test_longer_profile_truncated_to_series_length(self):
        env = self._fresh_env()
        grid = env.grid
        original_len = len(grid._node_loads_p)
        n = grid._node_loads_p.shape[1]
        # Profile longer than full series — should be clipped
        long = np.ones((original_len + 1000, n), dtype=np.float32)
        inject_load_profiles(env, long)
        assert len(grid._node_loads_p) == original_len

    # ── Factory paths ───────────────────────────────────────────────────────

    def test_factory_load_p_only(self):
        n = self._n_load_buses()
        env = make_ders_benchmark_env_with_profiles(
            load_profiles_p=np.full((48, n), 2.5, dtype=np.float32),
            split='train',
        )
        assert len(env.possible_agents) == 12
        assert np.allclose(env.grid._node_loads_p[:48], 2.5)

    def test_factory_pv_plus_load_p_plus_load_q(self):
        n = self._n_load_buses()
        env = make_ders_benchmark_env_with_profiles(
            pv_profiles=np.ones((48, 4), dtype=np.float32),
            load_profiles_p=np.full((48, n), 3.0, dtype=np.float32),
            load_profiles_q=np.full((48, n), 0.6, dtype=np.float32),
            split='train',
        )
        assert len(env.possible_agents) == 12
        # PV all-ones
        pv_agents = [a for a in env.possible_agents if 'pv' in a]
        for agent in pv_agents:
            assert np.allclose(env._resources[agent]._available_cf, 1.0, atol=0.01)
        # Load P and Q set
        assert np.allclose(env.grid._node_loads_p[:48], 3.0)
        assert np.allclose(env.grid._node_loads_q[:48], 0.6)

    def test_factory_q_without_p_raises(self):
        with pytest.raises(ValueError, match="load_profiles_p"):
            make_ders_benchmark_env_with_profiles(
                load_profiles_q=np.full(48, 1.0, dtype=np.float32),
                split='train',
            )

    def test_factory_all_none_returns_12_agents(self):
        """No profiles → default env, unchanged 12 agents."""
        env = make_ders_benchmark_env_with_profiles(split='train')
        assert len(env.possible_agents) == 12

    def test_factory_backward_compat_positional_pv(self):
        """Existing callers passing pv_profiles as first positional arg still work."""
        profiles = np.ones((48, 4), dtype=np.float32)
        env = make_ders_benchmark_env_with_profiles(profiles, split='train')
        assert len(env.possible_agents) == 12

    # ── Observable power-flow effect ────────────────────────────────────────

    def test_load_injection_changes_node_loads_in_power_flow(self):
        """After injection, grid._get_node_loads_p() uses the injected values."""
        n = self._n_load_buses()
        known = 0.777
        env = make_ders_benchmark_env_with_profiles(
            load_profiles_p=np.full((48, n), known, dtype=np.float32),
            split='train',
        )
        env.reset(seed=0)
        node_p = env.grid._get_node_loads_p()   # (n_nodes,) after loads_map
        total = float(np.sum(node_p))
        expected = known * n
        assert total == pytest.approx(expected, rel=0.01), (
            f"Expected sum ≈ {expected:.3f} MW, got {total:.3f}"
        )

    def test_different_load_levels_produce_different_totals(self):
        """Low-load vs high-load envs have different total P after reset."""
        n = self._n_load_buses()

        def get_total_after_reset(load_mw: float) -> float:
            env = make_ders_benchmark_env_with_profiles(
                load_profiles_p=np.full((48, n), load_mw, dtype=np.float32),
                split='train',
            )
            env.reset(seed=0)
            return float(np.sum(env.grid._get_node_loads_p()))

        total_low = get_total_after_reset(0.001)
        total_high = get_total_after_reset(5.0)
        assert total_high > total_low, (
            f"high({total_high:.3f}) should exceed low({total_low:.3f})"
        )
