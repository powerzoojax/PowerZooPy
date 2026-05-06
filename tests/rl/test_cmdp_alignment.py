"""CMDP benchmark-alignment smoke tests."""

from __future__ import annotations

import numpy as np

from powerzoo.benchmarks.evaluation import evaluate
from powerzoo.tasks.dso_task import make_dso_env
from powerzoo.tasks.public import get_public_task_info
from powerzoo.tasks.registry import make_task
from powerzoo.wrappers.safe_rl_wrapper import CMDPWrapper, GymnasiumSafeWrapper


class _ZeroPolicy:
    def __init__(self, action):
        self._action = np.asarray(action, dtype=np.float32)

    def act(self, obs, info):
        return self._action


def test_dso_safe_wrapper_projects_selected_scalar_cost():
    env = GymnasiumSafeWrapper(make_dso_env())
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(12, dtype=np.float32))
    assert tuple(info['selected_constraint_names']) == ('voltage_violation',)
    assert info['cost'] == info['cost_voltage_violation']


def test_cmdp_wrapper_returns_selected_vector():
    env = CMDPWrapper(make_dso_env())
    env.reset(seed=0)
    _, _, costs, _, _, info = env.step(np.zeros(12, dtype=np.float32))
    assert costs.shape == (1,)
    assert tuple(info['selected_constraint_names']) == ('voltage_violation',)
    assert float(costs[0]) == info['cost_voltage_violation']


def test_evaluate_returns_per_constraint_metrics():
    task = make_task('dc_microgrid_safe')
    env = task.create_env()
    policy = _ZeroPolicy(np.array([0.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float32))
    result = evaluate(policy, env, n_episodes=2, seed_start=0)
    assert result['constraint_names'] == ['sla', 'overtemp', 'power_deficit']
    assert set(result['mean_episode_cost_by_constraint']) == {'sla', 'overtemp', 'power_deficit'}
    assert 'mean_episode_cost' in result
    assert 'episode_costs' in result


def test_public_task_info_exposes_constraint_metadata():
    card = get_public_task_info('comparison_tso_centralized')
    assert card['constraint_names'] == ['thermal_overload', 'reserve_shortfall']
    assert card['cost_thresholds'] == [0.0, 5.0]
    assert card['training_contract'] == 'cmdp_env_plus_scalar_safe_projection'
