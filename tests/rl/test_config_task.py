"""Tests for ConfigTask and ConfigMultiAgentTask."""

import warnings

import pytest

from powerzoo.tasks.base import ConfigTask, ConfigMultiAgentTask


_SIMPLE_SINGLE_CONFIG = {
    'name': 'test_single',
    'grid': {'type': 'distribution', 'case': 'case33bw'},
    'resources': [
        {'type': 'battery', 'bus_id': 6, 'capacity_mwh': 1.0, 'charge_power_kw': 500}
    ],
    'reward': {'type': 'battery_arbitrage'},
    'episode': {'max_steps': 24},
}

_SIMPLE_MARL_CONFIG = {
    'name': 'test_marl',
    'grid': {'type': 'transmission', 'case': 'case5'},
    'resources': [],
    'agents': {
        'agent_type': 'unit',
        'reward_type': 'shared',
    },
}


class TestConfigTask:
    def test_instantiation(self):
        task = ConfigTask(_SIMPLE_SINGLE_CONFIG)
        assert task.agent_mode == 'single'
        assert task.name == '_config_task'

    def test_from_dict(self):
        task = ConfigTask.from_dict(_SIMPLE_SINGLE_CONFIG)
        assert isinstance(task, ConfigTask)

    def test_get_scenario_config(self):
        task = ConfigTask(_SIMPLE_SINGLE_CONFIG)
        sc = task.get_scenario_config()
        assert sc['name'] == 'test_single'
        assert sc['grid'] == _SIMPLE_SINGLE_CONFIG['grid']
        assert len(sc['resources']) == 1
        assert sc['reward'] == {'type': 'battery_arbitrage'}

    def test_get_agents_config(self):
        task = ConfigTask(_SIMPLE_SINGLE_CONFIG)
        ac = task.get_agents_config()
        assert ac['agent_type'] == 'single'

    def test_split_ignored_with_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            task = ConfigTask(_SIMPLE_SINGLE_CONFIG, split='train')
        assert any('split' in str(x.message).lower() for x in w)

    def test_constraint_tightness_ignored(self):
        # Should not raise even though ConfigTask has no presets
        task = ConfigTask(_SIMPLE_SINGLE_CONFIG, constraint_tightness='strict')
        assert isinstance(task, ConfigTask)

    def test_default_reward_zero(self):
        task = ConfigTask({'grid': {'type': 'distribution', 'case': 'case33bw'}})
        sc = task.get_scenario_config()
        assert sc['reward'] == {'type': 'zero'}


class TestConfigMultiAgentTask:
    def test_instantiation(self):
        task = ConfigMultiAgentTask(_SIMPLE_MARL_CONFIG)
        assert task.agent_mode == 'multi'
        assert task.name == '_config_marl_task'

    def test_from_dict(self):
        task = ConfigMultiAgentTask.from_dict(_SIMPLE_MARL_CONFIG)
        assert isinstance(task, ConfigMultiAgentTask)

    def test_explicit_agent_type_respected(self):
        task = ConfigMultiAgentTask(_SIMPLE_MARL_CONFIG)
        ac = task.get_agents_config()
        assert ac['agent_type'] == 'unit'

    def test_infer_agent_type_battery(self):
        cfg = {
            'grid': {'type': 'distribution', 'case': 'case33bw'},
            'resources': [{'type': 'battery'}, {'type': 'solar'}],
        }
        task = ConfigMultiAgentTask(cfg)
        assert task._infer_agent_type() == 'resource'

    def test_infer_agent_type_no_controllable(self):
        cfg = {
            'grid': {'type': 'transmission', 'case': 'case5'},
            'resources': [],
        }
        task = ConfigMultiAgentTask(cfg)
        assert task._infer_agent_type() == 'unit'

    def test_agents_config_defaults(self):
        cfg = {
            'grid': {'type': 'transmission', 'case': 'case5'},
            'resources': [],
        }
        task = ConfigMultiAgentTask(cfg)
        ac = task.get_agents_config()
        assert ac['reward_type'] == 'shared'
        assert ac['action_mode'] == 'score'

    def test_split_ignored_with_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ConfigMultiAgentTask(_SIMPLE_MARL_CONFIG, split='test')
        assert any('split' in str(x.message).lower() for x in w)
