"""Tests for info() and describe() — output structure and key coverage."""

import pytest

from powerzoo.rl.describe import info, describe


class TestInfo:
    def test_returns_dict(self):
        d = info('battery_arbitrage')
        assert isinstance(d, dict)

    def test_required_keys(self):
        d = info('battery_arbitrage')
        for key in ('task_id', 'description', 'agent_mode', 'difficulty',
                    'reward', 'cost', 'config_template'):
            assert key in d, f"Missing key: {key}"

    def test_task_id_matches(self):
        d = info('battery_arbitrage')
        assert d['task_id'] == 'battery_arbitrage'

    def test_agent_mode_single(self):
        d = info('battery_arbitrage')
        assert d['agent_mode'] == 'single'

    def test_agent_mode_multi(self):
        d = info('marl_opf')
        assert d['agent_mode'] == 'multi'

    def test_reward_default_key(self):
        d = info('battery_arbitrage')
        assert 'default' in d['reward']
        assert 'available' in d['reward']
        assert isinstance(d['reward']['available'], list)
        assert len(d['reward']['available']) > 0

    def test_cost_structure(self):
        d = info('battery_arbitrage')
        assert 'has_cmdp' in d['cost']
        assert 'threshold' in d['cost']
        assert 'constraint_names' in d['cost']
        assert 'training_contract' in d['cost']

    def test_vector_cmdp_metadata_exposed(self):
        d = info('dc_microgrid_safe')
        assert d['cost']['constraint_names'] == ['sla', 'overtemp', 'power_deficit']
        assert d['cost']['thresholds'] == [0.2, 0.15, 0.15]
        assert d['cost']['threshold'] == pytest.approx(0.5)

    def test_observation_space_structure(self):
        d = info('battery_arbitrage')
        obs = d.get('observation_space')
        if obs is not None:
            assert 'type' in obs
            assert 'shape' in obs

    def test_config_template_structure(self):
        d = info('battery_arbitrage')
        tmpl = d['config_template']
        assert 'task' in tmpl
        assert 'trainer' in tmpl

    def test_json_format(self):
        result = info('battery_arbitrage', format='json')
        assert isinstance(result, str)
        import json
        parsed = json.loads(result)
        assert 'task_id' in parsed

    def test_task_instance_input(self):
        from powerzoo.tasks.registry import make_task
        task = make_task('battery_arbitrage')
        d = info(task)
        assert isinstance(d, dict)
        assert d['agent_mode'] == 'single'

    def test_dict_config_input(self):
        config = {
            'name': 'custom',
            'grid': {'type': 'distribution', 'case': 'case33bw'},
            'resources': [],
        }
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            d = info(config)
        assert isinstance(d, dict)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            info(12345)


class TestDescribe:
    def test_returns_string(self):
        result = describe('battery_arbitrage')
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_task_id(self):
        result = describe('battery_arbitrage')
        assert 'battery_arbitrage' in result

    def test_contains_agent_mode(self):
        result = describe('marl_opf')
        assert 'multi' in result.lower()

    def test_multi_line(self):
        result = describe('battery_arbitrage')
        assert '\n' in result
