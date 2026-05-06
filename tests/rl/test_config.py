"""Tests for RLConfig dataclass — parsing, YAML round-trip, validation."""

import io
import os
import tempfile
import warnings

import pytest

from powerzoo.rl.config import RLConfig


# ── from_dict ──────────────────────────────────────────────────────────────

class TestRLConfigFromDict:
    def test_task_name_flat(self):
        cfg = RLConfig.from_dict({'task_name': 'battery_arbitrage'})
        assert cfg.task_name == 'battery_arbitrage'
        assert cfg.task_config is None

    def test_task_section_name(self):
        cfg = RLConfig.from_dict({'task': {'name': 'marl_opf', 'split': 'val'}})
        assert cfg.task_name == 'marl_opf'
        assert cfg.split == 'val'

    def test_wrappers_section(self):
        cfg = RLConfig.from_dict({
            'task': {'name': 'battery_arbitrage'},
            'wrappers': {'normalize': True, 'forecast_horizon': 6, 'safe_rl': True},
        })
        assert cfg.normalize is True
        assert cfg.forecast_horizon == 6
        assert cfg.safe_rl is True

    def test_trainer_section(self):
        cfg = RLConfig.from_dict({
            'task': {'name': 'battery_arbitrage'},
            'trainer': {
                'algorithm': 'PPO',
                'total_timesteps': 50_000,
                'hyperparams': {'learning_rate': 0.001},
            },
        })
        assert cfg.algorithm == 'PPO'
        assert cfg.total_timesteps == 50_000
        assert cfg.hyperparams == {'learning_rate': 0.001}

    def test_reward_section(self):
        cfg = RLConfig.from_dict({
            'task': {'name': 'battery_arbitrage'},
            'reward': {'type': 'lmp_arbitrage', 'profit_weight': 1.5},
        })
        assert cfg.reward == {'type': 'lmp_arbitrage', 'profit_weight': 1.5}

    def test_seed_and_framework(self):
        cfg = RLConfig.from_dict({
            'task': {'name': 'marl_opf'},
            'seed': 99,
            'framework': 'pettingzoo',
        })
        assert cfg.seed == 99
        assert cfg.framework == 'pettingzoo'

    def test_inline_task_config(self):
        inline = {'grid': {'type': 'distribution', 'case': 'case33bw'}, 'resources': []}
        cfg = RLConfig.from_dict({'task': inline})
        assert cfg.task_config is not None
        assert 'grid' in cfg.task_config

    def test_empty_dict_gives_defaults(self):
        """No task → no error from from_dict; validation catches the missing task."""
        cfg = RLConfig.from_dict({})
        assert cfg.task_name is None
        assert cfg.algorithm == 'SAC'


# ── to_dict / round-trip ───────────────────────────────────────────────────

class TestRLConfigRoundTrip:
    def test_dict_roundtrip(self):
        original = RLConfig(
            task_name='battery_arbitrage',
            algorithm='PPO',
            total_timesteps=20_000,
            normalize=True,
            seed=7,
        )
        restored = RLConfig.from_dict(original.to_dict())
        assert restored.task_name == original.task_name
        assert restored.algorithm == original.algorithm
        assert restored.total_timesteps == original.total_timesteps
        assert restored.normalize == original.normalize
        assert restored.seed == original.seed

    def test_yaml_roundtrip(self, tmp_path):
        import yaml
        original = RLConfig(
            task_name='marl_opf',
            algorithm='SAC',
            split='val',
            reward={'type': 'economic_dispatch'},
        )
        yaml_path = tmp_path / 'cfg.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(original.to_dict(), f)
        restored = RLConfig.from_yaml(yaml_path)
        assert restored.task_name == 'marl_opf'
        assert restored.split == 'val'
        assert restored.reward == {'type': 'economic_dispatch'}


# ── validate ──────────────────────────────────────────────────────────────

class TestRLConfigValidate:
    def test_missing_task_raises(self):
        cfg = RLConfig()
        with pytest.raises(ValueError, match="task_name.*task_config"):
            cfg.validate()

    def test_both_task_raises(self):
        cfg = RLConfig(task_name='battery_arbitrage', task_config={'grid': {}})
        with pytest.raises(ValueError, match="not both"):
            cfg.validate()

    def test_invalid_algorithm_raises(self):
        cfg = RLConfig(task_name='battery_arbitrage', algorithm='DDPG')
        with pytest.raises(ValueError, match="DDPG"):
            cfg.validate()

    def test_invalid_framework_raises(self):
        cfg = RLConfig(task_name='battery_arbitrage', framework='myfancyframework')
        with pytest.raises(ValueError, match="myfancyframework"):
            cfg.validate()

    def test_invalid_split_raises(self):
        cfg = RLConfig(task_name='battery_arbitrage', split='holdout')
        with pytest.raises(ValueError, match="holdout"):
            cfg.validate()

    def test_valid_config_passes(self):
        cfg = RLConfig(task_name='battery_arbitrage', algorithm='PPO', split='test')
        cfg.validate()  # should not raise

    def test_safe_rl_without_threshold_warns(self):
        cfg = RLConfig(task_name='battery_arbitrage', safe_rl=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cfg.validate()
        assert any('cost_threshold' in str(x.message) for x in w)
