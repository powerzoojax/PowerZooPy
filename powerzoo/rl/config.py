"""RLConfig — unified experiment configuration for PowerZoo RL tasks.

Covers task selection (by name or inline dict), optional wrapper stack,
reward override, trainer hyperparameters, and framework choice.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union


_VALID_ALGORITHMS = ('SAC', 'PPO', 'TD3')


@dataclass
class RLConfig:
    """Unified experiment configuration.

    One of ``task_name`` or ``task_config`` must be set, but not both.

    Example (Python)::

        cfg = RLConfig(task_name='battery_arbitrage', algorithm='SAC',
                       total_timesteps=200_000, save_path='./results/')

    Example (YAML)::

        task:
          name: battery_arbitrage
          split: train
        wrappers:
          normalize: false
        trainer:
          algorithm: SAC
          total_timesteps: 200000
          save_path: ./results/

    """

    # ── Task (exactly one must be set) ────────────────────────────────────
    task_name: Optional[str] = None          # e.g. 'battery_arbitrage'
    task_config: Optional[Dict] = None       # inline anonymous task dict

    # ── Wrappers ──────────────────────────────────────────────────────────
    normalize: bool = False
    flatten: bool = True
    forecast_horizon: int = 0
    safe_rl: bool = False
    cost_threshold: Optional[float] = None

    # ── Reward override ───────────────────────────────────────────────────
    reward: Optional[Dict] = None                # e.g. {'type': 'lmp_arbitrage'}
    custom_reward_fn: Optional[Callable] = None

    # ── Trainer ───────────────────────────────────────────────────────────
    algorithm: str = 'SAC'
    total_timesteps: int = 100_000
    policy: str = 'MlpPolicy'
    hyperparams: Dict = field(default_factory=dict)
    eval_episodes: int = 10
    save_path: Optional[str] = None

    # ── Framework ─────────────────────────────────────────────────────────
    framework: str = 'auto'
    split: str = 'train'
    seed: Optional[int] = None

    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'RLConfig':
        """Load an RLConfig from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            RLConfig instance.
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "pyyaml is required for YAML config loading. "
                "It should be included in the standard powerzoo install."
            ) from e

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data or {})

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RLConfig':
        """Build an RLConfig from a nested dict (e.g. from YAML parsing).

        Accepts the structured YAML schema::

            task:
              name: battery_arbitrage   # OR inline grid/resources/...
              split: train
            wrappers:
              normalize: false
              safe_rl: false
              forecast_horizon: 0
            reward:                     # optional override
              type: lmp_arbitrage
            trainer:
              algorithm: SAC
              total_timesteps: 500000
              hyperparams: { learning_rate: 0.0003 }
              save_path: ./results/
            framework: auto
            seed: 42

        Also accepts a flat dict (all keys at the top level).
        """
        d = copy.deepcopy(d)
        kwargs: Dict[str, Any] = {}

        # ── task section ──
        task_section = d.pop('task', None)
        if task_section is not None:
            if isinstance(task_section, str):
                kwargs['task_name'] = task_section
            elif isinstance(task_section, dict):
                name = task_section.pop('name', None)
                split = task_section.pop('split', None)
                if name is not None:
                    kwargs['task_name'] = name
                elif task_section:
                    # remaining keys are an inline config
                    kwargs['task_config'] = task_section
                if split is not None:
                    kwargs['split'] = split
        else:
            if 'task_name' in d:
                kwargs['task_name'] = d.pop('task_name')
            if 'task_config' in d:
                kwargs['task_config'] = d.pop('task_config')

        # ── wrappers section ──
        wrapper_section = d.pop('wrappers', None)
        if wrapper_section:
            for key in ('normalize', 'flatten', 'forecast_horizon', 'safe_rl', 'cost_threshold'):
                if key in wrapper_section:
                    kwargs[key] = wrapper_section[key]
        for key in ('normalize', 'flatten', 'forecast_horizon', 'safe_rl', 'cost_threshold'):
            if key in d:
                kwargs[key] = d.pop(key)

        # ── reward section ──
        reward_section = d.pop('reward', None)
        if reward_section is not None:
            kwargs['reward'] = reward_section
        elif 'reward' in d:
            kwargs['reward'] = d.pop('reward')

        # ── trainer section ──
        trainer_section = d.pop('trainer', None)
        if trainer_section:
            for key in ('algorithm', 'total_timesteps', 'policy', 'hyperparams',
                        'eval_episodes', 'save_path'):
                if key in trainer_section:
                    kwargs[key] = trainer_section[key]
        for key in ('algorithm', 'total_timesteps', 'policy', 'hyperparams',
                    'eval_episodes', 'save_path'):
            if key in d:
                kwargs[key] = d.pop(key)

        # ── top-level keys ──
        for key in ('framework', 'split', 'seed'):
            if key in d:
                kwargs[key] = d.pop(key)

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this config to a nested dict suitable for YAML serialization."""
        result: Dict[str, Any] = {}

        # task section
        task: Dict[str, Any] = {}
        if self.task_name is not None:
            task['name'] = self.task_name
        elif self.task_config is not None:
            task.update(self.task_config)
        task['split'] = self.split
        result['task'] = task

        # wrappers section
        wrappers: Dict[str, Any] = {
            'normalize': self.normalize,
            'flatten': self.flatten,
            'forecast_horizon': self.forecast_horizon,
            'safe_rl': self.safe_rl,
        }
        if self.cost_threshold is not None:
            wrappers['cost_threshold'] = self.cost_threshold
        result['wrappers'] = wrappers

        # reward section
        if self.reward is not None:
            result['reward'] = self.reward

        # trainer section
        trainer: Dict[str, Any] = {
            'algorithm': self.algorithm,
            'total_timesteps': self.total_timesteps,
            'policy': self.policy,
            'eval_episodes': self.eval_episodes,
        }
        if self.hyperparams:
            trainer['hyperparams'] = self.hyperparams
        if self.save_path is not None:
            trainer['save_path'] = self.save_path
        result['trainer'] = trainer

        result['framework'] = self.framework
        if self.seed is not None:
            result['seed'] = self.seed

        return result

    def validate(self) -> None:
        """Validate the config, raising ValueError on invalid combinations.

        Raises:
            ValueError: If validation fails.
        """
        if self.task_name is not None and self.task_config is not None:
            raise ValueError(
                "Specify either 'task_name' or 'task_config', not both."
            )
        if self.task_name is None and self.task_config is None:
            raise ValueError(
                "One of 'task_name' or 'task_config' must be set."
            )
        if self.algorithm.upper() not in _VALID_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{self.algorithm}'. "
                f"Supported: {_VALID_ALGORITHMS}."
            )
        if self.forecast_horizon < 0:
            raise ValueError("forecast_horizon must be >= 0.")
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be > 0.")
        if self.eval_episodes <= 0:
            raise ValueError("eval_episodes must be > 0.")
        if self.framework not in ('auto', 'rllib', 'pettingzoo'):
            raise ValueError(
                f"Unknown framework '{self.framework}'. "
                "Supported: 'auto', 'rllib', 'pettingzoo'."
            )
        if self.split not in ('train', 'val', 'test'):
            raise ValueError(
                f"Unknown split '{self.split}'. Supported: 'train', 'val', 'test'."
            )
        if self.safe_rl and self.cost_threshold is None:
            import warnings
            warnings.warn(
                "safe_rl=True but cost_threshold is not set; "
                "GymnasiumSafeWrapper will use its default threshold (25.0).",
                UserWarning,
                stacklevel=2,
            )
