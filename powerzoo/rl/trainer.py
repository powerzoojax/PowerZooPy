"""Trainer — thin SB3 wrapper for PowerZoo benchmark tasks.

Provides a high-level ``train()`` / ``evaluate()`` / ``save()`` / ``load()``
workflow on top of stable-baselines3.  The trainer is deliberately thin:
it delegates environment creation to :func:`~powerzoo.rl.env.make_env` and
evaluation to :func:`~powerzoo.benchmarks.evaluation.evaluate`.

SB3 is imported lazily inside ``__init__`` so that importing ``powerzoo.rl``
does **not** require stable-baselines3 to be installed.

Usage::

    from powerzoo.rl import Trainer

    # Single-agent
    t = Trainer('battery_arbitrage')
    t.train(total_timesteps=200_000)
    results = t.evaluate(split='test')
    t.save('./results/')

    # Independent-learners MARL (sequential SB3 .learn per agent)
    t = Trainer('marl_opf', framework='pettingzoo')
    t.train_il(total_timesteps=50_000)

    # Simultaneous MARL (one PettingZoo step, all agents update — SAC only)
    t = Trainer('marl_opf', framework='pettingzoo', algorithm='SAC')
    t.train_marl_simultaneous(total_timesteps=200_000)

    env = t.get_env()   # plug into EPyMARL / MAPPO / custom loops

"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import gymnasium

logger = logging.getLogger(__name__)


class Trainer:
    """High-level SB3 trainer for PowerZoo tasks.

    Args:
        config: Task name, inline config dict, YAML path, or
                :class:`~powerzoo.rl.config.RLConfig` instance.
        **kw:   Keyword overrides merged into the resolved
                :class:`~powerzoo.rl.config.RLConfig` (e.g.
                ``algorithm='PPO'``, ``total_timesteps=500_000``).

    """

    ALGORITHMS: Dict[str, Any] = {}   # populated lazily

    def __init__(
        self,
        config: Union[str, Dict[str, Any], 'RLConfig', Path],
        **kw,
    ):
        # ── resolve config ─────────────────────────────────────────────────
        from powerzoo.rl.config import RLConfig
        from powerzoo.rl.env import make_env

        if isinstance(config, RLConfig):
            cfg = config
        elif isinstance(config, str) and (
            config.endswith('.yaml') or config.endswith('.yml')
        ):
            cfg = RLConfig.from_yaml(config)
        elif isinstance(config, dict) and 'algorithm' in config:
            cfg = RLConfig.from_dict(config)
        elif isinstance(config, (str, dict)):
            # bare task name or inline task config — wrap in RLConfig
            if isinstance(config, str):
                cfg = RLConfig(task_name=config)
            else:
                cfg = RLConfig(task_config=config)
        else:
            cfg = RLConfig.from_yaml(str(config))

        # apply keyword overrides
        for k, v in kw.items():
            if hasattr(cfg, k):
                object.__setattr__(cfg, k, v)
            else:
                logger.warning("Trainer: unknown config key '%s' ignored.", k)

        cfg.validate()
        self._cfg = cfg
        self._model = None
        self._il_models: Optional[Dict[str, Any]] = None

        # ── import SB3 algorithms ─────────────────────────────────────────
        try:
            from stable_baselines3 import SAC, PPO, TD3
            Trainer.ALGORITHMS = {'SAC': SAC, 'PPO': PPO, 'TD3': TD3}
        except ImportError as e:
            raise ImportError(
                "stable-baselines3 is required for Trainer. "
                "Install it with: pip install 'powerzoo[rl]'"
            ) from e

        # ── (P1-β) Optionally register SBX algorithms (JAX-backed SB3 API) ─
        # SBX is API-compatible with SB3, so we just add SBX_PPO / SBX_SAC /
        # SBX_TD3 keys alongside the SB3 trio. Users select SBX by passing
        # ``algorithm='SBX_PPO'`` (etc) instead of ``'PPO'``. If sbx is not
        # installed, this block is silently skipped — Trainer keeps working
        # with SB3 only. Used by benchmarks/common/powerzoo_driver.py for
        # cross-backend speed/convergence comparisons.
        try:
            from sbx.ppo import PPO as _SBX_PPO
            from sbx.sac import SAC as _SBX_SAC
            from sbx.td3 import TD3 as _SBX_TD3
            Trainer.ALGORITHMS.update({
                'SBX_PPO': _SBX_PPO,
                'SBX_SAC': _SBX_SAC,
                'SBX_TD3': _SBX_TD3,
            })
        except ImportError:
            try:
                import sbx as _sbx
                Trainer.ALGORITHMS.update({
                    'SBX_PPO': _sbx.PPO,
                    'SBX_SAC': _sbx.SAC,
                    'SBX_TD3': _sbx.TD3,
                })
            except (ImportError, AttributeError):
                # sbx not installed or API unknown; SB3-only mode still works.
                pass

    # ── Environment helpers ──────────────────────────────────────────────

    def get_env(self, split: Optional[str] = None) -> Any:
        """Create and return the configured environment.

        Args:
            split: Override split (``'train'``, ``'val'``, ``'test'``).
                   Defaults to the split in the config.

        Returns:
            A Gymnasium or PettingZoo env, ready for interaction.
        """
        from powerzoo.rl.env import make_env
        cfg = self._cfg
        return make_env(
            cfg.task_name or cfg.task_config,
            split=split or cfg.split,
            framework=cfg.framework,
            reward=cfg.reward or cfg.custom_reward_fn,
            normalize=cfg.normalize,
            forecast_horizon=cfg.forecast_horizon,
            safe_rl=cfg.safe_rl,
            cost_threshold=cfg.cost_threshold,
            seed=cfg.seed,
        )

    # ── Training ─────────────────────────────────────────────────────────

    def train(
        self,
        total_timesteps: Optional[int] = None,
        progress_bar: bool = True,
        callback: Optional[Any] = None,
    ) -> 'Trainer':
        """Train a single-agent SB3 model.

        Args:
            total_timesteps: Override the config value.
            progress_bar:    Show a tqdm progress bar (requires ``tqdm``).
            callback:        Optional SB3 callback.

        Returns:
            ``self`` for chaining.

        Raises:
            TypeError: If the configured task is multi-agent.
        """
        from powerzoo.tasks.registry import make_task
        from powerzoo.rl.config import RLConfig

        cfg = self._cfg
        task_source = cfg.task_name or cfg.task_config
        task = make_task(task_source, split=cfg.split)  # type: ignore[arg-type]

        if task.agent_mode != 'single':
            raise TypeError(
                f"Trainer.train() is for single-agent tasks only. "
                f"Task '{task.name}' has agent_mode='{task.agent_mode}'. "
                f"For MARL, use Trainer.train_il() (sequential IL), "
                f"Trainer.train_marl_simultaneous() (joint-step SAC), or "
                f"get_env() and plug into EPyMARL / MAPPO / your preferred MARL framework."
            )

        env = self.get_env(split=cfg.split)
        algo_cls = Trainer.ALGORITHMS.get(cfg.algorithm.upper())
        if algo_cls is None:
            raise ValueError(
                f"Unknown algorithm '{cfg.algorithm}'. "
                f"Supported: {list(Trainer.ALGORITHMS.keys())}."
            )

        self._model = algo_cls(
            cfg.policy,
            env,
            verbose=1,
            seed=cfg.seed,
            **cfg.hyperparams,
        )
        ts = total_timesteps or cfg.total_timesteps
        self._model.learn(
            total_timesteps=ts,
            progress_bar=progress_bar,
            callback=callback,
        )
        return self

    def train_il(
        self,
        total_timesteps: Optional[int] = None,
        progress_bar: bool = True,
    ) -> 'Trainer':
        """Train independent-learner SB3 models for a MARL task.

        One SB3 model is created per agent.  During training for agent *A*
        all other agents act randomly (default SB3 policy before training).

        Supports **homogeneous** agent tasks (all agents share the same
        observation / action space).  For heterogeneous tasks, use
        :meth:`get_env` and implement your own training loop.

        Args:
            total_timesteps: Timesteps per agent (override config value).
            progress_bar:    Show a tqdm progress bar.

        Returns:
            ``self`` for chaining.

        Raises:
            ImportError: If pettingzoo is not installed.
            TypeError:   If the task is single-agent.
        """
        try:
            from pettingzoo import ParallelEnv
        except ImportError as e:
            raise ImportError(
                "pettingzoo is required for train_il(). "
                "Install with: pip install pettingzoo"
            ) from e

        from powerzoo.tasks.registry import make_task

        cfg = self._cfg
        task_source = cfg.task_name or cfg.task_config
        task = make_task(task_source, split=cfg.split)  # type: ignore[arg-type]

        if task.agent_mode != 'multi':
            raise TypeError(
                f"Trainer.train_il() is for multi-agent tasks. "
                f"Task '{task.name}' has agent_mode='{task.agent_mode}'. "
                f"Use Trainer.train() for single-agent tasks."
            )

        marl_env = self.get_env(split=cfg.split)
        marl_env.reset(seed=cfg.seed)

        agent_ids = list(marl_env.possible_agents)
        if not agent_ids:
            raise ValueError(
                "Could not determine agent IDs from the environment. "
                "Make sure framework='pettingzoo' for train_il()."
            )

        # check homogeneous spaces
        obs_spaces = [marl_env.observation_space(a) for a in agent_ids]
        act_spaces = [marl_env.action_space(a) for a in agent_ids]
        heterogeneous = (
            len({str(s) for s in obs_spaces}) > 1
            or len({str(s) for s in act_spaces}) > 1
        )
        if heterogeneous:
            logger.warning(
                "train_il(): heterogeneous agent spaces detected. "
                "Each agent gets its own SB3 model, but independent random "
                "exploration from other agents may not be meaningful. "
                "Consider a custom training loop via get_env()."
            )

        algo_cls = Trainer.ALGORITHMS.get(cfg.algorithm.upper())
        if algo_cls is None:
            raise ValueError(f"Unknown algorithm '{cfg.algorithm}'.")

        ts = total_timesteps or cfg.total_timesteps
        self._il_models = {}
        n_agents = len(agent_ids)

        for idx, agent_id in enumerate(agent_ids):
            logger.info(
                "train_il(): training agent %d/%d '%s'",
                idx + 1,
                n_agents,
                agent_id,
            )
            print(
                f"train_il(): agent {idx + 1}/{n_agents} '{agent_id}' "
                f"({ts:,} timesteps, others explore randomly) ...",
                flush=True,
            )
            agent_env = _AgentEnvWrapper(marl_env, agent_id)
            model = algo_cls(
                cfg.policy,
                agent_env,
                verbose=0,
                seed=cfg.seed,
                **cfg.hyperparams,
            )
            model.learn(total_timesteps=ts, progress_bar=progress_bar)
            self._il_models[agent_id] = model
            print(
                f"train_il(): agent {idx + 1}/{n_agents} '{agent_id}' done.",
                flush=True,
            )

        return self

    def train_marl_simultaneous(
        self,
        total_timesteps: Optional[int] = None,
        progress_bar: bool = True,
    ) -> 'Trainer':
        """Train homogeneous MARL with **simultaneous** joint env steps (SAC only).

        Uses the same ``make_env(..., framework='pettingzoo')`` config as
        :meth:`train_il`, but advances the PettingZoo env **once** per
        iteration while **all** agents act with their policies and each
        receives a transition in its own replay buffer (independent learners
        with co-adaptation — not MAPPO / centralized critic).

        ``total_timesteps`` counts **joint** environment steps (not per-agent
        multiples).  For the same wall time as ``train_il`` with
        ``T`` steps per agent and ``N`` agents, use roughly ``T`` here instead
        of ``T * N``.

        Args:
            total_timesteps: Joint steps (overrides config).
            progress_bar: tqdm over joint steps.

        Returns:
            ``self`` with ``_il_models`` populated like :meth:`train_il`.

        Raises:
            ImportError: If pettingzoo or stable-baselines3 is missing.
            NotImplementedError: If algorithm is not ``SAC``.
            TypeError: If the task is single-agent.
        """
        from powerzoo.tasks.registry import make_task
        from powerzoo.rl.marl_simultaneous_sb3 import run_simultaneous_sac

        cfg = self._cfg
        task_source = cfg.task_name or cfg.task_config
        task = make_task(task_source, split=cfg.split)  # type: ignore[arg-type]

        if task.agent_mode != 'multi':
            raise TypeError(
                f"Trainer.train_marl_simultaneous() is for multi-agent tasks. "
                f"Task '{task.name}' has agent_mode='{task.agent_mode}'. "
                f"Use Trainer.train() for single-agent tasks."
            )

        ts = total_timesteps or cfg.total_timesteps
        self._il_models = run_simultaneous_sac(self, int(ts), progress_bar=progress_bar)
        return self

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(
        self,
        n_episodes: Optional[int] = None,
        split: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate the trained model.

        Args:
            n_episodes: Number of episodes (override config value).
            split:      Data split (default: ``'test'``).

        Returns:
            Dict with benchmark metrics (``mean_reward``, ``mean_cost``, etc.).

        Raises:
            RuntimeError: If :meth:`train` has not been called yet.
        """
        from powerzoo.benchmarks.evaluation import evaluate as pz_evaluate

        if self._model is None and self._il_models is None:
            raise RuntimeError(
                "No trained model found. Call Trainer.train(), "
                "Trainer.train_il(), or Trainer.train_marl_simultaneous() "
                "before evaluate()."
            )

        effective_split = split or 'test'
        n_ep = n_episodes or self._cfg.eval_episodes

        eval_env = self.get_env(split=effective_split)

        if self._model is not None:
            policy = _SB3PolicyAdapter(self._model)
        else:
            # IL-MARL evaluation: use first agent's model as stand-in
            # for the single-env evaluate() call (best effort)
            first_model = next(iter(self._il_models.values()))  # type: ignore[union-attr]
            policy = _SB3PolicyAdapter(first_model)

        task_id: Optional[str] = self._cfg.task_name
        return pz_evaluate(
            policy,
            eval_env,
            n_episodes=n_ep,
            task_id=task_id,
            cost_threshold=self._cfg.cost_threshold,
        )

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Optional[Union[str, Path]] = None) -> str:
        """Save the trained model and config to disk.

        Creates two files:
        - ``<path>/model.zip``  — the SB3 model weights
        - ``<path>/config.yaml`` — the experiment config

        Args:
            path: Directory to save into (overrides ``config.save_path``).
                  Defaults to ``./powerzoo_results/``.

        Returns:
            The directory path as a string.

        Raises:
            RuntimeError: If no model is trained yet.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml is required for Trainer.save().")

        if self._model is None and self._il_models is None:
            raise RuntimeError(
                "No model to save. Call train(), train_il(), or "
                "train_marl_simultaneous() first."
            )

        save_dir = Path(path or self._cfg.save_path or './powerzoo_results/')
        save_dir.mkdir(parents=True, exist_ok=True)

        # save model weights
        if self._model is not None:
            self._model.save(str(save_dir / 'model'))
        else:
            for agent_id, model in (self._il_models or {}).items():
                safe_id = agent_id.replace('/', '_')
                model.save(str(save_dir / f'model_{safe_id}'))

        # save config
        config_path = save_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self._cfg.to_dict(), f, default_flow_style=False)

        logger.info("Trainer.save(): saved to '%s'", save_dir)
        return str(save_dir)

    def load(self, path: Union[str, Path]) -> 'Trainer':
        """Load a previously saved SB3 model.

        Args:
            path: Path to a ``model.zip`` file or a directory containing one.

        Returns:
            ``self`` for chaining.
        """
        load_path = Path(path)
        if load_path.is_dir():
            load_path = load_path / 'model.zip'

        algo_cls = Trainer.ALGORITHMS.get(self._cfg.algorithm.upper())
        if algo_cls is None:
            raise ValueError(f"Unknown algorithm '{self._cfg.algorithm}'.")

        env = self.get_env()
        self._model = algo_cls.load(str(load_path), env=env)
        logger.info("Trainer.load(): loaded model from '%s'", load_path)
        return self

    # ── Class constructors ────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'Trainer':
        """Create a Trainer from a YAML config file.

        Args:
            path: Path to the YAML config file.

        Returns:
            Trainer instance.
        """
        from powerzoo.rl.config import RLConfig
        cfg = RLConfig.from_yaml(path)
        return cls(cfg)

    def __repr__(self) -> str:
        cfg = self._cfg
        task = cfg.task_name or '<config>'
        trained = self._model is not None or self._il_models is not None
        return (
            f"Trainer(task={task!r}, algorithm={cfg.algorithm!r}, "
            f"trained={trained})"
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

class _SB3PolicyAdapter:
    """Wrap an SB3 model so it satisfies powerzoo's ``act(obs, info)`` interface."""

    def __init__(self, model: Any):
        self.model = model

    def act(self, obs: Any, info: Optional[Any] = None) -> Any:
        action, _ = self.model.predict(obs, deterministic=True)
        return action


class _AgentEnvWrapper(gymnasium.Env):
    """Wrap a PettingZoo ParallelEnv as a single-agent Gymnasium env.

    During ``step()``, all agents other than the target agent receive a
    randomly-sampled action from their action space (independent learner
    assumption).

    This is a minimal wrapper used internally by ``train_il()``.
    """

    def __init__(self, marl_env: Any, agent_id: str):
        super().__init__()
        self._env = marl_env
        self._agent_id = agent_id
        self.observation_space = marl_env.observation_space(agent_id)
        self.action_space = marl_env.action_space(agent_id)
        self._last_obs: Dict[str, Any] = {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Any] = None):
        obs_dict, info_dict = self._env.reset(seed=seed, options=options)
        self._last_obs = obs_dict
        obs = obs_dict.get(self._agent_id)
        info = info_dict.get(self._agent_id, {})
        return obs, info

    def step(self, action: Any):
        actions = {}
        for a in self._env.agents:
            if a == self._agent_id:
                actions[a] = action
            else:
                actions[a] = self._env.action_space(a).sample()

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._env.step(actions)
        self._last_obs = obs_dict

        obs = obs_dict.get(self._agent_id)
        reward = rew_dict.get(self._agent_id, 0.0)
        terminated = term_dict.get(self._agent_id, False)
        truncated = trunc_dict.get(self._agent_id, False)
        info = info_dict.get(self._agent_id, {})
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
