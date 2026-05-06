"""Simultaneous multi-agent SAC on a PettingZoo ParallelEnv (homogeneous agents).

Each environment timestep all agents choose actions from their own policies, the
parallel env advances once, and each agent stores a transition in its own SB3
replay buffer.  Gradient updates run in the same iteration (independent learners
with non-stationarity from co-learning partners — IPPO-style interleaving for
off-policy algorithms).

This is **not** MAPPO / centralized critic training; it is the standard
``one joint step → N transitions → N updates`` pattern with config-driven
``make_env(..., framework='pettingzoo')`` environments.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import gymnasium
import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from powerzoo.rl.trainer import Trainer


def _make_dummy_env(obs_space: gymnasium.spaces.Space, act_space: gymnasium.spaces.Space) -> gymnasium.Env:
    """Minimal Gymnasium env so SB3 can build networks (step/reset unused)."""

    class _SpaceEnv(gymnasium.Env):
        def __init__(self) -> None:
            super().__init__()
            self.observation_space = obs_space
            self.action_space = act_space

        def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
            super().reset(seed=seed)
            if isinstance(self.observation_space, gymnasium.spaces.Box):
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}
            raise TypeError('marl_simultaneous_sb3 only supports Box observations')

        def step(self, action):
            raise RuntimeError('internal dummy env — use train_marl_simultaneous() loop')

    return _SpaceEnv()


def run_simultaneous_sac(
    trainer: 'Trainer',
    total_timesteps: int,
    progress_bar: bool = True,
) -> Dict[str, Any]:
    """Build one SAC per agent and train on shared PettingZoo rollouts.

    Args:
        trainer: Configured :class:`~powerzoo.rl.trainer.Trainer` (``framework='pettingzoo'``).
        total_timesteps: Number of **joint** env steps (one step = all agents act once).
        progress_bar: tqdm over joint steps.

    Returns:
        ``{agent_id: sb3_model}`` stored on ``trainer._il_models``.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    from stable_baselines3 import SAC

    cfg = trainer._cfg
    algo_cls = trainer.ALGORITHMS.get(cfg.algorithm.upper())
    if algo_cls is not SAC:
        raise NotImplementedError(
            f"train_marl_simultaneous() currently supports SAC only; "
            f"got algorithm={cfg.algorithm!r}. "
            f"Use train_il() for sequential {cfg.algorithm} or plug a custom loop via get_env()."
        )

    marl_env = trainer.get_env(split=cfg.split)

    agent_ids: List[str] = list(marl_env.possible_agents)
    if not agent_ids:
        raise ValueError('No agents in PettingZoo env (possible_agents empty).')

    obs_spaces = [marl_env.observation_space(a) for a in agent_ids]
    act_spaces = [marl_env.action_space(a) for a in agent_ids]
    heterogeneous = len({str(s) for s in obs_spaces}) > 1 or len({str(s) for s in act_spaces}) > 1
    if heterogeneous:
        logger.warning(
            'train_marl_simultaneous(): heterogeneous spaces; each agent still gets '
            'its own SAC, but simultaneous training may be less stable.'
        )

    models: Dict[str, Any] = {}
    base_seed = cfg.seed

    for i, agent_id in enumerate(agent_ids):
        dummy = _make_dummy_env(
            marl_env.observation_space(agent_id),
            marl_env.action_space(agent_id),
        )
        seed_i = (base_seed + i) if base_seed is not None else None
        models[agent_id] = algo_cls(
            cfg.policy,
            dummy,
            verbose=0,
            seed=seed_i,
            **cfg.hyperparams,
        )
        models[agent_id]._total_timesteps = int(total_timesteps)

    # SB3 expects _setup_learn() before train() (logger, ep buffers, lr schedule).
    for model in models.values():
        model._setup_learn(
            total_timesteps=int(total_timesteps),
            callback=None,
            reset_num_timesteps=True,
            tb_log_name='run',
            progress_bar=False,
        )

    # Joint rollout loop
    iterator = range(int(total_timesteps))
    if progress_bar and tqdm is not None:
        iterator = tqdm(iterator, total=int(total_timesteps), desc='marl-sac (joint steps)')

    observations: Dict[str, Any]
    infos: Dict[str, Any]
    observations, infos = marl_env.reset(seed=cfg.seed)

    for _ in iterator:
        if not marl_env.agents:
            observations, infos = marl_env.reset()

        active = list(marl_env.agents)
        actions_dict: Dict[str, np.ndarray] = {}
        buffer_actions: Dict[str, np.ndarray] = {}

        for agent_id in active:
            model = models[agent_id]
            obs = np.asarray(observations[agent_id], dtype=np.float32)

            if model.num_timesteps < model.learning_starts:
                ua = model.action_space.sample()
            else:
                ua, _ = model.predict(obs, deterministic=False)

            ua = np.asarray(ua, dtype=np.float32).reshape(-1)
            ba = model.policy.scale_action(ua.reshape(1, -1))
            actions_dict[agent_id] = ua.astype(np.float32)
            buffer_actions[agent_id] = ba

        next_obs, rewards, terminations, truncations, step_infos = marl_env.step(actions_dict)

        for agent_id in active:
            model = models[agent_id]
            prev_obs = np.asarray(observations[agent_id], dtype=np.float32)
            r = float(rewards[agent_id])
            term = bool(terminations.get(agent_id, False))
            trunc = bool(truncations.get(agent_id, False))
            done = term or trunc

            if agent_id in next_obs:
                n_obs = np.asarray(next_obs[agent_id], dtype=np.float32)
            else:
                n_obs = prev_obs

            info_one = dict(step_infos.get(agent_id, {}))
            if trunc and not term:
                info_one['TimeLimit.truncated'] = True

            model.replay_buffer.add(
                prev_obs,
                n_obs,
                buffer_actions[agent_id],
                np.array([r], dtype=np.float32),
                np.array([done], dtype=np.float32),
                [info_one],
            )

        # One global step counter shared by all agents (same as one joint transition).
        ref = next(iter(models.values()))
        joint_step = int(ref.num_timesteps) + 1
        for model in models.values():
            model.num_timesteps = joint_step
            model._update_current_progress_remaining(joint_step, int(total_timesteps))

        if joint_step > ref.learning_starts:
            gs = ref.gradient_steps if ref.gradient_steps >= 0 else 1
            for model in models.values():
                model.train(gradient_steps=int(gs), batch_size=int(model.batch_size))

        if not marl_env.agents:
            observations, infos = marl_env.reset()
        else:
            observations = next_obs

    return models
