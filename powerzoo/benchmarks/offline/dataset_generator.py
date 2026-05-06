"""DatasetGenerator — collect offline experience from a PowerZoo environment.

Rolls out any Gymnasium-compatible PowerZoo env with a given policy and
stores the resulting trajectories in an HDF5 file following the standard
offline RL layout used elsewhere in ``powerzoo.offline``.

HDF5 layout
-----------
::

    /observations        float32 (N, obs_dim)
    /actions             float32 (N, act_dim)
    /rewards             float32 (N,)
    /next_observations   float32 (N, obs_dim)
    /terminals           bool    (N,)   — episode end due to env termination
    /truncations         bool    (N,)   — episode end due to time limit
    /infos/              group   — optional per-step info scalars (e.g. cost_sum)

where ``N`` is the total number of timesteps collected.

Usage::

    import gymnasium as gym
    from powerzoo.benchmarks.offline import DatasetGenerator

    env = gym.make('PowerZoo-OPF-v0')

    # Random policy
    gen = DatasetGenerator(env)
    stats = gen.collect(n_episodes=200, save_path='opf_random.h5', seed=42)
    print(stats)  # {'n_episodes': 200, 'n_steps': 9600, 'mean_return': -3.2}

    # Custom policy (callable: obs -> action)
    def my_policy(obs):
        return env.action_space.sample()

    gen = DatasetGenerator(env, policy=my_policy)
    gen.collect(n_episodes=100, save_path='opf_policy.h5')
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class DatasetGenerator:
    """Collect offline trajectories and persist them to HDF5.

    Parameters
    ----------
    env : gym.Env
        A Gymnasium-compatible PowerZoo environment.
    policy : callable or None
        ``policy(obs) -> action``.  If ``None``, uses
        ``env.action_space.sample()`` (random policy).
    info_keys : list of str
        Scalar fields to extract from ``info`` dict and store under
        ``/infos/`` in the HDF5 file.  Default: ``['cost_sum', 'is_safe']``.
    """

    def __init__(self, env, policy: Optional[Callable] = None,
                 info_keys: Optional[List[str]] = None):
        self.env = env
        self.policy = policy if policy is not None else lambda obs: env.action_space.sample()
        self.info_keys = info_keys if info_keys is not None else ['cost_sum', 'is_safe']

    def collect(self, n_episodes: int, save_path: str,
                seed: Optional[int] = None,
                verbose: bool = True) -> Dict[str, Any]:
        """Roll out the policy for ``n_episodes`` and save to ``save_path``.

        Parameters
        ----------
        n_episodes : int
            Number of episodes to collect.
        save_path : str
            Path to the output HDF5 file.  Directories must exist.
        seed : int or None
            Seed for the first episode reset.  Subsequent episodes use
            ``seed + episode_idx`` for reproducibility.
        verbose : bool
            Print progress every 10 % of episodes.  Default True.

        Returns
        -------
        dict with keys:
            ``n_episodes``, ``n_steps``, ``mean_return``, ``std_return``,
            ``wall_time_seconds``.
        """
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "h5py is required for DatasetGenerator. "
                "Install with: pip install h5py"
            ) from e

        buffers: Dict[str, List] = {
            'observations': [], 'actions': [], 'rewards': [],
            'next_observations': [], 'terminals': [], 'truncations': [],
        }
        info_buffers: Dict[str, List] = {k: [] for k in self.info_keys}

        episode_returns: List[float] = []
        t0 = time.time()
        log_interval = max(1, n_episodes // 10)

        for ep in range(n_episodes):
            ep_seed = None if seed is None else (seed + ep)
            obs, _ = self.env.reset(seed=ep_seed)
            ep_return = 0.0

            while True:
                action = self.policy(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                ep_return += reward

                buffers['observations'].append(obs)
                buffers['actions'].append(np.atleast_1d(action))
                buffers['rewards'].append(float(reward))
                buffers['next_observations'].append(next_obs)
                buffers['terminals'].append(bool(terminated))
                buffers['truncations'].append(bool(truncated))

                for key in self.info_keys:
                    info_buffers[key].append(float(info.get(key, 0.0)))

                obs = next_obs
                if terminated or truncated:
                    break

            episode_returns.append(ep_return)
            if verbose and (ep + 1) % log_interval == 0:
                avg = float(np.mean(episode_returns[-log_interval:]))
                print(f"  DatasetGenerator: episode {ep + 1}/{n_episodes}  "
                      f"avg_return={avg:.3f}")

        # Stack arrays
        obs_arr      = np.array(buffers['observations'],      dtype=np.float32)
        act_arr      = np.array(buffers['actions'],           dtype=np.float32)
        rew_arr      = np.array(buffers['rewards'],           dtype=np.float32)
        next_obs_arr = np.array(buffers['next_observations'], dtype=np.float32)
        term_arr     = np.array(buffers['terminals'],         dtype=bool)
        trunc_arr    = np.array(buffers['truncations'],       dtype=bool)

        # Write HDF5
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('observations',      data=obs_arr,      compression='gzip')
            f.create_dataset('actions',           data=act_arr,      compression='gzip')
            f.create_dataset('rewards',           data=rew_arr,      compression='gzip')
            f.create_dataset('next_observations', data=next_obs_arr, compression='gzip')
            f.create_dataset('terminals',         data=term_arr,     compression='gzip')
            f.create_dataset('truncations',       data=trunc_arr,    compression='gzip')

            info_grp = f.create_group('infos')
            for key, vals in info_buffers.items():
                info_grp.create_dataset(key, data=np.array(vals, dtype=np.float32),
                                        compression='gzip')

            # Metadata
            meta = f.create_group('metadata')
            meta.attrs['n_episodes']  = n_episodes
            meta.attrs['n_steps']     = len(rew_arr)
            meta.attrs['mean_return'] = float(np.mean(episode_returns))
            meta.attrs['std_return']  = float(np.std(episode_returns))
            meta.attrs['seed']        = seed if seed is not None else -1

        elapsed = time.time() - t0
        stats = {
            'n_episodes':        n_episodes,
            'n_steps':           len(rew_arr),
            'mean_return':       float(np.mean(episode_returns)),
            'std_return':        float(np.std(episode_returns)),
            'wall_time_seconds': round(elapsed, 2),
            'save_path':         save_path,
        }
        if verbose:
            print(f"DatasetGenerator: saved {stats['n_steps']} steps "
                  f"({stats['n_episodes']} episodes) → {save_path}  "
                  f"[{elapsed:.1f}s]")
        return stats
