"""DatasetLoader — offline dataset interface for PowerZoo.

Reads an HDF5 file produced by ``DatasetGenerator`` and exposes the data
through a conventional dict-of-arrays layout so offline RL code that expects
``observations`` / ``actions`` / ``rewards`` / ``terminals`` / … can consume
PowerZoo data without extra glue.

Standard dict keys
------------------
``observations``        float32 (N, obs_dim)
``actions``             float32 (N, act_dim)
``rewards``             float32 (N,)
``next_observations``   float32 (N, obs_dim)
``terminals``           float32 (N,)   — 1.0 if ``terminated`` or ``truncated``

Additional keys (PowerZoo-specific)
-------------------------------------
``truncations``         float32 (N,)   — 1.0 only when ``truncated``
``infos/<key>``         float32 (N,)   — any stored info field

Usage::

    from powerzoo.benchmarks.offline import DatasetLoader

    loader = DatasetLoader('data/opf_train.h5')
    dataset = loader.get_dataset()            # dict of NumPy arrays
    print(dataset['observations'].shape)      # (N, obs_dim)

    # Subset by episode return
    good = loader.filter(min_return=-1.0)
    print(good['rewards'].mean())

    # Metadata
    print(loader.metadata())
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class DatasetLoader:
    """Load a PowerZoo HDF5 offline dataset.

    Parameters
    ----------
    path : str
        Path to an HDF5 file created by ``DatasetGenerator``.
    """

    def __init__(self, path: str):
        self.path = path
        self._cache: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def get_dataset(self, load_infos: bool = False) -> Dict[str, Any]:
        """Return a dataset dict with standard offline RL keys.

        Parameters
        ----------
        load_infos : bool
            If True, also load the ``/infos/`` group fields.

        Returns
        -------
        dict with numpy arrays.
        """
        if self._cache is not None:
            return self._cache

        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "h5py is required for DatasetLoader. "
                "Install with: pip install h5py"
            ) from e

        with h5py.File(self.path, 'r') as f:
            dataset: Dict[str, Any] = {
                'observations':      f['observations'][:].astype(np.float32),
                'actions':           f['actions'][:].astype(np.float32),
                'rewards':           f['rewards'][:].astype(np.float32),
                'next_observations': f['next_observations'][:].astype(np.float32),
                # terminals = terminated OR truncated (common offline RL convention)
                'terminals': (
                    f['terminals'][:].astype(np.float32) |
                    f['truncations'][:].astype(np.float32)
                ).astype(np.float32),
                'truncations':       f['truncations'][:].astype(np.float32),
            }

            if load_infos and 'infos' in f:
                for key in f['infos']:
                    dataset[f'infos/{key}'] = f['infos'][key][:].astype(np.float32)

        self._cache = dataset
        return dataset

    def metadata(self) -> Dict[str, Any]:
        """Return metadata stored in the HDF5 file."""
        try:
            import h5py
        except ImportError as e:
            raise ImportError("h5py is required for DatasetLoader.") from e

        with h5py.File(self.path, 'r') as f:
            if 'metadata' in f:
                return dict(f['metadata'].attrs)
            # Compute on the fly if metadata group is missing
            return {
                'n_steps': len(f['rewards']),
                'obs_dim': f['observations'].shape[1] if f['observations'].ndim > 1 else 1,
                'act_dim': f['actions'].shape[1] if f['actions'].ndim > 1 else 1,
            }

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter(self, min_return: Optional[float] = None,
               max_return: Optional[float] = None) -> Dict[str, Any]:
        """Return a dataset filtered to episodes within a return range.

        Parameters
        ----------
        min_return : float or None
            Drop episodes with total return < min_return.
        max_return : float or None
            Drop episodes with total return > max_return.

        Returns
        -------
        Filtered dataset dict (same keys as ``get_dataset()``).
        """
        dataset = self.get_dataset()
        terminals = dataset['terminals'].astype(bool)
        rewards   = dataset['rewards']

        # Identify episode boundaries (end of each episode)
        episode_ends = np.where(terminals)[0]
        if len(episode_ends) == 0 or episode_ends[-1] != len(rewards) - 1:
            episode_ends = np.append(episode_ends, len(rewards) - 1)

        # Compute per-episode returns
        starts = np.concatenate([[0], episode_ends[:-1] + 1])
        keep_mask = np.zeros(len(rewards), dtype=bool)

        for start, end in zip(starts, episode_ends):
            ep_return = float(rewards[start:end + 1].sum())
            above_min = min_return is None or ep_return >= min_return
            below_max = max_return is None or ep_return <= max_return
            if above_min and below_max:
                keep_mask[start:end + 1] = True

        return {k: v[keep_mask] for k, v in dataset.items()}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        try:
            meta = self.metadata()
            return (f"DatasetLoader(path='{self.path}', "
                    f"n_steps={meta.get('n_steps', '?')})")
        except Exception:
            return f"DatasetLoader(path='{self.path}')"
