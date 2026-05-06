"""PowerZoo Offline Dataset — HDF5 offline RL data with a standard dict interface.

Modules
-------
``dataset_generator``
    Roll out any PowerZoo environment with a given policy, collect
    ``(obs, action, reward, next_obs, terminated, truncated, info)``
    tuples and save them to an HDF5 file.

``dataset_loader``
    Load a saved HDF5 dataset and expose a standard offline RL dataset dict
    (``observations``, ``actions``, ``rewards``, ``next_observations``,
    ``terminals``).

Quick start::

    from powerzoo.benchmarks.offline import DatasetGenerator, DatasetLoader

    # Collect data
    gen = DatasetGenerator(env, policy=random_policy)
    gen.collect(n_episodes=500, save_path='data/opf_train.h5')

    # Load for offline RL
    loader = DatasetLoader('data/opf_train.h5')
    dataset = loader.get_dataset()   # dict of NumPy arrays
"""

from powerzoo.benchmarks.offline.dataset_generator import DatasetGenerator
from powerzoo.benchmarks.offline.dataset_loader import DatasetLoader

__all__ = ['DatasetGenerator', 'DatasetLoader']
