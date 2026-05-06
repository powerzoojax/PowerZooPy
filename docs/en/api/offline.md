# Offline Datasets

`powerzoo.benchmarks.offline` provides tools for collecting and replaying offline RL datasets. Data are stored as **HDF5** files and loaded as a **standard offline RL dataset dict** with the usual keys (`observations`, `actions`, `rewards`, `next_observations`, `terminals`, …).

> **Optional dependency**: `pip install h5py`

```python
from powerzoo.benchmarks.offline import DatasetGenerator, DatasetLoader
```

---

::: powerzoo.benchmarks.offline.DatasetGenerator
    options:
      show_source: false
      members:
        - __init__
        - collect

Rolls out any Gymnasium-compatible PowerZoo environment with a given policy and writes an HDF5 dataset.

**Parameters**

| Parameter | Default | Description |
|---|---|---|
| `env` | — | A Gymnasium-wrapped PowerZoo env |
| `policy` | `None` | Callable `policy(obs) → action`; defaults to `env.action_space.sample()` if `None` |
| `info_keys` | `['cost_sum', 'is_safe']` | Keys from `info` to store in `/infos/` |

**`collect(n_episodes, save_path, seed=None, verbose=True) → dict`**

Runs `n_episodes` episodes and writes the dataset to `save_path`. Returns the same dataset dict that `DatasetLoader.get_dataset()` would return.

```python
from powerzoo.benchmarks.offline import DatasetGenerator
from powerzoo.wrappers import GymnasiumWrapper
from powerzoo.tasks import make_task_env

gen = DatasetGenerator(
    env=GymnasiumWrapper(make_task_env("marl_opf")),
    info_keys=["cost_sum", "is_safe"],
)
dataset = gen.collect(n_episodes=500, save_path="data/marl_opf_random.h5", seed=42)
```

### HDF5 Schema

```
/observations            float32  [N, obs_dim]    — observation at each step
/actions                 float32  [N, act_dim]    — action taken
/rewards                 float32  [N]             — scalar reward
/next_observations       float32  [N, obs_dim]    — observation after step
/terminals               bool     [N]             — episode end (done)
/truncations             bool     [N]             — episode truncated
/infos/
    cost_sum             float32  [N]             — total violation cost (F4)
    is_safe              bool     [N]             — safety flag (F4)
    <extra_key>          ...      [N]             — any additional info_keys
/metadata/
    n_episodes           int                      — number of episodes
    n_steps              int                      — total transition count
    mean_return          float32                  — mean episode return
    std_return           float32                  — std of episode returns
    seed                 int                      — RNG seed used
```

`N` = total number of environment steps across all episodes.

---

::: powerzoo.benchmarks.offline.DatasetLoader
    options:
      show_source: false
      members:
        - __init__
        - get_dataset
        - metadata
        - filter

Loads an HDF5 dataset and exposes a standard offline RL dataset dict. Supports lazy loading and result caching for repeated access.

**Parameters**

| Parameter | Default | Description |
|---|---|---|
| `path` | — | Path to the `.h5` file |

### `get_dataset(load_infos=False) → dict`

Returns a dataset dictionary with the following keys:

```python
{
    "observations":       np.ndarray,  # [N, obs_dim]
    "actions":            np.ndarray,  # [N, act_dim]
    "rewards":            np.ndarray,  # [N]
    "next_observations":  np.ndarray,  # [N, obs_dim]
    "terminals":          np.ndarray,  # [N]  bool
    "truncations":        np.ndarray,  # [N]  bool  (PowerZoo extension)
    # if load_infos=True:
    "infos": {
        "cost_sum":  np.ndarray,
        "is_safe":   np.ndarray,
        ...
    }
}
```

### `metadata() → dict`

```python
{
    "n_steps":    int,
    "n_episodes": int,
    "obs_dim":    int,
    "act_dim":    int,
    "mean_return": float,
    "std_return":  float,
}
```

### `filter(min_return=None, max_return=None) → dict`

Returns a filtered dataset dict containing only transitions from episodes whose return falls within `[min_return, max_return]`. Useful for quality-filtered offline learning.

```python
from powerzoo.benchmarks.offline import DatasetLoader

loader = DatasetLoader("data/marl_opf_random.h5")

# Full dataset
dataset = loader.get_dataset()

# Only transitions from episodes with return > −50
good_data = loader.filter(min_return=-50.0)

# Metadata summary
print(loader.metadata())
```

---

## Full Collect → Train Example

```python
from powerzoo.benchmarks.offline import DatasetGenerator, DatasetLoader
from powerzoo.wrappers import GymnasiumWrapper
from powerzoo.tasks import make_task_env

# 1. Collect offline data with a random policy
gen = DatasetGenerator(GymnasiumWrapper(make_task_env("marl_opf")))
gen.collect(n_episodes=1000, save_path="data/random.h5", seed=0)

# 2. Load and inspect
loader = DatasetLoader("data/random.h5")
print(loader.metadata())

# 3. Filter high-quality trajectories for offline training
dataset = loader.filter(min_return=-30.0)
# → pass dataset to your offline RL algorithm (IQL, TD3+BC, DT, …)
```
