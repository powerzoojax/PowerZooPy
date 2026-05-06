# 离线数据集

`powerzoo.benchmarks.offline` 提供收集与回放离线 RL 数据集的工具。数据以 **HDF5** 文件存储，并以**标准离线 RL 数据集 dict**（含常见键 `observations`、`actions`、`rewards`、`next_observations`、`terminals` ……）的形式加载。

> **可选依赖**：`pip install h5py`

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

用给定 policy 在任意 Gymnasium 兼容的 PowerZoo env 上做 rollout，并写出 HDF5 数据集。

**参数**

| 参数 | 默认 | 描述 |
|---|---|---|
| `env` | — | 一个经过 Gymnasium 包装的 PowerZoo env |
| `policy` | `None` | Callable `policy(obs) → action`；为 `None` 时默认使用 `env.action_space.sample()` |
| `info_keys` | `['cost_sum', 'is_safe']` | 从 `info` 中保存到 `/infos/` 的键 |

**`collect(n_episodes, save_path, seed=None, verbose=True) → dict`**

运行 `n_episodes` 个 episode，把数据集写到 `save_path`，并返回与 `DatasetLoader.get_dataset()` 一致的数据集 dict。

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

`N` = 所有 episode 的环境步数总和。

---

::: powerzoo.benchmarks.offline.DatasetLoader
    options:
      show_source: false
      members:
        - __init__
        - get_dataset
        - metadata
        - filter

加载一个 HDF5 数据集，并提供标准的离线 RL 数据集 dict。支持懒加载与结果缓存，便于重复访问。

**参数**

| 参数 | 默认 | 描述 |
|---|---|---|
| `path` | — | `.h5` 文件的路径 |

### `get_dataset(load_infos=False) → dict`

返回一个数据集字典，含以下键：

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

返回过滤后的数据集 dict，只包含 return 落在 `[min_return, max_return]` 区间内的 episode 的 transition。适用于按质量过滤后再做的离线学习。

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

## 完整 Collect → Train 示例

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
