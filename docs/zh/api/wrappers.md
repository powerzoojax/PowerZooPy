# 包装器

所有 wrapper 都可从 `powerzoo.wrappers` 导入。

面向基准任务、任务感知的 PettingZoo 包装目前位于 `powerzoo.tasks.interfaces.TaskPettingZooWrapper`，仍从 `powerzoo.wrappers` 重新导出以保持向后兼容。

```python
from powerzoo.wrappers import (
    GymnasiumWrapper,
    NormalizationWrapper,
    SafeRLWrapper,
    GymnasiumSafeWrapper,
    ForecastWrapper,
    MARLWrapper,
    FlattenWrapper,
)
```

---

::: powerzoo.wrappers.gym_wrappers.GymnasiumWrapper
    options:
      show_source: false
      members:
        - __init__
        - reset
        - step

把任意 PowerZoo `GridEnv` 适配为标准 Gymnasium 五元组 API：

```python
from powerzoo.envs.grid.trans import TransGridEnv
from powerzoo.wrappers import GymnasiumWrapper

env = GymnasiumWrapper(TransGridEnv())
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

---

::: powerzoo.wrappers.gym_wrappers.NormalizationWrapper
    options:
      show_source: false
      members:
        - __init__
        - reset
        - step

利用滑动统计把 observation（可选 action）归一化到 `[−1, 1]`。堆叠在 `GymnasiumWrapper` 之上：

```python
from powerzoo.wrappers import GymnasiumWrapper, NormalizationWrapper

env = NormalizationWrapper(GymnasiumWrapper(TransGridEnv()))
```

---

::: powerzoo.wrappers.safe_rl_wrapper.SafeRLWrapper
    options:
      show_source: false
      members:
        - __init__
        - step

返回与 [OmniSafe](https://github.com/PKU-Alignment/omnisafe) 和 Safety-Gymnasium 兼容的 **6 元组** `(obs, reward, cost, terminated, truncated, info)`。

Cost 提取优先级：

1. `info['selected_constraint_costs']` — task 选中的 CMDP 向量
2. `info['constraint_costs']` — core env 完整向量
3. `info['cost_sum']` 或兼容 `info['cost']`
4. `0.0` — 安全回退

```python
from powerzoo.wrappers import GymnasiumWrapper, SafeRLWrapper

env = SafeRLWrapper(GymnasiumWrapper(TransGridEnv()), cost_threshold=25.0)
obs, info = env.reset(seed=0)
obs, reward, cost, terminated, truncated, info = env.step(env.action_space.sample())
```

| 参数 | 默认 | 描述 |
|---|---|---|
| `cost_threshold` | `25.0` | 标量阈值，以 `env.cost_threshold` 形式提供给 OmniSafe |

---

::: powerzoo.wrappers.safe_rl_wrapper.GymnasiumSafeWrapper
    options:
      show_source: false
      members:
        - __init__
        - step

返回标准 Gymnasium **5 元组**，并把 `cost` 注入 `info['cost']`。当算法从 `info` 读取 cost 而不是接收单独返回值时使用它。

```python
from powerzoo.wrappers import GymnasiumWrapper, GymnasiumSafeWrapper

env = GymnasiumSafeWrapper(GymnasiumWrapper(TransGridEnv()))
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(info['cost'])  # non-negative scalar
```

---

::: powerzoo.wrappers.forecast_wrapper.ForecastWrapper
    options:
      show_source: false
      members:
        - __init__
        - reset
        - observation

在每个 observation 末尾追加一个 `horizon` 长度的需求预测，并自动扩展 `observation_space`（base dim + `horizon`）。

| 参数 | 默认 | 描述 |
|---|---|---|
| `horizon` | `6` | 追加的未来步数 |
| `mode` | `'perfect'` | `'perfect'`（真值）、`'noisy'`（高斯噪声）或 `'none'`（零） |
| `noise_std` | `0.02` | `mode='noisy'` 的分数噪声 std（如 0.02 = 2 %） |
| `normalize` | `True` | 把预测值除以数据集最大值 |

```python
from powerzoo.wrappers import GymnasiumWrapper, ForecastWrapper

env = ForecastWrapper(GymnasiumWrapper(TransGridEnv()), horizon=6, mode='noisy')
obs, info = env.reset(seed=0)
# obs[-6:] contains the next 6 half-hour demand values
```

---

::: powerzoo.wrappers.marl_wrapper.MARLWrapper
    options:
      show_source: false
      members:
        - __init__
        - reset
        - step

把单 agent 的 PowerZoo env 转换为 **PettingZoo Parallel API**。底层 `GridEnv` 中注册的每个 resource 都对应一个独立 agent。

```python
from powerzoo.wrappers import MARLWrapper

env = MARLWrapper(TransGridEnv(), agent_type='generators')
obs, info = env.reset(seed=0)
actions = {a: env.action_space(a).sample() for a in env.agents}
obs, rewards, terminations, truncations, info = env.step(actions)
```

---

::: powerzoo.wrappers.flatten.FlattenWrapper
    options:
      show_source: false
      members:
        - __init__
        - reset
        - step

把 dict / 嵌套的 `observation_space` 与 `action_space` 扁平化为 1-D `Box`。适用于要求扁平向量输入的算法。

```python
from powerzoo.wrappers import GymnasiumWrapper, FlattenWrapper

env = FlattenWrapper(GymnasiumWrapper(TransGridEnv()))
```
