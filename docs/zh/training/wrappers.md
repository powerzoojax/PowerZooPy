# Wrappers

`powerzoo/wrappers/` 中的 wrapper 把 [Python contract](../concepts/python-contract.md) 的合约适配到具体 RL 算法所期望的接口形式。它们不修改物理或任务语义，只调整 API 的形态。

本页是各 wrapper 功能的实用参考。各类签名见 [API · Wrappers](../api/wrappers.md)。

## 每个 wrapper 都保留哪些内容

所有 PowerZoo wrapper 都遵守 [Python contract](../concepts/python-contract.md) 中的合约：

- reward 通道仍然只承载经济目标。
- benchmark-core cost 通过命名 `cost_*`、固定顺序 `info['constraint_costs']` 和诊断别名 `info['cost_sum']` 暴露；标量 `info['cost']` 只在兼容 wrapper 中生成。
- 5 种 observation 模式（`global` / `local` / `local_plus_forecast` / `local_plus_voltage` / `ders_local`）不会被修改；改变的只是**容器形式**（Dict → 扁平 Box、单 agent → PettingZoo Parallel ……）。

wrapper 可以彼此堆叠；规范的堆叠顺序见后文。

## `GymnasiumWrapper` — 适配原始 `GridEnv`

把原始 `GridEnv`（state-dict 返回）适配为标准 Gymnasium 五元组：

```python
from powerzoo.envs.grid.trans import TransGridEnv
from powerzoo.wrappers import GymnasiumWrapper

env = GymnasiumWrapper(TransGridEnv())
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

需要直接驱动 `GridEnv`（不带任务封装）时使用它；做基准实验时，`make_task_env(...)` 已经返回 Gymnasium 风格 env。

## `NormalizationWrapper` — 滑动统计归一化

利用滑动统计把 observation（可选 action）归一化到 `[-1, 1]`。堆叠在 `GymnasiumWrapper` 之上：

```python
from powerzoo.wrappers import GymnasiumWrapper, NormalizationWrapper

env = NormalizationWrapper(GymnasiumWrapper(TransGridEnv()))
```

对 task env，`powerzoo.rl.make_env(name, normalize=True)` 等价于上面的写法。

## `ForecastWrapper` — 追加负荷预测窗口

在每个 observation 末尾追加一个 `horizon` 长度的需求预测，并自动扩展 `observation_space`（base dim + `horizon`）。

| 参数 | 默认 | 描述 |
|---|---|---|
| `horizon` | `6` | 追加的未来步数。 |
| `mode` | `'perfect'` | `'perfect'`（真值）、`'noisy'`（高斯噪声）、`'none'`（零）。 |
| `noise_std` | `0.02` | `mode='noisy'` 的分数噪声 std（如 0.02 = 2 %）。 |
| `normalize` | `True` | 把预测值除以数据集最大值。 |

```python
from powerzoo.wrappers import GymnasiumWrapper, ForecastWrapper

env = ForecastWrapper(GymnasiumWrapper(TransGridEnv()), horizon=6, mode='noisy')
obs, info = env.reset(seed=0)
# obs[-6:] 是接下来 6 个半小时的需求值
```

三种 forecast 模式可让你在不改变底层物理的前提下，度量同一任务对预测质量的敏感性。

## `TaskCMDPWrapper` — task 级约束选择

保持标准 Gymnasium **5 元组**不变，并把 task 的 `ConstraintSpec` 附加到 `info` 中：

- `constraint_names` / `constraint_costs` — core env 的完整向量。
- `selected_constraint_names` / `selected_constraint_costs` — benchmark task 选中的子集。
- `selected_cost_sum` — 选中子集的标量和，仅作诊断或兼容投影使用。

`make_task_env(...)` 对带 `constraint_spec()` 的单 agent task 会自动使用它。

## `CMDPWrapper` — 显式 6 元组向量 cost

返回 `(obs, reward, costs, terminated, truncated, info)`，其中 `costs` 是 `info['selected_constraint_costs']` 中的 task 选中向量；没有 task spec 时回退到完整 `constraint_costs`。

## `SafeRLWrapper` — 面向 OmniSafe / Safety-Gymnasium 的 6 元组

返回 **6 元组** `(obs, reward, cost, terminated, truncated, info)`，把选中向量投影成当前安全 RL 库常用的标量 cost。Cost 提取优先级：

1. `info['selected_constraint_costs']` — task 选中的 CMDP 子集。
2. `info['constraint_costs']` — core env 完整向量。
3. `info['cost_sum']` 或兼容 `info['cost']`。
4. `0.0` — 安全回退。

```python
from powerzoo.wrappers import GymnasiumWrapper, SafeRLWrapper

env = SafeRLWrapper(GymnasiumWrapper(TransGridEnv()), cost_threshold=25.0)
obs, info = env.reset(seed=0)
obs, reward, cost, terminated, truncated, info = env.step(env.action_space.sample())
```

| 参数 | 默认 | 描述 |
|---|---|---|
| `cost_threshold` | `25.0` | 标量阈值，以 `env.cost_threshold` 形式提供给 OmniSafe。 |

## `GymnasiumSafeWrapper` — 5 元组，cost 通过 `info` 暴露

返回标准 Gymnasium **5 元组**，并把选中向量的标量投影注入 `info['cost']`。当算法从 `info` 读取 cost 而不是接收单独返回值时使用它。

```python
from powerzoo.wrappers import GymnasiumWrapper, GymnasiumSafeWrapper

env = GymnasiumSafeWrapper(GymnasiumWrapper(TransGridEnv()))
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(info['cost'])  # non-negative scalar
```

## `MARLWrapper` — 单 agent → PettingZoo Parallel

把单 agent 的 PowerZoo env 转换为 **PettingZoo Parallel API**。底层 `GridEnv` 中注册的每个 resource 都变成一个独立 agent：

```python
from powerzoo.wrappers import MARLWrapper

env = MARLWrapper(TransGridEnv(), agent_type='generators')
obs, info = env.reset(seed=0)
actions = {a: env.action_space(a).sample() for a in env.agents}
obs, rewards, terminations, truncations, info = env.step(actions)
```

对 task env，**任务感知**的 `TaskPettingZooWrapper`（定义在 `powerzoo/tasks/interfaces/pettingzoo.py`，同时从 `powerzoo.wrappers` 重新导出以兼容）就是 `make_task_env(..., framework='pettingzoo')` 底层使用的 wrapper。它保留底层 task adapter 的 reward / cost / observation / `__all__` 语义。

## `FlattenWrapper` — Dict → 扁平 Box

把 dict / 嵌套的 `observation_space` 与 `action_space` 扁平化为 1-D `Box`。适用于要求扁平向量输入的算法：

```python
from powerzoo.wrappers import GymnasiumWrapper, FlattenWrapper

env = FlattenWrapper(GymnasiumWrapper(TransGridEnv()))
```

`PowerEnv` 返回 Dict observation 并接受 Dict action；`FlattenWrapper` 正是让单 agent 任务 `battery_arbitrage`、`dc_scheduling`、`dc_microgrid*` 对外暴露扁平 `Box` 的关键。

## 标准堆叠

常见的堆叠模式：

```python
from powerzoo.envs.grid.trans import TransGridEnv
from powerzoo.wrappers import (
    GymnasiumWrapper, NormalizationWrapper, ForecastWrapper, SafeRLWrapper,
)

# Single-agent vanilla
env = GymnasiumWrapper(TransGridEnv())

# Single-agent with normalisation and forecast
env = ForecastWrapper(NormalizationWrapper(GymnasiumWrapper(TransGridEnv())))

# Single-agent Safe RL
env = SafeRLWrapper(NormalizationWrapper(GymnasiumWrapper(TransGridEnv())),
                    cost_threshold=25.0)
```

对 task env（推荐路径），使用 `powerzoo.rl.make_env(...)` 并传入 `normalize=True`、`forecast_horizon=N`、`safe_rl=True`、`cost_threshold=...`；它会自动构建好对应的 wrapper 栈。

## 另见

- [Python contract](../concepts/python-contract.md) — 每个 wrapper 必须保留的内容。
- [Trainers](trainers.md) — `Trainer` 与 `make_env` 如何自动构建 wrapper 栈。
- [API · Wrappers](../api/wrappers.md) — 类签名。
