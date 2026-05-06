# Python API 合约

PowerZoo 在 Gymnasium / PettingZoo / RLlib 之上遵循一组稳定约定。本页是这些约定的**正式描述**；其他页面默认它们已经成立。

合约分四层：

1. 基础 env 接口（`BaseEnv`）。
2. 单智能体任务 env（Gymnasium 五元组）。
3. 多智能体任务 env（PettingZoo Parallel 或 RLlib `MultiAgentEnv`）。
4. 每个任务都遵守的 framework / observation / cost 合约。

reward / cost 分离足够重要，单独成页：[Reward and cost split](reward-cost-split.md)。

## 1. `BaseEnv` — 抽象父类

`BaseEnv`（`powerzoo/envs/base.py`）继承自 `gymnasium.Env`，并新增两个 PowerZoo 专属属性：

| 属性 / 方法 | 用途 |
|---|---|
| `time_step` | 当前 episode 内的步数计数。 |
| `delta_t_minutes` | 步长（分钟），必须能整除 1440。默认 30。 |
| `action_space` / `observation_space` | 由子类填充。 |
| `reset(seed, options)` | 重置 `time_step`，返回 `(state, info)`（子类决定具体内容）。 |
| `step(action)` | 子类决定。在任务层返回 Gymnasium 风格的五元组。 |
| `obs()` / `reward()` / `cost()` | 钩子方法；`cost()` 默认返回 0（与 CMDP 兼容）。 |

子类不会把可变状态存放在任意属性中。`GridEnv` 与 `ResourceEnv` 把状态保存在定义明确的字段里（case 数据、`current_p_mw`、`soc`……），因此 reset 是可复现的。

`BaseEnv` 直接被 `GridEnv`、`ResourceEnv`、`PowerEnv`、`MarketEnv` 与 `DCMicrogridEnv` 使用。前三个在 [Architecture · Environment stack](../architecture/env-stack.md) 介绍；后两个分别有专页 [Physics](../physics/markets.md) 与 [Physics · Microgrid](../physics/microgrid.md)。

## 2. 单智能体任务 env（Gymnasium 五元组）

单智能体任务（`battery_arbitrage`、`dc_scheduling`、`dc_microgrid`、`dc_microgrid_safe`）返回标准 `gymnasium.Env`。按常规写法循环：

```python
from powerzoo.tasks import make_task_env

env = make_task_env("battery_arbitrage", split="train")
obs, info = env.reset(seed=0)
terminated = truncated = False
while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

扁平的 observation 是把 `FlattenWrapper` 套在 `PowerEnv`（它本身绑定一个 `GridEnv` 与一个或多个 `ResourceEnv`）之外得到的。`info` 字典含 §4 描述的完整物理违反明细。

## 3. 多智能体任务 env

多智能体任务有两种兼容接口。两者的 reward、cost 与 observation 语义完全一致——只在 `step()` 的返回形式以及 `done` 信号的传递方式上有差别。

| `framework=` | 返回的 env | 何时使用 |
|---|---|---|
| `'auto'`（默认） | 专用任务 adapter（安装了 `ray` 时同时兼容 RLlib） | 默认；未装 RLlib 也能用。`terminateds.get("__all__")` 或 `truncateds.get("__all__")` 为真时 episode 结束。 |
| `'pettingzoo'` | 围绕同一 adapter 的、任务感知的 PettingZoo Parallel API wrapper | 想用 `while env.agents:` 写法；wrapper 会在 episode 结束时清空 `env.agents`。 |
| `'rllib'` | 与 `'auto'` 相同，但若缺 `ray[rllib]` 直接报错 | 在生产环境中显式声明依赖。 |

PettingZoo 路径：

```python
env = make_task_env("marl_opf", framework="pettingzoo")
obs, info = env.reset(seed=42)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
```

Auto / RLlib 路径：

```python
env = make_task_env("marl_opf", framework="auto")
obs, infos = env.reset(seed=42)
terminated = truncated = False
while not (terminated or truncated):
    actions = {a: env.action_space[a].sample() for a in env.possible_agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    terminated = bool(terms.get("__all__", False))
    truncated  = bool(truncs.get("__all__", False))
```

驱动 `framework='pettingzoo'` 的 PettingZoo 桥接代码位于 `powerzoo/tasks/interfaces/pettingzoo.py`，并以 `powerzoo.wrappers.TaskPettingZooWrapper` 重新导出以保持向后兼容。它必须完整保留底层 adapter 合约的每一个细节。

## 4. `info` 字典

每个 grid env 在 `info` 中至少填以下字段。任务和 `PowerEnv` 可以加，但不能减。

| 键 | 类型 | 含义 |
|---|---|---|
| `is_safe` | bool | 本步所有物理限制都满足。 |
| `pf_converged` | bool | 潮流求解器收敛。 |
| `cost_exception` | bool | PF 求解中抛出过异常。 |
| `cost_thermal_overload` | float (MW) | 线路潮流越限的求和。 |
| `cost_voltage_violation` | float (pu) | 节点电压越带的幅值求和。 |
| `cost_sum` | float | 物理违反 cost 的总和（grid + resources 中所有 `cost_*` 之和）。 |
| `p_slack_MW` / `q_slack_MVAr` | float | Slack 节点的有功 / 无功注入（配电中是馈线首端交换功率）。 |
| `is_diverged` | bool | BFS 在到达容差前先到达 `max_iter`。 |
| `voltage_collapse` | bool | 检测到严重的低电压未钳位（视为 PF 失败）。 |

`PowerEnv` 接着把 resource 的 cost 贡献聚合到同一个字典里——完整数据流见 [Reward and cost split](reward-cost-split.md)。

## 5. Observation 模式

PowerZoo 定义了**五种规范 observation 模式**。代码中的列表在 `powerzoo.tasks.observation`：

```python
from powerzoo.tasks.observation import OBSERVATION_MODES
print(OBSERVATION_MODES)
# ('global', 'local', 'local_plus_forecast', 'local_plus_voltage', 'ders_local')
```

每种模式是一组特征名的元组，任务 adapter 据此为每个 agent 构造一个 per-agent `Box` observation。任务通过 `make_observation_config(...)` 声明它支持哪些模式；可以通过 adapter 的 `get_observation_fields()` 查询实际字段顺序。

| 模式 | agent 看到的内容 | 适用场景 |
|---|---|---|
| `global` | 共享的电网摘要（总负荷、归一化线路潮流、时间特征）加上 agent 自身的不可变参数。 | *最容易*的设置；最接近中心化训练（CTDE）。 |
| `local` | 只看到 agent 自己的 state 与相邻电网信号，无系统级摘要。 | *最难*的默认设置；通常需要依赖习得的通信才能解决。 |
| `local_plus_forecast` | `local` 加上任务声明的预测窗口（负荷 / 电价 / 可用度）。 | *中等*设置；用于研究预测质量对性能的影响。 |
| `local_plus_voltage` | `local` 加上按馈线或按区的电压摘要。 | 配电侧任务，电压是关键安全信号但又不希望泄露全局 state。 |
| `ders_local` | 适用于异构 DER 群组的紧凑 `local` 布局，按 resource 角色统一编排的 per-agent 向量。 | `marl_ders_benchmark` 与类似的混合资源任务。 |

每个公开任务的默认模式定义在任务类内部；`get_public_task_info(name)['default_observation_mode']` 给出当前值。

| 任务 | 默认模式 | 通常支持的其他模式 |
|---|---|---|
| `marl_opf`、`marl_uc`、`opf_118`、`opf_118_7d` | `global` | `local`、`local_plus_forecast` |
| `marl_der_arbitrage` | `local_plus_forecast` | `local`、`local_plus_voltage` |
| `marl_ders_benchmark` | `ders_local` | `local`、`local_plus_voltage` |
| `marl_ev_v2g` | `local_plus_forecast` | `local` |
| `battery_arbitrage`、`dc_scheduling`、`dc_microgrid*` | `flattened`（单智能体） | n/a |

只需切换模式，就能把同一任务变成一条难度阶梯：

```python
from powerzoo.tasks import make_task_env

easy   = make_task_env("marl_opf", split="train", obs_mode="global")
medium = make_task_env("marl_opf", split="train", obs_mode="local_plus_forecast")
hard   = make_task_env("marl_opf", split="train", obs_mode="local")
```

## 6. `envs / tasks / wrappers` 边界

PowerZoo 把三个包严格分开，每个包只负责一件事：

```mermaid
flowchart LR
    subgraph envs ["envs/ — physical simulation"]
      E1[GridEnv / ResourceEnv / PowerEnv]
      E2[time progression, PF, cost_*]
    end
    subgraph tasks ["tasks/ — benchmark presets + adapters"]
      T1[Task definitions]
      T2[Adapters\n(OPF / UC / Resource / EV)]
      T3[Public surface + registry]
    end
    subgraph wrappers ["wrappers/ — generic API adaptation"]
      W1[Gymnasium / Flatten / Normalize]
      W2[SafeRL / Forecast / MARL]
    end
    envs --> tasks
    tasks --> wrappers
```

箭头是单向的。`envs/` 必须能独立工作（不 import `tasks/`）。`wrappers/` 不应去修补任务专属 bug——这类修复应放在 `tasks/` 或 `envs/`。`powerzoo/tasks/adapters/` 中的 adapter 解析任务级动作、调用底层 `PowerEnv`、组装 per-agent 的 `info`、`cost` 与 `costs`，并提供 `get_observation_fields()`。它们不会重新实现物理。

## 7. 公开基准面

明确、稳定的公开基准面是 `powerzoo.tasks.public.PUBLIC_TASKS`：

```python
from powerzoo.tasks import PUBLIC_TASKS, list_public_tasks, get_public_task_catalog

print(PUBLIC_TASKS)
print(list_public_tasks())
catalog = get_public_task_catalog()
print(catalog[0]['task_id'], catalog[0]['default_episode_horizon_steps'])
```

要保留在 `PUBLIC_TASKS` 中，一个任务必须满足：

1. 在 `powerzoo.tasks.registry` 中注册。
2. 有文档，且可通过 `make_task_env(name, split=...)` 实例化。
3. 在 `train` / `val` / `test` 上各通过至少一次 smoke test。
4. 与本页合约一致。

已注册但尚不完整的任务（`joint_trans_dist*`、原子验证 preset 等）仍可通过 `list_tasks()` 与 `make_task_env(...)` 访问，但**不**属于公开基准面。
