# 任务

PowerZoo 提供现成的基准任务，将 grid case、数据切分、agent 设计与评估协议组合到一个对象中。使用 `make_task_env` 按名称实例化任意任务：

```python
from powerzoo.tasks import make_task_env, list_public_tasks

print(list_public_tasks())
env = make_task_env("marl_opf", split="train")

# Use PettingZoo Parallel API (no RLlib required)
env = make_task_env("marl_opf", split="train", framework="pettingzoo")
```

多智能体任务默认始终使用**专用任务 adapter**（`TaskOPFMultiAgentEnv`、`TaskUCMultiAgentEnv`、`TaskEVMultiAgentEnv` 等）。这些 adapter 在装与不装 RLlib 时都能正常工作；装了 `ray[rllib]` 后，返回对象会同时满足 RLlib `MultiAgentEnv` 接口。要在同一 adapter 语义之上得到任务感知的 **PettingZoo Parallel API** wrapper，传 `framework='pettingzoo'`。

明确的公开基准面通过 `powerzoo.tasks.public` 或它重新导出的 helper 提供：

```python
from powerzoo.tasks import PUBLIC_TASKS, list_public_tasks, get_public_task_catalog

print(PUBLIC_TASKS)
print(list_public_tasks())
print(get_public_task_catalog()[0]["task_id"])
print(get_public_task_catalog()[0]["default_episode_horizon_steps"])
```

只有满足基准合约的任务才会留在 `PUBLIC_TASKS`：有文档、已注册、可实例化、通过 smoke test。已注册但尚未完整的任务仍可通过 `list_tasks()` / `make_task_env(...)` 访问，但不属于公开基准面。

`framework` 参数决定使用哪种多智能体接口：

| 取值 | 描述 |
|---|---|
| `'auto'`（默认） | 专用任务 adapter（安装 `ray` 时同时兼容 RLlib） |
| `'pettingzoo'` | 经 `powerzoo.tasks.interfaces.TaskPettingZooWrapper` 包装的任务感知 PettingZoo Parallel API（轻量，无需 RLlib） |
| `'rllib'` | 与 `'auto'` 相同，但缺 `ray[rllib]` 时直接报错 |

---

## 注册任务路由

| 任务名 | `make_task_env()` / `create_env()` 返回 |
|---|---|
| `battery_arbitrage` | 围绕 `PowerEnv` 的 `FlattenWrapper`（单 agent Gymnasium） |
| `marl_opf` | `TaskOPFMultiAgentEnv` |
| `marl_der_arbitrage`、`marl_ders_benchmark` | `TaskResourceMultiAgentEnv` |
| `marl_ev_v2g` | `TaskEVMultiAgentEnv` |
| `dc_scheduling` | 围绕 `PowerEnv` 的 `FlattenWrapper`（单 agent Gymnasium） |
| `dc_microgrid`、`dc_microgrid_safe` | `DCMicrogridEnv`（单 agent Gymnasium，自包含） |
| `gencos_bidding` | `GenCosMARLEnv`（PettingZoo Parallel API；竞争式 5-agent 市场） |
| `marl_uc` | `TaskUCMultiAgentEnv` |
| `opf_118` / `opf_118_7d` | `TaskOPFMultiAgentEnv` |
| `joint_trans_dist` / `joint_trans_dist_7d` | 仅实验性 — 仍处于注册状态，但不属于 `PUBLIC_TASKS`；在 joint adapter / reward 路径正式上线前，当前实例化会失败 |

---

## 公开基准任务卡片

`get_public_task_catalog()` 为当前公开基准面返回稳定的任务卡片元数据。在构建实验菜单、基准摘要、或需要与真实公开任务保持同步的文档时，把它作为权威数据源使用。

```python
from powerzoo.tasks import get_public_task_catalog

for card in get_public_task_catalog():
    print(card["task_id"], card["grid_case"], card["default_episode_horizon_steps"])
```

| 任务 | Grid | Agent 模式 | 默认 observation | Reward / cost 合约 | Horizon | Frameworks |
|---|---|---|---|---|---|---|
| `battery_arbitrage` | distribution / `Case33bw` | single | `flattened` | 仅目标 peak / off-peak 套利利润，带 SOC 目标 shaping；SOC 违反在 `info['cost']` | 48 | `gymnasium` |
| `marl_opf` | transmission / `Case5` | multi | `global` | 共享经济调度 reward；物理违反在 `info['cost']` | 48 | `auto`、`rllib`、`pettingzoo` |
| `marl_der_arbitrage` | distribution / `Case33bw` | multi | `local_plus_forecast` | 共享电池套利 reward；电压 / SOC 违反在 `info['cost']` | 48 | `auto`、`rllib`、`pettingzoo` |
| `marl_ev_v2g` | distribution / `Case33bw` | multi | `local_plus_forecast` | 共享 EV 套利与出发就绪 reward；grid / EV 违反在 `info['cost']` | 168 | `auto`、`rllib`、`pettingzoo` |
| `dc_scheduling` | distribution / `Case33bw` | single | `flattened` | 仅目标的单 agent energy-SLA-PUE reward；grid 与 datacenter 热稳违反在 `info['cost_sum']` | 48 | `gymnasium` |
| `dc_microgrid` | self-contained DC microgrid | single | `flattened` | 标量化 `r_energy + w_cost·r_cost + w_carbon·r_carbon`；向量在 `info['reward_vector']`；SLA / overtemp / power-deficit 在 `info['cost']` | 288 | `gymnasium` |
| `dc_microgrid_safe` | self-contained DC microgrid | single | `flattened` | 与 `dc_microgrid` 相同，CMDP `cost_threshold = 0.5` | 288 | `gymnasium` |
| `marl_uc` | transmission / `Case5` | multi | `global` | 共享 UC 经济 reward；物理违反在 `info['cost']` | 48 | `auto`、`rllib`、`pettingzoo` |
| `opf_118` | transmission / `Case118` | multi | `global` | 共享大规模经济调度 reward；物理违反在 `info['cost']` | 48 | `auto`、`rllib`、`pettingzoo` |
| `opf_118_7d` | transmission / `Case118` | multi | `global` | 共享大规模经济调度 reward；物理违反在 `info['cost']` | 336 | `auto`、`rllib`、`pettingzoo` |

内部原子验证 preset 位于 `powerzoo.tasks.atomic` 下，但有意不纳入公开基准面。

---

## Simple Tasks

::: powerzoo.tasks.simple.MARLOPFTask
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - get_scenario_config
        - get_agents_config
        - create_env

**关键参数**

| 参数 | 默认 | 描述 |
|---|---|---|
| `case` | `'Case5'` | Grid case 名称 |
| `split` | `'train'` | 数据切分：`'train'`、`'val'` 或 `'test'` |
| `action_mode` | `'score'` | `'score'`（softmax 分配）或 `'direct'`（直接给出 MW 出力） |
| `max_load_ratio` | `0.9` | 最大负荷占总发电容量的比例 |
| `max_steps` | `48` | 每 episode 步数（48 = 30 分钟分辨率下的 1 天） |

**Agent 设计**

- **Action**：score ∈ [0, 1] — 用 softmax 把净负荷分配到各发电机
- **Observation**：全局特征（总负荷、线路潮流、时间）+ 本地特征（机组下标、p_min、p_max、成本系数）
- **Reward**：−(发电成本) / 1000（共享、合作式）

**数据切分**（不重叠，固定不变以保证基准可复现）

| Split | 日期范围 |
|---|---|
| train | 2023-07-05 – 2024-12-31 |
| val | 2025-01-01 – 2025-06-30 |
| test | 2025-07-01 – 2025-12-15 |

---

## Middle Tasks

::: powerzoo.tasks.middle.MARLUCTask
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - get_scenario_config
        - get_agents_config
        - create_env

在 `MARLOPFTask` 之上扩展**机组组合**决策。每个发电机 agent 既要决定*出力多少*，也要决定*是否在线*。

**关键参数**

| 参数 | 默认 | 描述 |
|---|---|---|
| `case` | `'Case5'` | Grid case 名称 |
| `split` | `'train'` | 数据切分 |
| `max_load_ratio` | `0.9` | 最大负荷比例 |
| `max_steps` | `48` | 每 episode 步数 |

**UC 默认值**（当 `case.units` 列中未给出时使用）

| 列 | 默认 | 单位 |
|---|---|---|
| `startup_cost` | 500 | $/start |
| `shutdown_cost` | 200 | $/stop |
| `ramp_rate` | 999 | MW/step |
| `min_up_time` | 1 | steps |
| `min_down_time` | 1 | steps |

**Agent 设计**

- **Action**：`[score, on_off]` — 2 元向量；`on_off ≥ 0.5` 时投运机组
- **Observation**：全局 + 本地 + commitment 向量（所有机组当前 on/off 状态）
- **Reward**：−(发电成本 + 启动成本 + 停机成本) / 1000（仅经济目标）
- **Cost 信号**：物理约束违反 → `info['cost']`（CMDP 分离）

---

## Complex Tasks

::: powerzoo.tasks.complex.OPF118Task
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - get_scenario_config
        - get_agents_config
        - create_env

在 IEEE 118-bus 系统上的大规模合作 OPF：54 个发电机、186 条输电线。

**关键参数**

| 参数 | 默认 | 描述 |
|---|---|---|
| `split` | `'train'` | 数据切分 |
| `max_load_ratio` | `0.85` | 最大负荷比例（系统规模更大，比 5-bus 略低） |
| `max_steps` | `48` | 每 episode 步数 |
| `action_mode` | `'score'` | `'score'` 或 `'direct'` |

**Agent 设计**

- 54 个合作 agent（每个发电机一个）
- 使用与 `MARLOPFTask` 相同的 score 动作 / OPF observation 协议
- 共享合作 reward

### OPF118Task7Days

`OPF118Task` 的 7 天（336 步）变体。所有参数继承自 `OPF118Task`；`max_steps` 默认 `336`。

```python
env = make_task_env("opf_118_7d", split="train")
```
