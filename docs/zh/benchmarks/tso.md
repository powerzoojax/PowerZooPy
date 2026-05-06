# TSO — 安全调度

TSO 系列覆盖输电系统运营商任务：在 IEEE `Case5` 与 `Case118` 网络上做经济调度（ED）与机组组合（UC）。统一的 RL 问题是**含离散-连续混合动作的 Safe RL**——agent 必须在热稳限制下（UC 还要附加最小开机时间 / 最小停机时间 / 爬坡约束）最小化发电成本。

> **术语速查**。*TSO*（Transmission System Operator，输电系统运营商）——负责运行主干电网。*ED / OPF*（Economic Dispatch / Optimal Power Flow）——在限制下选择发电机出力以最小化成本。*UC / SCUC*（Unit Commitment / Security-Constrained UC）——在 ED 之上增加开/关决策、启停成本、爬坡限制、最小开机 / 停机时间。

## 本系列下的任务

| 任务名 | 载体 | Agent 数 | Steps | Action | 说明 |
|---|---|---|---|---|---|
| `marl_uc` | `Case5` | 5 个发电机 | 48 × 30 min | 每 agent `[score, on_off]` | 最小的 UC 基准。 |
| `opf_118` | `Case118` | 54 个发电机 | 48 × 30 min | 每 agent `[score]` | 大规模 ED。 |
| `opf_118_7d` | `Case118` | 54 个发电机 | 336 × 30 min | 每 agent `[score]` | `opf_118` 的 7 天变体。 |

`marl_opf`（`Case5` 上的 5-agent ED）是更轻量的起步任务；它不在旗舰基准内，但与旗舰任务共用同一个 adapter（`TaskOPFMultiAgentEnv`）。

## 为什么是这个系列

- 唯一一个要求单 agent（或小团队）同时满足**密集且同时绑定的约束集**（热稳 + 备用 + UC 约束）的系列。
- 唯一一个在 `marl_uc` 中提供明确的**离散-连续混合动作空间**的系列（`on_off ∈ {0, 1}` × `score ∈ [0, 1]`）。
- 标准的 Safe-RL 测试场景：`opf_118` 的约束足够紧，随机策略在多数 step 上都不可行。

## 物理设置

- **载体**：输电网（`TransGridEnv`），默认 DC 物理。传 `physics='ac'` 切换到 AC。
- **资源**：纯发电；默认不挂电池 / EV / DR。Agent 的工作是分配，不是套利。
- **约束**：线路热稳限制（DC 为线性，AC 为非线性）。`marl_uc` 还从 `units` 表读取最小开机时间、最小停机时间、爬坡、启停成本等约束。

## Agent 设计

| 任务 | 每 agent action | 每 agent observation |
|---|---|---|
| `marl_uc` | `Box(2)` `[score, on_off]`（`on_off ≥ 0.5` 投运机组） | 全局摘要 + 本机参数 + commitment 向量 + 时间 |
| `opf_118` / `opf_118_7d` | `Box(1)` `[score]`（按 softmax 分配净负荷份额） | 全局特征（总负荷、线路潮流、时间）+ 本发电机参数（`p_min`、`p_max`、成本系数） |

Reward 是**共享、合作式**的——总发电成本的负值（UC 还加上启停成本）：

\[
r_t \;=\; -\frac{1}{1000}\,\sum_i \bigl[ C_i(P_{g,i,t}) \cdot u_{i,t} + S_i^{\text{up}} z_{i,t} + S_i^{\text{dn}} w_{i,t} \bigr]
\]

按 [Reward and cost split](../concepts/reward-cost-split.md) 的约定，core env 与 task wrapper 使用显式向量 cost，而不是把 `info["cost"]` 当作规范通道：

- `cost_thermal_overload` — 各线路 `max(|F_k| - F_k^max, 0)` 之和（MW）。
- `cost_voltage_violation` — 仅 AC 模式（pu）。

与 JAX 对齐的 Python TSO CMDP 基准面是 `comparison_tso_centralized`，因为它同时暴露两个选中约束：

- `selected_constraint_names = ("thermal_overload", "reserve_shortfall")`。
- `cost_thresholds = (0.0, 5.0)`。
- `fallback_weights = (1.0, 1.0)`。

旧的 MARL OPF / UC 任务仍然可运行，并且 per-agent cost 字典使用准确的约束名；但本轮不把它们表述为真正的多约束优化器训练。

## 变体与难度维度

`marl_uc` 通过 `obs_mode` 提供一条内置的难度阶梯：

```python
from powerzoo.tasks import make_task_env
easy   = make_task_env("marl_uc", split="train", obs_mode="global")
medium = make_task_env("marl_uc", split="train", obs_mode="local_plus_forecast")
hard   = make_task_env("marl_uc", split="train", obs_mode="local")
```

`opf_118` 的 7 天变体把 `max_steps` 从 48 延长到 336。更长的 horizon 显著加大信用分配难度，因为预测误差与可再生波动会在一周内累积。

## 切分

三个 TSO 任务都用标准的 GB 需求切分：

| 切分 | 日期范围 | 用途 |
|---|---|---|
| `train` | 2023-07-05 – 2024-12-31 | 算法训练 |
| `val` | 2025-01-01 – 2025-06-30 | 超参调优 |
| `test` | 2025-07-01 – 2025-12-15 | 官方基准评估 |

## Baseline

- **Random** — `RandomPolicy(env.action_space)`。定义 `normalized_score = 0`。
- **Rule-based** — `RuleBasedPolicy`（按比例分配 + 始终在线）。一个合理的起步启发式。
- **Oracle (DC-OPF)** — `OraclePolicy` 求解 env 内部使用的同一 DCOPF。定义 `normalized_score = 1`。

## 应报告的指标

- 平均 episode reward（= 负的成本）。
- 对 `OraclePolicy` 的 `normalized_score`。
- 按约束拆分的平均 episode cost，来自 `info["constraint_costs"]` / `selected_constraint_costs`。
- 标量平均 episode cost 仍保留为向后兼容的聚合别名。

## 代码配方

```python
from powerzoo.rl import Trainer

t = Trainer("opf_118", framework="pettingzoo", algorithm="SAC", total_timesteps=2_000_000)
t.train_il()
results = t.evaluate(split="test")
print(results)
```

与 JAX 对齐的中心化 CMDP 基准：

```python
from powerzoo.tasks import make_task_env
env = make_task_env("comparison_tso_centralized")
print(env.selected_constraint_names, env.cost_thresholds)
```

## 另见

- [Transmission physics](../physics/transmission.md) — DC / AC OPF / PF 细节。
- [Python contract](../concepts/python-contract.md) — observation 模式（TSO 默认 `global`）。
- [Training · Trainers](../training/trainers.md)、[Training · Wrappers](../training/wrappers.md)。
- [API · Tasks](../api/tasks.md) — 类签名。
