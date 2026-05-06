# DERs — 电压 / DER 协调

DERs 系列覆盖**配电馈线上的可扩展 Safe MARL**：多个可控资产共享同一条馈线，各自只看本地信号，必须协同把电压保持在带内。

> **术语速查**。*DER*（Distributed Energy Resource，分布式能源资源）——配电网上的电池、PV、EV 或可控负荷。*Dec-POMDP*（Decentralised, Partially Observable MDP，分布式部分可观测 MDP）——"agent 只看本地状态、共享全局 reward" 这种 MARL 的标准模型。

## 本系列下的任务


| 任务名                   | 载体          | Agent 数                                   | Steps             | 说明                      |
| --------------------- | ----------- | ----------------------------------------- | ----------------- | ----------------------- |
| `marl_der_arbitrage`  | `Case33bw`  | 3 个电池（bus 6 / 12 / 18）                    | 48 × 30 min       | 仅电池组成的 DER 群组，最简单的入口。   |
| `marl_ders_benchmark` | `Case118zh` | 12 个异构 DER（4 Battery + 4 PV + 4 FlexLoad） | 48 × 30 min       | 更大的馈线，混合资源类型，异构策略 MARL。 |
| `marl_ev_v2g`         | `Case33bw`  | 5 辆带 V2G / G2V 的 EV                       | 168 × 60 min（1 周） | 加入可用度屏蔽与出发 SOC 硬截止。     |


三个任务合起来覆盖 3 → 12 的 agent 数维度，并分别独立引入混合资源与 EV 截止两类复杂度。

## 为什么是这个系列

- 唯一一个**所有耦合都通过电压介导**（无拥塞价格、无共享 LMP）的系列，耦合完全由物理驱动。
- 唯一一个具有清晰**可扩展性维度**的系列（同一系列内从 3 → 5 → 12 个 agent）。
- 唯一一个包含**异构 agent 类型**（电池 / PV / FlexLoad）但仍共享全局 reward 的系列。
- 唯一一个在 DER 物理之上叠加**硬截止约束**（EV 出发 SOC）的系列。

## 物理设置


| 任务                    | 载体                      | 资源                                                    |
| --------------------- | ----------------------- | ----------------------------------------------------- |
| `marl_der_arbitrage`  | `Case33bw`（33-bus 辐射）   | 3× `BatteryEnv`                                       |
| `marl_ders_benchmark` | `Case118zh`（118-bus 辐射） | 4× `BatteryEnv`、4× `SolarEnv`（或 PV 逆变器）、4× `FlexLoad` |
| `marl_ev_v2g`         | `Case33bw`              | 5× `VehicleEnv`                                       |


电压限制默认为 `v_min = 0.94 pu`，`v_max = 1.06 pu`。BFS solver 即 [Distribution physics](../physics/distribution.md) 中描述的单相配电模型。

## Agent 设计


| 任务                    | 每 agent action                  | 默认 observation                          |
| --------------------- | ------------------------------- | --------------------------------------- |
| `marl_der_arbitrage`  | `Box(1)` 电池设定 `[-P_max, P_max]` | `local_plus_forecast`（SOC + 电价 + 负荷预测）  |
| `marl_ders_benchmark` | `Box(2)`，按 resource 角色类型化       | `ders_local`（按资源类型组织的 per-agent 向量）     |
| `marl_ev_v2g`         | `Box(1)`（EV 离家时屏蔽为 0）           | `local_plus_forecast`（SOC + 出发就绪度 + 预测） |


三个任务的 reward 都是**共享、合作式**：每个 agent 收到团队的整体经济价值作为 reward。

- `marl_der_arbitrage`：基于本地 LMP 的套利利润。
- `marl_ders_benchmark`：套利利润 + 削减成本的加权和。
- `marl_ev_v2g`：套利利润 + 出发就绪奖励。

按 [Reward and cost split](../concepts/reward-cost-split.md) 的约定，benchmark 使用显式 CMDP 向量 cost：

- `cost_voltage_violation`（三个任务都有）。
- `cost_clipped_power`（电池 / EV SOC 截断）。
- EV 专属：出发 SOC 违反、在家可用度违反（通过任务 adapter 给出）。

与 JAX 对齐的旗舰 DERs benchmark `marl_ders_benchmark` 冻结为：

- `selected_constraint_names = ("voltage_violation", "thermal_overload", "resource")`。
- `cost_thresholds = (0.25, 0.125, 0.125)`。
- `fallback_weights = (4.0, 1.0, 1.0)`。

当前 Python MARL 训练在需要时通过标准化的 MDP fallback reward 使用这些 cost；本轮不把它表述为一阶 constrained MARL optimizer。

## 变体与难度维度

主要的两条难度调节方式是 observation 模式与资源数量：

```python
from powerzoo.tasks import make_task_env
small  = make_task_env("marl_der_arbitrage")
medium = make_task_env("marl_ders_benchmark")
large_horizon = make_task_env("marl_ev_v2g")  # 168 步
```

对 `marl_der_arbitrage`，切换 `obs_mode="local"`（无预测）会让同样的物理问题明显变难；`obs_mode="local_plus_voltage"` 则补上一份馈线电压摘要。

## 切分

三个 DER 任务都用标准切分：


| 切分      | 日期范围                    |
| ------- | ----------------------- |
| `train` | 2023-07-05 – 2024-12-31 |
| `val`   | 2025-01-01 – 2025-06-30 |
| `test`  | 2025-07-01 – 2025-12-15 |


## Baseline

- **Random** — `RandomPolicy(env.action_space)`。通常接近可行（在随机电池设定下电压限制只是轻度绑定），但套利利润为零。
- **Rule-based** — `RuleBasedPolicy`（按固定时段规则做峰谷套利）。
- **Oracle（适用时）** — 对 `marl_der_arbitrage`，完美预知 LP 给出套利利润的上界。

## 应报告的指标

- 平均 episode reward（per agent 与 team 各算一次）。
- `voltage_violation_rate` 与 `max_voltage_deviation` — 安全方面的核心指标。
- 相对于 random / oracle baseline 的 `normalized_score`。
- 对 `marl_ev_v2g`：`departure_readiness_rate`（达到出发 SOC 的 EV 比例）。
- 如同时跑了中心化策略：`decentralization_gap` = `perf(centralised) − perf(MARL)`。

## 代码配方

```python
from powerzoo.rl import Trainer

t = Trainer("marl_der_arbitrage", framework="pettingzoo", algorithm="SAC")
t.train_il(total_timesteps=500_000)
results = t.evaluate(split="test")
```

使用标准 fallback 合约的旗舰 DERs benchmark：

```python
from powerzoo.rl import make_env
env = make_env("marl_ders_benchmark", framework="pettingzoo")
print(env.unwrapped.task.constraint_spec())
```

## 另见

- [Distribution physics](../physics/distribution.md) — 形成电压耦合的 BFS solver。
- [Resources](../physics/resources.md) — `BatteryEnv`、`VehicleEnv`、`SolarEnv`、`FlexLoad` 的参数表。
- [Python contract · Observation modes](../concepts/python-contract.md#5-observation-模式) — `local`、`local_plus_forecast`、`local_plus_voltage`、`ders_local`。
- [Training · Wrappers](../training/wrappers.md) — `MARLWrapper`、`TaskPettingZooWrapper`、`SafeRLWrapper`。
