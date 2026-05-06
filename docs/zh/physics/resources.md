# 资源

一个 **resource** 是挂载到电网上的可控物理资产。PowerZoo 的资源层（`powerzoo/envs/resource/`）覆盖六种资产类型：电池、电动汽车（EV）、光伏、风电、柔性负荷与数据中心。

共同设计原则是：**一个 resource 不是独立 RL env**。它的 `step()` 更新内部状态（SOC、队列、温度……），但*不*返回 reward 或终止信号。Gymnasium `(obs, reward, terminated, truncated, info)` 合约与 `info` 中的 CMDP cost 向量由 `PowerEnv` 加上一个 `Task` 组装。要在单个资源上训练 RL agent，请用一个任务（如对单电池用 `battery_arbitrage`）或写一个自定义 `PowerEnv` 配置——绝不要把 `BatteryEnv` 当成 `gymnasium.Env` 直接调用。

> **术语速查**。*SOC*（State Of Charge，荷电状态）——电池当前储能占容量的比例，0–1。*G2V / V2G* — Grid-to-Vehicle / Vehicle-to-Grid。*DR*（Demand Response，需求响应）——根据电网信号削减或转移负荷。*PUE*（Power Usage Effectiveness）——总设施功率 / IT 设备功率，越低越好。*COP*（Coefficient of Performance，性能系数）——每单位电输入移除的热量。

## `BatteryEnv` — 储能

电池荷电状态演化：

\[
\text{SOC}_{t+1} \;=\; \text{SOC}_t + \frac{\Delta t}{E_{\text{cap}}}
\begin{cases}
-P_t \cdot \eta_{\text{charge}} & \text{if charging } (P_t < 0) \\
-P_t / \eta_{\text{discharge}} & \text{if discharging } (P_t > 0)
\end{cases}
\]

约束 `SOC_min ≤ SOC ≤ SOC_max` 与 `P_min ≤ P_t ≤ P_max`。

默认值：单向 `eta_charge = eta_discharge = 0.95`（因此实际往返效率 ≈ 0.9025）。可选参数 `eta_roundtrip` 是 sqrt 简写：设 `eta_roundtrip = 0.9` 等价于 `eta_charge = eta_discharge = sqrt(0.9) ≈ 0.949`。遗留的 `efficiency` 关键字已弃用。

动作空间为 1D `[P_norm]` ∈ `[-1, 1]`，映射到 `[-power_mw, +power_mw]`。设 `enable_q_control=True` 后动作变为 2D `[P_norm, Q_norm]`。Cost 分量：`cost_clipped_power` = `|desired − feasible|`，即经 SOC / 功率截断后的差值。

## `VehicleEnv` — 电动汽车（G2V / V2G）

SOC 动态与 `BatteryEnv` 相同，并在其上附加三个 EV 专属约束：

- **可用度**：EV 只有停在家时才能充/放电。通勤时段动作被屏蔽为零。
- **出发 SOC**：EV 出发前必须达到 `SOC ≥ SOC_departure`。错过则记为硬违反。
- **随机日程**：出发 / 到达时间在不同 episode 间可能不同。

动作：1D `[P_norm]` ∈ `[-1, 1]`。Observation 为 9 维，包含在家 / 离家标志、出发就绪度、距出发时间以及底层 SOC。Cost 贡献：`cost_clipped_power`（在 EV 离家但仍下达动作时也非零），以及通过任务 adapter 给出的 EV 专属"出发"与"在家可用度"两项 cost。

## `SolarEnv` 与 `WindEnv` — 可再生

两者都继承 `RenewableEnv`。出力由曲线驱动（来自数据管线的 `SOLAR_AVAILABLE_MW` / `WIND_AVAILABLE_MW`），上限为铭牌 `capacity_mw`。Agent 唯一可控的是**弃光/弃风**：1D 动作 `[curtail_frac]` ∈ `[0, 1]`，将出力限制在可用值之下（1.0 = 不弃，0.0 = 全弃）。设 `enable_q_control=True` 后动作变为 2D `[curtail_frac, Q_norm]`。

Observation 为 4 维（启用 Q 控制时 5 维），包含底层容量因子、当前功率与一项弃电 cost 罚。没有 SOC 积分器；唯一的内部 state 是时序索引。

## `FlexLoad` — 需求响应

`FlexLoad` 是 DR 资源。每台设备有两个独立的控制维度：

- **削减**——直接削减需求，上限为 `curtail_cap_mw`。
- **需求转移**——在 `shift_horizon` 窗口内推迟消费，上限为 `shift_cap_mw`。被推迟的能量进入一个**缓冲区**，必须在 horizon 内偿还。

动作 2D `[curtail_mw, shift_out_mw]`，提供三种 scaling 模式（`physical`、`unit`、`tanh`）。Observation 8D — `[curtail_norm, shift_out_norm, shift_in_norm, buffer_fill_ratio, buffer_energy_norm, time_sin, time_cos, price_norm]`。Cost 分量：

| Cost 字段 | 单位 | 含义 |
|---|---|---|
| `cost_buffer_overflow` | MWh | 推迟需求超出 shift horizon。 |
| `cost_curtailment` | $ | 削减能量的不便补偿。 |
| `cost_shift_discomfort` | $ | 缓冲区中推迟需求的持有成本。 |
| `cost_simultaneous` | $ | 互补性违反（同一步内既削减又转移）。 |

`FlexLoad` 还提供一个 LMP 注入接口（`set_lmp`、`get_bid`），与 SCUC / SCED 兼容——见 [Markets](markets.md)。

## `DataCenterEnv` — AI 数据中心作为可控负荷

`DataCenterEnv` 比其他资源更复杂，建模了：

- **GPU 级 IT 功率**：每 GPU 的空闲与活跃功耗、训练与微调任务队列、外生的推理负荷。
- **制冷**：基于 COP 的制冷功率，随冷却设定与室外温度变化。
- **热动力学**：带临界阈值的一阶区域温度模型。
- **Workload**：一个 EDF（Earliest-Deadline-First）调度器消费任务队列。

动作 3D `[r_train, r_finetune, T_cool_setpoint_norm]`：

- `r_train`、`r_finetune` ∈ `[0, 1]` — 分配给各 workload 的可用 GPU 比例。
- `T_cool_setpoint_norm` ∈ `[0, 1]` — 归一化的冷却设定值。

Observation 11D（利用率、队列、温度、COP、电价、时间）。推理 workload 是外生的（呈昼夜变化）。Cost 分量：`cost_overtemp` = `max(t_zone − t_critical, 0)`。

`DataCenterEnv` 被两个任务使用：`dc_scheduling`（部署在配电网上作为柔性负荷）和 `dc_microgrid`（部署在自包含直流微电网内）。

## Cost 汇总

每个 resource 暴露的 cost 字段都自动流入 `info['cost_resource']`，再进入固定顺序的 `info['constraint_costs']` 向量。过渡期仍保留旧别名 `info['cost_resource_violation']`。完整表：

| Resource | Cost 字段 | 单位 |
|---|---|---|
| `BatteryEnv` | `cost_clipped_power` | MW |
| `VehicleEnv` | `cost_clipped_power`（+ 经 adapter 的 EV 专属） | MW |
| `DataCenterEnv` | `cost_overtemp` | °C |
| `FlexLoad` | `cost_buffer_overflow` | MWh |
| `FlexLoad` | `cost_curtailment` | $ |
| `FlexLoad` | `cost_shift_discomfort` | $ |
| `FlexLoad` | `cost_simultaneous` | $ |

`status()` 中任何 `cost_*` 键都会被自动收集；完整约定见 [Reward and cost split](../concepts/reward-cost-split.md)。

## 把 resource 挂到电网上

```python
from powerzoo.envs.grid import TransGridEnv
from powerzoo.envs.resource import BatteryEnv

grid = TransGridEnv()
battery = BatteryEnv(capacity_mwh=50.0, power_mw=20.0, parent=grid, bus_id=2)
print(battery.resource_id)  # 'battery_0'
```

resource 通过 `parent` / `bus_id` 自动完成注册。中途修改 `bus_id` 会经由 property setter 触发映射重建。完整注册数据流见 [Architecture · Environment stack](../architecture/env-stack.md) §3。

## 另见

- [Transmission physics](transmission.md)、[Distribution physics](distribution.md) — resource 挂载的载体。
- [Markets](markets.md) — `FlexLoad` 与 `BatteryEnv` 在市场 env 中的用法。
- [Microgrid](microgrid.md) — `DataCenterEnv` + `BatteryEnv` 在自包含直流微电网中的组合。
- [API · Resources](../api/resource.md) — 各类签名与参数表。
