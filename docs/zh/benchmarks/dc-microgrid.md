# DC microgrid

`dc_microgrid`（与 `dc_microgrid_safe`）是 PowerZoo 的**多目标稳健 RL** 基准。两个任务共享同一个 env，即 `DCMicrogridEnv`：一个自包含的表后微电网（无外部电网），由 `DataCenterEnv` + `BatteryEnv` + 内联光伏 + 柴油机组合而成。safe 任务冻结同一套向量 CMDP 合约，并为仍要求单一 cost 的库暴露标量兼容预算。

物理模型——功率平衡、动作 / 观测 / reward 向量、cost 分量——见 [Physics · Microgrid](../physics/microgrid.md)。本页聚焦面向 agent 的基准设置：变体、切分、baseline 与指标。

> **术语速查**。*PUE*（Power Usage Effectiveness）——总设施功率 / IT 设备功率，越低越好。*SLA*（Service Level Agreement，服务等级协议）——对 workload 完成的保证；错过截止时间即视为 SLA 违反。*Pareto front*（帕累托前沿）——非占优的 `(reward1, reward2, …)` 元组集合；多目标 RL 即在近似该前沿。

## 为什么是这个系列

- 唯一一个**没有外部电网作为后备**的系列：功率平衡是硬内部约束，过度承诺 IT 负荷会直接产生 `cost_power_deficit`。
- 唯一一个在 `info["reward_vector"] = [r_energy, r_cost, r_carbon]` 中给出**向量 reward** 的系列；多目标与 Pareto 方法可直接使用。
- 唯一一个采用**5 分钟分辨率、24 小时 horizon**（288 步）的系列，需要在热惯性下做长时序信用分配。
- 唯一一个把 **AI workload 建模**（含 EDF 调度的训练 + 微调 + 推理队列）与能源管理结合在一起的系列。

## 两个注册任务

| 任务 | CMDP thresholds | 何时使用 |
|---|---|---|
| `dc_microgrid` | 无 | 标准 reward-only 训练；做多目标研究时直接读 reward 向量。 |
| `dc_microgrid_safe` | 对 `("sla", "overtemp", "power_deficit")` 使用 `(0.2, 0.15, 0.15)`；标量别名 `0.5` | 在显式 per-constraint 预算下做 Safe-RL / fallback 基准。 |

两个任务都是单 agent Gymnasium env。

## 物理设置

汇总（完整细节见 [Physics · Microgrid](../physics/microgrid.md)）：

- **载体**：`DCMicrogridEnv` — 自包含，下层没有 `GridEnv`。
- **资源**：`DataCenterEnv`（默认 1000 GPU）、`BatteryEnv`（2 MWh / 0.5 MW）、内联 PV（0.4 MW）、内联柴油机（0.6 MW）。
- **Episode**：288 步 × 5 min = 24 小时。

## Agent 设计

| 项 | 值 |
|---|---|
| Action | `Box(5)` `[r_train, r_finetune, T_cool_norm, P_batt, P_dg]` |
| Observation | `Box(18)`（利用率、队列、热状态、发电可用度、上一步 action、时间） |
| Reward | 标量化 `r_energy + w_cost · r_cost + w_carbon · r_carbon`；同时在 `info["reward_vector"]` 中给出向量 |
| Core constraints | `constraint_names = ("sla", "overtemp", "power_deficit")`；`info["constraint_costs"]` 按这个顺序排列。 |
| Task constraints | `selected_constraint_costs` 使用完整向量，thresholds 为 `(0.2, 0.15, 0.15)`，fallback weights 为 `(1.0, 1.0, 1.0)`。 |
| 标量兼容 | `info["cost"]` 只由兼容 wrapper 写入；`cost_sum` 保留为聚合诊断别名。 |

## 变体

`dc_microgrid` 仅提供一份 env 配置；可派生的变体：

- **Reward 加权**。在 env 构造时调整 `w_cost` 与 `w_carbon`，沿 Pareto 前沿移动。
- **Workload OOD**。通过 `set_profiles(...)` 把 Google 的轨迹替换为 Azure 或 Alibaba DC 轨迹，测试 workload 分布漂移。
- **太阳干旱**。把太阳曲线乘以一个小常数若干天，测试高碳排放体制下的表现。
- **冷却压力**。把室外温度抬高几 °C，迫使 agent 进入高 COP 体制。

## 切分

三条 Google DC 轨迹合在一起作为数据源，按滚动窗口取用。约定：

| 切分 | 轨迹窗口 | 用途 |
|---|---|---|
| `train` | 取自 Google DC 2019-05 之后的 30 天滚动窗口 | 训练 |
| `val` | 不同的 7 天窗口 | 超参调优 |
| `test` | 不同的 7 天窗口 | 报告 |

OOD 切分（workload 换 Azure / Alibaba、太阳干旱、冷却压力）按实验通过 `set_profiles(...)` 以及 `powerzoo.data.dc_microgrid_profiles` 中的 OOD 变换进行配置。

## 代码配方

```python
from powerzoo.tasks import make_task_env

env = make_task_env("dc_microgrid_safe", split="train")
obs, info = env.reset(seed=0)

terminated = truncated = False
while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(info["reward_vector"], info["selected_constraint_costs"])
```

对于需要标量 cost 的 Safe-RL 库，可保持 Gym 5 元组并由 wrapper 写入兼容 `info["cost"]`：

```python
from powerzoo.rl import Trainer

t = Trainer("dc_microgrid_safe", algorithm="SAC", total_timesteps=2_000_000,
            safe_rl=True, cost_threshold=0.5)
t.train()
results = t.evaluate(split="test")
```

多目标训练请用读取 `info["reward_vector"]` 的自定义循环（见 [Training · Custom loops](../training/custom-loops.md)）。

## Baseline

- **Random** — `RandomPolicy`。通常不可行（`cost_power_deficit` 较大）。
- **EDF + 固定冷却** — earliest-deadline-first 调度器加上固定温控设定，是"基于规则"的基线。
- **Solar-aware 启发式** — 光伏出力高时调度训练，其余时段保守运行。
- **完美预知 LP / MILP planner** — 标量 reward 的离线上界。

## 应报告的指标

多目标任务需要完整的向量。核心指标：

- `total_energy_mwh` — 设施总能耗。
- `total_operating_cost_$` — 燃料 + 电池磨损。
- `total_carbon_kg` — 柴油排放量。
- `sla`、`overtemp`、`power_deficit` 三个 per-constraint episode cost。
- `sla_violation_rate`、`thermal_safety_rate`、`power_balance_deficit_mwh` — 三个运行约束维度。
- `renewable_utilization_pct`、`battery_cycling_count`、`diesel_runtime_hours` — 诊断量。
- `pareto_hypervolume` — 多目标方法专用。
- `robustness_gap` — `perf(IID) - perf(workload_swap)`。

## 另见

- [Physics · Microgrid](../physics/microgrid.md) — env 级模型与方程。
- [Resources · DataCenterEnv](../physics/resources.md#datacenterenv-ai-数据中心作为可控负荷)。
- [Architecture · Data pipeline](../architecture/data-pipeline.md) — `dc_microgrid_profiles` loader。
- [Training · Custom loops](../training/custom-loops.md) — 使用 `info["reward_vector"]` 的多目标循环。
