# GenCos — 市场竞价

`gencos_bidding` 是 PowerZoo 的**竞争式 MARL** 基准。5 个发电机 agent 在一个 5-bus 输电网上，每步各自提交一条 3 段单调报价曲线；市场通过网络约束 SCED 清算，每个 agent 收到自己**私有**的分配利润作为 reward。

env 级机制见 [Physics · Markets](../physics/markets.md)。本页聚焦面向 agent 的基准设置。

> **术语速查**。*SCED*（Security-Constrained Economic Dispatch，安全约束经济调度）——把提交的报价作为 LP 目标函数、在热稳限制下的 OPF。*LMP*（Locational Marginal Price，节点边际电价）——节点平衡约束的对偶变量；线路拥塞时不同 bus 的 LMP 不同。*Markup*（加成）——在真实边际成本之上的报价加成比例，是 agent 的策略性手段。*Nash equilibrium*（纳什均衡）——联合策略组合，任何 agent 单方面改变策略都无法提升自身收益。

## 为什么是这个系列

- PowerZoo 中**唯一一个竞争式**系列（其他四个均为合作或单 agent）。
- **唯一带私有 reward** 的系列——每个 agent 的分配利润取决于自己的报价曲线、市场清算结果和所在 bus 的网络 LMP。
- **唯一 reward 取决于报价**（而非真实成本）的系列——策略性报价是核心挑战。
- **唯一在相邻市场区间之间显式带爬坡耦合**的系列——`t` 步的分配会约束 `t+1` 步的可行范围。

综上，`gencos_bidding` 适合用作 self-play、population-based training（PSRO-lite）、independent learner 与博弈论 MARL 方法的测试场景。

## 物理设置

| 项 | 值 |
|---|---|
| 载体 | `Case5`（5-bus IEEE），由 `GenCosMARLEnv` 包装。 |
| Agent | 5 个发电机（`genco_0` … `genco_4`）。 |
| Episode | 48 步 × 30 min = 1 天。 |
| 清算 | `solve_piecewise_ed_opf`（在 3 段报价上的网络约束 SCED）。 |
| 跨步耦合 | 由上一步分配派生的爬坡限制 `[p_min_rt, p_max_rt]`。 |

选用 `Case5` 的原因是线路 4 → 5 存在结构性拥塞，会让 bus 5（便宜的发电机 G5）与 bus 2 / 3 / 4（负荷中心）之间产生持续的 LMP 价差。负荷侧的若干发电机因此获得真正的本地市场力，所以纯 price-taker 难以获得收益。

## Agent 设计

| 项 | 值 |
|---|---|
| 每 agent action | `Box(3) ∈ [-1, 1]` 加成标量；排序后形成单调的 3 段报价曲线。 |
| 每 agent observation | 12 维私有向量——自身成本 + 容量 + 上轮分配 + 上轮利润 + 爬坡余量 + 4 步 LMP 历史 + 需求预测 + 时间。 |
| 每 agent reward | `LMP[node_i] · P_i · Δt - TC_i(P_i) · Δt`（私有利润）。 |
| Constraint cost | `constraint_names = ("thermal_overload",)`；per-agent `info["costs"]["thermal_overload"]` 与固定顺序 `constraint_costs` 暴露同一通道。 |

动作映射：

```
offer_curve[k] = true_mc × (1 + sorted_action[k] × max_markup)   for k in 0..2
```

`max_markup = 2.0`（即报价最高可达真实边际成本的 3 倍）。

## 变体

- **`gencos_bidding`**（主任务，5-agent 部分信息竞争）。
- 2-agent 变体——`G1` 与 `G5` 走策略性报价、其余按真实成本计价——通过把自定义 `case` 与策略性 agent mask 一起传给 `make_gencos_env(...)` 实现。这个变体适合用于 self-play 收敛的健全性测试。

## 切分

基准用标准 GB 需求切分驱动每个 bus 的负荷：

| 切分 | 日期范围 |
|---|---|
| `train` | 2023-07-05 – 2024-12-31 |
| `val` | 2025-01-01 – 2025-06-30 |
| `test` | 2025-07-01 – 2025-12-15 |

env 原生支持的 OOD 维度：

- **需求漂移** — `load × constant`，使系统进入更深的拥塞。
- **可再生冲击** — 缩放太阳 / 风容量因子，改变净负荷形状。
- **爬坡压力** — 构造时收紧爬坡限制，使时间耦合更强。
- **对手漂移** — 用一组对手 seed 训练，再用另一组 seed 评估。

## 代码配方

```python
from powerzoo.envs.market import make_gencos_env

env = make_gencos_env()
obs, info = env.reset(seed=0)
while env.agents:
    actions = {ag: env.action_spaces[ag].sample() for ag in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
```

或通过 registry：

```python
from powerzoo.tasks import make_task_env
env = make_task_env("gencos_bidding", framework="pettingzoo")
```

竞争式训练：

```python
from powerzoo.rl import Trainer

t = Trainer("gencos_bidding", framework="pettingzoo", algorithm="PPO")
t.train_il(total_timesteps=5_000_000)   # independent PPO baseline
```

当前 Python 训练把它作为 CMDP env + reward-only 或 fallback-MDP 学习路径；本轮没有为 GenCos 增加一阶多约束 MARL optimizer。

如需 self-play / PSRO 风格训练：使用 `t.get_env()`，在自己代码中编写种群循环（见 [Training · Custom loops](../training/custom-loops.md)）。

## Baseline

- **Truthful bidding** — 每个 agent 按真实边际成本报价（`markup = 0`）。"社会规划者"参考；等价于中心化 DCOPF 的社会福利。
- **Fixed markup** — 每个 agent 使用 `markup = 0.2`。朴素的策略性基线。
- **Myopic best-response** — 每步，agent 在"假设对手沿用上一步动作"的前提下求解一个 1 步 LP。
- **Random markup** — `markup ~ U[0, 1]`。
- **Independent PPO** — `Trainer.train_il`。
- **Self-play PPO** — 单一共享策略与自身对战训练。

## 应报告的指标

- `cumulative_profit_per_agent` — 跨 seed 的均值 ± std（per agent）。
- `social_welfare_ratio` — `(RL welfare) / (planner welfare)` ∈ `[0, 1]`。
- `price_volatility` — `std(LMP) / mean(LMP)`。
- `market_share_dynamics` — per-agent 的分配份额轨迹。
- `exploitability_proxy` — `max_i (BR_i(π_{-i}) - π_i)`，使用 1 步 myopic best-response。
- `HHI`（Herfindahl-Hirschman Index，赫芬达尔指数）—— 市场集中度。
- `ramp_binding_rate` — 分配触及爬坡边界的步数比例，用于验证时间耦合处于激活状态。

## 另见

- [Physics · Markets](../physics/markets.md) — `CostBasedMarketEnv`、`BidBasedMarketEnv`、`GenCosMARLEnv` 的 env 机制。
- [Transmission physics](../physics/transmission.md) — DC-OPF 与 LMP 推导。
- [Training · Custom loops](../training/custom-loops.md) — self-play 与种群训练循环。
- [API · Markets](../api/market.md)。
