# 配电网物理

配电层假设**辐射状拓扑**（无环），使用**前推回代（BFS）**而不是输电网常用的稀疏 Newton-Raphson。PowerZoo 提供两个 env：

- `DistGridEnv`（`powerzoo/envs/grid/dist.py`）——单相平衡 BFS。
- `DistGrid3PhaseEnv`（`powerzoo/envs/grid/dist_3phase.py`）——基于 BIBC / BCBV 矩阵的三相不平衡 BFS。

两者都继承自 `GridEnv`；env 栈合约（resource 注册、`info` schema、reset → step 流程）与 `TransGridEnv` 一致。

> **术语速查**。*Radial*（辐射状）——网络是一棵树；每个负荷只有唯一一条路径回变电站。*Feeder*（馈线）——树上的一支，从变电站到叶子节点。*BFS*（Backward-Forward Sweep，前推回代）——迭代 PF：沿树向后求和电流，再向前更新电压。*DistFlow*——本 env 用的线性化 BFS 递推。*VUF*（Voltage Unbalance Factor，电压不平衡因子）——三相线路上各相幅值差异。

## `DistGridEnv` — 单相 BFS

单相 env 求解平衡的 **DistFlow 风格**模型。资源被建模为 **PQ 注入**：每个挂载的资源暴露 `current_p_mw`（可选 `current_q_mvar`），env 在每次 PF 迭代前把它们聚合到对应 bus 上。

关键约定：

- 净负荷采用**负荷为正 / 注入为负**的符号约定（因此放电中的电池作为负的负荷进入 BFS）。
- 馈线首端的交换功率通过 `info['p_slack_MW']` 与 `info['q_slack_MVAr']` 给出。
- 非辐射输入默认会自动剪成 BFS 首访生成树。传 `allow_mesh_pruning=False` 则立即报错。
- 可选的 `load.reactive_mvar` 时序会覆盖推断出的 Q 缩放；否则 env 保留 case 基线中各节点的功率因数。
- 默认的标量 reward 为**仅网损**：`-loss_penalty_weight * p_loss_MW`（`loss_penalty_weight=0.1`）。电压与热稳违反量保留在 `info['cost_voltage_violation']` 与 `info['cost_thermal_overload']`，除非显式启用 soft-penalty 权重（基准实验不建议启用）。

### 收敛与电压崩溃

`DistGridEnv` 区分两种 PF 失败：

| `info` 键 | 含义 |
|---|---|
| `pf_converged` | BFS 达到迭代容差。 |
| `is_diverged` | BFS 在满足容差前先到达 `max_iter`。 |
| `voltage_collapse` | 未做钳位的电压更新进入严重低电压区（即便 BFS 仍在迭代）。 |

数值电压钳位让状态机不至崩溃，但严重低电压会通过 `voltage_collapse=True` 报告，并在 env 层面视为 PF 失败（resource `step()` 仍会被调用，但 agent 应将结果视为不可行）。

### 可用 case

| Case | Bus 数 | 说明 |
|---|---|---|
| `Case33bw` | 33 | IEEE 33-bus 辐射（Baran & Wu）。默认。 |
| `Case118zh` | 118 | 118-bus 配电（Zhang）。`marl_ders_benchmark` 的默认 case。 |
| `Case141` | 141 | 141-bus Caracas 配电。 |
| `Case533mt_lo`、`Case533mt_hi` | 533 | 瑞典 533-bus（低 / 高负荷变体）。 |

## `DistGrid3PhaseEnv` — 三相不平衡

`DistGrid3PhaseEnv` 在 `DistGridEnv` 之上扩展为分相形式。内部对 Kron 展开的 `A/B/C` 状态向量求解 BIBC / BCBV 矩阵递推；分相电压、电流和潮流通过 `obs()` 返回给 agent。

约定：

- 核心 solver 向量采用 **node-major `ABC` 顺序**（`[node1_A, node1_B, node1_C, node2_A, ...]`）。
- `env.topo3ph` 提供物理节点到矩阵的映射，便于检查。
- 串联 `3x3` 阻抗块内的互耦合完全支持。
- 非额定变压器抽头、支路并联 `B`、相移（`ratio` / `angle`）目前**被忽略**——阻抗必须已经编码所需的变压器行为。
- 真正缺相的支线必须在上游 `3x3` 块中编为零阻抗；env 不会自动合成。
- 当 BFS 不收敛时，返回的电压和潮流**仅为末次迭代的诊断量**。信任这些数值前务必先检查 `info['pf_converged']`。
- `safety_check` 在单相基础上增加分相电压限制、VUF 与分相热稳限制。

### 可用三相 case

| Case | Bus 数 | 说明 |
|---|---|---|
| `Case123` | 123 | IEEE 123-bus 三相配电。默认。 |

挂载到三相电网的 resource 可附带可选的 `phase` 参数（`A` / `B` / `C` / `ABC`）。`ABC` 表示把 resource 功率均分到三相。

## DistFlow 物理：一段话概览

对一条从 bus `i` 到 bus `j` 的辐射支路，承载有功 `P` 与无功 `Q`，线路电阻 `R`、电抗 `X`，BFS 更新公式为：

\[
V_j^2 \;\approx\; V_i^2 \;-\; 2\,(R \cdot P + X \cdot Q) \;+\; (R^2 + X^2) \cdot \frac{P^2 + Q^2}{V_i^2}
\]

完整非线性项在配电电压接近 1 pu 时数值很小，但 `DistGridEnv` 仍保留它，以避免系统性低估馈线末端的压降。由于配电网的 `R / X` ≈ 1.0（输电网约为 0.1），有功注入在配电网中对电压有可观影响。这就是为什么配电的 DER 协调本质上是电压问题，而输电则主要是热稳极限问题。

## `info` 中有什么

除标准键（见 [Python contract](../concepts/python-contract.md) §4）外，还有：

- `voltages` — 各 bus 电压幅值（pu）。
- `branch_loading` — 每条线的视在功率 / 额定值之比。
- `p_loss_MW`、`q_loss_MVAr` — 总网损。
- `voltage_collapse`、`is_diverged` — 前述的两个 BFS 失败标志。
- 对 `DistGrid3PhaseEnv`：上述各项的分相版本，再加上 `vuf_pct`。

## 另见

- [Transmission physics](transmission.md) — 输电网 DC / AC OPF / PF 的配套页。
- [Resources](resources.md) — 挂载在配电馈线上的可控资产。
- [Benchmarks · DSO](../benchmarks/dso.md)、[Benchmarks · DERs](../benchmarks/ders.md)。
- [API · Grid](../api/grid.md) — 各方法签名。
