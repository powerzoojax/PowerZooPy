# 输电网物理

`TransGridEnv`（`powerzoo/envs/grid/trans.py`）是输电网 env。它支持四种 solver 模式，由两个正交参数组合而成：

- `physics ∈ {'dc', 'ac'}` — 线性化 vs 完整非线性 AC 方程。
- `solver_mode ∈ {'opf', 'pf'}` — env 是*优化*分配（`opf`），还是仅*评估*给定分配（`pf`）。

四种组合覆盖了标准的输电任务：

| `physics` | `solver_mode` | 对应 solver | 典型 RL 用法 |
|---|---|---|---|
| `'dc'` | `'opf'` | **DCOPF** — 线性规划（Gurobi / SciPy / CVXPY） | Agent 学习报价或 commitment；分配由 env 完成。 |
| `'ac'` | `'opf'` | **ACOPF** — 非线性规划（cyipopt / SLSQP） | Agent 学习带无功与电压的报价。 |
| `'dc'` | `'pf'` | **DCPF** — `line_flow = PTDF · injection`（矩阵-向量乘） | Agent 直接学习分配，env 负责评估可行性。 |
| `'ac'` | `'pf'` | **ACPF** — Newton-Raphson | Agent 学习带完整 AC 物理的分配。 |

OPF LP 后端由 `solver_type ∈ {'auto', 'gurobi', 'scipy', 'cvxpy'}` 决定。它与 `physics` / `solver_mode` 正交，仅决定 OPF 模式下使用哪个 LP 库。

> **术语速查**。*PF*（Power Flow，潮流）——给定固定注入求电压和线路潮流。*OPF*（Optimal Power Flow，最优潮流）——同时调度发电机使成本最小。这里 *DC* 指*线性化*（电压恒为 1 pu，无无功，无损耗），不是直流电。*PTDF*（Power Transfer Distribution Factor）——把节点注入连到线路潮流的灵敏度矩阵。*LMP*（Locational Marginal Price，节点边际电价）——节点功率平衡的对偶变量。

## DC 潮流

DC PF 假设电压为 1 pu，忽略无功，把损耗近似为零：

\[
P_{\text{line}} \;=\; \text{PTDF} \cdot P_{\text{injection}}
\]

PTDF 矩阵在 `case.init()` 阶段由线路电抗预计算。一次 DC PF 调用就是一次矩阵-向量乘：快速、可微，并且与 DCOPF LP 中线路潮流约束所采用的形式完全一致。

## AC 潮流

AC PF 在每个 bus 上求解非线性的节点功率平衡：

\[
P_i = V_i \sum_j V_j (G_{ij} \cos\theta_{ij} + B_{ij} \sin\theta_{ij})
\]

\[
Q_i = V_i \sum_j V_j (G_{ij} \sin\theta_{ij} - B_{ij} \cos\theta_{ij})
\]

其中 \(V_i\) 是电压幅值，\(\theta_{ij}\) 是电压相角差，\(G_{ij}\)、\(B_{ij}\) 来自导纳矩阵。PowerZoo 默认 Newton-Raphson。

分配不可行时 AC PF 可能不收敛。`info['pf_converged']` 报告实际收敛结果；PF 失败本身也是一种实际 cost（见 [Reward and cost split](../concepts/reward-cost-split.md)）。

## DCOPF

DCOPF 求解一个线性规划：

\[
\min_{P_g} \; \sum_i C_i(P_{g,i}) \quad \text{s.t.} \quad
\mathbf{1}^\top P_g = \mathbf{1}^\top D, \quad
|\text{PTDF} \cdot (P_g - D)| \leq \overline{S}, \quad
P_g^{\min} \leq P_g \leq P_g^{\max}
\]

二次成本 \(C_i(P) = mc\_a_i P^2 + mc\_b_i P + mc\_c_i\)。当 `mc_a = mc_b = 0`（`Case5` 与多数 IEEE case）时，成本为常值边际：\(C_i(P) = mc\_c_i \cdot P\)。LMP 由节点平衡约束的对偶变量恢复。

## ACOPF

ACOPF 在 AC 方程与电压限制下求解相同目标。PowerZoo 通过 cyipopt（首选）或 SLSQP 调用。ACOPF 非凸，可能返回局部最优。

## `physics` × `solver_mode` 决策树

```mermaid
flowchart TD
    Q1{Does the agent decide\nunit dispatch (P)?}
    Q1 -->|yes| Q2{Need voltage and Q?}
    Q1 -->|no, env optimises| Q3{Need voltage and Q?}
    Q2 -->|yes| ACPF["physics='ac' + solver_mode='pf'\n→ ACPF"]
    Q2 -->|no| DCPF["physics='dc' + solver_mode='pf'\n→ DCPF"]
    Q3 -->|yes| ACOPF["physics='ac' + solver_mode='opf'\n→ ACOPF"]
    Q3 -->|no| DCOPF["physics='dc' + solver_mode='opf'\n→ DCOPF"]
```

## 可用 case 规模

| Case | Bus 数 | 线路数 | 发电机数 | 默认用于 |
|---|---|---|---|---|
| `Case5` | 5 | 6 | 5 | `marl_opf`、`marl_uc`、`gencos_bidding` |
| `Case14` | 14 | 20 | 5 | sandbox / OOD 规模变体 |
| `Case118` | 118 | 186 | 54 | `opf_118`、`opf_118_7d` |
| `Case300`、`Case1354pegase`、`Case2383wp` | 300+ | 411+ | 80+ | 可扩展性压力测试 |
| `Case29GB` | 29 | 99 | 66 | GB 简化输电网（MATPOWER reduced） |
| `Case552GB` | 552 | 673 | 2385 | GB 大规模输电网 |

`Case5` 与 `Case118` 是两个主要的公开基准 case。更大的 case 用于可扩展性研究，不属于标准任务集。

## `info` 中有什么

每次 `TransGridEnv.step()` 都会填入标准键（见 [Python contract](../concepts/python-contract.md) §4），并在适用时补充以下字段：

- `lmp`（np.ndarray，$/MWh）——OPF 对偶变量给出的节点 LMP。
- `lmp_quality` — `'gurobi_dual'` / `'scipy_recovered'` / `'cvxpy'` — LMP 的计算来源。
- `solver_backend` — 实际使用的 LP solver（如 `'highs'`）。
- `opf_cost`（$/h）——OPF 目标函数给出的总发电成本。

## 难度预设

`TransGridEnv` 接受 `difficulty='easy' / 'medium' / 'hard'`，对应不同的负荷比例与步长：

| Preset | 负荷比例 | `delta_t_minutes` | 效果 |
|---|---|---|---|
| `easy` | 0.7 | 60 | 约束宽松，步数较少。 |
| `medium` | 0.85 | 30 | 标准基准设置。 |
| `hard` | 0.95 | 30 | 多条线路接近极限。 |

预设便于做健全性测试；做基准实验时，建议直接使用显式任务名（如 `marl_opf`），它通过 `SPLIT_DATES` 固定全部超参。

## 另见

- [Distribution physics](distribution.md) — 配电的 BFS / DistFlow 配套页。
- [Resources](resources.md) — 挂在输电网上的可控资产。
- [Markets](markets.md) — `TransGridEnv` 之上的 LMP 驱动结算。
- [Benchmarks · TSO](../benchmarks/tso.md), [Benchmarks · GenCos](../benchmarks/gencos.md)。
- [API · Grid](../api/grid.md) — 各方法签名。
