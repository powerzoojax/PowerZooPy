# 02 — 潮流

**脚本**：`examples/02_create_grid_power_flow.py`

本示例创建一个 `TransGridEnv`，运行内置的 OPF 风格潮流，并打印线路潮流与节点注入。

## 输电（DC OPF）

```python
from powerzoo.envs.grid import TransGridEnv

env = TransGridEnv(
    delta_t_minutes=30,
    start_date="2024-01-01",
    end_date="2024-01-31",
    solver_type="scipy",
)

state, info = env.reset(day_id=0)

lines = state["lines"]
nodes = state["nodes"]

print("Line flows (MW):")
print(lines[["#id", "from_bus", "to_bus", "line_flow_mw", "cap"]].to_string(index=False))

print("\nNode injections (MW):")
print(nodes[["#id", "node_inj_mw"]].to_string(index=False))

print(f"\nOPF total cost : {state['opf_cost']:.2f} $/h")
print(f"Safety         : {'OK' if state['is_safe'] else 'VIOLATION'}")
print(f"Solver backend : {state['solver_backend']}")
print(f"LMP quality    : {state['lmp_quality']}")
print(f"LMP ($/MWh)    :\n{state['lmp']}")
```

!!! note "physics / solver_mode / solver_type"
    `TransGridEnv` 提供三个正交的 solver 参数：

    - `physics ∈ {'dc', 'ac'}` — 线性化 vs 完整非线性 AC 方程。
    - `solver_mode ∈ {'opf', 'pf'}` — env 是优化分配（`opf`）还是仅评估给定分配（`pf`）。
    - `solver_type ∈ {'auto', 'gurobi', 'scipy', 'cvxpy'}` — OPF 使用哪个 **LP 后端**。

    前两个决定物理模型与 solver 角色；第三个只决定使用哪个 LP 库。基准 LMP 任务使用 `solver_type='auto'`、`'gurobi'` 或 `'scipy'`。Gurobi 直接返回节点对偶 LMP；SciPy 从 HiGHS 的对偶边际重构。`cvxpy` 路径可用于 OPF-cost 实验，但不作为基准级 LMP 支持。完整映射见 [Physics · Transmission](../physics/transmission.md)。

## 配电（AC BFS）

```python
from powerzoo.envs.grid import DistGridEnv

env = DistGridEnv()
state, info = env.reset(day_id=0)

print("Node voltages (pu):")
print(state["nodes"][["#id", "voltage"]].to_string(index=False))

print("\nBranch loading (%):")
branches = state["lines"].copy()
branches["loading_%"] = (branches["line_flow_mw"].abs() / branches["cap"] * 100).round(1)
print(branches[["#id", "from_bus", "to_bus", "loading_%"]].to_string(index=False))
```

## 手动调用 cal_pf（进阶）

可以绕过 `step` 循环，直接调用 `cal_pf` 测试某个特定分配：

```python
import numpy as np
from powerzoo.envs.grid import TransGridEnv

env = TransGridEnv()
state, info = env.reset(day_id=0)

# Manually specify unit dispatch
p_min = env.case.units["p_min"].values
p_max = env.case.units["p_max"].values
node_load = env.case.loads["d_max"].values

# Proportional dispatch
unit_power_mw = p_min + (p_max - p_min) / (p_max - p_min).sum() * (node_load.sum() - p_min.sum())

lines, nodes = env.cal_pf(unit_power_mw, node_load, df=True)
line_safe, info = env.safety_check(lines, with_info=True)

print("Unsafe lines:", info["unsafe_line_ids"])
```

!!! note
    `cal_pf` 使用 PTDF 形式：`line_flow_mw = PTDF @ node_net_injection_mw`。`step` 期间调用的 `_run_power_flow` 默认走 OPF；只有当 action dict 中预先指定了 `unit_power_mw` 时，才会回退到 `cal_pf`。
