# 02 — Power Flow

**Script:** `examples/02_create_grid_power_flow.py`

This example creates a `TransGridEnv`, runs the built-in OPF-based power flow, and prints the line flows and node injections.

## Transmission (DC OPF)

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
    `TransGridEnv` exposes three orthogonal solver knobs:

    - `physics ∈ {'dc', 'ac'}` — linearised vs full nonlinear AC equations.
    - `solver_mode ∈ {'opf', 'pf'}` — does the env optimise dispatch (`opf`) or only evaluate a given dispatch (`pf`)?
    - `solver_type ∈ {'auto', 'gurobi', 'scipy', 'cvxpy'}` — which **LP backend** runs the OPF.

    The first two pick the physical model and the solver's role; the third only chooses the LP library. For benchmark LMP tasks use `solver_type='auto'`, `'gurobi'` or `'scipy'`. Gurobi returns nodal dual LMPs directly; SciPy reconstructs them from HiGHS dual marginals. The `cvxpy` path is acceptable for OPF-cost experiments but is not treated as benchmark-grade LMP support. The full mapping is in [Physics · Transmission](../physics/transmission.md).

## Distribution (AC BFS)

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

## Manual Cal-PF (advanced)

You can bypass the `step` loop and call `cal_pf` directly to test a specific dispatch:

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
    `cal_pf` uses the PTDF formulation: `line_flow_mw = PTDF @ node_net_injection_mw`.
    The `_run_power_flow` method called during `step` uses OPF by default and falls
    back to `cal_pf` only when `unit_power_mw` is pre-specified in the action dict.
