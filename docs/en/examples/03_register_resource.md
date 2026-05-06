# 03 — Register Resources

**Script:** `examples/03_create_grid_and_register_resource.py`

Demonstrates how to attach, inspect and detach resources from a grid environment. This is the lowest-level example — for the same wiring with task semantics, use `make_task_env(...)`; for the full reset → resource step → power-flow → cost flow, see [Architecture · Environment stack](../architecture/env-stack.md).

## Attach a Battery

```python
from powerzoo.envs.grid import TransGridEnv
from powerzoo.envs.resource import BatteryEnv

grid = TransGridEnv()

# Attach battery at bus 1 — registration is automatic
battery = BatteryEnv(
    capacity_mwh=50.0,
    power_mw=20.0,
    parent=grid,
    bus_id=1,
)

print(f"resource_id : {battery.resource_id}")  # battery_0
print(f"bus_id      : {battery.bus_id}")       # 1
print(f"sub_resources: {list(grid.sub_resources.keys())}")
print(battery.status())
```

## Inspect the Bus Map

`nodes_resources_map` is a sparse `(n_nodes, n_resources)` matrix. A `1` in row `i`, column `j` means resource `j` is connected at node `i`. The solver aligns resource power vectors to this column order explicitly rather than relying on plain dictionary iteration order.

```python
print(f"Map shape: {grid.nodes_resources_map.shape}")
print(grid.nodes_resources_map)
# [[0. 0.]
#  [1. 0.]   ← battery_0 at node index 1
#  [0. 1.]   ← battery_1 at node index 2
#  ...]
```

## Add a Second Battery

```python
battery2 = BatteryEnv(parent=grid, bus_id=2)
print(f"second id: {battery2.resource_id}")  # battery_1
print(f"map shape: {grid.nodes_resources_map.shape}")  # (5, 2)
```

## Move a Resource to Another Bus

```python
battery.bus_id = 4   # triggers automatic map rebuild
print(grid.nodes_resources_map)
```

## Detach a Resource

```python
battery2.detach()
print(f"sub_resources after detach: {list(grid.sub_resources.keys())}")
print(f"battery2.resource_id: {battery2.resource_id}")  # None
print(f"battery2.parent: {battery2.parent}")             # None
```

## Step with Resource Action

```python
state, info = grid.reset(day_id=0)

action = {
    battery.resource_id: {"p_mw": 10.0},   # discharge 10 MW
}
state, reward, terminated, truncated, info = grid.step(action)

print(f"Battery SOC  : {battery.soc:.2%}")
print(f"Battery power: {battery.current_p_mw:.1f} MW")
print(f"Reward       : {reward:.2f}")
```

!!! tip
    Resources that are not included in the `action` dict are auto-stepped with
    `action=None` (idle for batteries, profile-driven for renewables).
