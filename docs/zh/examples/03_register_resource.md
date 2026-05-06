# 03 — 注册资源

**脚本**：`examples/03_create_grid_and_register_resource.py`

演示如何在 grid env 上挂载、查看与卸载 resource。这是最低层的示例——若需要带任务语义的同样接线，请使用 `make_task_env(...)`；完整的 reset → resource step → 潮流 → cost 流程见 [Architecture · Environment stack](../architecture/env-stack.md)。

## 挂载一个电池

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

## 查看 bus 映射

`nodes_resources_map` 是一个稀疏 `(n_nodes, n_resources)` 矩阵。第 `i` 行第 `j` 列为 `1` 表示 resource `j` 挂在节点 `i`。Solver 显式地按这个列顺序对齐 resource 功率向量，不依赖普通 dict 的迭代顺序。

```python
print(f"Map shape: {grid.nodes_resources_map.shape}")
print(grid.nodes_resources_map)
# [[0. 0.]
#  [1. 0.]   ← battery_0 at node index 1
#  [0. 1.]   ← battery_1 at node index 2
#  ...]
```

## 再挂一个电池

```python
battery2 = BatteryEnv(parent=grid, bus_id=2)
print(f"second id: {battery2.resource_id}")  # battery_1
print(f"map shape: {grid.nodes_resources_map.shape}")  # (5, 2)
```

## 把 resource 移到另一个 bus

```python
battery.bus_id = 4   # triggers automatic map rebuild
print(grid.nodes_resources_map)
```

## 卸载 resource

```python
battery2.detach()
print(f"sub_resources after detach: {list(grid.sub_resources.keys())}")
print(f"battery2.resource_id: {battery2.resource_id}")  # None
print(f"battery2.parent: {battery2.parent}")             # None
```

## 带 resource action 的一步 step

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
    在 `action` dict 中没有出现的 resource 会自动以 `action=None` 进行 step（电池保持空闲，可再生由曲线驱动）。
