from powerzoo.envs.grid import TransGridEnv
from powerzoo.envs.resource import BatteryEnv

trans_grid = TransGridEnv()
print(trans_grid.PTDF)

# ===== Add a battery resource to the grid =====
battery = BatteryEnv(parent=trans_grid, bus_id=1)
print(f"Battery resource_id: {battery.resource_id}")
print(f"Battery bus_id: {battery.bus_id}")
print(f"Grid sub_resources: {list(trans_grid.sub_resources.keys())}")
print(battery.status())

# ===== Test nodes_resources_map =====
print(f"\nNodes-resources map shape: {trans_grid.nodes_resources_map.shape if trans_grid.nodes_resources_map is not None else None}")
print(f"Nodes-resources map:\n{trans_grid.nodes_resources_map}")

# ===== Add another battery resource to test naming =====
battery2 = BatteryEnv(parent=trans_grid, bus_id=2)
print(f"\nSecond battery resource_id: {battery2.resource_id}")
print(f"Second battery bus_id: {battery2.bus_id}")
print(f"Grid sub_resources: {list(trans_grid.sub_resources.keys())}")
print(f"Nodes-resources map shape after adding second battery: {trans_grid.nodes_resources_map.shape if trans_grid.nodes_resources_map is not None else None}")
print(f"Nodes-resources map:\n{trans_grid.nodes_resources_map}")

# ===== Test bus_id setter =====
print(f"\nBefore: battery.bus_id = {battery.bus_id}")
battery.bus_id = 5
print(f"After: battery.bus_id = {battery.bus_id}")

# ===== Test detach =====
print(f"\nBefore detach: Grid sub_resources = {list(trans_grid.sub_resources.keys())}")
print(f"Nodes-resources map shape before detach: {trans_grid.nodes_resources_map.shape if trans_grid.nodes_resources_map is not None else None}")
battery2.detach()
print(f"After detach: Grid sub_resources = {list(trans_grid.sub_resources.keys())}")
print(f"Nodes-resources map shape after detach: {trans_grid.nodes_resources_map.shape if trans_grid.nodes_resources_map is not None else None}")
print(f"Nodes-resources map after detach:\n{trans_grid.nodes_resources_map}")
print(f"Battery2 resource_id after detach: {battery2.resource_id}")
print(f"Battery2 parent after detach: {battery2.parent}")
