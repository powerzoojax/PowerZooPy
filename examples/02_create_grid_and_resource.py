import sys
from pathlib import Path

# Ensure project root is importable when running this file directly from examples/.
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

from powerzoo.envs.grid import TransGridEnv
from powerzoo.envs.resource import BatteryEnv
from powerzoo.case import load_case

# Create a default case and print basic info
c = load_case("552GB")  # other ids: 5, '33bw', ...
print(c)


trans_grid = TransGridEnv(case=c)
ptdf = trans_grid.PTDF

node_load_mw = trans_grid.case.loads['d_max'].values
p_min = trans_grid.case.units['p_min'].values
p_max = trans_grid.case.units['p_max'].values
unit_power_mw = p_min + (p_max - p_min) / sum((p_max - p_min)) * (sum(node_load_mw) - sum(p_min))
print(f'Total gen {sum(unit_power_mw):.2f}MW == {sum(node_load_mw):.2f}MW Total load')

print('=' * 40)
line_flow_mw, node_inj_mw = trans_grid.cal_pf(unit_power_mw, node_load_mw, df=True)
print("line_flow_mw")
print(line_flow_mw)
print("node_inj_mw")
print(node_inj_mw)
line_flow_safe, info = trans_grid.safety_check(line_flow_mw, with_info=True)
print("line_flow_safe:", info)
print('=' * 40)

# ===== Add a battery resource to the grid =====
# battery = BatteryEnv(parent=trans_grid, bus_id=2)
# battery.step(20)
# print(f"Battery resource_id: {battery.resource_id}")
# print(f"Battery bus_id: {battery.bus_id}")
# print(f"Grid sub_resources: {list(trans_grid.sub_resources.keys())}")
# print(battery.status())

# print(trans_grid.nodes_resources_map)

# print('=' * 40)
# line_flow_mw, node_inj_mw = trans_grid.cal_pf(unit_power_mw, node_load_mw, df=True)
# print("line_flow_mw")
# print(line_flow_mw)
# print("node_inj_mw")
# print(node_inj_mw)
# line_flow_safe, info = trans_grid.safety_check(line_flow_mw, with_info=True)
# print("line_flow_safe:", info)
# print('=' * 40)
