import sys
from pathlib import Path

# Ensure project root is importable when running this file directly from examples/.
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

from powerzoo.envs.grid import TransGridEnv
from powerzoo.envs.resource import BatteryEnv
from powerzoo.case import load_case
from powerzoo.envs.grid.cal_dcopf_trans import solve_ed_opf_detailed
import numpy as np
import pandas as pd
import json
# Create a default case and print basic info
c = load_case("552GB")  # other ids: 5, '33bw', ...
print(c)


trans_grid = TransGridEnv(case=c)
ptdf = trans_grid.PTDF

#round to 8
node_load_mw = np.round(trans_grid.case.loads['d_max'].values.astype(float), 8)
p_min = trans_grid.case.units['p_min'].values
p_max = trans_grid.case.units['p_max'].values
# normalize
node_load_mw = node_load_mw / np.sum(node_load_mw)
node_load_mw = node_load_mw * sum(p_max)


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

node_net_load_mw = node_load_mw * 0.5 
# node_net_load_mw = node_net_load_mw * unit_power_mw

print("=" * 80)
print("Testing ED-OPF Solver")
print("=" * 80)

result = solve_ed_opf_detailed(c, node_net_load_mw, verbose=True, solver_type='scipy')

print(f"\nOptimization Status: {result['status']}")
print(f"Total Cost: ${result['total_cost']:.2f}")
print(f"Slack Violation: {result['slack_violation']:.6f}")

print(f"\nUnit Power Output:")
case = c
for i, p in enumerate(result['unit_power_mw']):
    print(
        f"  Unit {i + 1}: {p:.2f} MW (p_min={case.units.iloc[i]['p_min']:.1f}, p_max={case.units.iloc[i]['p_max']:.1f}, mc={case.units.iloc[i]['mc_c']:.1f})")

print(f"\nLine Flow:")
for i, flow in enumerate(result['line_flow_mw']):
    floor = case.lines.iloc[i]['floor']
    cap = case.lines.iloc[i]['cap']
    print(f"  Line {i + 1}: {flow:.2f} MW (floor={floor:.1f}, cap={cap:.1f})")

print(f"\nSystem Balance Check:")
print(f"  Total Generation: {result['unit_power_mw'].sum():.2f} MW")
print(f"  Total Net Load: {node_net_load_mw.sum():.2f} MW")
print(f"  Difference: {abs(result['unit_power_mw'].sum() - node_net_load_mw.sum()):.6f} MW")

# Save optimization results
output_dir = Path(__file__).resolve().parent / "x_OPF_demo_output" 
output_dir.mkdir(parents=True, exist_ok=True)

summary = {
    "status": result["status"],
    "total_cost": float(result["total_cost"]),
    "slack_violation": float(result["slack_violation"]),
    "total_generation_mw": float(result["unit_power_mw"].sum()),
    "total_net_load": float(node_net_load_mw.sum()),
    "difference": float(abs(result["unit_power_mw"].sum() - node_net_load_mw.sum())),
}

unit_df = pd.DataFrame({
    "unit_id": np.arange(1, len(result["unit_power_mw"]) + 1),
    "unit_power_mw": result["unit_power_mw"],
    "p_min_mw": case.units["p_min"].values,
    "p_max_mw": case.units["p_max"].values,
    "mc_c": case.units["mc_c"].values,
})

line_df = pd.DataFrame({
    "line_id": np.arange(1, len(result["line_flow_mw"]) + 1),
    "line_flow_mw": result["line_flow_mw"],
    "floor_mw": case.lines["floor"].values,
    "cap_mw": case.lines["cap"].values,
})

with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

unit_df.to_csv(output_dir / "unit_power_mw.csv", index=False)
line_df.to_csv(output_dir / "line_flow.csv", index=False)

print(f"\nSaved results to: {output_dir}")

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
