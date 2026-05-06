"""Example: Distribution Grid with Battery Resource

This example demonstrates adding a battery to the distribution grid
and comparing power flow results before and after battery operation.
"""
import numpy as np
from powerzoo.envs.grid import DistGridEnv
from powerzoo.envs.resource import BatteryEnv


def print_separator(title=""):
    print("\n" + "=" * 80)
    if title:
        print(title)
        print("=" * 80)


def compare_results(nodes_before, lines_before, nodes_after, lines_after, battery_bus):
    """Compare power flow results before and after battery operation"""
    
    # Voltage comparison
    v_before = nodes_before['v_mag'].values
    v_after = nodes_after['v_mag'].values
    v_diff = v_after - v_before
    
    print(f"\n{'Bus':>5} {'V_before':>12} {'V_after':>12} {'Change':>12} {'Change%':>10}")
    print("-" * 55)
    
    for i in range(len(v_before)):
        bus_id = int(nodes_before.iloc[i]['id'])
        change_pct = (v_after[i] - v_before[i]) / v_before[i] * 100 if v_before[i] > 0 else 0
        marker = " <-- Battery" if bus_id == battery_bus else ""
        print(f"{bus_id:>5} {v_before[i]:>12.4f} {v_after[i]:>12.4f} {v_diff[i]:>12.4f} {change_pct:>9.2f}%{marker}")
    
    print("\n" + "-" * 55)
    print(f"Min voltage: {v_before.min():.4f} -> {v_after.min():.4f} (delta {v_after.min() - v_before.min():+.4f})")
    print(f"Max voltage: {v_before.max():.4f} -> {v_after.max():.4f} (delta {v_after.max() - v_before.max():+.4f})")
    
    # Loss comparison
    p_loss_before = lines_before['p_loss_MW'].sum()
    p_loss_after = lines_after['p_loss_MW'].sum()
    q_loss_before = lines_before['q_loss_MVAr'].sum()
    q_loss_after = lines_after['q_loss_MVAr'].sum()
    
    print(f"\nP Loss: {p_loss_before:.4f} MW -> {p_loss_after:.4f} MW (delta {p_loss_after - p_loss_before:+.4f} MW)")
    print(f"Q Loss: {q_loss_before:.4f} MVAr -> {q_loss_after:.4f} MVAr (delta {q_loss_after - q_loss_before:+.4f} MVAr)")
    
    return v_diff


def main():
    print_separator("Distribution Grid with Battery Resource")
    
    # Create distribution grid
    env = DistGridEnv()
    print(f"\nSystem Info:")
    print(f"  Buses: {env.n_nodes}")
    print(f"  Branches: {env.n_lines}")
    print(f"  Base MVA: {env.baseMVA}")
    
    # =========================================================================
    # Run power flow WITHOUT battery
    # =========================================================================
    print_separator("1. Power Flow WITHOUT Battery")
    
    nodes_before, lines_before = env.cal_pf(df=True)
    p_loss_before, q_loss_before = env.get_total_loss(lines_before)
    
    print(f"Converged: {env._converged} in {env._iterations} iterations")
    print(f"Voltage range: {nodes_before['v_mag'].min():.4f} - {nodes_before['v_mag'].max():.4f} p.u.")
    print(f"Min voltage at bus: {int(nodes_before.loc[nodes_before['v_mag'].idxmin(), 'id'])}")
    print(f"Total P loss: {p_loss_before:.4f} MW")
    print(f"Total Q loss: {q_loss_before:.4f} MVAr")
    
    # Safety check
    is_safe, info = env.safety_check(nodes_before, lines_before, v_min=0.95, v_max=1.05, with_info=True)
    print(f"System safe (0.95-1.05): {is_safe}")
    if info['v_violation_nodes']:
        print(f"  Voltage violations at buses: {[i+1 for i in info['v_violation_nodes']]}")
    
    # =========================================================================
    # Add battery to the grid
    # =========================================================================
    print_separator("2. Adding Battery Resource")
    
    # Add battery at bus 18 (lowest voltage bus)
    battery_bus = 18
    battery = BatteryEnv(
        parent=env,
        bus_id=battery_bus,
        capacity_mwh=2.0,  # 2 MWh capacity
        power_mw=1.0,  # 50% of capacity per hour
        initial_soc=0.8,  # Start at 80% SOC
        eta_charge=0.95,
        eta_discharge=0.95,
        normalize_actions=False,
    )
    
    print(f"Battery added:")
    print(f"  Resource ID: {battery.resource_id}")
    print(f"  Bus ID: {battery.bus_id}")
    print(f"  Capacity: {battery.capacity_mwh} MWh")
    print(f"  Max charge/discharge: {battery.power_mw} MW")
    print(f"  Initial SOC: {battery.soc * 100:.1f}%")
    print(f"\nGrid sub_resources: {list(env.sub_resources.keys())}")
    print(f"Nodes-resources map shape: {env.nodes_resources_map.shape}")
    
    # =========================================================================
    # Discharge battery and run power flow
    # =========================================================================
    print_separator("3. Power Flow WITH Battery Discharging")
    
    # Discharge 0.5 MW (500 kW) to support local load
    discharge_power = 0.5  # MW
    battery.step(discharge_power)
    
    print(f"Battery status after discharge command ({discharge_power} MW):")
    print(battery.status())
    
    nodes_after, lines_after = env.cal_pf(df=True)
    p_loss_after, q_loss_after = env.get_total_loss(lines_after)
    
    print(f"\nConverged: {env._converged} in {env._iterations} iterations")
    print(f"Voltage range: {nodes_after['v_mag'].min():.4f} - {nodes_after['v_mag'].max():.4f} p.u.")
    print(f"Min voltage at bus: {int(nodes_after.loc[nodes_after['v_mag'].idxmin(), 'id'])}")
    print(f"Total P loss: {p_loss_after:.4f} MW")
    print(f"Total Q loss: {q_loss_after:.4f} MVAr")
    
    # Safety check
    is_safe, info = env.safety_check(nodes_after, lines_after, v_min=0.95, v_max=1.05, with_info=True)
    print(f"System safe (0.95-1.05): {is_safe}")
    if info['v_violation_nodes']:
        print(f"  Voltage violations at buses: {[i+1 for i in info['v_violation_nodes']]}")
    
    # =========================================================================
    # Compare results
    # =========================================================================
    print_separator("4. Comparison: Before vs After Battery Discharge")
    
    compare_results(nodes_before, lines_before, nodes_after, lines_after, battery_bus)
    
    # =========================================================================
    # Branch flow comparison at key branches
    # =========================================================================
    print_separator("5. Branch Flow Changes (Main Feeder)")
    
    # Show main feeder branches (1-2-3-...-18) and branches with significant changes
    print(f"\n{'Branch':<10} {'P_before':>12} {'P_after':>12} {'dP':>12} {'Q_before':>12} {'Q_after':>12} {'dQ':>12}")
    print("-" * 82)
    
    for idx, row in lines_before.iterrows():
        from_bus = int(row['#from'])
        to_bus = int(row['#to'])
        
        p_before = lines_before.loc[idx, 'p_flow_MW']
        p_after = lines_after.loc[idx, 'p_flow_MW']
        q_before = lines_before.loc[idx, 'q_flow_MVAr']
        q_after = lines_after.loc[idx, 'q_flow_MVAr']
        
        delta_p = p_after - p_before
        
        # Show branches on main path to bus 18 or with significant change
        is_main_path = (from_bus <= 18 and to_bus <= 18 and abs(to_bus - from_bus) == 1)
        is_connected = (from_bus == battery_bus or to_bus == battery_bus)
        has_change = abs(delta_p) > 0.01
        
        if is_main_path or is_connected or (has_change and abs(delta_p) > 0.1):
            marker = " <-- Battery" if is_connected else ""
            print(f"{from_bus:>3}->{to_bus:<4} {p_before:>12.4f} {p_after:>12.4f} {delta_p:>12.4f} "
                  f"{q_before:>12.4f} {q_after:>12.4f} {q_after - q_before:>12.4f}{marker}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_separator("6. Summary")
    
    v_improvement = nodes_after.iloc[battery_bus - 1]['v_mag'] - nodes_before.iloc[battery_bus - 1]['v_mag']
    loss_reduction = p_loss_before - p_loss_after
    
    print(f"\nBattery at bus {battery_bus}:")
    print(f"  Discharge power: {battery.current_p_mw:.3f} MW ({battery.current_p_mw * 1000:.1f} kW)")
    print(f"  SOC: {battery.soc * 100:.1f}%")
    
    print(f"\nVoltage improvement at bus {battery_bus}:")
    print(f"  Before: {nodes_before.iloc[battery_bus - 1]['v_mag']:.4f} p.u.")
    print(f"  After:  {nodes_after.iloc[battery_bus - 1]['v_mag']:.4f} p.u.")
    print(f"  Improvement: {v_improvement * 100:.2f}%")
    
    print(f"\nSystem-wide minimum voltage:")
    print(f"  Before: {nodes_before['v_mag'].min():.4f} p.u.")
    print(f"  After:  {nodes_after['v_mag'].min():.4f} p.u.")
    
    print(f"\nPower loss reduction:")
    print(f"  P loss: {p_loss_before:.4f} -> {p_loss_after:.4f} MW (saved {loss_reduction * 1000:.2f} kW)")
    print(f"  Reduction: {loss_reduction / p_loss_before * 100:.1f}%")
    
    print_separator()
    print("Example completed!")


if __name__ == "__main__":
    main()
