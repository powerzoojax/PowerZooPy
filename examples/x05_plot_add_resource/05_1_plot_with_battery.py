"""Example: Plot Distribution Grid with Battery Resource

Visualize power flow results before and after adding a battery.
Battery is shown as a square marker connected to its bus.
"""
import os.path
import sys
sys.path.insert(0, '.')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from powerzoo.envs.grid import DistGridEnv
from powerzoo.envs.resource import BatteryEnv

OUTPUT_DIR = os.path.dirname(__file__)


def main():
    print("=" * 80)
    print("Distribution Grid with Battery - Visualization")
    print("=" * 80)
    
    # Create environment
    env = DistGridEnv()
    plotter = env.case.plotter
    
    # =========================================================================
    # 1. Power flow WITHOUT battery
    # =========================================================================
    print("\n[1] Running power flow WITHOUT battery...")
    nodes_before, lines_before = env.cal_pf(df=True)
    p_loss_before, _ = env.get_total_loss(lines_before)
    v_min_before = nodes_before['v_mag'].min()
    
    print(f"    Min voltage: {v_min_before:.4f} p.u.")
    print(f"    P loss: {p_loss_before:.4f} MW")
    
    # Plot before
    fig1, ax1 = plotter.plot_power_flow(
        nodes_df=nodes_before,
        lines_df=lines_before,
        figsize=(18, 10),
        layout='feeder',
        v_min=0.90,
        v_max=1.02,
        flow_label_format='pq',
        title='IEEE 33-Bus - Before Battery (No Resources)'
    )
    plt.savefig(f'{OUTPUT_DIR}/05_1_before_battery.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: {OUTPUT_DIR}/05_1_before_battery.png")
    
    # =========================================================================
    # 2. Add battery and discharge
    # =========================================================================
    print("\n[2] Adding battery at bus 18...")
    
    battery_bus = 18
    battery = BatteryEnv(
        parent=env,
        bus_id=battery_bus,
        E_max_MWh=2.0,
        p_charge_max_per_hour=0.5,
        p_discharge_max_per_hour=0.5,
        soc_init=0.8,
    )
    
    # Discharge 0.5 MW
    battery.step(0.5)
    print(f"    Battery ID: {battery.resource_id}")
    print(f"    Discharging: {battery.current_p_mw:.2f} MW")
    print(f"    SOC: {battery.soc * 100:.1f}%")
    
    # =========================================================================
    # 3. Power flow WITH battery
    # =========================================================================
    print("\n[3] Running power flow WITH battery...")
    nodes_after, lines_after = env.cal_pf(df=True)
    p_loss_after, _ = env.get_total_loss(lines_after)
    v_min_after = nodes_after['v_mag'].min()
    
    print(f"    Min voltage: {v_min_after:.4f} p.u. (Δ{v_min_after - v_min_before:+.4f})")
    print(f"    P loss: {p_loss_after:.4f} MW (Δ{p_loss_after - p_loss_before:+.4f})")
    
    # Plot with battery using integrated method
    plotter.pos = None  # Reset layout cache
    fig2, ax2 = plotter.plot_power_flow_with_resources(
        nodes_df=nodes_after,
        lines_df=lines_after,
        grid_env=env,
        figsize=(18, 10),
        layout='feeder',
        v_min=0.90,
        v_max=1.02,
        flow_label_format='pq',
        title='IEEE 33-Bus - With Battery Discharging 0.5 MW',
        resource_offset=(0.8, 1.4),
        resource_size=0.9
    )
    plt.savefig(f'{OUTPUT_DIR}/05_1_with_battery.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: {OUTPUT_DIR}/05_1_with_battery.png")
    
    # =========================================================================
    # 4. Comparison plot (side by side)
    # =========================================================================
    print("\n[4] Creating comparison plot...")
    
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Plot before
    plotter.pos = None
    plotter.plot_power_flow(
        nodes_df=nodes_before,
        lines_df=lines_before,
        ax=ax3,
        layout='feeder',
        v_min=0.90,
        v_max=1.02,
        flow_label_format='pq',
        show_colorbar=False,
        title=f'Before Battery\nVmin={v_min_before:.4f} p.u., Ploss={p_loss_before:.4f} MW'
    )
    
    # Plot after with resources
    plotter.pos = None
    plotter.plot_power_flow_with_resources(
        nodes_df=nodes_after,
        lines_df=lines_after,
        grid_env=env,
        ax=ax4,
        layout='feeder',
        v_min=0.90,
        v_max=1.02,
        flow_label_format='pq',
        show_colorbar=False,
        title=f'With Battery (0.5 MW discharge)\nVmin={v_min_after:.4f} p.u., Ploss={p_loss_after:.4f} MW',
        resource_offset=(0.8, 1.4),
        resource_size=0.9
    )
    
    fig3.suptitle('IEEE 33-Bus: Effect of Battery on Power Flow', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_1_comparison.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: {OUTPUT_DIR}/05_1_comparison.png")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Generated files:")
    print(f"  - {OUTPUT_DIR}/05_1_before_battery.png : Power flow without battery")
    print(f"  - {OUTPUT_DIR}/05_1_with_battery.png   : Power flow with battery (square marker)")
    print(f"  - {OUTPUT_DIR}/05_1_comparison.png     : Side-by-side comparison")
    print("=" * 80)
    
    # plt.show()


if __name__ == "__main__":
    main()
