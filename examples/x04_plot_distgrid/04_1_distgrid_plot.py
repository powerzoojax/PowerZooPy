"""Example: Visualize Distribution Grid Power Flow Results

This example demonstrates how to use CasePlotter to visualize:
1. Network topology
2. Power flow results with voltage coloring
3. Voltage profile bar chart
4. Branch loading chart
5. Comprehensive summary plot

Layout options:
- 'feeder': Best for radial distribution networks (horizontal trunk with branches)
- 'tree': Hierarchical top-down tree layout
- 'radial': Concentric circles
- 'spring', 'kamada_kawai', 'spectral': General graph layouts
"""
import sys
import os
sys.path.insert(0, '.')

import matplotlib.pyplot as plt
from powerzoo.envs.grid import DistGridEnv

# Output directory
OUTPUT_DIR = os.path.dirname(__file__)


def main():
    print("=" * 80)
    print("IEEE 33-Bus Distribution System Visualization")
    print("=" * 80)
    
    # =========================================================================
    # 1. Create distribution grid environment and run power flow
    # =========================================================================
    print("\n[1] Creating DistGridEnv and running power flow...")
    
    env = DistGridEnv()
    nodes_df, lines_df = env.cal_pf(df=True)
    p_loss, q_loss = env.get_total_loss(lines_df)
    
    print(f"    Converged: {env._converged} (iterations: {env._iterations})")
    print(f"    Min voltage: {nodes_df['v_mag'].min():.4f} p.u. (Bus {nodes_df.loc[nodes_df['v_mag'].idxmin(), 'id']:.0f})")
    print(f"    Max voltage: {nodes_df['v_mag'].max():.4f} p.u.")
    print(f"    Total P loss: {p_loss:.4f} MW")
    print(f"    Total Q loss: {q_loss:.4f} MVAr")
    
    # =========================================================================
    # 2. Get the plotter from case
    # =========================================================================
    print("\n[2] Accessing CasePlotter from case...")
    
    case = env.case
    plotter = case.plotter  # Lazy initialization via property
    
    print(f"    Case: {case.__class__.__name__}")
    print(f"    Plotter: {plotter.__class__.__name__}")
    
    # =========================================================================
    # 3. Plot network topology with different layouts
    # =========================================================================
    print("\n[3] Plotting network topology (feeder layout)...")
    
    # Use 'feeder' layout - best for radial distribution networks
    fig1, ax1 = plotter.plot_topology(
        figsize=(16, 8),
        layout='feeder',  # New: optimized for radial distribution
        title='IEEE 33-Bus Distribution System Topology (Feeder Layout)',
        highlight_slack=True,
        highlight_loads=True
    )
    plt.savefig(f'{OUTPUT_DIR}/04_1_topology.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: {OUTPUT_DIR}/04_1_topology.png")
    
    # =========================================================================
    # 4. Plot power flow results on topology
    # =========================================================================
    print("\n[4] Plotting power flow results on topology...")
    
    fig2, ax2 = plotter.plot_power_flow(
        nodes_df=nodes_df,
        lines_df=lines_df,
        figsize=(18, 10),
        layout='feeder',  # Use feeder layout
        colormap='RdYlGn_r',
        v_min=0.90,
        v_max=1.02,
        show_voltage_values=True,
        show_flow_values=True,
        flow_label_format='pq',  # Show P and Q values
        show_colorbar=True,
        title='IEEE 33-Bus - Power Flow Results (P/Q in MW/MVAr)',
        edge_width_scale=4.0
    )
    plt.savefig(f'{OUTPUT_DIR}/04_1_powerflow.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: {OUTPUT_DIR}/04_1_powerflow.png")
    
    # =========================================================================
    # 5. Plot voltage profile
    # =========================================================================
    print("\n[5] Plotting voltage profile...")
    
    fig3, ax3 = plotter.plot_voltage_profile(
        nodes_df=nodes_df,
        figsize=(14, 5),
        show_limits=True,
        v_min_limit=0.90,
        v_max_limit=1.10,
        highlight_violations=True,
        title='IEEE 33-Bus Voltage Profile'
    )
    plt.savefig(f'{OUTPUT_DIR}/04_1_voltage_profile.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: {OUTPUT_DIR}/04_1_voltage_profile.png")
    
    # =========================================================================
    # 6. Plot branch loading
    # =========================================================================
    print("\n[6] Plotting branch loading...")
    
    fig4, ax4 = plotter.plot_branch_loading(
        lines_df=lines_df,
        figsize=(16, 5),
        show_p=True,
        show_q=True,
        title='IEEE 33-Bus Branch Power Flow'
    )
    plt.savefig(f'{OUTPUT_DIR}/04_1_branch_loading.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: {OUTPUT_DIR}/04_1_branch_loading.png")
    
    # =========================================================================
    # 7. Create comprehensive summary plot
    # =========================================================================
    print("\n[7] Creating comprehensive summary plot...")
    
    fig5 = plotter.plot_summary(
        nodes_df=nodes_df,
        lines_df=lines_df,
        figsize=(20, 14),
        layout='feeder',  # Use feeder layout
        save_path=f'{OUTPUT_DIR}/04_1_summary.png'
    )
    
    # =========================================================================
    # 8. Compare different layouts (optional)
    # =========================================================================
    print("\n[8] Comparing different layouts...")
    
    fig_compare, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    layouts = ['feeder', 'tree', 'radial', 'kamada_kawai']
    titles = ['Feeder (Best for Distribution)', 'Hierarchical Tree', 
              'Radial (Concentric)', 'Kamada-Kawai']
    
    for ax, layout, title in zip(axes.flatten(), layouts, titles):
        plotter.pos = None  # Reset position cache
        plotter.plot_topology(ax=ax, layout=layout, title=title,
                            highlight_slack=True, highlight_loads=True)
    
    fig_compare.suptitle('Layout Comparison for IEEE 33-Bus System', 
                        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_1_layout_comparison.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: {OUTPUT_DIR}/04_1_layout_comparison.png")
    
    # =========================================================================
    # 9. Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("All plots saved:")
    print(f"  - {OUTPUT_DIR}/04_1_topology.png         : Network topology (feeder)")
    print(f"  - {OUTPUT_DIR}/04_1_powerflow.png        : Power flow results")
    print(f"  - {OUTPUT_DIR}/04_1_voltage_profile.png  : Voltage bar chart")
    print(f"  - {OUTPUT_DIR}/04_1_branch_loading.png   : Branch P/Q flows")
    print(f"  - {OUTPUT_DIR}/04_1_summary.png          : Summary (4-in-1)")
    print(f"  - {OUTPUT_DIR}/04_1_layout_comparison.png: Layout comparison")
    print("=" * 80)
    
    # plt.show()


if __name__ == "__main__":
    main()
