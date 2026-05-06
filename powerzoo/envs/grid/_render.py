"""Visualization helpers for grid environments.

Separated from the environment classes to keep physics/RL logic free of
matplotlib/networkx imports.  Each function receives the full environment
object so callers do not need to extract dozens of attributes individually.

Public functions
----------------
render_trans_grid(env, mode)
    Two-panel figure for TransGridEnv: network topology + unit dispatch chart.

render_dist_grid(env, mode)
    Two-panel figure for DistGridEnv: radial network + voltage profile chart.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import numpy as np

if TYPE_CHECKING:
    from powerzoo.envs.grid.trans import TransGridEnv
    from powerzoo.envs.grid.dist import DistGridEnv


def _import_vis_libs(mode: str):
    """Lazily import matplotlib and networkx; raise a clear error if missing."""
    try:
        import matplotlib
        if mode == 'rgb_array':
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx
        return plt, mpatches, nx
    except ImportError:
        raise ImportError(
            "matplotlib and networkx are required for render(). "
            "Install with: pip install matplotlib networkx"
        )


def _fig_to_rgb_array(fig) -> np.ndarray:
    """Convert a matplotlib figure to an (H, W, 3) uint8 RGB ndarray."""
    import matplotlib.pyplot as plt
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(h, w, 4)[:, :, :3].copy()
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Transmission grid
# ---------------------------------------------------------------------------

def render_trans_grid(env: 'TransGridEnv', mode: str = 'human') -> Optional[np.ndarray]:
    """Render a TransGridEnv state as a two-panel figure.

    Left panel  — Network topology (NetworkX spring layout).  Buses are
                  coloured by net injection; lines are coloured by loading
                  ratio (green < 80 %, yellow 80–95 %, red > 95 %).
    Right panel — Bar chart: unit power output vs. capacity (MW).

    Args:
        env:  A ``TransGridEnv`` instance after at least one solve step.
        mode: ``'human'`` (display interactively) or ``'rgb_array'``
              (return a ``(H, W, 3)`` uint8 ndarray without a window).

    Returns:
        ``None`` for *human* mode, or a numpy ndarray for *rgb_array*.
    """
    plt, mpatches, nx = _import_vis_libs(mode)

    case = env.case
    n_nodes = len(case.nodes)
    n_lines = len(case.lines)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"TransGridEnv — step {env.time_step}  "
        f"({'safe' if env._is_safe else 'UNSAFE'})",
        fontsize=13, fontweight='bold',
    )

    # ---- Left: network topology ----
    ax_net = axes[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    for _, row in case.lines.iterrows():
        G.add_edge(int(row['#from']), int(row['#to']))

    try:
        pos = nx.spring_layout(G, seed=42)
    except Exception:
        pos = {i: (i % 3, i // 3) for i in range(n_nodes)}

    node_colors = ['#AED6F1'] * n_nodes
    if env._nodes is not None and 'node_inj_mw' in env._nodes.columns:
        inj = env._nodes['node_inj_mw'].values
        node_colors = ['#A9DFBF' if inj[i] >= 0 else '#F1948A' for i in range(n_nodes)]

    nx.draw_networkx_nodes(G, pos, ax=ax_net, node_color=node_colors, node_size=400, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax_net, font_size=8)

    edge_colors = ['#7F8C8D'] * n_lines
    edge_widths = [1.5] * n_lines
    if env._lines is not None and 'line_flow_mw' in env._lines.columns:
        flows = env._lines['line_flow_mw'].values
        caps = np.where(case.lines['cap'].values > 0, case.lines['cap'].values, 1e5)
        ratios = np.abs(flows) / caps
        edge_colors = np.where(ratios < 0.80, '#27AE60',
                      np.where(ratios < 0.95, '#F39C12', '#E74C3C')).tolist()
        edge_widths = (1.0 + 3.0 * np.minimum(ratios, 1.0)).tolist()

    nx.draw_networkx_edges(G, pos, ax=ax_net, edge_color=edge_colors,
                           width=edge_widths, arrows=False, alpha=0.8)
    ax_net.legend(handles=[
        mpatches.Patch(color='#27AE60', label='<80 %'),
        mpatches.Patch(color='#F39C12', label='80–95 %'),
        mpatches.Patch(color='#E74C3C', label='>95 % (risk)'),
    ], loc='lower right', fontsize=7)
    ax_net.set_title('Network (line loading)', fontsize=10)
    ax_net.axis('off')

    # ---- Right: unit dispatch bar chart ----
    ax_bar = axes[1]
    units = case.units
    n_units = len(units)
    x = list(range(n_units))

    p_max = units['p_max'].values.astype(float)
    p_min = units['p_min'].values.astype(float)
    p_out = env._unit_power_mw if env._unit_power_mw is not None else np.zeros(n_units)

    bar_colors = ['#3498DB' if p <= pmax * 0.95 else '#E74C3C'
                  for p, pmax in zip(p_out, p_max)]

    ax_bar.bar(x, p_max, color='#BDC3C7', alpha=0.4, label='Capacity (p_max)', zorder=1)
    ax_bar.bar(x, p_out, color=bar_colors, alpha=0.85, label='Output', zorder=2)
    ax_bar.bar(x, p_min, color='#F39C12', alpha=0.4, label='p_min', zorder=3)

    unit_ids = (units['#id'].astype(int).tolist()
                if '#id' in units.columns else list(range(n_units)))
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f'G{i}' for i in unit_ids], fontsize=8)
    ax_bar.set_ylabel('Power (MW)')
    ax_bar.set_title('Unit dispatch', fontsize=10)
    ax_bar.legend(fontsize=7)

    if env._opf_result is not None:
        ax_bar.set_xlabel(f'OPF cost: ${env._opf_result["total_cost"]:.1f}', fontsize=9)

    plt.tight_layout()

    if mode == 'rgb_array':
        return _fig_to_rgb_array(fig)
    return None


# ---------------------------------------------------------------------------
# Distribution grid
# ---------------------------------------------------------------------------

def render_dist_grid(env: 'DistGridEnv', mode: str = 'human') -> Optional[np.ndarray]:
    """Render a DistGridEnv state as a two-panel figure.

    Left panel  — Radial network (spring layout).  Nodes are coloured by
                  voltage magnitude (blue = under-voltage, green = nominal,
                  red = over-voltage).
    Right panel — Voltage profile bar chart with violation bands.

    Args:
        env:  A ``DistGridEnv`` instance after at least one solve step.
        mode: ``'human'`` (display interactively) or ``'rgb_array'``
              (return a ``(H, W, 3)`` uint8 ndarray without a window).

    Returns:
        ``None`` for *human* mode, or a numpy ndarray for *rgb_array*.
    """
    plt, mpatches, nx = _import_vis_libs(mode)

    n_nodes = env.n_nodes

    v_mag = (env._nodes['v_mag'].values
             if env._nodes is not None and 'v_mag' in env._nodes.columns
             else np.ones(n_nodes))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"DistGridEnv — step {env.time_step}  "
        f"({'safe' if env._is_safe else 'UNSAFE'})  "
        f"v_min={env.v_min:.2f}  v_max={env.v_max:.2f}",
        fontsize=12, fontweight='bold',
    )

    # ---- Left: radial network ----
    ax_net = axes[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    for _, row in env.case.lines.iterrows():
        G.add_edge(int(row['#from']), int(row['#to']))

    try:
        pos = nx.spring_layout(G, seed=0, k=2.0)
    except Exception:
        pos = {i: (i % 6, -(i // 6)) for i in range(n_nodes)}

    def _vcol(v):
        if v < env.v_min:
            return '#2980B9'
        if v > env.v_max:
            return '#C0392B'
        return '#27AE60'

    node_colors = [_vcol(v) for v in v_mag]
    nx.draw_networkx_nodes(G, pos, ax=ax_net, node_color=node_colors, node_size=200, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax_net, edge_color='#95A5A6', width=1.2, arrows=False)
    label_subset = {i: str(i) for i in range(0, n_nodes, max(1, n_nodes // 10))}
    nx.draw_networkx_labels(G, pos, labels=label_subset, ax=ax_net, font_size=7)
    ax_net.legend(handles=[
        mpatches.Patch(color='#27AE60', label='Nominal voltage'),
        mpatches.Patch(color='#2980B9', label='Under-voltage'),
        mpatches.Patch(color='#C0392B', label='Over-voltage'),
    ], loc='lower right', fontsize=7)
    ax_net.set_title('Network (voltage)', fontsize=10)
    ax_net.axis('off')

    # ---- Right: voltage profile ----
    ax_v = axes[1]
    bar_colors = [_vcol(v) for v in v_mag]
    ax_v.bar(list(range(n_nodes)), v_mag, color=bar_colors, alpha=0.85, zorder=2)
    ax_v.axhline(1.0,        color='#2ECC71', linewidth=1.2, linestyle='--', label='Nominal 1.0 p.u.')
    ax_v.axhline(env.v_min,  color='#E74C3C', linewidth=1.2, linestyle=':', label=f'v_min={env.v_min}')
    ax_v.axhline(env.v_max,  color='#E74C3C', linewidth=1.2, linestyle=':', label=f'v_max={env.v_max}')
    ax_v.fill_between([-0.5, n_nodes - 0.5], env.v_min, env.v_max,
                      color='#A9DFBF', alpha=0.15, zorder=0, label='Permissible voltage band')
    ax_v.set_xlabel('Node index')
    ax_v.set_ylabel('Voltage magnitude (p.u.)')
    ax_v.set_title(f'Voltage profile  (P_loss={env._p_loss:.3f} MW)', fontsize=10)
    ax_v.legend(fontsize=7)
    ax_v.set_xlim(-0.5, n_nodes - 0.5)
    ax_v.set_ylim(
        min(v_mag.min(), env.v_min) - 0.02,
        max(v_mag.max(), env.v_max) + 0.02,
    )

    plt.tight_layout()

    if mode == 'rgb_array':
        return _fig_to_rgb_array(fig)
    return None
