"""Power System Case Plotter

This module provides visualization utilities for power system networks.
Supports topology plotting with optional power flow annotations.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Any, List
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not installed. Plotting will not be available.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("networkx not installed. Plotting will not be available.")


class CasePlotter:
    """Plotter for power system case visualization
    
    Provides methods to visualize network topology and power flow results.
    
    Attributes:
        case: The power system case object
        G: NetworkX graph representation of the network
        pos: Node positions for plotting
    """
    
    def __init__(self, case: Any):
        """Initialize plotter with a case object
        
        Args:
            case: Power system case object (must have nodes and lines DataFrames)
        """
        if not MATPLOTLIB_AVAILABLE or not NETWORKX_AVAILABLE:
            raise ImportError("matplotlib and networkx are required for plotting")
        
        self.case = case
        self.G = None
        self.pos = None
        self._build_graph()
    
    def _build_graph(self) -> None:
        """Build NetworkX graph from case data"""
        self.G = nx.Graph()
        
        # Add nodes
        for idx, row in self.case.nodes.iterrows():
            node_id = int(row['id'])
            self.G.add_node(node_id, **row.to_dict())
        
        # Add edges (only in-service lines)
        for idx, row in self.case.lines.iterrows():
            if row.get('status', 1) == 1:
                from_node = int(row['from'])
                to_node = int(row['to'])
                self.G.add_edge(from_node, to_node, **row.to_dict())
    
    def _has_geo_coords(self) -> bool:
        """Check if case nodes have meaningful (x, y) coordinates."""
        nodes = self.case.nodes
        if 'x' not in nodes.columns or 'y' not in nodes.columns:
            return False
        # At least two distinct non-zero positions
        coords = nodes[['x', 'y']].values.astype(float)
        nonzero = (coords != 0).any(axis=1).sum()
        return nonzero >= 2 and len(set(map(tuple, coords))) >= 2

    def _geo_layout(self) -> Dict[int, Tuple[float, float]]:
        """Build layout from the (x, y) columns in case.nodes."""
        pos = {}
        for _, row in self.case.nodes.iterrows():
            pos[int(row['id'])] = (float(row['x']), float(row['y']))
        return pos

    def _compute_layout(self, layout: str = 'auto') -> Dict[int, Tuple[float, float]]:
        """Compute node positions for plotting

        Args:
            layout: Layout algorithm ('auto', 'geo', 'spring', 'kamada_kawai',
                    'spectral', 'radial', 'tree', 'feeder').
                    ``'geo'`` reads (x, y) from case.nodes directly.
                    ``'auto'`` uses geo if available, else heuristic.

        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        # --- resolve 'auto' ---
        if layout == 'auto':
            if self._has_geo_coords():
                layout = 'geo'
            elif nx.is_tree(self.G):
                layout = 'feeder'
            else:
                layout = 'kamada_kawai'

        # --- explicit 'geo' ---
        if layout == 'geo':
            if self._has_geo_coords():
                self.pos = self._geo_layout()
                return self.pos
            # fallback if requested but missing
            layout = 'kamada_kawai'

        if layout == 'spring':
            self.pos = nx.spring_layout(self.G, seed=42, k=2.0, iterations=100)
        elif layout == 'kamada_kawai':
            self.pos = nx.kamada_kawai_layout(self.G)
        elif layout == 'spectral':
            self.pos = nx.spectral_layout(self.G)
        elif layout == 'radial':
            # Concentric circles layout
            root = min(self.G.nodes())
            try:
                bfs_tree = nx.bfs_tree(self.G, root)
                self.pos = self._radial_tree_layout(bfs_tree, root)
            except Exception:
                self.pos = nx.kamada_kawai_layout(self.G)
        elif layout == 'tree':
            # Hierarchical tree layout (top-down)
            root = min(self.G.nodes())
            try:
                bfs_tree = nx.bfs_tree(self.G, root)
                self.pos = self._hierarchical_tree_layout(bfs_tree, root)
            except Exception:
                self.pos = nx.kamada_kawai_layout(self.G)
        elif layout == 'feeder':
            # Distribution feeder layout (best for radial networks)
            root = min(self.G.nodes())
            try:
                bfs_tree = nx.bfs_tree(self.G, root)
                self.pos = self._feeder_layout(bfs_tree, root)
            except Exception:
                self.pos = nx.kamada_kawai_layout(self.G)
        else:
            self.pos = nx.spring_layout(self.G, seed=42)
        
        return self.pos
    
    def _radial_tree_layout(self, tree: nx.DiGraph, root: int) -> Dict[int, Tuple[float, float]]:
        """Compute radial layout for a tree graph (concentric circles)
        
        Args:
            tree: Directed tree graph
            root: Root node
        
        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        pos = {}
        
        # Get levels using BFS
        levels = {root: 0}
        queue = [root]
        while queue:
            node = queue.pop(0)
            for child in tree.successors(node):
                if child not in levels:
                    levels[child] = levels[node] + 1
                    queue.append(child)
        
        # Group nodes by level
        level_nodes = {}
        for node, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)
        
        max_level = max(levels.values())
        
        # Position root at center
        pos[root] = (0, 0)
        
        # Position other nodes in concentric circles
        for level in range(1, max_level + 1):
            nodes = level_nodes.get(level, [])
            n_nodes = len(nodes)
            if n_nodes == 0:
                continue
            
            radius = level * 1.2
            
            # Sort by parent position for better layout
            parent_angles = {}
            for node in nodes:
                parents = list(tree.predecessors(node))
                if parents and parents[0] in pos:
                    parent = parents[0]
                    parent_angles[node] = np.arctan2(pos[parent][1], pos[parent][0])
                else:
                    parent_angles[node] = 0
            
            nodes_sorted = sorted(nodes, key=lambda n: parent_angles[n])
            
            for i, node in enumerate(nodes_sorted):
                angle = -np.pi/2 + (i + 0.5) * 2 * np.pi / n_nodes
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                pos[node] = (x, y)
        
        return pos
    
    def _hierarchical_tree_layout(self, tree: nx.DiGraph, root: int) -> Dict[int, Tuple[float, float]]:
        """Compute hierarchical tree layout (top-down)
        
        Args:
            tree: Directed tree graph
            root: Root node
        
        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        pos = {}
        
        # BFS to get levels
        levels = {root: 0}
        queue = [root]
        while queue:
            node = queue.pop(0)
            for child in tree.successors(node):
                if child not in levels:
                    levels[child] = levels[node] + 1
                    queue.append(child)
        
        # Group by level
        level_nodes = {}
        for node, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)
        
        max_level = max(levels.values())
        
        # Calculate subtree sizes for better horizontal spacing
        subtree_size = {}
        def calc_subtree_size(node):
            children = list(tree.successors(node))
            if not children:
                subtree_size[node] = 1
            else:
                subtree_size[node] = sum(calc_subtree_size(c) for c in children)
            return subtree_size[node]
        calc_subtree_size(root)
        
        # Position nodes
        def position_subtree(node, x_min, x_max, y):
            children = list(tree.successors(node))
            x = (x_min + x_max) / 2
            pos[node] = (x, y)
            
            if children:
                total_size = sum(subtree_size[c] for c in children)
                current_x = x_min
                for child in children:
                    child_width = (x_max - x_min) * subtree_size[child] / total_size
                    position_subtree(child, current_x, current_x + child_width, y - 1.5)
                    current_x += child_width
        
        total_width = subtree_size[root] * 2
        position_subtree(root, -total_width/2, total_width/2, 0)
        
        return pos
    
    def _feeder_layout(self, tree: nx.DiGraph, root: int) -> Dict[int, Tuple[float, float]]:
        """Compute feeder layout optimized for radial distribution networks
        
        Places main feeder horizontally, with branches going up/down.
        
        Args:
            tree: Directed tree graph
            root: Root node
        
        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        pos = {}
        
        # Find the main trunk (longest path from root)
        def find_longest_path(node, path=[]):
            path = path + [node]
            children = list(tree.successors(node))
            if not children:
                return path
            
            longest = path
            for child in children:
                child_path = find_longest_path(child, path)
                if len(child_path) > len(longest):
                    longest = child_path
            return longest
        
        main_trunk = find_longest_path(root)
        main_trunk_set = set(main_trunk)
        
        # Position main trunk horizontally
        for i, node in enumerate(main_trunk):
            pos[node] = (i * 1.2, 0)
        
        # Find all branch points and their branches
        def get_subtree_nodes(node, exclude_set):
            """Get all nodes in subtree excluding certain nodes"""
            nodes = [node]
            for child in tree.successors(node):
                if child not in exclude_set:
                    nodes.extend(get_subtree_nodes(child, exclude_set))
            return nodes
        
        # Process each branch point on main trunk
        branch_direction = 1  # Alternate up/down
        branch_y_offset = 1.5
        
        for trunk_node in main_trunk:
            children = list(tree.successors(trunk_node))
            
            # Find children that are NOT on main trunk (these are branch starts)
            branch_children = [c for c in children if c not in main_trunk_set]
            
            for _, branch_start in enumerate(branch_children):
                # Get all nodes in this branch
                branch_nodes = get_subtree_nodes(branch_start, main_trunk_set)
                
                # Determine direction for this branch
                direction = branch_direction
                branch_direction *= -1  # Alternate
                
                # Position branch nodes
                trunk_x = pos[trunk_node][0]
                
                # BFS within the branch
                branch_levels = {branch_start: 0}
                queue = [branch_start]
                while queue:
                    node = queue.pop(0)
                    for child in tree.successors(node):
                        if child not in branch_levels and child not in main_trunk_set:
                            branch_levels[child] = branch_levels[node] + 1
                            queue.append(child)
                
                # Group by level
                level_nodes = {}
                for node, level in branch_levels.items():
                    if level not in level_nodes:
                        level_nodes[level] = []
                    level_nodes[level].append(node)
                
                # Position branch nodes
                for level, nodes in level_nodes.items():
                    for i, node in enumerate(nodes):
                        y = direction * (branch_y_offset + level * 1.0)
                        # Spread horizontally if multiple nodes at same level
                        x = trunk_x + (level + 1) * 1.2 + i * 0.3
                        pos[node] = (x, y)
        
        return pos
    
    def _build_bus_annotations(self) -> Dict[int, str]:
        """Build per-bus annotation strings from units and nodes data."""
        ann: Dict[int, List[str]] = {}
        # Generators (from units table)
        if hasattr(self.case, 'units') and self.case.units is not None:
            for _, u in self.case.units.iterrows():
                bid = int(u['bus_id'])
                p_max = u.get('p_max', None)
                label = f"G{int(u['id'])}"
                if p_max is not None:
                    label += f" {p_max:.0f}MW"
                ann.setdefault(bid, []).append(label)
        # Loads (from nodes.Pd if present)
        for _, row in self.case.nodes.iterrows():
            nid = int(row['id'])
            pd_val = row.get('Pd', 0)
            if pd_val and float(pd_val) > 0:
                ann.setdefault(nid, []).append(f"L {float(pd_val):.0f}MW")
        return {k: "\n".join(v) for k, v in ann.items()}

    def plot_topology(self,
                      ax: Optional[plt.Axes] = None,
                      figsize: Tuple[int, int] = (14, 10),
                      layout: str = 'auto',
                      node_size: int = 500,
                      node_color: str = '#4A90D9',
                      edge_color: str = '#666666',
                      edge_width: float = 1.5,
                      show_node_labels: bool = True,
                      show_edge_labels: bool = False,
                      show_annotations: bool = True,
                      title: str = None,
                      highlight_slack: bool = True,
                      highlight_loads: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """Plot network topology
        
        Args:
            ax: Matplotlib axes (creates new figure if None)
            figsize: Figure size (width, height)
            layout: Layout algorithm ('auto', 'spring', 'kamada_kawai', 'spectral', 'radial')
            node_size: Size of nodes
            node_color: Default color for PQ buses
            edge_color: Color of edges
            edge_width: Width of edges
            show_node_labels: Whether to show node ID labels
            show_edge_labels: Whether to show edge labels (from-to)
            show_annotations: Whether to show generator/load annotations
            title: Plot title
            highlight_slack: Highlight slack bus in different color
            highlight_loads: Show load magnitude with node size
        
        Returns:
            (figure, axes) tuple
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=300)
        else:
            fig = ax.get_figure()
        
        # Compute layout
        self._compute_layout(layout)
        
        # Determine node colors
        # type 3 = reference bus (sets θ=0 in DCOPF; slack in ACPF)
        node_colors = []
        for node in self.G.nodes():
            node_type = self.G.nodes[node].get('type', 1)
            if node_type == 3 and highlight_slack:
                node_colors.append('#E74C3C')  # Red for reference
            elif node_type == 2:
                node_colors.append('#27AE60')  # Green for PV / gen
            else:
                node_colors.append(node_color)  # Blue for PQ / load
        
        # Determine node sizes based on load (capped to avoid extreme sizes)
        if highlight_loads:
            node_sizes = []
            loads = []
            for node in self.G.nodes():
                pd_val = self.G.nodes[node].get('Pd', 0)
                loads.append(float(pd_val) if pd_val else 0.0)
            max_load = max(loads) if max(loads) > 0 else 1
            for load in loads:
                size = node_size * (0.6 + 0.8 * load / max_load)
                node_sizes.append(size)
        else:
            node_sizes = [node_size] * len(self.G.nodes())
        
        # Draw edges
        nx.draw_networkx_edges(self.G, self.pos, ax=ax,
                              edge_color=edge_color,
                              width=edge_width,
                              alpha=0.7)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax,
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.9,
                              edgecolors='white',
                              linewidths=2)
        
        # Draw node labels
        if show_node_labels:
            nx.draw_networkx_labels(self.G, self.pos, ax=ax,
                                   font_size=9,
                                   font_color='white',
                                   font_weight='bold')
        
        # Draw edge labels
        if show_edge_labels:
            edge_labels = {(u, v): f"{u}-{v}" for u, v in self.G.edges()}
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels,
                                        ax=ax, font_size=7)

        # Annotate generators and loads
        if show_annotations:
            annotations = self._build_bus_annotations()
            for node, text in annotations.items():
                if node in self.pos:
                    x, y = self.pos[node]
                    ax.annotate(
                        text, (x, y),
                        textcoords="offset points", xytext=(0, -18),
                        ha='center', va='top',
                        fontsize=6.5, color='#333333',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  fc='#FFFFFFCC', ec='#CCCCCC', lw=0.5),
                    )

        # Legend — only show bus types actually present
        type_set = {self.G.nodes[n].get('type', 1) for n in self.G.nodes()}
        legend_elements = []
        if 3 in type_set:
            legend_elements.append(mpatches.Patch(color='#E74C3C', label='Ref Bus'))
        if 2 in type_set:
            legend_elements.append(mpatches.Patch(color='#27AE60', label='Gen Bus'))
        if 1 in type_set:
            legend_elements.append(mpatches.Patch(color=node_color, label='Load Bus'))
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            case_name = getattr(self.case, 'name',
                                type(self.case).__name__)
            n_bus = len(self.case.nodes)
            n_line = len(self.case.lines) if hasattr(self.case, 'lines') else 0
            ax.set_title(f"{case_name}  ({n_bus} buses, {n_line} lines)",
                         fontsize=13, fontweight='bold')
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        return fig, ax
    
    def plot_power_flow(self,
                        nodes_df: pd.DataFrame,
                        lines_df: pd.DataFrame,
                        ax: Optional[plt.Axes] = None,
                        figsize: Tuple[int, int] = (16, 12),
                        layout: str = 'auto',
                        colormap: str = 'RdYlGn_r',
                        v_min: float = 0.9,
                        v_max: float = 1.1,
                        show_voltage_values: bool = True,
                        show_flow_values: bool = True,
                        flow_label_format: str = 'pq',
                        show_colorbar: bool = True,
                        title: str = None,
                        edge_width_scale: float = 3.0) -> Tuple[plt.Figure, plt.Axes]:
        """Plot power flow results on network topology
        
        Args:
            nodes_df: DataFrame with node results (must have 'v_mag' column)
            lines_df: DataFrame with line results (must have 'p_flow_MW' column)
            ax: Matplotlib axes (creates new figure if None)
            figsize: Figure size
            layout: Layout algorithm
            colormap: Colormap for voltage magnitude
            v_min: Min voltage for colormap
            v_max: Max voltage for colormap
            show_voltage_values: Show voltage magnitude on nodes
            show_flow_values: Show power flow on edges
            flow_label_format: Format for flow labels:
                - 'p': Show only P (MW)
                - 'pq': Show P/Q (MW/MVAr)
                - 's': Show apparent power S (MVA)
            show_colorbar: Show colorbar for voltage
            title: Plot title
            edge_width_scale: Scale factor for edge width based on flow
        
        Returns:
            (figure, axes) tuple
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        # Compute layout
        self._compute_layout(layout)
        
        # Create voltage color map
        cmap = plt.get_cmap(colormap)
        norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
        
        # Get voltage values for each node
        node_voltages = {}
        for idx, row in nodes_df.iterrows():
            node_id = int(row.get('id', idx))
            node_voltages[node_id] = row['v_mag']
        
        # Node colors based on voltage
        node_colors = [cmap(norm(node_voltages.get(node, 1.0))) for node in self.G.nodes()]
        
        # Get power flow values (P and Q)
        line_flows_p = {}
        line_flows_q = {}
        for idx, row in lines_df.iterrows():
            from_node = int(row.get('#from', row.get('from', 0)))
            to_node = int(row.get('#to', row.get('to', 0)))
            p_flow = row.get('p_flow_MW', 0)
            q_flow = row.get('q_flow_MVAr', 0)
            line_flows_p[(from_node, to_node)] = abs(p_flow)
            line_flows_p[(to_node, from_node)] = abs(p_flow)
            line_flows_q[(from_node, to_node)] = abs(q_flow)
            line_flows_q[(to_node, from_node)] = abs(q_flow)
        
        # Edge widths based on power flow
        max_flow = max(line_flows_p.values()) if line_flows_p else 1
        edge_widths = []
        for u, v in self.G.edges():
            flow = line_flows_p.get((u, v), 0)
            width = 1 + edge_width_scale * flow / max_flow
            edge_widths.append(width)
        
        # Draw edges with varying width
        edges = list(self.G.edges())
        edge_colors = ['#555555'] * len(edges)
        
        nx.draw_networkx_edges(self.G, self.pos, ax=ax,
                              edge_color=edge_colors,
                              width=edge_widths,
                              alpha=0.6)
        
        # Draw nodes
        node_size = 600
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax,
                              node_color=node_colors,
                              node_size=node_size,
                              alpha=0.95,
                              edgecolors='#333333',
                              linewidths=2)
        
        # Draw node labels (node ID)
        nx.draw_networkx_labels(self.G, self.pos, ax=ax,
                               font_size=8,
                               font_color='black',
                               font_weight='bold')
        
        # Add voltage values as annotations
        if show_voltage_values:
            for node in self.G.nodes():
                x, y = self.pos[node]
                v = node_voltages.get(node, 1.0)
                ax.annotate(f'{v:.3f}', xy=(x, y), xytext=(0, -15),
                           textcoords='offset points',
                           ha='center', va='top',
                           fontsize=7, color='#333333',
                           bbox=dict(boxstyle='round,pad=0.2',
                                   facecolor='white', alpha=0.7,
                                   edgecolor='none'))
        
        # Add power flow values on edges
        if show_flow_values:
            for u, v in self.G.edges():
                p_flow = line_flows_p.get((u, v), 0)
                q_flow = line_flows_q.get((u, v), 0)
                
                if p_flow > 0.01:  # Only show significant flows
                    x = (self.pos[u][0] + self.pos[v][0]) / 2
                    y = (self.pos[u][1] + self.pos[v][1]) / 2
                    
                    # Format label based on flow_label_format
                    if flow_label_format == 'p':
                        label = f'{p_flow:.2f} MW'
                    elif flow_label_format == 'pq':
                        label = f'P={p_flow:.2f}\nQ={q_flow:.2f}'
                    elif flow_label_format == 's':
                        s_flow = np.sqrt(p_flow**2 + q_flow**2)
                        label = f'{s_flow:.2f} MVA'
                    else:
                        label = f'{p_flow:.2f}'
                    
                    # Offset perpendicular to edge
                    dx = self.pos[v][0] - self.pos[u][0]
                    dy = self.pos[v][1] - self.pos[u][1]
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        offset_x = -dy / length * 0.2
                        offset_y = dx / length * 0.3
                    else:
                        offset_x = offset_y = 0
                    
                    ax.annotate(label, xy=(x + offset_x, y + offset_y),
                               ha='center', va='center',
                               fontsize=6, color='#333333',
                               bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor='#FFFFCC', alpha=0.85,
                                       edgecolor='#CCCC00', linewidth=0.5))
        
        # Colorbar
        if show_colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
            cbar.set_label('Voltage Magnitude (p.u.)', fontsize=10)
        
        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            case_name = getattr(self.case, 'name', 'Power System')
            ax.set_title(f"{case_name} - Power Flow Results", fontsize=14, fontweight='bold')
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        return fig, ax
    
    def plot_power_flow_with_resources(self,
                                       nodes_df: pd.DataFrame,
                                       lines_df: pd.DataFrame,
                                       grid_env: Any,
                                       ax: Optional[plt.Axes] = None,
                                       figsize: Tuple[int, int] = (18, 10),
                                       layout: str = 'auto',
                                       colormap: str = 'RdYlGn_r',
                                       v_min: float = 0.9,
                                       v_max: float = 1.1,
                                       show_voltage_values: bool = True,
                                       show_flow_values: bool = True,
                                       flow_label_format: str = 'pq',
                                       show_colorbar: bool = True,
                                       title: str = None,
                                       edge_width_scale: float = 4.0,
                                       resource_offset: Tuple[float, float] = (0.8, 0.6),
                                       resource_size: float = 0.6) -> Tuple[plt.Figure, plt.Axes]:
        """Plot power flow results with resources shown as squares
        
        Args:
            nodes_df: DataFrame with node results
            lines_df: DataFrame with line results
            grid_env: Grid environment with sub_resources
            ax: Matplotlib axes
            figsize: Figure size
            layout: Layout algorithm
            colormap: Colormap for voltage
            v_min: Min voltage for colormap
            v_max: Max voltage for colormap
            show_voltage_values: Show voltage on nodes
            show_flow_values: Show power flow on edges
            flow_label_format: Flow label format ('p', 'pq', 's')
            show_colorbar: Show colorbar
            title: Plot title
            edge_width_scale: Scale for edge width
            resource_offset: (x, y) offset for resource position from node
            resource_size: Size of resource square
        
        Returns:
            (figure, axes) tuple
        """
        # First plot the power flow
        fig, ax = self.plot_power_flow(
            nodes_df=nodes_df,
            lines_df=lines_df,
            ax=ax,
            figsize=figsize,
            layout=layout,
            colormap=colormap,
            v_min=v_min,
            v_max=v_max,
            show_voltage_values=show_voltage_values,
            show_flow_values=show_flow_values,
            flow_label_format=flow_label_format,
            show_colorbar=show_colorbar,
            title=title,
            edge_width_scale=edge_width_scale
        )
        
        # Draw resources as squares
        if hasattr(grid_env, 'sub_resources') and grid_env.sub_resources:
            for res_id, resource in grid_env.sub_resources.items():
                bus_id = resource.bus_id
                
                if bus_id in self.pos:
                    node_x, node_y = self.pos[bus_id]
                    
                    # Offset resource position
                    res_x = node_x + resource_offset[0]
                    res_y = node_y + resource_offset[1]
                    
                    # Draw connection line
                    ax.plot([node_x, res_x], [node_y, res_y], 
                           color='#8B4513', linewidth=2, linestyle='-', zorder=1)
                    
                    # Draw resource as square
                    square = mpatches.FancyBboxPatch(
                        (res_x - resource_size/2, res_y - resource_size/2),
                        resource_size, resource_size,
                        boxstyle="square,pad=0.02",
                        facecolor='#9370DB',  # Purple for battery
                        edgecolor='#4B0082',
                        linewidth=2,
                        zorder=5
                    )
                    ax.add_patch(square)
                    
                    # Add resource label
                    res_name = getattr(resource, 'name', 'resource')
                    if hasattr(resource, 'current_p_mw'):
                        power_mw = resource.current_p_mw
                        if abs(power_mw) < 0.1:
                            power_text = f"{power_mw * 1000:.1f} kW"
                        else:
                            power_text = f"{power_mw:.2f} MW"
                    else:
                        power_text = ""
                    
                    ax.annotate(f'{res_name}\n{power_text}', 
                               xy=(res_x, res_y),
                               ha='center', va='center',
                               fontsize=8, fontweight='bold',
                               color='white',
                               zorder=6)
            
            # Add legend for resources
            resource_patch = mpatches.Patch(color='#9370DB', label='Battery/Resource')
            ax.legend(handles=[resource_patch], loc='upper left', fontsize=9)
        
        return fig, ax
    
    def plot_voltage_profile(self,
                            nodes_df: pd.DataFrame,
                            ax: Optional[plt.Axes] = None,
                            figsize: Tuple[int, int] = (12, 6),
                            show_limits: bool = True,
                            v_min_limit: float = 0.9,
                            v_max_limit: float = 1.1,
                            highlight_violations: bool = True,
                            title: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot voltage profile as bar chart
        
        Args:
            nodes_df: DataFrame with node results
            ax: Matplotlib axes
            figsize: Figure size
            show_limits: Show voltage limit lines
            v_min_limit: Minimum voltage limit
            v_max_limit: Maximum voltage limit
            highlight_violations: Highlight buses with voltage violations
            title: Plot title
        
        Returns:
            (figure, axes) tuple
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        # Get bus IDs and voltages
        bus_ids = nodes_df['id'].astype(int).values
        voltages = nodes_df['v_mag'].values
        
        # Determine bar colors
        colors = []
        for v in voltages:
            if highlight_violations and (v < v_min_limit or v > v_max_limit):
                colors.append('#E74C3C')  # Red for violations
            elif v < 0.95:
                colors.append('#F39C12')  # Orange for low voltage
            else:
                colors.append('#4A90D9')  # Blue for normal
        
        # Create bar chart
        x = np.arange(len(bus_ids))
        ax.bar(x, voltages, color=colors, edgecolor='white', linewidth=0.5)
        
        # Voltage limits
        if show_limits:
            ax.axhline(y=v_min_limit, color='#E74C3C', linestyle='--',
                      linewidth=1.5, label=f'Vmin = {v_min_limit} p.u.')
            ax.axhline(y=v_max_limit, color='#E74C3C', linestyle='--',
                      linewidth=1.5, label=f'Vmax = {v_max_limit} p.u.')
            ax.axhline(y=1.0, color='#27AE60', linestyle='-',
                      linewidth=1, alpha=0.5, label='V = 1.0 p.u.')
        
        # Labels
        ax.set_xlabel('Bus ID', fontsize=11)
        ax.set_ylabel('Voltage Magnitude (p.u.)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(bus_ids, fontsize=8, rotation=45)
        ax.set_ylim(min(0.85, min(voltages) - 0.02), max(1.15, max(voltages) + 0.02))
        
        # Legend
        ax.legend(loc='upper right', fontsize=9)
        
        # Grid
        ax.grid(axis='y', alpha=0.3)
        
        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('Voltage Profile', fontsize=14, fontweight='bold')
        
        # Annotate min voltage
        min_idx = np.argmin(voltages)
        ax.annotate(f'Min: {voltages[min_idx]:.4f}\nBus {bus_ids[min_idx]}',
                   xy=(min_idx, voltages[min_idx]),
                   xytext=(min_idx + 2, voltages[min_idx] - 0.02),
                   ha='left', fontsize=9,
                   arrowprops=dict(arrowstyle='->', color='#333333'))
        
        plt.tight_layout()
        return fig, ax
    
    def plot_branch_loading(self,
                           lines_df: pd.DataFrame,
                           ax: Optional[plt.Axes] = None,
                           figsize: Tuple[int, int] = (14, 6),
                           show_p: bool = True,
                           show_q: bool = True,
                           title: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot branch loading (P and Q flows)
        
        Args:
            lines_df: DataFrame with line results
            ax: Matplotlib axes
            figsize: Figure size
            show_p: Show active power flow
            show_q: Show reactive power flow
            title: Plot title
        
        Returns:
            (figure, axes) tuple
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        # Get branch data
        from_nodes = lines_df['#from'].astype(int).values
        to_nodes = lines_df['#to'].astype(int).values
        branch_labels = [f'{f}-{t}' for f, t in zip(from_nodes, to_nodes)]
        
        x = np.arange(len(branch_labels))
        width = 0.35
        
        if show_p and show_q:
            p_flows = lines_df['p_flow_MW'].values
            q_flows = lines_df['q_flow_MVAr'].values
            
            ax.bar(x - width/2, p_flows, width, label='P (MW)',
                   color='#3498DB', edgecolor='white')
            ax.bar(x + width/2, q_flows, width, label='Q (MVAr)',
                   color='#E67E22', edgecolor='white')
        elif show_p:
            p_flows = lines_df['p_flow_MW'].values
            ax.bar(x, p_flows, width, label='P (MW)',
                   color='#3498DB', edgecolor='white')
        else:
            q_flows = lines_df['q_flow_MVAr'].values
            ax.bar(x, q_flows, width, label='Q (MVAr)',
                   color='#E67E22', edgecolor='white')
        
        # Labels
        ax.set_xlabel('Branch (From-To)', fontsize=11)
        ax.set_ylabel('Power Flow', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(branch_labels, fontsize=7, rotation=45, ha='right')
        
        # Legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Grid
        ax.grid(axis='y', alpha=0.3)
        
        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('Branch Power Flow', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_summary(self,
                    nodes_df: pd.DataFrame,
                    lines_df: pd.DataFrame,
                    figsize: Tuple[int, int] = (18, 14),
                    layout: str = 'auto',
                    save_path: str = None) -> plt.Figure:
        """Create a comprehensive summary plot
        
        Includes: topology with power flow, voltage profile, and branch loading.
        
        Args:
            nodes_df: DataFrame with node results
            lines_df: DataFrame with line results
            figsize: Figure size
            layout: Layout algorithm for topology
            save_path: Path to save the figure
        
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=figsize)
        
        # Grid layout: 2x2
        ax1 = fig.add_subplot(2, 2, 1)  # Topology
        ax2 = fig.add_subplot(2, 2, 2)  # Topology with PF
        ax3 = fig.add_subplot(2, 2, 3)  # Voltage profile
        ax4 = fig.add_subplot(2, 2, 4)  # Branch loading
        
        # 1. Network topology
        self.plot_topology(ax=ax1, layout=layout, title='Network Topology')
        
        # 2. Power flow results
        self.plot_power_flow(nodes_df, lines_df, ax=ax2, layout=layout,
                            title='Power Flow Results')
        
        # 3. Voltage profile
        self.plot_voltage_profile(nodes_df, ax=ax3)
        
        # 4. Branch loading
        self.plot_branch_loading(lines_df, ax=ax4)
        
        # Overall title
        case_name = getattr(self.case, 'name', 'Power System')
        fig.suptitle(f'{case_name} Analysis Summary', fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig

