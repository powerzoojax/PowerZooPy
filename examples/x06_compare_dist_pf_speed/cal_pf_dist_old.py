"""Forward-Backward Sweep Power Flow Calculation Functions

Core computational functions for radial distribution network power flow.
These are stateless functions that can be used independently.
"""
from typing import Tuple, List, Dict
from dataclasses import dataclass

import numpy as np


@dataclass
class RadialTopology:
    """Data structure for radial network topology"""
    n_nodes: int
    n_lines: int
    parent: np.ndarray          # parent[i] = parent node of node i (-1 for root)
    parent_line: np.ndarray     # parent_line[i] = line index connecting node i to parent
    children: List[List[int]]   # children[i] = list of child nodes of node i
    children_lines: List[List[int]]  # children_lines[i] = list of line indices to children
    node_order: np.ndarray      # BFS order from root to leaves
    from_nodes: np.ndarray      # from node for each line
    to_nodes: np.ndarray        # to node for each line
    r_pu: np.ndarray           # resistance in p.u.
    x_pu: np.ndarray           # reactance in p.u.
    active_line_indices: List[int]  # original line indices in case data


def build_radial_topology(n_nodes: int, from_nodes: np.ndarray, to_nodes: np.ndarray,
                          r_pu: np.ndarray, x_pu: np.ndarray, slack_bus_id: int = 0,
                          active_line_indices: List[int] = None) -> RadialTopology:
    """Build radial network topology using BFS from slack bus

    Args:
        n_nodes: Number of nodes
        from_nodes: Array of from-node indices for each line
        to_nodes: Array of to-node indices for each line
        r_pu: Resistance array in p.u.
        x_pu: Reactance array in p.u.
        slack_bus_id: Index of slack bus (default: 0)
        active_line_indices: Original line indices (default: range(n_lines))

    Returns:
        RadialTopology: Topology data structure
    """
    n_lines = len(from_nodes)
    if active_line_indices is None:
        active_line_indices = list(range(n_lines))

    # Build adjacency list
    adj = {i: [] for i in range(n_nodes)}
    for i in range(n_lines):
        f, t = from_nodes[i], to_nodes[i]
        adj[f].append((t, i))
        adj[t].append((f, i))

    # BFS to build tree structure
    parent = np.full(n_nodes, -1, dtype=int)
    parent_line = np.full(n_nodes, -1, dtype=int)
    children = [[] for _ in range(n_nodes)]
    children_lines = [[] for _ in range(n_nodes)]
    node_order = []

    visited = np.zeros(n_nodes, dtype=bool)
    queue = [slack_bus_id]
    visited[slack_bus_id] = True

    while queue:
        node = queue.pop(0)
        node_order.append(node)

        for neighbor, line_idx in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = node
                parent_line[neighbor] = line_idx
                children[node].append(neighbor)
                children_lines[node].append(line_idx)
                queue.append(neighbor)

    return RadialTopology(
        n_nodes=n_nodes,
        n_lines=n_lines,
        parent=parent,
        parent_line=parent_line,
        children=children,
        children_lines=children_lines,
        node_order=np.array(node_order),
        from_nodes=from_nodes,
        to_nodes=to_nodes,
        r_pu=r_pu,
        x_pu=x_pu,
        active_line_indices=active_line_indices
    )


def backward_sweep(topo: RadialTopology, p_load_pu: np.ndarray, q_load_pu: np.ndarray,
                   v_sq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Backward sweep: Calculate branch power flows from leaves to root

    Includes line losses in the branch power calculation.

    Args:
        topo: RadialTopology structure
        p_load_pu: Active power load at each node (p.u.)
        q_load_pu: Reactive power load at each node (p.u.)
        v_sq: Voltage squared at each node (p.u.^2)

    Returns:
        p_branch: Active power flow on each branch (p.u.)
        q_branch: Reactive power flow on each branch (p.u.)
    """
    p_branch = np.zeros(topo.n_lines)
    q_branch = np.zeros(topo.n_lines)

    # Process nodes in reverse order (leaves to root)
    for node in reversed(topo.node_order[1:]):  # Skip root node
        line_idx = topo.parent_line[node]

        # Sum of downstream loads
        p_sum = p_load_pu[node]
        q_sum = q_load_pu[node]

        # Add children branch flows and their losses
        for i, child_line in enumerate(topo.children_lines[node]):
            # Add downstream branch flow
            p_sum += p_branch[child_line]
            q_sum += q_branch[child_line]

            # Add line losses (computed from previous iteration)
            v_sq_node = max(v_sq[node], 0.5)
            i_sq = (p_branch[child_line] ** 2 + q_branch[child_line] ** 2) / v_sq_node
            p_sum += topo.r_pu[child_line] * i_sq
            q_sum += topo.x_pu[child_line] * i_sq

        p_branch[line_idx] = p_sum
        q_branch[line_idx] = q_sum

    return p_branch, q_branch


def forward_sweep(topo: RadialTopology, p_branch: np.ndarray, q_branch: np.ndarray,
                  v_sq: np.ndarray, v_slack: float = 1.0, slack_bus_id: int = 0) -> np.ndarray:
    """Forward sweep: Calculate node voltages from root to leaves

    Uses the complete DistFlow voltage drop equation:
    V_j^2 = V_i^2 - 2*(r*P + x*Q) + (r^2 + x^2)*(P^2 + Q^2)/V_i^2

    Args:
        topo: RadialTopology structure
        p_branch: Active power flow on each branch (p.u.)
        q_branch: Reactive power flow on each branch (p.u.)
        v_sq: Previous voltage squared at each node (p.u.^2)
        v_slack: Slack bus voltage magnitude (p.u.)
        slack_bus_id: Index of slack bus

    Returns:
        v_sq_new: Updated voltage squared at each node (p.u.^2)
    """
    v_sq_new = v_sq.copy()
    v_sq_new[slack_bus_id] = v_slack ** 2

    # Process nodes in order (root to leaves)
    for node in topo.node_order[1:]:  # Skip root node
        parent = topo.parent[node]
        line_idx = topo.parent_line[node]

        r_pu = topo.r_pu[line_idx]
        x_pu = topo.x_pu[line_idx]

        p = p_branch[line_idx]
        q = q_branch[line_idx]
        v_p_sq = v_sq_new[parent]

        # DistFlow voltage drop equation
        z_sq = r_pu ** 2 + x_pu ** 2
        s_sq = p ** 2 + q ** 2
        v_sq_new[node] = v_p_sq - 2 * (r_pu * p + x_pu * q) + z_sq * s_sq / max(v_p_sq, 0.5)

        # Ensure voltage is positive
        v_sq_new[node] = max(v_sq_new[node], 0.25)

    return v_sq_new


def run_fbs_power_flow(topo: RadialTopology, p_load_pu: np.ndarray, q_load_pu: np.ndarray,
                       v_slack: float = 1.0, slack_bus_id: int = 0,
                       max_iter: int = 100, tol: float = 1e-6) -> Dict:
    """Run Forward-Backward Sweep power flow iteration

    Args:
        topo: RadialTopology structure
        p_load_pu: Active power load at each node (p.u.)
        q_load_pu: Reactive power load at each node (p.u.)
        v_slack: Slack bus voltage magnitude (p.u.)
        slack_bus_id: Index of slack bus
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Dict with keys:
            - v_sq: Voltage squared at each node (p.u.^2)
            - v_mag: Voltage magnitude at each node (p.u.)
            - p_branch: Active power flow on each branch (p.u.)
            - q_branch: Reactive power flow on each branch (p.u.)
            - converged: Whether converged
            - iterations: Number of iterations
    """
    # Initialize voltage (flat start)
    v_sq = np.ones(topo.n_nodes)

    converged = False
    for iteration in range(max_iter):
        v_sq_old = v_sq.copy()

        # Backward sweep
        p_branch, q_branch = backward_sweep(topo, p_load_pu, q_load_pu, v_sq)

        # Forward sweep
        v_sq = forward_sweep(topo, p_branch, q_branch, v_sq, v_slack, slack_bus_id)

        # Check convergence
        max_diff = np.max(np.abs(v_sq - v_sq_old))
        if max_diff < tol:
            converged = True
            break

    return {
        'v_sq': v_sq,
        'v_mag': np.sqrt(np.maximum(v_sq, 0)),
        'p_branch': p_branch,
        'q_branch': q_branch,
        'converged': converged,
        'iterations': iteration + 1
    }


def calculate_line_losses(topo: RadialTopology, p_branch: np.ndarray, q_branch: np.ndarray,
                          v_sq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate line losses

    Args:
        topo: RadialTopology structure
        p_branch: Active power flow on each branch (p.u.)
        q_branch: Reactive power flow on each branch (p.u.)
        v_sq: Voltage squared at each node (p.u.^2)

    Returns:
        p_loss_pu: Active power loss on each line (p.u.)
        q_loss_pu: Reactive power loss on each line (p.u.)
    """
    p_loss_pu = np.zeros(topo.n_lines)
    q_loss_pu = np.zeros(topo.n_lines)

    for line_idx in range(topo.n_lines):
        # Get sending end node
        from_node = get_line_from_node(topo, line_idx)
        v_from_sq = v_sq[from_node] if from_node >= 0 else 1.0

        # I^2 = S^2 / V^2
        i_sq = (p_branch[line_idx] ** 2 + q_branch[line_idx] ** 2) / max(v_from_sq, 0.5)
        p_loss_pu[line_idx] = topo.r_pu[line_idx] * i_sq
        q_loss_pu[line_idx] = topo.x_pu[line_idx] * i_sq

    return p_loss_pu, q_loss_pu


def get_line_from_node(topo: RadialTopology, line_idx: int) -> int:
    """Get the sending-end node for a line in the radial tree

    Args:
        topo: RadialTopology structure
        line_idx: Line index

    Returns:
        Node index of the sending end (parent in the tree)
    """
    from_node = int(topo.from_nodes[line_idx])
    to_node = int(topo.to_nodes[line_idx])

    if topo.parent[to_node] == from_node:
        return from_node
    elif topo.parent[from_node] == to_node:
        return to_node
    else:
        return from_node

