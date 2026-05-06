"""Backward/Forward Sweep (BFS) power flow for radial distribution grids.

This solver targets **single-phase balanced radial feeders** and uses a
DistFlow-style power-summation formulation:

- ``p_load_pu`` / ``q_load_pu`` are **net loads** in per-unit
- positive values mean net consumption at the node
- negative values mean net injection (for example DER export)
- the squared-voltage update keeps the quadratic loss term
  ``(r^2 + x^2) * (P^2 + Q^2) / V^2``, so this is not the linear LinDistFlow
  approximation
- explicit voltage-angle states, phase coupling, and unbalance are not modeled

The public solver result keeps the internal numerical guardrails
(``v_sq`` denominator floor and voltage clamp), but also exposes whether the
unclamped voltage trajectory entered a severe low-voltage regime via
``voltage_collapse`` so RL environments can terminate the episode explicitly
instead of silently treating the clamped state as physically valid.
"""
import collections
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass

import numpy as np
from scipy import sparse

# These hard ``max()`` guards keep benchmark solves finite under catastrophic
# actions, but they are intentionally non-smooth and therefore not a good fit
# for differentiable / autodiff-based power-flow experiments.
_CURRENT_DENOMINATOR_FLOOR_V_SQ = 0.5
_VOLTAGE_CLAMP_FLOOR_V_SQ = 0.25
_VOLTAGE_COLLAPSE_THRESHOLD_V_SQ = 0.3

# Networks up to this size use dense numpy arrays for path/downstream/loss
# matrices. For benchmark-sized feeders (33–123 nodes), numpy matmul
# outperforms scipy sparse due to lower dispatch overhead.  Larger grids
# retain sparse storage automatically.
_DENSE_THRESHOLD = 256


@dataclass
class RadialTopology:
    """Data structure for radial network topology"""
    n_nodes: int
    n_lines: int
    parent: np.ndarray  # parent[i] = parent node of node i (-1 for root)
    parent_line: np.ndarray  # parent_line[i] = line index connecting node i to parent
    children: List[List[int]]  # children[i] = list of child nodes of node i
    children_lines: List[List[int]]  # children_lines[i] = list of line indices to children
    node_order: np.ndarray  # BFS order from root to leaves
    from_nodes: np.ndarray  # from node for each line (sending end in tree)
    to_nodes: np.ndarray  # to node for each line (receiving end in tree)
    r_pu: np.ndarray  # resistance in p.u.
    x_pu: np.ndarray  # reactance in p.u.
    z_sq_pu: np.ndarray  # r_pu**2 + x_pu**2 — precomputed for forward sweep hot path
    active_line_indices: List[int]  # original line indices in case data
    # Precomputed matrices for vectorization.
    # For networks with n_nodes <= _DENSE_THRESHOLD these are plain np.ndarray;
    # larger networks retain sparse.csr_matrix.  All usages are `@` matmul,
    # which is compatible with both representations.
    sending_nodes: np.ndarray  # sending end node index for each line
    receiving_nodes: np.ndarray  # receiving end node index for each line
    path_matrix: Union[sparse.csr_matrix, np.ndarray]  # path_matrix[node, line] = 1 if line is on path to node
    downstream_matrix: Union[sparse.csr_matrix, np.ndarray]  # downstream_matrix[line, node] = 1 if node is downstream of line
    loss_matrix: Union[sparse.csr_matrix, np.ndarray]  # loss_matrix[upstream_line, line] = 1 if that line's loss contributes upstream


def build_radial_topology(n_nodes: int, from_nodes: np.ndarray, to_nodes: np.ndarray,
                          r_pu: np.ndarray, x_pu: np.ndarray, slack_bus_id: int = 0,
                          active_line_indices: List[int] = None,
                          allow_mesh_pruning: bool = True) -> RadialTopology:
    """Build radial network topology using BFS from slack bus

    Args:
        n_nodes: Number of nodes
        from_nodes: Array of from-node indices for each line
        to_nodes: Array of to-node indices for each line
        r_pu: Resistance array in p.u.
        x_pu: Reactance array in p.u.
        slack_bus_id: Index of slack bus (default: 0)
        active_line_indices: Original line indices (default: range(n_lines))
        allow_mesh_pruning: If True (default), extra lines beyond the BFS
            spanning tree are ignored with a warning. The retained tree follows
            the *first-visit order* of the BFS rooted at ``slack_bus_id``.
            If False, non-radial inputs raise ``ValueError`` instead.

    Returns:
        RadialTopology: Topology data structure with precomputed matrices
    """
    n_lines = len(from_nodes)
    if active_line_indices is None:
        active_line_indices = list(range(n_lines))

    # Ensure integer type
    from_nodes = np.asarray(from_nodes, dtype=int)
    to_nodes = np.asarray(to_nodes, dtype=int)

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
    queue = collections.deque([slack_bus_id])
    visited[slack_bus_id] = True

    while queue:
        node = queue.popleft()
        node_order.append(node)

        for neighbor, line_idx in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = node
                parent_line[neighbor] = line_idx
                children[node].append(neighbor)
                children_lines[node].append(line_idx)
                queue.append(neighbor)

    node_order = np.array(node_order)

    # Validate: all nodes must be reachable from slack bus
    n_visited = int(visited.sum())
    if n_visited < n_nodes:
        unreachable = np.where(~visited)[0]
        raise ValueError(
            f"{n_nodes - n_visited} unreachable node(s) from slack bus "
            f"{slack_bus_id}: {unreachable.tolist()}"
        )

    # Warn if there are extra lines (loops) beyond the spanning tree
    n_tree_edges = n_nodes - 1  # a tree with n_nodes has n_nodes-1 edges
    if n_lines > n_tree_edges:
        message = (
            f"Extra lines detected: {n_lines} lines for {n_nodes} nodes "
            f"({n_lines - n_tree_edges} loop(s)). BFS keeps the first-visit "
            f"spanning tree rooted at slack bus {slack_bus_id}; extra lines "
            f"are ignored."
        )
        if not allow_mesh_pruning:
            raise ValueError(
                f"{message} Pass allow_mesh_pruning=True to keep the current "
                f"BFS pruning behavior explicitly."
            )
        import warnings
        warnings.warn(
            f"{message} Pass allow_mesh_pruning=False to reject non-radial input.",
            stacklevel=2,
        )

    # Compute sending/receiving nodes for each line (parent/child direction in tree)
    sending_nodes = np.where(parent[from_nodes] == to_nodes, to_nodes, from_nodes)
    receiving_nodes = np.where(sending_nodes == from_nodes, to_nodes, from_nodes)

    # Build Path Matrix (for Forward Sweep)
    # path_matrix[node, line] = 1 if line is on the path from root to node
    path_rows = []
    path_cols = []
    path_data = []

    for node in range(n_nodes):
        curr = node
        while parent[curr] != -1:
            line = parent_line[curr]
            if line != -1:
                path_rows.append(node)
                path_cols.append(line)
                path_data.append(1.0)
            curr = parent[curr]

    path_matrix = sparse.csr_matrix(
        (path_data, (path_rows, path_cols)),
        shape=(n_nodes, n_lines),
        dtype=np.float64
    )

    # Downstream Matrix (for Backward Sweep)
    # downstream_matrix[line, node] = 1 if node is downstream of line
    # This is essentially the transpose of path_matrix
    downstream_matrix = path_matrix.T.tocsr()

    # Collapse "line loss -> receiving node -> upstream branches" into one
    # sparse operator so backward_sweep can add downstream losses without
    # allocating node-length temporary arrays or using np.add.at each iteration.
    receiving_node_matrix = sparse.csr_matrix(
        (
            np.ones(n_lines, dtype=np.float64),
            (receiving_nodes, np.arange(n_lines)),
        ),
        shape=(n_nodes, n_lines),
        dtype=np.float64,
    )
    loss_matrix = (downstream_matrix @ receiving_node_matrix).tocsr()

    # For small networks convert to dense arrays: numpy matmul has lower
    # dispatch overhead than scipy sparse on benchmark-sized feeders.
    # All downstream consumers use `@` which works for both representations.
    if n_nodes <= _DENSE_THRESHOLD:
        path_matrix = path_matrix.toarray()
        downstream_matrix = downstream_matrix.toarray()
        loss_matrix = loss_matrix.toarray()

    return RadialTopology(
        n_nodes=n_nodes,
        n_lines=n_lines,
        parent=parent,
        parent_line=parent_line,
        children=children,
        children_lines=children_lines,
        node_order=node_order,
        from_nodes=from_nodes,
        to_nodes=to_nodes,
        r_pu=r_pu,
        x_pu=x_pu,
        z_sq_pu=r_pu ** 2 + x_pu ** 2,
        active_line_indices=active_line_indices,
        sending_nodes=sending_nodes,
        receiving_nodes=receiving_nodes,
        path_matrix=path_matrix,
        downstream_matrix=downstream_matrix,
        loss_matrix=loss_matrix,
    )


def backward_sweep(
    topo: RadialTopology,
    p_load_pu: np.ndarray,
    q_load_pu: np.ndarray,
    v_sq: np.ndarray,
    p_branch_old: Optional[np.ndarray] = None,
    q_branch_old: Optional[np.ndarray] = None,
    p_branch_base: Optional[np.ndarray] = None,
    q_branch_base: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward sweep: Calculate branch power flows from leaves to root (matrix version)

    Uses ``downstream_matrix`` for vectorized accumulation.
    Branch losses are evaluated from the previous iteration's branch powers and
    then propagated upstream through the same downstream matrix. This keeps the
    branch head power consistent with downstream losses at convergence.

    Args:
        topo: RadialTopology structure
        p_load_pu: Net active load at each node (p.u.). Positive means demand,
            negative means net injection from DER.
        q_load_pu: Net reactive load at each node (p.u.). Positive means demand,
            negative means net injection from DER.
        v_sq: Voltage squared at each node (p.u.^2)
        p_branch_old: Previous-iteration active branch flows (p.u.). If None,
            initializes from the downstream net-load sum.
        q_branch_old: Previous-iteration reactive branch flows (p.u.). If None,
            initializes from the downstream net-load sum.
        p_branch_base: Optional precomputed downstream net active load by branch
            (p.u.) for the current ``p_load_pu``.
        q_branch_base: Optional precomputed downstream net reactive load by branch
            (p.u.) for the current ``q_load_pu``.

    Returns:
        p_branch: Active power flow on each branch (p.u.)
        q_branch: Reactive power flow on each branch (p.u.)
    """
    # Base branch flow from net loads only: each line sees the sum of all
    # downstream node net demand/injection. This is also the first-iteration
    # initialization when no previous branch estimate is available.
    if p_branch_base is None:
        p_branch_base = topo.downstream_matrix @ p_load_pu
    if q_branch_base is None:
        q_branch_base = topo.downstream_matrix @ q_load_pu

    if p_branch_old is None:
        p_branch_old = p_branch_base
    if q_branch_old is None:
        q_branch_old = q_branch_base

    # Evaluate line current/loss from the previous branch-power iterate. Using
    # p_branch_base here would systematically under-estimate upstream current
    # because child-line losses would never feed back into parent branch power.
    v_sending_sq = np.maximum(v_sq[topo.sending_nodes], _CURRENT_DENOMINATOR_FLOOR_V_SQ)
    i_sq = (p_branch_old ** 2 + q_branch_old ** 2) / v_sending_sq
    p_loss = topo.r_pu * i_sq
    q_loss = topo.x_pu * i_sq

    # Each branch loss contributes to that branch and every upstream branch.
    # ``loss_matrix`` precomputes this relation directly in line space.
    p_branch = p_branch_base + topo.loss_matrix @ p_loss
    q_branch = q_branch_base + topo.loss_matrix @ q_loss

    return p_branch, q_branch


def forward_sweep(topo: RadialTopology, p_branch: np.ndarray, q_branch: np.ndarray,
                  v_sq: np.ndarray, v_slack: float = 1.0, slack_bus_id: int = 0) -> Tuple[np.ndarray, bool]:
    """Forward sweep: Calculate node voltages from root to leaves (matrix version)

    Uses path_matrix for vectorized computation of voltage drops.

    Args:
        topo: RadialTopology structure
        p_branch: Active power flow on each branch (p.u.)
        q_branch: Reactive power flow on each branch (p.u.)
        v_sq: Previous voltage squared at each node (p.u.^2)
        v_slack: Slack bus voltage magnitude (p.u.)
        slack_bus_id: Index of slack bus

    Returns:
        v_sq_new: Updated voltage squared at each node (p.u.^2)
        voltage_collapse: True if the *unclamped* voltage update entered a
            severe low-voltage regime (``v_sq < 0.3``) anywhere in the feeder
            before the numerical clamp was applied.
    """
    # Get sending end voltages for all lines
    v_sending_sq = np.maximum(v_sq[topo.sending_nodes], _CURRENT_DENOMINATOR_FLOOR_V_SQ)

    # Compute voltage drop terms for each line (DistFlow equation)
    # ΔV² ≈ 2(rP + xQ) - (r² + x²)(P² + Q²)/V²
    s_sq = p_branch ** 2 + q_branch ** 2

    term1 = 2 * (topo.r_pu * p_branch + topo.x_pu * q_branch)
    term2 = topo.z_sq_pu * s_sq / v_sending_sq
    delta_v_sq = term1 - term2

    # Accumulate voltage drops along paths (matrix multiplication)
    # Each node's voltage drop = sum of voltage drops on lines in its path
    total_drop = topo.path_matrix @ delta_v_sq

    # Calculate new voltages
    v_sq_raw = v_slack ** 2 - total_drop

    voltage_collapse = bool(np.any(v_sq_raw < _VOLTAGE_COLLAPSE_THRESHOLD_V_SQ))

    # Keep the numerical guard so current/loss calculations stay finite even
    # after a catastrophic action, but expose the collapse separately.
    v_sq_new = np.maximum(v_sq_raw, _VOLTAGE_CLAMP_FLOOR_V_SQ)

    return v_sq_new, voltage_collapse


def run_bfs_power_flow(topo: RadialTopology, p_load_pu: np.ndarray, q_load_pu: np.ndarray,
                       v_slack: float = 1.0, slack_bus_id: int = 0,
                       max_iter: int = 100, tol: float = 1e-6,
                       v_sq_init: Optional[np.ndarray] = None) -> Dict:
    """Run Backward/Forward Sweep (BFS) power flow.

    Args:
        topo: RadialTopology structure
        p_load_pu: Net active load at each node (p.u.). Positive means demand,
            negative means net injection from DER/export.
        q_load_pu: Net reactive load at each node (p.u.). Positive means demand,
            negative means net injection from DER/export.
        v_slack: Slack bus voltage magnitude (p.u.)
        slack_bus_id: Index of slack bus
        max_iter: Maximum iterations
        tol: Convergence tolerance
        v_sq_init: Optional warm-start voltage-squared array (p.u.^2, shape
            ``(n_nodes,)``).  When provided, the iteration begins from this
            estimate instead of the flat start (all ones).  Supplying the
            converged ``v_sq`` from the previous time step typically reduces
            the number of BFS iterations in RL rollouts.

    Returns:
        Dict with keys:
            - v_sq: Voltage squared at each node (p.u.^2)
            - v_mag: Voltage magnitude at each node (p.u.)
            - p_branch: Active power flow on each branch (p.u.)
            - q_branch: Reactive power flow on each branch (p.u.)
            - p_slack: Total active power injected by the slack bus (p.u.).
              Includes both outflows into child branches and any local load
              directly at the slack node (``p_load_pu[slack_bus_id]``).
            - q_slack: Total reactive power injected by the slack bus (p.u.).
              Includes branch outflows and local reactive load at the slack
              node (``q_load_pu[slack_bus_id]``).
            - converged: Whether the BFS solve produced a valid operating point
              (tolerance reached and no voltage-collapse event)
            - is_diverged: Whether the iteration failed to satisfy ``tol``
              before reaching ``max_iter``
            - voltage_collapse: Whether the unclamped voltage update fell below
              the collapse threshold at any point during iteration
            - iterations: Number of iterations
    """
    # Dtype safety: RL frameworks typically produce float32 actions; ensure
    # float64 throughout to avoid precision issues with sparse matrix ops.
    p_load_pu = np.asarray(p_load_pu, dtype=np.float64)
    q_load_pu = np.asarray(q_load_pu, dtype=np.float64)

    # Initialize voltage: use provided warm-start if available, else flat start.
    if v_sq_init is not None:
        v_sq = np.asarray(v_sq_init, dtype=np.float64).copy()
    else:
        v_sq = np.ones(topo.n_nodes)
    # Calibrate slack bus to the correct voltage reference.  Flat start sets
    # v_sq[slack] = 1.0; warm-start values come from a previous solve.
    # Either can differ from v_slack**2 when v_slack != 1.0, causing the first
    # backward sweep to use a wrong sending-end voltage for root-adjacent lines.
    v_sq[slack_bus_id] = v_slack ** 2

    # Base branch-power contribution from net load is constant within one BFS
    # solve, so reuse it across iterations.
    p_branch_base = topo.downstream_matrix @ p_load_pu
    q_branch_base = topo.downstream_matrix @ q_load_pu
    p_branch = p_branch_base.copy()
    q_branch = q_branch_base.copy()

    reached_tolerance = False
    voltage_collapse = False
    iteration_count = 0

    for iteration in range(max_iter):
        iteration_count = iteration + 1
        v_sq_old = v_sq.copy()

        p_branch, q_branch = backward_sweep(
            topo,
            p_load_pu,
            q_load_pu,
            v_sq,
            p_branch_old=p_branch,
            q_branch_old=q_branch,
            p_branch_base=p_branch_base,
            q_branch_base=q_branch_base,
        )
        v_sq, collapse_now = forward_sweep(
            topo, p_branch, q_branch, v_sq, v_slack, slack_bus_id
        )
        voltage_collapse = voltage_collapse or collapse_now

        max_diff = np.max(np.abs(v_sq - v_sq_old))
        if max_diff < tol:
            reached_tolerance = True
            break

    is_diverged = not reached_tolerance
    converged = reached_tolerance and not voltage_collapse

    # p_slack / q_slack: total power the slack bus must inject.
    # The backward sweep only accumulates loads *downstream* of each line, so
    # any load sitting directly at the slack node (root) is not captured by
    # branch flows alone — it must be added explicitly.
    slack_mask = topo.sending_nodes == slack_bus_id
    p_slack = float(np.sum(p_branch[slack_mask])) + float(p_load_pu[slack_bus_id])
    q_slack = float(np.sum(q_branch[slack_mask])) + float(q_load_pu[slack_bus_id])

    return {
        'v_sq': v_sq,
        'v_mag': np.sqrt(np.maximum(v_sq, 0)),
        'p_branch': p_branch,
        'q_branch': q_branch,
        'p_slack': p_slack,
        'q_slack': q_slack,
        'converged': converged,
        'is_diverged': is_diverged,
        'voltage_collapse': voltage_collapse,
        'iterations': iteration_count,
    }


def calculate_line_losses(topo: RadialTopology, p_branch: np.ndarray, q_branch: np.ndarray,
                          v_sq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate line losses (fully vectorized)

    Args:
        topo: RadialTopology structure
        p_branch: Active power flow on each branch (p.u.)
        q_branch: Reactive power flow on each branch (p.u.)
        v_sq: Voltage squared at each node (p.u.^2)

    Returns:
        p_loss_pu: Active power loss on each line (p.u.)
        q_loss_pu: Reactive power loss on each line (p.u.)
    """
    # Vectorized: get all sending end voltages at once
    v_from_sq = np.maximum(v_sq[topo.sending_nodes], _CURRENT_DENOMINATOR_FLOOR_V_SQ)

    # Vectorized calculation of I² = S² / V²
    i_sq = (p_branch ** 2 + q_branch ** 2) / v_from_sq

    # Vectorized loss calculation
    p_loss_pu = topo.r_pu * i_sq
    q_loss_pu = topo.x_pu * i_sq

    return p_loss_pu, q_loss_pu


def get_line_from_node(topo: RadialTopology, line_idx: int) -> int:
    """Get the sending-end node for a line in the radial tree

    Args:
        topo: RadialTopology structure
        line_idx: Line index

    Returns:
        Node index of the sending end (parent in the tree)
    """
    return topo.sending_nodes[line_idx]


# Backward-compatible alias
run_fbs_power_flow = run_bfs_power_flow
