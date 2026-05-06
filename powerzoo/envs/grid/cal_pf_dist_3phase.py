"""Three-phase BFS power flow helpers for radial distribution feeders.

This module uses a Kron-expanded BIBC/BCBV formulation: every non-reference bus
is represented by an ``A/B/C`` state triplet, and each branch contributes a
``3x3`` series-impedance block. That keeps the benchmark core compact, but it
also means the topology builder does **not** natively drop missing phases from
the graph. Feeders with true single-/two-phase laterals must therefore encode
phase availability upstream inside ``Z_3ph_pu`` (PowerZoo's bundled Case123 does
this via sparse/zero-padded ``3x3`` line-configuration matrices).

The matrices are intentionally stored as dense ``ndarray`` objects. Although
``BIBC``/``BCBV`` start sparse, the combined ``DLF`` used in every BFS
iteration becomes dense on benchmark feeders such as Case123, so switching the
runtime path to ``scipy.sparse`` does not improve ``step()`` performance here.

Vector layout convention:
    All flattened ``3*n_lines`` solver vectors use node-major ``ABC`` order:
    ``[node1_A, node1_B, node1_C, node2_A, node2_B, node2_C, ...]``, where the
    nodes are the non-reference buses sorted by internal matrix index.

Model boundary:
    The BFS core supports full mutual coupling inside the supplied series
    impedance block ``Z_3ph``. It does **not** apply branch shunt charging
    ``B`` terms, off-nominal transformer tap ratios, or phase shifts from
    branch metadata such as ``ratio`` / ``angle``.
"""
from typing import Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np

# Reuse topology building from single-phase module
from powerzoo.envs.grid.cal_pf_dist import build_radial_topology, RadialTopology


@dataclass
class ThreePhaseTopology:
    """Data structure for the three-phase radial-network solve surface."""
    n_nodes: int  # Total bus count including the reference bus
    n_lines: int  # In-service radial branch count; equals n_nodes - 1 for a tree
    ref_bus: int  # Reference bus index (0-based)
    ref_node_id: Any  # Physical ID / label of the reference bus
    
    # Basic topology from single-phase
    topo: RadialTopology

    # Explicit node/index metadata for node-major ABC solver vectors
    node_ids: np.ndarray  # (n_nodes,) physical node IDs / labels
    non_ref_node_indices: np.ndarray  # (n_lines,) internal node indices excluding ref_bus
    non_ref_node_ids: np.ndarray  # (n_lines,) physical IDs matching non_ref_node_indices
    node_id_to_matrix_index: Dict[Any, int]  # physical node ID -> reduced solver row
    vector_layout: str  # flattened vector convention, currently "node_major_abc"
    phase_order: Tuple[str, str, str]  # phase order used inside each node block
    
    # Three-phase specific matrices (stored dense; see module docstring)
    Z_3ph: np.ndarray  # (n_lines, 3, 3) complex impedance matrices in p.u.
    BIBC: np.ndarray  # (3*n_lines, 3*n_lines) Bus Injection to Branch Current matrix
    BCBV: np.ndarray  # (3*n_lines, 3*n_lines) Branch Current to Bus Voltage matrix (complex)
    DLF: np.ndarray  # (3*n_lines, 3*n_lines) BCBV @ BIBC combined matrix

    # Precomputed masks / gather indices for the runtime solve path
    non_ref_phase_mask: np.ndarray  # (n_lines, 3) valid phases for each non-ref bus
    non_ref_phase_mask_flat: np.ndarray  # (3*n_lines,) flattened node-major phase mask
    sending_bus_is_ref: np.ndarray  # (n_lines,) whether the branch sending bus is the reference bus
    sending_bus_gather_indices: np.ndarray  # (n_lines,) safe reduced-bus gather indices for V_from
    
    # Reference voltage phasor (per-phase magnitudes, positive-sequence angles by default)
    V_ref_3ph: np.ndarray  # (3,) complex reference voltage for phases A/B/C


def build_3phase_topology(n_nodes: int, from_nodes: np.ndarray, to_nodes: np.ndarray,
                          Z_3ph_pu: np.ndarray, ref_bus: int = 0,
                          v_ref_mag: float | np.ndarray = 1.0,
                          node_ids: np.ndarray | None = None) -> ThreePhaseTopology:
    """Build a three-phase radial topology with Kron-expanded BIBC/BCBV matrices.

    Args:
        n_nodes: Number of nodes in the network
        from_nodes: Array of from-node indices for each line
        to_nodes: Array of to-node indices for each line
        Z_3ph_pu: ``(n_lines, 3, 3)`` complex impedance matrices in p.u.
            Missing phases are not removed from the topology here; callers must
            already encode any absent phase in these ``3x3`` branch blocks.
        ref_bus: Reference (slack) bus index (default: 0)
        v_ref_mag: Reference voltage magnitude.
            Accepts either a scalar (balanced-magnitude mode) or a length-3
            array-like ``[V_A, V_B, V_C]`` for per-phase magnitudes. The phase
            angles remain the default positive-sequence ``0/-120/+120`` basis.
        node_ids: Optional physical node IDs / labels with length ``n_nodes``.
            When omitted, internal ``0..n_nodes-1`` indices are used.

    Returns:
        ThreePhaseTopology: Topology data structure with precomputed matrices
    """
    n_lines = len(from_nodes)
    node_ids_arr = np.asarray(
        np.arange(n_nodes) if node_ids is None else node_ids
    ).reshape(-1)
    if node_ids_arr.size != n_nodes:
        raise ValueError(
            f"node_ids must contain {n_nodes} elements. Got {node_ids_arr.size}."
        )
    
    # Build basic topology using single-phase method (dummy r, x)
    # This gives us the tree structure, parent/child relationships
    r_dummy = np.ones(n_lines)
    x_dummy = np.ones(n_lines)
    topo = build_radial_topology(
        n_nodes=n_nodes,
        from_nodes=from_nodes,
        to_nodes=to_nodes,
        r_pu=r_dummy,
        x_pu=x_dummy,
        slack_bus_id=ref_bus
    )
    
    # Build incidence matrix K (n_lines x n_nodes)
    # K[br, from_node] = ±1, K[br, to_node] = −1   (direction depends on ref_bus)
    K = np.zeros((n_lines, n_nodes))
    rows = np.arange(n_lines)
    
    # Logic:
    # if to_node <= ref_bus: K[br, from] = -1, K[br, to] = 1
    # else:                  K[br, from] = 1,  K[br, to] = -1
    mask_to_le_ref = (to_nodes <= ref_bus)
    val_from = np.where(mask_to_le_ref, -1, 1)
    val_to = np.where(mask_to_le_ref, 1, -1)
    
    K[rows, from_nodes] = val_from
    K[rows, to_nodes] = val_to
    
    # Remove reference bus column to get Gamma
    Gamma = np.delete(K, ref_bus, axis=1)            # reduced incidence matrix (n_lines × n_lines)
    
    # BIBC_0 = -inv(Gamma')  — single-phase Bus Injection to Branch Current
    BIBC_0 = -np.linalg.inv(Gamma.T)
    
    # Expand to 3-phase BIBC matrix (3*n_lines x 3*n_lines)
    # Use Kronecker product for efficient expansion
    BIBC = np.kron(BIBC_0, np.eye(3))              # 3-phase BIBC matrix
    
    # Build BCBV matrix (complex)  — Branch Current to Bus Voltage
    BCBV = np.zeros((3 * n_lines, 3 * n_lines), dtype=np.complex128)
    
    # Create a view of shape (n_lines, n_lines, 3, 3)
    # Steps: (3*n, 3*n) -> (n, 3, n, 3) -> (n, n, 3, 3)
    # This view allows us to access the 3x3 block at [b, bb] directly
    BCBV_view = BCBV.reshape(n_lines, 3, n_lines, 3).swapaxes(1, 2)
    
    # Find indices where connection exists (path matrix is non-zero)
    # BIBC_0[bb, b] == 1 means bb is on the path to b
    # We want to fill BCBV[b, bb] with Z_3ph_pu[bb]
    # So we look for non-zeros in BIBC_0.T (which maps b -> bb)
    rows_b, cols_bb = np.nonzero(BIBC_0.T)
    
    # Direct block assignment using advanced indexing
    # This copies Z matrices only to valid positions, skipping zeros
    BCBV_view[rows_b, cols_bb] = Z_3ph_pu[cols_bb]
    
    # Precompute DLF = BCBV @ BIBC for efficiency
    DLF = BCBV @ BIBC                              # combined ΔV = DLF @ I_inj matrix

    non_ref_node_indices = np.delete(np.arange(n_nodes, dtype=int), ref_bus)
    non_ref_node_ids = node_ids_arr[non_ref_node_indices]
    node_id_to_matrix_index = {
        node_id: idx for idx, node_id in enumerate(non_ref_node_ids.tolist())
    }

    # A non-ref bus inherits its valid phase set from its unique parent branch.
    line_phase_mask = np.abs(np.diagonal(Z_3ph_pu, axis1=1, axis2=2)) > 1e-12
    parent_lines = topo.parent_line[non_ref_node_indices]
    if np.any(parent_lines < 0):
        raise ValueError(
            "Each non-reference bus must have a valid parent line in a radial topology."
        )
    non_ref_phase_mask = line_phase_mask[parent_lines]
    non_ref_phase_mask_flat = non_ref_phase_mask.reshape(-1)

    sending_nodes = np.asarray(topo.sending_nodes, dtype=int)
    sending_bus_is_ref = sending_nodes == ref_bus
    sending_bus_gather_indices = np.where(
        sending_bus_is_ref,
        0,
        np.where(sending_nodes < ref_bus, sending_nodes, sending_nodes - 1)
    )
    non_ref_sending_idx = sending_bus_gather_indices[~sending_bus_is_ref]
    if non_ref_sending_idx.size > 0:
        bad_mask = (non_ref_sending_idx < 0) | (non_ref_sending_idx >= n_lines)
        if bad_mask.any():
            raise IndexError(
                f"Sending-bus reduced indices {non_ref_sending_idx[bad_mask].tolist()} "
                f"are out of range [0, {n_lines})."
            )
    
    # Reference voltage phasor (default positive-sequence angles).
    V_ref_3ph = _coerce_reference_voltage_magnitudes(v_ref_mag) * np.exp(
        1j * np.deg2rad([0, -120, 120])
    )
    
    return ThreePhaseTopology(
        n_nodes=n_nodes,
        n_lines=n_lines,
        ref_bus=ref_bus,
        ref_node_id=node_ids_arr[ref_bus],
        topo=topo,
        node_ids=node_ids_arr,
        non_ref_node_indices=non_ref_node_indices,
        non_ref_node_ids=non_ref_node_ids,
        node_id_to_matrix_index=node_id_to_matrix_index,
        vector_layout="node_major_abc",
        phase_order=("A", "B", "C"),
        Z_3ph=Z_3ph_pu,
        BIBC=BIBC,
        BCBV=BCBV,
        DLF=DLF,
        non_ref_phase_mask=non_ref_phase_mask,
        non_ref_phase_mask_flat=non_ref_phase_mask_flat,
        sending_bus_is_ref=sending_bus_is_ref,
        sending_bus_gather_indices=sending_bus_gather_indices,
        V_ref_3ph=V_ref_3ph
    )


def _regularize_voltage_for_division(
    V: np.ndarray,
    ref_direction: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, int]:
    """Protect ``S / V`` against zero / non-finite voltages without losing phase layout.

    The magnitude floor uses the current phasor direction when it is still
    finite and non-zero; otherwise it falls back to the corresponding reference
    phase angle from the precomputed ``ref_direction``.
    """
    V_arr = np.asarray(V, dtype=np.complex128)
    ref_dir_arr = np.asarray(ref_direction, dtype=np.complex128)

    abs_v = np.abs(V_arr)
    current_direction = np.where(abs_v > 0.0, V_arr / np.maximum(abs_v, eps), ref_dir_arr)
    current_direction = np.where(np.isfinite(current_direction), current_direction, ref_dir_arr)

    bad_mask = (~np.isfinite(abs_v)) | (abs_v < eps)
    V_safe = np.where(bad_mask, eps * current_direction, V_arr)
    return V_safe, int(np.count_nonzero(bad_mask))


def _coerce_reference_voltage_magnitudes(v_ref_mag: float | np.ndarray) -> np.ndarray:
    """Return a validated ``(3,)`` per-phase reference-magnitude vector."""
    arr = np.asarray(v_ref_mag, dtype=float).reshape(-1)
    if arr.size == 1:
        mags = np.repeat(arr[0], 3)
    elif arr.size == 3:
        mags = arr.copy()
    else:
        raise ValueError(
            "v_ref_mag must be either a scalar or an array-like with 3 elements "
            "for phases A/B/C."
        )
    if not np.all(np.isfinite(mags)):
        raise ValueError("v_ref_mag must contain only finite values.")
    if np.any(mags <= 0.0):
        raise ValueError("v_ref_mag must be strictly positive on every phase.")
    return mags


def run_3phase_bfs_power_flow(topo3ph: ThreePhaseTopology,
                              P_3ph_pu: np.ndarray, Q_3ph_pu: np.ndarray,
                              v_ref_mag: float | np.ndarray = 1.0,
                              max_iter: int = 100, tol: float = 1e-6) -> Dict:
    """Run three-phase Backward/Forward Sweep (BFS) power flow using BIBC/BCBV.

    Conventions:
      - Input P/Q are LOADS per bus-phase (positive means consumption).
      - Negative P/Q mean net generation / injection under the same load-positive
        convention. For example, PV inverters or battery discharge should be
        passed as negative active power, and reactive injection should be passed
        as negative Q.
      - Internally we use injection S_inj = -(P + jQ).
      - I_inj = conj(S_inj / V_bus)
      - ΔV = DLF @ I_inj
      - V_bus = V_ref + ΔV   (sign consistent with how DLF was built in build_3phase_topology)

    Args:
        topo3ph: Precomputed three-phase radial topology.
        P_3ph_pu: Net bus-phase active demand in p.u. using node-major ABC order.
        Q_3ph_pu: Net bus-phase reactive demand in p.u. using node-major ABC order.
        v_ref_mag: Reference voltage magnitude. Accepts either a scalar
            (balanced-magnitude mode) or a length-3 array-like
            ``[V_A, V_B, V_C]`` for per-phase magnitudes. The phase angles stay
            on the default positive-sequence ``0/-120/+120`` basis.
        max_iter: Maximum BFS iterations.
        tol: Convergence tolerance on the maximum voltage update.

    Returns arrays are flattened length ``(3*n_lines,)`` in node-major ABC
    order, where ``n_lines == (n_nodes - 1)`` for radial networks.

    Non-converged runs still return the final iterate so callers can inspect the
    collapse mode, but those voltages/currents/flows are **diagnostic only**.
    Upper layers must gate on ``result['converged']`` (or, in the env wrapper,
    ``info['pf_converged']`` / ``self._converged``) before treating the outputs
    as a physically valid power-flow solution.
    When divergence is triggered by extreme RL exploration actions, these
    diagnostic tensors may still contain very large finite values. A bounded RL
    environment must therefore replace or mask them before they reach the
    policy network; in PowerZoo this responsibility lives at the env / wrapper
    layer rather than in the raw solver helper.

    Missing-phase contract:
        Any input power assigned to a phase that is absent on the bus's parent
        branch is clamped to zero before the BFS iterations. This prevents RL
        policies from injecting power into phases that do not physically exist
        in the Kron-expanded feeder representation.
    """

    n_lines = topo3ph.n_lines

    # ---- reshape & validate -------------------------------------------------
    Pbus_load = np.asarray(P_3ph_pu).reshape(-1)
    Qbus_load = np.asarray(Q_3ph_pu).reshape(-1)

    expected = 3 * n_lines
    if Pbus_load.size != expected or Qbus_load.size != expected:
        raise ValueError(
            f"P_3ph_pu/Q_3ph_pu must contain {expected} elements (= 3*n_lines). "
            f"Got P={Pbus_load.size}, Q={Qbus_load.size}. "
            f"Hint: use shape (n_lines,3) or (3*n_lines,)."
        )

    Pbus_load = Pbus_load.reshape(-1, 1)
    Qbus_load = Qbus_load.reshape(-1, 1)

    # Load -> negative injection, then clamp inactive phases to zero.
    S_inj_raw = -(Pbus_load + 1j * Qbus_load)  # (3*n_lines, 1)
    phase_mask_flat = topo3ph.non_ref_phase_mask_flat.reshape(-1, 1)
    S_inj = np.where(phase_mask_flat, S_inj_raw, 0.0 + 0.0j)
    inactive_phase_input_count = int(np.count_nonzero(
        (~phase_mask_flat.reshape(-1)) & (np.abs(S_inj_raw.reshape(-1)) > 1e-12)
    ))

    # ---- reference voltage --------------------------------------------------
    eps = 1e-12

    # Prefer topology's reference phasor definition; allow magnitude override.
    V_ref_3ph_unit = topo3ph.V_ref_3ph / np.maximum(np.abs(topo3ph.V_ref_3ph), eps)
    # Normalize to magnitude 1 (per phase) then scale to v_ref_mag
    V_ref_3ph = _coerce_reference_voltage_magnitudes(v_ref_mag) * V_ref_3ph_unit

    # Bus voltages for NON-REF buses only, ordered as "all node indices except ref_bus (ascending)"
    Vr_n = np.tile(V_ref_3ph, n_lines).reshape(-1, 1)  # (3*n_lines, 1)
    # Phase-direction fallback is iteration-invariant, so precompute it once.
    ref_direction = np.tile(V_ref_3ph_unit, n_lines).reshape(-1, 1)

    # ---- iteration ----------------------------------------------------------
    V = Vr_n.copy()
    converged = False
    iterations = 0
    max_voltage_update_pu = 0.0
    voltage_regularization_count = 0

    for _ in range(max_iter):
        iterations += 1
        V_safe, n_regularized = _regularize_voltage_for_division(V, ref_direction, eps)
        voltage_regularization_count += n_regularized
        I_inj = np.conj(S_inj / V_safe)  # (3*n_lines, 1)

        delta_V = topo3ph.DLF @ I_inj
        V_new = Vr_n + delta_V

        max_diff = float(np.max(np.abs(V_new - V)))
        max_voltage_update_pu = max(max_voltage_update_pu, max_diff)
        V = V_new

        if max_diff < tol:
            converged = True
            break

    # Recompute injection current using FINAL voltage to keep outputs consistent
    V_safe, n_regularized = _regularize_voltage_for_division(V, ref_direction, eps)
    voltage_regularization_count += n_regularized
    I_inj = np.conj(S_inj / V_safe)

    # ---- branch currents ----------------------------------------------------
    I_branch = topo3ph.BIBC @ I_inj  # (3*n_lines, 1)

    # ---- branch powers (sending-end, per phase) -----------------------------
    # Build sending-end voltage vector per line/phase (vectorized)
    V_bus_mat = V.reshape(n_lines, 3)  # bus voltages excluding reference
    V_from = V_bus_mat[topo3ph.sending_bus_gather_indices].copy()
    V_from[topo3ph.sending_bus_is_ref] = V_ref_3ph

    I_line_mat = I_branch.reshape(n_lines, 3)
    S_branch_send = V_from * np.conj(I_line_mat)  # (n_lines, 3)
    P_branch = np.real(S_branch_send).reshape(-1, 1)
    Q_branch = np.imag(S_branch_send).reshape(-1, 1)

    # ---- pack results -------------------------------------------------------
    V_flat = V.reshape(-1)
    if converged:
        convergence_status = "converged"
        convergence_message = f"Converged in {iterations} iterations."
    elif iterations == 0:
        convergence_status = "not_started"
        convergence_message = (
            "No BFS iterations were executed because max_iter=0; outputs are "
            "flat-start diagnostics only."
        )
    else:
        convergence_status = "max_iter_exhausted"
        convergence_message = (
            f"Did not converge within {max_iter} iterations; outputs are "
            "last-iterate diagnostics only."
        )

    min_v_mag = float(np.min(np.abs(V_flat))) if V_flat.size > 0 else float('nan')
    # Consistent with single-phase BFS: sqrt(0.3) ≈ 0.547 p.u.
    _VOLTAGE_COLLAPSE_THRESHOLD_PU = 0.5
    voltage_collapse = bool(min_v_mag < _VOLTAGE_COLLAPSE_THRESHOLD_PU)

    return {
        'V': V_flat,
        'V_mag': np.abs(V_flat),
        'V_angle': np.angle(V_flat, deg=True),
        'I_branch': I_branch.reshape(-1),
        'P_branch': P_branch.reshape(-1),
        'Q_branch': Q_branch.reshape(-1),
        'converged': converged,
        'voltage_collapse': voltage_collapse,
        'iterations': iterations,
        'convergence_status': convergence_status,
        'convergence_message': convergence_message,
        'max_voltage_update_pu': max_voltage_update_pu,
        'min_voltage_magnitude_pu': min_v_mag,
        'used_voltage_regularization': bool(voltage_regularization_count > 0),
        'voltage_regularization_count': voltage_regularization_count,
        'inactive_phase_inputs_clamped': bool(inactive_phase_input_count > 0),
        'inactive_phase_input_count': inactive_phase_input_count,
        'vector_layout': topo3ph.vector_layout,
        'phase_order': topo3ph.phase_order,
    }


def calculate_3phase_losses(topo3ph: ThreePhaseTopology, result: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate three-phase branch losses from the full ``3x3`` impedance blocks.

    Args:
        topo3ph: ThreePhaseTopology structure
        result: Result dict from run_3phase_bfs_power_flow
        
    Returns:
        P_loss: Active power loss contribution per phase ``(3*n_lines,)`` in p.u.
        Q_loss: Reactive power loss contribution per phase ``(3*n_lines,)`` in p.u.
    """
    I_branch = np.asarray(result['I_branch']).reshape(-1)
    n_lines = topo3ph.n_lines
    expected = 3 * n_lines
    if I_branch.size != expected:
        raise ValueError(
            f"result['I_branch'] must contain {expected} elements (= 3*n_lines). "
            f"Got {I_branch.size}."
        )

    # Branch complex loss uses the full mutual-coupled voltage drop:
    #   ΔV_branch = Z_3ph @ I_branch
    #   S_loss,ph = ΔV_ph * conj(I_ph)
    # Summing over phases recovers I^H Z I for each branch.
    I_matrix = I_branch.reshape(n_lines, 3)
    delta_v_matrix = np.einsum('bij,bj->bi', topo3ph.Z_3ph, I_matrix, optimize=True)
    s_loss_matrix = delta_v_matrix * np.conj(I_matrix)

    return np.real(s_loss_matrix).reshape(-1), np.imag(s_loss_matrix).reshape(-1)


def reshape_3phase_to_per_node(data_3ph: np.ndarray, n_lines: int) -> np.ndarray:
    """Reshape a node-major ABC vector ``(3*n_lines,)`` to ``(n_lines, 3)``.
    
    Args:
        data_3ph: Flattened three-phase data in
            ``[node1_A, node1_B, node1_C, node2_A, ...]`` order.
        n_lines: Number of non-reference nodes (equal to the radial line count).
        
    Returns:
        Reshaped data (n_lines, 3) where columns are [Phase A, Phase B, Phase C]
    """
    return data_3ph.reshape(n_lines, 3)


def get_phase_results(result: Dict, n_lines: int) -> Dict:
    """Extract per-phase results from power flow result
    
    Args:
        result: Result dict from run_3phase_bfs_power_flow
        n_lines: Number of lines
        
    Returns:
        Dict with phase-separated results:
            - V_A, V_B, V_C: Voltage magnitudes per phase (n_lines,)
            - angle_A, angle_B, angle_C: Voltage angles per phase (n_lines,)
            - P_A, P_B, P_C: Branch power per phase (n_lines,)

        The source arrays in ``result`` are assumed to use the node-major ABC
        layout declared by ``ThreePhaseTopology.vector_layout``.
    """
    V_mag = reshape_3phase_to_per_node(result['V_mag'], n_lines)
    V_angle = reshape_3phase_to_per_node(result['V_angle'], n_lines)
    P_branch = reshape_3phase_to_per_node(result['P_branch'], n_lines)
    Q_branch = reshape_3phase_to_per_node(result['Q_branch'], n_lines)
    
    return {
        'V_A': V_mag[:, 0], 'V_B': V_mag[:, 1], 'V_C': V_mag[:, 2],
        'angle_A': V_angle[:, 0], 'angle_B': V_angle[:, 1], 'angle_C': V_angle[:, 2],
        'P_A': P_branch[:, 0], 'P_B': P_branch[:, 1], 'P_C': P_branch[:, 2],
        'Q_A': Q_branch[:, 0], 'Q_B': Q_branch[:, 1], 'Q_C': Q_branch[:, 2],
    }
