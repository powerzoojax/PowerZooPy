"""Three-Phase Forward-Backward Sweep Power Flow Calculation

Core computational functions for three-phase radial distribution network power flow.
Based on BIBC (Bus Injection to Branch Current) and BCBV (Branch Current to Bus Voltage) method.

Reference: three_phase_BFS.py and IEEE 123-bus test system.
"""
from typing import Tuple, Dict
from dataclasses import dataclass

import numpy as np
from scipy import sparse

# Reuse topology building from single-phase module
from powerzoo.envs.grid.cal_pf_dist import build_radial_topology, RadialTopology


@dataclass
class ThreePhaseTopology:
    """Data structure for three-phase radial network topology"""
    n_nodes: int  # Number of nodes (excluding reference)
    n_lines: int  # Number of lines
    ref_bus: int  # Reference bus index (0-based)
    
    # Basic topology from single-phase
    topo: RadialTopology
    
    # Three-phase specific matrices
    Z_3ph: np.ndarray  # (n_lines, 3, 3) complex impedance matrices in p.u.
    BIBC: np.ndarray  # (3*n_lines, 3*n_lines) Bus Injection to Branch Current matrix
    BCBV: np.ndarray  # (3*n_lines, 3*n_lines) Branch Current to Bus Voltage matrix (complex)
    DLF: np.ndarray  # (3*n_lines, 3*n_lines) BCBV @ BIBC combined matrix
    
    # Reference voltage (3-phase balanced)
    V_ref_3ph: np.ndarray  # (3,) complex reference voltage for three phases


def build_3phase_topology(n_nodes: int, from_nodes: np.ndarray, to_nodes: np.ndarray,
                          Z_3ph_pu: np.ndarray, ref_bus: int = 0,
                          v_ref_mag: float = 1.0) -> ThreePhaseTopology:
    """Build three-phase radial network topology with BIBC/BCBV matrices
    
    Args:
        n_nodes: Number of nodes in the network
        from_nodes: Array of from-node indices for each line
        to_nodes: Array of to-node indices for each line
        Z_3ph_pu: (n_lines, 3, 3) complex impedance matrices in p.u.
        ref_bus: Reference (slack) bus index (default: 0)
        v_ref_mag: Reference voltage magnitude (default: 1.0)
        
    Returns:
        ThreePhaseTopology: Topology data structure with precomputed matrices
    """
    n_lines = len(from_nodes)
    
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
    # Following the convention from three_phase_BFS.py:
    # K[br, a] = 1, K[br, b] = -1 by default
    # K[br, a] = -1, K[br, b] = 1 if to_node <= ref_bus
    K = np.zeros((n_lines, n_nodes))
    
    for br in range(n_lines):
        a = from_nodes[br]
        b = to_nodes[br]
        if b <= ref_bus:
            K[br, a] = -1
            K[br, b] = 1
        else:
            K[br, a] = 1
            K[br, b] = -1
    
    # Remove reference bus column to get Gamma
    Gamma = np.delete(K, ref_bus, axis=1)
    
    # BIBC_0 = -inv(Gamma')
    # BIBC_0[b, bb] = 1 means current on branch b depends on injection at node bb
    BIBC_0 = -np.linalg.inv(Gamma.T)
    
    # Expand to 3-phase BIBC matrix (3*n_lines x 3*n_lines)
    # Each element of BIBC_0 becomes a 3x3 identity block (diagonal coupling)
    BIBC = np.zeros((3 * n_lines, 3 * n_lines))
    for b in range(n_lines):
        for bb in range(n_lines):
            BIBC[b*3:(b+1)*3, bb*3:(bb+1)*3] = BIBC_0[b, bb] * np.eye(3)
    
    # Build BCBV matrix (complex)
    # BCBV[b, bb] = Z_3ph[bb] if branch bb is on path from root to branch b's receiving end
    BCBV = np.zeros((3 * n_lines, 3 * n_lines), dtype=np.complex128)
    for b in range(n_lines):
        for bb in range(n_lines):
            if BIBC_0[bb, b] == 1:  # Branch bb contributes to voltage drop at node b+1
                BCBV[b*3:(b+1)*3, bb*3:(bb+1)*3] = Z_3ph_pu[bb]
    
    # Precompute DLF = BCBV @ BIBC for efficiency
    DLF = BCBV @ BIBC
    
    # Reference voltage (3-phase balanced)
    # Phase A: V∠0°, Phase B: V∠-120°, Phase C: V∠120°
    V_ref_3ph = v_ref_mag * np.array([
        1.0,
        np.cos(np.deg2rad(-120)) + 1j * np.sin(np.deg2rad(-120)),
        np.cos(np.deg2rad(120)) + 1j * np.sin(np.deg2rad(120))
    ], dtype=np.complex128)
    
    return ThreePhaseTopology(
        n_nodes=n_nodes,
        n_lines=n_lines,
        ref_bus=ref_bus,
        topo=topo,
        Z_3ph=Z_3ph_pu,
        BIBC=BIBC,
        BCBV=BCBV,
        DLF=DLF,
        V_ref_3ph=V_ref_3ph
    )


def run_3phase_bfs_power_flow(topo3ph: ThreePhaseTopology,
                              P_3ph_pu: np.ndarray, Q_3ph_pu: np.ndarray,
                              v_ref_mag: float = 1.0,
                              max_iter: int = 100, tol: float = 1e-6) -> Dict:
    """Run three-phase Forward-Backward Sweep power flow using BIBC/BCBV method
    
    Uses complex calculation: I = conj((P + jQ) / V), delta_V = DLF @ I
    
    Args:
        topo3ph: ThreePhaseTopology structure
        P_3ph_pu: (n_lines, 3) or (3*n_lines,) active power load at each node/phase (p.u.)
                  Note: n_lines = n_nodes - 1 (excluding reference bus)
        Q_3ph_pu: (n_lines, 3) or (3*n_lines,) reactive power load at each node/phase (p.u.)
        v_ref_mag: Reference voltage magnitude (default: 1.0)
        max_iter: Maximum iterations (default: 100)
        tol: Convergence tolerance (default: 1e-6)
        
    Returns:
        Dict with keys:
            - V: Complex voltage at each node/phase (3*n_lines,)
            - V_mag: Voltage magnitude (3*n_lines,)
            - V_angle: Voltage angle in degrees (3*n_lines,)
            - I_branch: Complex branch current (3*n_lines,)
            - P_branch: Active power flow on each branch/phase (3*n_lines,)
            - Q_branch: Reactive power flow on each branch/phase (3*n_lines,)
            - converged: Whether converged
            - iterations: Number of iterations
    """
    n_lines = topo3ph.n_lines
    
    # Reshape power arrays to (3*n_lines,) column vectors
    if P_3ph_pu.ndim == 2:
        Pbus = P_3ph_pu.reshape(-1, 1)  # (3*n_lines, 1)
    else:
        Pbus = P_3ph_pu.reshape(-1, 1)
    
    if Q_3ph_pu.ndim == 2:
        Qbus = Q_3ph_pu.reshape(-1, 1)
    else:
        Qbus = Q_3ph_pu.reshape(-1, 1)
    
    # Negative because loads consume power (convention in BFS)
    Pbus = -Pbus
    Qbus = -Qbus
    
    # Reference voltage for all nodes (3-phase balanced)
    V_ref_3ph = v_ref_mag * np.array([
        1.0,
        np.cos(np.deg2rad(-120)) + 1j * np.sin(np.deg2rad(-120)),
        np.cos(np.deg2rad(120)) + 1j * np.sin(np.deg2rad(120))
    ], dtype=np.complex128)
    
    # Initialize all node voltages to reference
    Vr_n = np.zeros((3 * n_lines, 1), dtype=np.complex128)
    for b in range(n_lines):
        Vr_n[b*3:(b+1)*3, 0] = V_ref_3ph
    
    # Iterative solution
    V = Vr_n.copy()
    converged = False
    
    for iteration in range(max_iter):
        V_old_real = V.real.copy()
        
        # Calculate injection current: I = conj((P + jQ) / V)
        S = Pbus + 1j * Qbus
        I = np.conj(S / V)
        
        # Calculate voltage drop: delta_V = DLF @ I
        delta_V = topo3ph.DLF @ I
        
        # Update voltage: V = Vr_n + delta_V
        V_new = Vr_n + delta_V
        
        # Check convergence
        max_diff = np.max(np.abs(V_new.real - V_old_real))
        V = V_new
        
        if max_diff < tol:
            converged = True
            break
    
    # Calculate final branch currents
    I_branch = topo3ph.BIBC @ I
    
    # Calculate branch power
    S_branch = V * np.conj(I_branch)
    P_branch = -S_branch.real
    Q_branch = -S_branch.imag
    
    # Flatten results
    V_flat = V.flatten()
    V_mag = np.abs(V_flat)
    V_angle = np.angle(V_flat, deg=True)
    
    return {
        'V': V_flat,
        'V_mag': V_mag,
        'V_angle': V_angle,
        'I_branch': I_branch.flatten(),
        'P_branch': P_branch.flatten(),
        'Q_branch': Q_branch.flatten(),
        'converged': converged,
        'iterations': iteration + 1
    }


def calculate_3phase_losses(topo3ph: ThreePhaseTopology, result: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate three-phase line losses
    
    Args:
        topo3ph: ThreePhaseTopology structure
        result: Result dict from run_3phase_bfs_power_flow
        
    Returns:
        P_loss: Active power loss per phase (3*n_lines,) in p.u.
        Q_loss: Reactive power loss per phase (3*n_lines,) in p.u.
    """
    # Loss = I^2 * R for each phase
    I_branch = result['I_branch']
    I_sq = np.abs(I_branch) ** 2
    
    n_lines = topo3ph.n_lines
    P_loss = np.zeros(3 * n_lines)
    Q_loss = np.zeros(3 * n_lines)
    
    for br in range(n_lines):
        Z_br = topo3ph.Z_3ph[br]  # 3x3 complex
        I_br = I_branch[br*3:(br+1)*3]  # 3 phases
        
        # S_loss = I * Z * I^H (simplified to diagonal terms for loss)
        for ph in range(3):
            P_loss[br*3 + ph] = np.real(Z_br[ph, ph]) * np.abs(I_br[ph])**2
            Q_loss[br*3 + ph] = np.imag(Z_br[ph, ph]) * np.abs(I_br[ph])**2
    
    return P_loss, Q_loss


def reshape_3phase_to_per_node(data_3ph: np.ndarray, n_lines: int) -> np.ndarray:
    """Reshape (3*n_lines,) array to (n_lines, 3) for per-node analysis
    
    Args:
        data_3ph: Flattened 3-phase data (3*n_lines,)
        n_lines: Number of lines/nodes
        
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

