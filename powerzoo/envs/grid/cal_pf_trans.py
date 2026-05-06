"""AC & DC Power Flow solver for transmission grids.

Implements Newton-Raphson AC power flow using the MATPOWER branch model
(pi-model with complex tap ratio). No external dependencies beyond
numpy/scipy.

MATPOWER branch admittance model
---------------------------------
For each branch with parameters (r, x, b, ratio, angle):

    Ys = 1 / (r + jx)           # series admittance
    Bc = b                       # total line charging susceptance
    tap = ratio * exp(j*angle)   # complex tap; ratio=0 means regular line (tap=1)

    Yff = (Ys + jBc/2) / |tap|^2
    Yft = -Ys / conj(tap)
    Ytf = -Ys / tap
    Ytt =  Ys + jBc/2

References:
    MATPOWER Technical Note 2 and MATPOWER source code (makeYbus.m).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Any


# ---------------------------------------------------------------------------
# Helpers: extract branch arrays from case (used by both AC and DC)
# ---------------------------------------------------------------------------

def _extract_branch_arrays(case: Any):
    """Extract branch parameter arrays and internal bus index mapping.

    Returns:
        (n_bus, n_branch, n_gen, bus_id_to_idx, bus_type,
         br_from, br_to, br_r, br_x, br_b, br_ratio, br_angle, br_status)
    """
    nodes = case.nodes
    lines = case.lines

    n_bus = len(nodes)
    n_branch = len(lines)
    n_gen = len(case.units)

    bus_ids = nodes['id'].values.astype(int)
    bus_id_to_idx = {int(bid): i for i, bid in enumerate(bus_ids)}
    bus_type = nodes['type'].values.astype(int)

    br_from = np.array([bus_id_to_idx[int(b)] for b in lines['from'].values])
    br_to = np.array([bus_id_to_idx[int(b)] for b in lines['to'].values])

    def _col(name, default=0.0):
        return lines[name].values.copy() if name in lines.columns else np.full(n_branch, default)

    br_r = _col('r')
    br_x = lines['x'].values.copy()
    br_b = _col('b')
    br_ratio = _col('ratio')
    br_angle = _col('angle')
    br_status = _col('status', 1.0)

    return (n_bus, n_branch, n_gen, bus_id_to_idx, bus_type,
            br_from, br_to, br_r, br_x, br_b, br_ratio, br_angle, br_status)


def _branch_admittances(br_r, br_x, br_b, br_ratio, br_angle):
    """Compute per-branch pi-model admittances (Yff, Yft, Ytf, Ytt).

    Handles ratio=0 as regular line (tap=1).
    Returns (Yff, Yft, Ytf, Ytt, tap) — all complex arrays of length n_branch.
    """
    z = br_r + 1j * br_x                          # branch series impedance
    Ys = np.where(z != 0, 1.0 / z, 0j)            # branch series admittance

    ratio = br_ratio.copy()
    ratio[ratio == 0] = 1.0
    tap = ratio * np.exp(1j * np.radians(br_angle))  # complex tap ratio
    tap_mag2 = (tap * np.conj(tap)).real           # |tap|²

    Yff = (Ys + 0.5j * br_b) / tap_mag2           # from-from pi-model admittance
    Yft = -Ys / np.conj(tap)                       # from-to pi-model admittance
    Ytf = -Ys / tap                                # to-from pi-model admittance
    Ytt = Ys + 0.5j * br_b                         # to-to pi-model admittance

    return Yff, Yft, Ytf, Ytt, tap


# ---------------------------------------------------------------------------
# Ybus construction (vectorised)
# ---------------------------------------------------------------------------

def build_ybus(n_bus: int,
               br_from: np.ndarray,
               br_to: np.ndarray,
               br_r: np.ndarray,
               br_x: np.ndarray,
               br_b: np.ndarray,
               br_ratio: np.ndarray,
               br_angle: np.ndarray,
               br_status: np.ndarray,
               bus_gs: np.ndarray,
               bus_bs: np.ndarray,
               baseMVA: float = 100.0) -> np.ndarray:
    """Build the complex bus admittance matrix (Ybus) — fully vectorised."""
    on = br_status > 0
    f = br_from[on].astype(int)
    t = br_to[on].astype(int)

    Yff, Yft, Ytf, Ytt, _ = _branch_admittances(
        br_r[on], br_x[on], br_b[on], br_ratio[on], br_angle[on])

    # Assemble Ybus via np.add.at (handles duplicate indices)
    Ybus = np.zeros((n_bus, n_bus), dtype=complex)
    np.add.at(Ybus, (f, f), Yff)
    np.add.at(Ybus, (f, t), Yft)
    np.add.at(Ybus, (t, f), Ytf)
    np.add.at(Ybus, (t, t), Ytt)

    # Bus shunts
    diag_idx = np.arange(n_bus)
    Ybus[diag_idx, diag_idx] += (bus_gs + 1j * bus_bs) / baseMVA

    return Ybus


# ---------------------------------------------------------------------------
# Newton-Raphson AC Power Flow (vectorised Jacobian)
# ---------------------------------------------------------------------------

def _run_nr_inner(Vm, Va, P_sched, Q_sched, pv_pq_idx, pq_idx,
                  Ybus, tol, max_iter, verbose):
    """Run the Newton-Raphson inner loop.

    Args:
        Vm, Va:        Voltage magnitude/angle arrays (modified in-place).
        P_sched, Q_sched: Scheduled net injections [p.u.], bus-indexed.
        pv_pq_idx:     Indices of all non-slack buses (PV first, then PQ).
        pq_idx:        Indices of PQ buses only.
        Ybus:          Complex bus admittance matrix.
        tol:           Convergence tolerance on max |mismatch| [p.u.].
        max_iter:      Maximum NR iterations.
        verbose:       Print per-iteration mismatch if True.

    Returns:
        (Vm, Va, converged, iteration)
    """
    G = Ybus.real
    B = Ybus.imag
    n_pq = len(pq_idx)
    n_pvpq = len(pv_pq_idx)

    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        V = Vm * np.exp(1j * Va)
        S = V * np.conj(Ybus @ V)
        P_calc = S.real
        Q_calc = S.imag

        mismatch = np.concatenate([
            P_sched[pv_pq_idx] - P_calc[pv_pq_idx],
            Q_sched[pq_idx] - Q_calc[pq_idx],
        ])

        max_mis = np.max(np.abs(mismatch))
        if verbose:
            print(f"  Iter {iteration}: max |mismatch| = {max_mis:.2e}")
        if max_mis < tol:
            converged = True
            break

        J = _build_jacobian(Vm, Va, G, B, P_calc, Q_calc,
                            pv_pq_idx, pq_idx, n_pvpq, n_pq)
        try:
            dx = np.linalg.solve(J, mismatch)
        except np.linalg.LinAlgError:
            break

        Va[pv_pq_idx] += dx[:n_pvpq]
        Vm[pq_idx] += dx[n_pvpq:]

    if not converged and verbose:
        print(f"  WARNING: did not converge in {max_iter} iterations")

    return Vm, Va, converged, iteration


def run_acpf(case: Any,
             Pd_mw: Optional[np.ndarray] = None,
             Pg_mw: Optional[np.ndarray] = None,
             tol: float = 1e-8,
             max_iter: int = 30,
             verbose: bool = False,
             enforce_q_lim: bool = False,
             max_qlim_iter: int = 10) -> Dict[str, np.ndarray]:
    """Run Newton-Raphson AC power flow on a ClearCase object.

    Bus types (MATPOWER convention stored in ``case.nodes['type']``):
        1 = PQ,  2 = PV,  3 = Slack (reference).

    Args:
        case:           ClearCase instance (provides topology, branch parameters, Vg).
        Pd_mw:          Active load per node (MW), shape (n_bus,).  When provided,
                        overrides ``case.nodes['Pd']``.
        Pg_mw:          Active dispatch per unit (MW), shape (n_gen,).  When provided,
                        overrides ``case.units['Pg']``.
        enforce_q_lim:  If True, enforce generator Qmin/Qmax.  PV buses whose
                        calculated Q exceeds their limit are switched to PQ in an
                        outer loop (up to ``max_qlim_iter`` rounds).  Requires
                        ``case.units`` to have ``Qmin``/``Qmax`` columns.
                        Defaults to False (unlimited Q, backward-compatible).
        max_qlim_iter:  Maximum outer Q-limit switching iterations (default 10).
                        PQ→PV voltage recovery is not attempted.

    **Divergence handling:** If the Jacobian becomes singular or the iteration
    limit is reached, the solver sets ``converged=False`` and returns zero-filled
    placeholder arrays.  Callers must check ``result['converged']`` before using
    physical fields.

    Returns a dict with keys:
        vm, va, va_deg, p_gen, q_gen,
        pf_from, qf_from, pf_to, qf_to,
        p_loss, q_loss, q_loss_net,
        converged, iterations, Ybus
    """
    baseMVA = getattr(case, 'baseMVA', 100.0)
    nodes = case.nodes
    units = case.units

    (n_bus, n_branch, n_gen, bus_id_to_idx, bus_type,
     br_from, br_to, br_r, br_x, br_b,
     br_ratio, br_angle, br_status) = _extract_branch_arrays(case)

    # --- Bus data ---
    Pd = (Pd_mw if Pd_mw is not None else nodes['Pd'].values) / baseMVA
    Qd = nodes['Qd'].values / baseMVA
    Gs = nodes['Gs'].values if 'Gs' in nodes.columns else np.zeros(n_bus)
    Bs = nodes['Bs'].values if 'Bs' in nodes.columns else np.zeros(n_bus)

    # --- Generator data → aggregate to buses (vectorised) ---
    gen_bus_idx = np.array([bus_id_to_idx[int(b)] for b in units['bus_id'].values])
    Pg_bus = np.zeros(n_bus)
    Qg_bus = np.zeros(n_bus)
    Vm_setpoint = np.ones(n_bus)

    pg_values = Pg_mw if Pg_mw is not None else units['Pg'].values
    np.add.at(Pg_bus, gen_bus_idx, pg_values / baseMVA)
    np.add.at(Qg_bus, gen_bus_idx, units['Qg'].values / baseMVA)
    # Voltage setpoint (last gen on each bus wins — fine for well-formed data)
    Vm_setpoint[gen_bus_idx] = units['Vg'].values

    P_sched = Pg_bus - Pd
    Q_sched = Qg_bus - Qd

    # --- Build Ybus ---
    Ybus = build_ybus(n_bus, br_from, br_to, br_r, br_x, br_b,
                       br_ratio, br_angle, br_status, Gs, Bs, baseMVA)
    G = Ybus.real                                  # Ybus conductance (real part)
    B = Ybus.imag                                  # Ybus susceptance (imag part)

    # --- Classify buses ---
    pv_idx_base = np.where(bus_type == 2)[0]       # PV bus indices (all, before switching)
    pq_idx_base = np.where(bus_type == 1)[0]       # PQ bus indices

    # --- Initial guess ---
    Vm = np.ones(n_bus)
    Va = np.zeros(n_bus)
    Vm[pv_idx_base] = Vm_setpoint[pv_idx_base]
    Vm[bus_type == 3] = Vm_setpoint[bus_type == 3]

    # --- Q-limit data (read once, used only when enforce_q_lim=True) ---
    if enforce_q_lim:
        has_q_cols = ('Qmin' in units.columns and 'Qmax' in units.columns)
        if has_q_cols:
            Qg_min_gen = units['Qmin'].values / baseMVA
            Qg_max_gen = units['Qmax'].values / baseMVA
        else:
            Qg_min_gen = np.full(n_gen, -np.inf)
            Qg_max_gen = np.full(n_gen,  np.inf)

        Qg_min_bus = np.full(n_bus,  np.inf)
        Qg_max_bus = np.full(n_bus, -np.inf)
        np.minimum.at(Qg_min_bus, gen_bus_idx, Qg_min_gen)
        np.maximum.at(Qg_max_bus, gen_bus_idx, Qg_max_gen)
        Qg_min_bus[pq_idx_base] = -np.inf
        Qg_max_bus[pq_idx_base] =  np.inf
        Qg_min_bus[bus_type == 3] = -np.inf
        Qg_max_bus[bus_type == 3] =  np.inf
        # PV buses with no generator assigned: treat as unlimited
        Qg_min_bus[pv_idx_base[Qg_min_bus[pv_idx_base] == np.inf]] = -np.inf
        Qg_max_bus[pv_idx_base[Qg_max_bus[pv_idx_base] == -np.inf]] =  np.inf

        # Convert generator Q limits to net-injection limits: Q_net = Q_gen - Q_load.
        # Q_calc in NR is the net bus injection. Comparing against these corrected
        # limits is equivalent to checking Q_gen = Q_calc + Qd against generator limits.
        Qg_min_bus -= Qd
        Qg_max_bus -= Qd

        pv_switched = np.zeros(n_bus, dtype=bool)
        Q_sched_eff = Q_sched.copy()
    else:
        Q_sched_eff = Q_sched

    # --- Outer Q-limit loop (runs once when enforce_q_lim=False) ---
    converged = False
    iteration = 0
    n_qlim_iters = max_qlim_iter if enforce_q_lim else 1

    for _qlim_iter in range(n_qlim_iters):
        # Build current bus classification (some PV may have switched to PQ)
        if enforce_q_lim:
            current_pq_mask = (bus_type == 1) | pv_switched
            pq_idx = np.where(current_pq_mask)[0]
            pv_idx = np.where((bus_type == 2) & ~pv_switched)[0]
            # Re-seat non-switched PV bus voltages at setpoints for warm-start
            Vm[pv_idx] = Vm_setpoint[pv_idx]
        else:
            pv_idx = pv_idx_base
            pq_idx = pq_idx_base
        pv_pq_idx = np.concatenate([pv_idx, pq_idx])

        Vm, Va, converged, iteration = _run_nr_inner(
            Vm, Va, P_sched, Q_sched_eff, pv_pq_idx, pq_idx,
            Ybus, tol, max_iter, verbose)

        if not converged or not enforce_q_lim:
            break

        # --- Q-limit check (only still-PV buses) ---
        V = Vm * np.exp(1j * Va)
        Q_calc_full = (V * np.conj(Ybus @ V)).imag

        switched_this_round = False
        for bus in pv_idx:
            Q_at_bus = Q_calc_full[bus]
            if Q_at_bus > Qg_max_bus[bus]:
                Q_sched_eff[bus] = Qg_max_bus[bus]
                pv_switched[bus] = True
                switched_this_round = True
                if verbose:
                    print(f"  Q-lim: bus {bus} → PQ "
                          f"(Q={Q_at_bus:.3f} > Qmax={Qg_max_bus[bus]:.3f})")
            elif Q_at_bus < Qg_min_bus[bus]:
                Q_sched_eff[bus] = Qg_min_bus[bus]
                pv_switched[bus] = True
                switched_this_round = True
                if verbose:
                    print(f"  Q-lim: bus {bus} → PQ "
                          f"(Q={Q_at_bus:.3f} < Qmin={Qg_min_bus[bus]:.3f})")

        if not switched_this_round:
            break  # no new switches: solution is Q-limit feasible

    # --- Early exit on divergence ---
    # When the solver has not converged (singular Jacobian or iteration
    # limit exceeded), the intermediate Vm/Va values are not a physical
    # solution.  Return a zero-filled dict immediately.
    if not converged:
        # Each key gets its own array to prevent in-place modification of one
        # key from silently corrupting others (aliasing hazard in RL loops).
        return {
            'vm':         np.ones(n_bus),           # flat-start placeholder
            'va':         np.zeros(n_bus),
            'va_deg':     np.zeros(n_bus),
            'p_gen':      np.zeros(n_gen),
            'q_gen':      np.zeros(n_gen),
            'pf_from':    np.zeros(n_branch),
            'qf_from':    np.zeros(n_branch),
            'pf_to':      np.zeros(n_branch),
            'qf_to':      np.zeros(n_branch),
            'p_loss':     np.zeros(n_branch),
            'q_loss':     np.zeros(n_branch),
            'q_loss_net': np.zeros(n_branch),
            'converged':  False,
            'iterations': iteration,
            'Ybus':       Ybus,
        }

    # --- Post-process (converged solution only) ---
    V = Vm * np.exp(1j * Va)
    S_bus = V * np.conj(Ybus @ V) * baseMVA

    # Generator outputs (vectorised)
    p_gen, q_gen = _distribute_gen_power(
        S_bus, Pd * baseMVA, Qd * baseMVA, units, gen_bus_idx, n_gen)

    # Branch flows (vectorised)
    pf_from, qf_from, pf_to, qf_to = _calc_branch_flows(
        V, br_from, br_to, br_r, br_x, br_b,
        br_ratio, br_angle, br_status, baseMVA)
    p_loss, q_loss = _calc_branch_losses(
        V, br_from, br_to, br_r, br_x,
        br_ratio, br_angle, br_status, baseMVA)

    return {
        'vm': Vm,
        'va': Va,
        'va_deg': np.degrees(Va),
        'p_gen': p_gen,
        'q_gen': q_gen,
        'pf_from': pf_from,
        'qf_from': qf_from,
        'pf_to': pf_to,
        'qf_to': qf_to,
        'p_loss': p_loss,
        'q_loss': q_loss,
        'q_loss_net': qf_from + qf_to,
        'converged': converged,
        'iterations': iteration,
        'Ybus': Ybus,
    }


def _build_jacobian(Vm, Va, G, B, P_calc, Q_calc,
                    pv_pq_idx, pq_idx, n_pvpq, n_pq):
    """Build the NR Jacobian using vectorised outer-product formulas.

    Full-bus matrices are computed first, then sliced to the relevant
    rows/columns.  This trades a little extra memory for eliminating
    all Python-level double loops.
    """
    n = len(Vm)

    # Angle difference matrix  dVa[i,j] = Va[i] - Va[j]
    dVa = Va[:, None] - Va[None, :]

    # Outer products of Vm
    VmVm = Vm[:, None] * Vm[None, :]               # Vm_i * Vm_j product matrix

    sin_dVa = np.sin(dVa)
    cos_dVa = np.cos(dVa)

    # --- Full-bus Jacobian sub-matrices (n x n) ---
    # Off-diagonal terms
    H = VmVm * (G * sin_dVa - B * cos_dVa)        # J11: dP/dVa (off-diagonal)
    N = VmVm * (G * cos_dVa + B * sin_dVa)        # J12: dP/dVm (off-diagonal, pre-/Vm_j)
    M = -N                                         # J21: dQ/dVa (off-diagonal)
    L = H.copy()                                   # J22: dQ/dVm (off-diagonal, pre-/Vm_j)

    # Diagonal terms
    diag = np.arange(n)
    H[diag, diag] = -Q_calc - B[diag, diag] * Vm ** 2
    N[diag, diag] = P_calc + G[diag, diag] * Vm ** 2
    M[diag, diag] = P_calc - G[diag, diag] * Vm ** 2
    L[diag, diag] = Q_calc - B[diag, diag] * Vm ** 2

    # N and L need to be divided by Vm[j] for the off-diagonal,
    # but the diagonal already has the correct formula.
    # Actually the standard formulas are:
    #   dP_i/dVm_j (i!=j) = Vm[i] * (G[i,j]*cos + B[i,j]*sin)
    #   dQ_i/dVm_j (i!=j) = Vm[i] * (G[i,j]*sin - B[i,j]*cos)
    # The VmVm product gives Vm[i]*Vm[j], so divide column j by Vm[j]:
    N_col = N / Vm[None, :]   # (n, n)
    L_col = L / Vm[None, :]

    # Restore diagonals (they use different formulas)
    N_col[diag, diag] = P_calc / Vm + G[diag, diag] * Vm
    L_col[diag, diag] = Q_calc / Vm - B[diag, diag] * Vm

    # Slice to relevant indices and assemble
    J11 = H[np.ix_(pv_pq_idx, pv_pq_idx)]         # dP/dVa sub-block
    J12 = N_col[np.ix_(pv_pq_idx, pq_idx)]        # dP/dVm sub-block
    J21 = M[np.ix_(pq_idx, pv_pq_idx)]            # dQ/dVa sub-block
    J22 = L_col[np.ix_(pq_idx, pq_idx)]           # dQ/dVm sub-block

    return np.block([[J11, J12],
                     [J21, J22]])


_PG_SUM_EPS = 1e-9  # threshold below which Σ|Pg_sched| is treated as zero


def _distribute_gen_power(S_bus, Pd_mw, Qd_mw, units, gen_bus_idx, n_gen):
    """Compute per-generator P and Q from solved bus injections.

    **Algorithm — "base + deviation" distribution:**

    Each device keeps its scheduled output as a baseline; any deviation
    between the bus-level solved total and the sum of schedules is
    distributed in proportion to the devices' absolute scheduled values.
    P and Q use *separate* weight vectors so that pure reactive devices
    (e.g. SVGs with Pg = 0) can absorb reactive deviations even when their
    active-power schedule is zero.

        weights_p = |Pg_sched| / Σ|Pg_sched|        (Σ = 1, positive)
        P_delta   = P_total  − Σ Pg_sched
        p_gen     = Pg_sched + P_delta × weights_p   → Σ p_gen = P_total  ✓

        weights_q = |Qg_sched| / Σ|Qg_sched|        (Σ = 1, positive)
        Q_delta   = Q_total  − Σ Qg_sched
        q_gen     = Qg_sched + Q_delta × weights_q   → Σ q_gen = Q_total  ✓

    When ``Σ|Pg_sched|`` or ``Σ|Qg_sched|`` is near-zero (< ``_PG_SUM_EPS``),
    equal weights are used (standard equal-split fallback).

    Key properties:

    1. **Power conservation** — ``Σ p_gen = P_total`` and ``Σ q_gen = Q_total``
       for all device mixes (generators, storage, SVGs).

    2. **Direction preservation** — when scheduled and solved totals agree
       (delta = 0), each device returns exactly its scheduled value.

    3. **SVG / STATCOM compatibility** — a device with Pg = 0 but Qg ≠ 0
       gets zero P-weight (correct: it carries no active power) but a
       proportional Q-weight, so it absorbs reactive deviations normally.

    Example — gen (+10 MW, 3 MVAr) + SVG (0 MW, 5 MVAr), Q_total = 9 MVAr:
        weights_q = [3/8, 5/8],  Q_delta = 9 − 8 = 1
        q_gen     = [3 + 0.375,  5 + 0.625] = [3.375, 5.625],  sum = 9  ✓
    """
    # Bus-level generation = bus injection + bus load
    Pgen_bus = S_bus.real + Pd_mw
    Qgen_bus = S_bus.imag + Qd_mw

    p_gen = np.zeros(n_gen)
    q_gen = np.zeros(n_gen)

    unique_buses = np.unique(gen_bus_idx)

    for bidx in unique_buses:
        gen_mask = gen_bus_idx == bidx
        gen_indices = np.where(gen_mask)[0]
        P_total = Pgen_bus[bidx]
        Q_total = Qgen_bus[bidx]

        if len(gen_indices) == 1:
            p_gen[gen_indices[0]] = P_total
            q_gen[gen_indices[0]] = Q_total
        else:
            Pg_sched = units['Pg'].values[gen_indices]
            Qg_sched = units['Qg'].values[gen_indices]

            n_dev = len(gen_indices)

            # P weights: proportional to |Pg_sched|
            abs_sum_p = np.abs(Pg_sched).sum()
            weights_p = (np.abs(Pg_sched) / abs_sum_p if abs_sum_p > _PG_SUM_EPS
                         else np.full(n_dev, 1.0 / n_dev))

            # Q weights: proportional to |Qg_sched| (independent of P weights)
            # Allows pure reactive devices (SVG, Pg=0) to absorb Q deviations.
            abs_sum_q = np.abs(Qg_sched).sum()
            weights_q = (np.abs(Qg_sched) / abs_sum_q if abs_sum_q > _PG_SUM_EPS
                         else np.full(n_dev, 1.0 / n_dev))

            P_delta = P_total - Pg_sched.sum()
            p_gen[gen_indices] = Pg_sched + P_delta * weights_p

            Q_delta = Q_total - Qg_sched.sum()
            q_gen[gen_indices] = Qg_sched + Q_delta * weights_q

    return p_gen, q_gen


def _calc_branch_flows(V, br_from, br_to, br_r, br_x, br_b,
                       br_ratio, br_angle, br_status, baseMVA):
    """Compute branch power flows — fully vectorised."""
    Yff, Yft, Ytf, Ytt, _ = _branch_admittances(
        br_r, br_x, br_b, br_ratio, br_angle)

    # Bus voltages at branch terminals
    Vf = V[br_from]
    Vt = V[br_to]

    # Complex power
    Sf = Vf * np.conj(Yff * Vf + Yft * Vt) * baseMVA
    St = Vt * np.conj(Ytf * Vf + Ytt * Vt) * baseMVA

    # Zero out inactive branches
    inactive = br_status <= 0
    Sf[inactive] = 0
    St[inactive] = 0

    return Sf.real, Sf.imag, St.real, St.imag


def _calc_branch_losses(V, br_from, br_to, br_r, br_x,
                        br_ratio, br_angle, br_status, baseMVA):
    """Compute MATPOWER-style branch losses from the series element only."""
    z = br_r + 1j * br_x
    Ys = np.where(z != 0, 1.0 / z, 0j)

    ratio = br_ratio.copy()
    ratio[ratio == 0] = 1.0
    tap = ratio * np.exp(1j * np.radians(br_angle))

    Vf_series = V[br_from] / tap
    Vt = V[br_to]
    I_series = Ys * (Vf_series - Vt)
    i_sq = np.abs(I_series) ** 2

    p_loss = i_sq * br_r * baseMVA
    q_loss = i_sq * br_x * baseMVA

    inactive = br_status <= 0
    p_loss[inactive] = 0.0
    q_loss[inactive] = 0.0

    return p_loss, q_loss


# ---------------------------------------------------------------------------
# DC Power Flow — PTDF-based (precompute once, then one matmul per call)
# ---------------------------------------------------------------------------

def build_dcpf_ptdf(case: Any,
                    ref_bus: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Precompute the PTDF matrix for DC power flow.

    Handles transformer tap ratios (MATPOWER ``makeBdc`` model):
        b = 1 / (x * tap),   Bbus = Cft' * diag(b) * Cft   (symmetric)

    The PTDF maps bus net injection (p.u.) to branch flow (p.u.):
        line_flow_mw = PTDF @ node_injection_mw

    Args:
        case:    ClearCase instance.
        ref_bus: Internal index of reference bus (default: first type-3 bus).

    Returns a dict (cache and reuse):
        PTDF, Bf, Bp_red_inv, ref
    """
    (n_bus, n_branch, _n_gen, _bus_id_to_idx, bus_type,
     br_from, br_to, _br_r, br_x, _br_b,
     br_ratio, _br_angle, br_status) = _extract_branch_arrays(case)

    # b = 1/(x*tap) for active branches, 0 for inactive
    tap = np.where(br_ratio == 0, 1.0, br_ratio)   # transformer tap ratio (0 → 1 = regular line)
    b_vec = np.where(br_status > 0, 1.0 / (br_x * tap), 0.0)  # branch susceptance (p.u.)

    # Bf: branch-bus susceptance matrix (vectorised)
    k_idx = np.arange(n_branch)
    Bf = np.zeros((n_branch, n_bus))               # branch-bus susceptance (n_branch × n_bus)
    np.add.at(Bf, (k_idx, br_from), b_vec)
    np.add.at(Bf, (k_idx, br_to), -b_vec)

    # Bbus = Cft' * diag(b) * Cft  (vectorised via np.add.at)
    active = br_status > 0
    fa, ta, ba = br_from[active], br_to[active], b_vec[active]
    Bbus = np.zeros((n_bus, n_bus))                # bus susceptance matrix (n_bus × n_bus)
    np.add.at(Bbus, (fa, fa), ba)
    np.add.at(Bbus, (fa, ta), -ba)
    np.add.at(Bbus, (ta, fa), -ba)
    np.add.at(Bbus, (ta, ta), ba)

    # Reference bus
    if ref_bus is not None:
        ref = ref_bus
    else:
        ref_candidates = np.where(bus_type == 3)[0]
        ref = int(ref_candidates[0]) if len(ref_candidates) > 0 else 0

    # Remove ref → reduced B', compute its inverse via solve (numerically
    # more stable than np.linalg.inv for ill-conditioned cases).
    # Scale note: np.linalg.solve is adequate for benchmark-scale systems
    # (up to a few hundred buses).  For larger grids (≥ ~1000 buses) a
    # sparse LU factorisation (scipy.sparse.linalg) would be preferable,
    # but that is outside the current PowerZoo benchmark scope.
    non_ref = np.concatenate([np.arange(ref), np.arange(ref + 1, n_bus)])
    Bbus_red = Bbus[np.ix_(non_ref, non_ref)]
    n_red = len(non_ref)
    Bp_red_inv = np.linalg.solve(Bbus_red, np.eye(n_red))  # inverse of reduced B' matrix

    PTDF = np.zeros((n_branch, n_bus))             # Power Transfer Distribution Factor
    PTDF[:, non_ref] = Bf[:, non_ref] @ Bp_red_inv

    return {'PTDF': PTDF, 'Bf': Bf, 'Bp_red_inv': Bp_red_inv, 'ref': ref}


def run_dcpf(case: Any,
             ptdf_cache: Optional[Dict] = None,
             verbose: bool = False) -> Dict[str, np.ndarray]:
    """Run DC power flow using the PTDF matrix.

    If ``ptdf_cache`` is provided (from ``build_dcpf_ptdf``), the core
    computation is a single matrix-vector multiply.

    Returns a dict with keys:
        vm, va, va_deg, p_gen, pf_from, pf_to, p_loss, converged, iterations
    """
    if ptdf_cache is None:
        ptdf_cache = build_dcpf_ptdf(case)

    PTDF = ptdf_cache['PTDF']
    ref = ptdf_cache['ref']
    Bp_red_inv = ptdf_cache['Bp_red_inv']

    baseMVA = getattr(case, 'baseMVA', 100.0)
    nodes = case.nodes
    units = case.units

    (n_bus, _n_branch, n_gen, bus_id_to_idx, bus_type,
     *_br) = _extract_branch_arrays(case)

    # Net injection per bus (p.u.) — vectorised
    gen_bus_idx = np.array([bus_id_to_idx[int(b)] for b in units['bus_id'].values])
    Pg_bus = np.zeros(n_bus)
    np.add.at(Pg_bus, gen_bus_idx, units['Pg'].values / baseMVA)
    Pd = nodes['Pd'].values / baseMVA

    P_inj = Pg_bus - Pd
    P_inj[ref] = 0.0

    # Core: one matmul
    pf_from = PTDF @ P_inj * baseMVA

    # Voltage angles
    non_ref = np.concatenate([np.arange(ref), np.arange(ref + 1, n_bus)])
    Va = np.zeros(n_bus)
    Va[non_ref] = Bp_red_inv @ P_inj[non_ref]

    # Generator outputs — slack absorbs mismatch
    p_gen = units['Pg'].values.copy()
    is_slack = bus_type[gen_bus_idx] == 3
    total_load_mw = nodes['Pd'].values.sum()
    total_non_slack = p_gen[~is_slack].sum()
    n_slack = is_slack.sum()
    if n_slack > 0:
        p_gen[is_slack] = (total_load_mw - total_non_slack) / n_slack

    return {
        'vm': np.ones(n_bus),
        'va': Va,
        'va_deg': np.degrees(Va),
        'p_gen': p_gen,
        'pf_from': pf_from,
        'pf_to': -pf_from,
        'p_loss': np.zeros(len(case.lines)),
        'converged': True,
        'iterations': 0,
    }
