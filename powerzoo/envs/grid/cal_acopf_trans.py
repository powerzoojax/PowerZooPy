"""Built-in AC Optimal Power Flow solver (canonical PowerZoo AC-OPF backend).

Solves the full nonlinear AC-OPF in polar coordinates.

Formulation (all in per-unit on baseMVA)
-----------------------------------------
Decision variables:  x = [Va (nb), Vm (nb), Pg (ng), Qg (ng), Ps (nb), Qs (nb)]

    min  sum_i  tc_a_i * Pg_i^2 + tc_b_i * Pg_i          (generation cost, in MW)
       + penalty * sum_j  (Ps_j * baseMVA)^2 + (Qs_j * baseMVA)^2   (slack penalty)
    where tc_a = mc_b / 2, tc_b = mc_c  (TC = integral of MC; exact when mc_a = 0)
    s.t.
        P_inj(Va, Vm) - Cg*Pg + Pd - Ps = 0   (nb equalities; Ps absorbs P imbalance)
        Q_inj(Va, Vm) - Cg*Qg + Qd - Qs = 0   (nb equalities; Qs absorbs Q imbalance)
        Va[ref] = 0                              (1  equality)
        Pg_min <= Pg <= Pg_max                   (box)
        Qg_min <= Qg <= Qg_max                  (box)
        Vm_min <= Vm <= Vm_max                   (box)
        |Sf_k|^2 <= Sf_max_k^2                  (per limited branch)
        |St_k|^2 <= St_max_k^2                  (per limited branch)
        Ps, Qs  free (unbounded)

Slack-variable design
---------------------
  Ps_j and Qs_j are per-bus virtual power injections that make the power-balance
  equalities always feasible.  A quadratic penalty (default penalty=1e6 ¥/MW²)
  drives them to zero when the system is physically solvable.  When the RL agent
  pushes the system into an overloaded or topologically infeasible state, IPOPT
  still converges and returns a result with ``slack_violation > 0`` instead of
  a hard failure, giving the RL training loop continuous, differentiable cost
  signals.  The ``status`` field is ``'optimal'`` when slacks are negligible
  (< 1 MW) and ``'feasibility_restored'`` otherwise.

Backend options
---------------
  'auto'   — use cyipopt (IPOPT) if available, else scipy SLSQP with a one-time warning.
             SLSQP is ~50-100x slower and unsuitable for RL training loops. Use
             backend='ipopt' to raise ImportError immediately when cyipopt is absent.
  'ipopt'  — always use cyipopt (raises ImportError if not installed).
  'slsqp'  — always use scipy SLSQP.

LMP computation
---------------
  When using the cyipopt backend, LMPs are extracted directly from the IPOPT KKT
  multipliers (``info['mult_g'][:nb] / baseMVA``) for the P-balance equalities.
  This gives exact nodal prices at the optimal solution.  The result dict reports
  ``lmp_quality='exact_kkt'`` in this case.  If extraction fails (e.g. abnormal
  termination), the solver falls back to the heuristic marginal-cost propagation
  method (``lmp_quality='approximate'``).  The SLSQP backend always uses the
  heuristic method.

Cache invalidation
------------------
  ``solve_acopf`` caches one solver per case object (keyed by ``id(case)``).
  If you modify the case topology or parameters **in-place** between calls,
  the cached solver becomes stale.  Pass ``rebuild=True`` or call
  ``clear_acopf_cache(case)`` to force a rebuild.

Public API
----------
    solve_acopf(case, node_net_load_mw, ..., backend='auto') -> dict
    solve_acopf_detailed(case, node_net_load_mw, ..., backend='auto') -> dict
    clear_acopf_cache(case=None)

Dependencies: numpy, scipy  (+ optional cyipopt for IPOPT backend).
"""

from __future__ import annotations
import logging
import warnings
from typing import Dict, Any, Optional

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional cyipopt (IPOPT interior-point solver)
# ---------------------------------------------------------------------------
try:
    import cyipopt as _cyipopt
    HAS_CYIPOPT = True
except ImportError:
    _cyipopt = None
    HAS_CYIPOPT = False

_CYIPOPT_WARNED = False  # emit the "slow fallback" warning only once


# ======================================================================
# Ybus construction
# ======================================================================

def _build_ybus(n_bus, from_bus, to_bus, r, x, b_ch, tap, shift_deg):
    """Build Ybus, Yf, Yt as sparse CSC matrices (MATPOWER conventions)."""
    n_br = len(from_bus)
    ys = 1.0 / (r + 1j * x)
    bc = 1j * b_ch / 2.0

    tap_c = tap.astype(complex)
    tap_c[tap_c == 0] = 1.0
    tap_c = tap_c * np.exp(1j * np.deg2rad(shift_deg))

    Ytt = ys + bc
    Yff = (ys + bc) / (tap_c * np.conj(tap_c))
    Yft = -ys / np.conj(tap_c)
    Ytf = -ys / tap_c

    br_idx = np.arange(n_br)
    Yf = sp.csc_matrix(
        (np.concatenate([Yff, Yft]),
         (np.concatenate([br_idx, br_idx]),
          np.concatenate([from_bus, to_bus]))),
        shape=(n_br, n_bus),
    )
    Yt = sp.csc_matrix(
        (np.concatenate([Ytf, Ytt]),
         (np.concatenate([br_idx, br_idx]),
          np.concatenate([from_bus, to_bus]))),
        shape=(n_br, n_bus),
    )

    Cf = sp.csc_matrix((np.ones(n_br), (br_idx, from_bus)), shape=(n_br, n_bus))
    Ct = sp.csc_matrix((np.ones(n_br), (br_idx, to_bus)), shape=(n_br, n_bus))
    Ybus = Cf.T @ Yf + Ct.T @ Yt

    return Ybus, Yf, Yt


# ======================================================================
# cyipopt problem class — sparse Jacobian, L-BFGS Hessian
# ======================================================================

class _IPOPTProblem:
    """Callback class for cyipopt (AC-OPF with slack variables, sparse Jacobian).

    Variable layout
    ---------------
    x = [Va(nb), Vm(nb), Pg(ng), Qg(ng), Ps(nb), Qs(nb)]
    nx = 4*nb + 2*ng

    Constraint layout
    -----------------
    [0 .. nb-1]              P power balance  (equality = 0)
    [nb .. 2*nb-1]           Q power balance  (equality = 0)
    [2*nb .. 2*nb+n_lim-1]  |Sf_k|²  ≤ Smax_k²  (inequality)
    [2*nb+n_lim .. 2*nb+2*n_lim-1]  |St_k|²  ≤ Smax_k²

    All parameters must be in per-unit (on baseMVA).
    """

    def __init__(
        self, nb, ng, Ybus,
        yb_row, yb_col, yb_conj,   # Ybus COO (canonical, no duplicates)
        Cg, gen_bus, tc_a, tc_b, baseMVA,
        Pd, Qd,
        n_lim, fb_lim, tb_lim,
        Yff_lim, Yft_lim, Ytf_lim, Ytt_lim, Smax2_lim,
        penalty_p=1e6,              # ¥/MW² — quadratic P-slack penalty
        penalty_q=None,             # ¥/MW² — quadratic Q-slack penalty (defaults to penalty_p)
    ):
        self.nb, self.ng = nb, ng                  # number of buses / generators
        self.Ybus = Ybus                              # bus admittance matrix (sparse)
        self.yb_row = yb_row                          # Ybus COO row indices
        self.yb_col = yb_col                          # Ybus COO column indices
        self.yb_conj = yb_conj                        # conj(Ybus data) for Jacobian
        self.n_Y = len(yb_row)                        # number of Ybus non-zeros
        self.is_diag_Y = (yb_row == yb_col)           # mask: diagonal entries of Ybus

        self.Cg = Cg                                  # generator-bus incidence matrix (nb × ng)
        self.gen_bus = gen_bus                         # bus index of each generator
        self.tc_a = tc_a                              # TC quadratic coeff (= mc_b/2)
        self.tc_b = tc_b                              # TC linear coeff   (= mc_c)
        self.baseMVA = baseMVA

        self.Pd = Pd                                  # active load at each bus (p.u.)
        self.Qd = Qd                                  # reactive load at each bus (p.u.)
        self.penalty_p = float(penalty_p)             # quadratic P-slack penalty (¥/MW²)
        self.penalty_q = float(penalty_q) if penalty_q is not None else self.penalty_p  # Q-slack penalty

        self.n_lim = n_lim                            # number of flow-limited branches
        if n_lim > 0:
            self.fb_lim = fb_lim                      # from-bus indices of limited branches
            self.tb_lim = tb_lim                      # to-bus indices of limited branches
            self.Yff_lim = Yff_lim                    # Yff of limited branches (complex)
            self.Yft_lim = Yft_lim                    # Yft of limited branches (complex)
            self.Ytf_lim = Ytf_lim                    # Ytf of limited branches (complex)
            self.Ytt_lim = Ytt_lim                    # Ytt of limited branches (complex)
            self.Smax2_lim = Smax2_lim                # (Sf_max)² of limited branches (p.u.²)

        # Variable index slices  — nx = 4*nb + 2*ng
        self.iVa = slice(0, nb)                       # bus voltage angles
        self.iVm = slice(nb, 2 * nb)                  # bus voltage magnitudes
        self.iPg = slice(2 * nb, 2 * nb + ng)         # generator active power
        self.iQg = slice(2 * nb + ng, 2 * nb + 2 * ng)  # generator reactive power
        self.iPs = slice(2 * nb + 2 * ng, 3 * nb + 2 * ng)  # P slack (one per bus)
        self.iQs = slice(3 * nb + 2 * ng, 4 * nb + 2 * ng)  # Q slack (one per bus)

        self._build_jac_structure()

    # ------------------------------------------------------------------
    def _build_jac_structure(self):
        """Precompute sparse Jacobian (row, col) indices — called once."""
        nb, ng = self.nb, self.ng
        n_Y = self.n_Y
        n_lim = self.n_lim
        yb_row, yb_col = self.yb_row, self.yb_col
        gen_bus = self.gen_bus
        gen_idx = np.arange(ng)

        # Equality part — variable layout: [Va, Vm, Pg, Qg, Ps, Qs]
        # dP/dVa: rows=yb_row,     cols=yb_col
        # dP/dVm: rows=yb_row,     cols=yb_col+nb
        # dP/dPg: rows=gen_bus,    cols=2*nb+gen_idx        (value = -1)
        # dP/dPs: rows=bus_idx,    cols=2*nb+2*ng+bus_idx   (value = -1, diagonal)
        # dQ/dVa: rows=yb_row+nb,  cols=yb_col
        # dQ/dVm: rows=yb_row+nb,  cols=yb_col+nb
        # dQ/dQg: rows=gen_bus+nb, cols=2*nb+ng+gen_idx     (value = -1)
        # dQ/dQs: rows=nb+bus_idx, cols=3*nb+2*ng+bus_idx   (value = -1, diagonal)
        bus_idx = np.arange(nb, dtype=np.int32)
        eq_rows = np.concatenate([
            yb_row, yb_row, gen_bus, bus_idx,
            yb_row + nb, yb_row + nb, gen_bus + nb, bus_idx + nb,
        ]).astype(np.int32)
        eq_cols = np.concatenate([
            yb_col,   yb_col + nb,  gen_idx + 2*nb,    bus_idx + 2*nb + 2*ng,
            yb_col,   yb_col + nb,  gen_idx + 2*nb + ng, bus_idx + 3*nb + 2*ng,
        ]).astype(np.int32)

        if n_lim > 0:
            fb_lim, tb_lim = self.fb_lim, self.tb_lim
            # Each Sf/St constraint has 4 non-zeros: Va_fb, Vm_fb, Va_tb, Vm_tb
            sf_rows = np.repeat(np.arange(n_lim, dtype=np.int32) + 2*nb, 4)
            st_rows = np.repeat(np.arange(n_lim, dtype=np.int32) + 2*nb + n_lim, 4)
            br_cols = np.zeros(4 * n_lim, dtype=np.int32)
            br_cols[0::4] = fb_lim
            br_cols[1::4] = fb_lim + nb
            br_cols[2::4] = tb_lim
            br_cols[3::4] = tb_lim + nb

            self._jac_rows = np.concatenate([eq_rows, sf_rows, st_rows])
            self._jac_cols = np.concatenate([eq_cols, br_cols, br_cols])
        else:
            self._jac_rows = eq_rows
            self._jac_cols = eq_cols

    # ------------------------------------------------------------------
    # cyipopt callbacks
    # ------------------------------------------------------------------

    def objective(self, x):
        bMVA = self.baseMVA
        Pg_mw = x[self.iPg] * bMVA
        gen_cost = float(np.sum(self.tc_a * Pg_mw**2 + self.tc_b * Pg_mw))
        # Slack penalty (quadratic, in MW²): drives Ps/Qs → 0 when feasible
        Ps_mw = x[self.iPs] * bMVA
        Qs_mw = x[self.iQs] * bMVA
        slack_cost = (self.penalty_p * float(np.sum(Ps_mw**2))
                      + self.penalty_q * float(np.sum(Qs_mw**2)))
        return gen_cost + slack_cost

    def gradient(self, x):
        nb, ng, bMVA = self.nb, self.ng, self.baseMVA
        g = np.zeros(4*nb + 2*ng)          # expanded: Va, Vm, Pg, Qg, Ps, Qs
        Pg_mw = x[self.iPg] * bMVA
        g[self.iPg] = (2 * self.tc_a * Pg_mw + self.tc_b) * bMVA
        # d(penalty_p * Ps_mw^2)/dPs_pu = 2 * penalty_p * Ps_mw * bMVA
        Ps_mw = x[self.iPs] * bMVA
        Qs_mw = x[self.iQs] * bMVA
        g[self.iPs] = 2 * self.penalty_p * Ps_mw * bMVA
        g[self.iQs] = 2 * self.penalty_q * Qs_mw * bMVA
        return g

    def constraints(self, x):
        Va, Vm = x[self.iVa], x[self.iVm]
        V = Vm * np.exp(1j * Va)
        S = V * np.conj(self.Ybus @ V)
        # Slack variables absorb any power imbalance, making balance always satisfiable
        gP = S.real - self.Cg @ x[self.iPg] + self.Pd - x[self.iPs]
        gQ = S.imag - self.Cg @ x[self.iQg] + self.Qd - x[self.iQs]
        if self.n_lim == 0:
            return np.concatenate([gP, gQ])
        fb, tb = self.fb_lim, self.tb_lim
        If = self.Yff_lim * V[fb] + self.Yft_lim * V[tb]
        It = self.Ytf_lim * V[fb] + self.Ytt_lim * V[tb]
        Sf = V[fb] * np.conj(If)
        St = V[tb] * np.conj(It)
        return np.concatenate([gP, gQ, np.abs(Sf)**2, np.abs(St)**2])

    def jacobianstructure(self):
        return (self._jac_rows, self._jac_cols)

    def jacobian(self, x):
        nb, ng = self.nb, self.ng
        Va_, Vm_ = x[self.iVa], x[self.iVm]
        V = Vm_ * np.exp(1j * Va_)
        I = np.asarray(self.Ybus @ V).ravel()
        Vnorm = V / Vm_  # = exp(j*Va), avoids division by |V|

        yb_row = self.yb_row
        yb_col = self.yb_col
        yb_conj = self.yb_conj
        is_diag = self.is_diag_Y

        # dS/dVa[i,k] = -j * V[i] * conj(V[k]) * conj(Y[i,k])
        # diagonal: add j * V[i] * conj(I[i])
        dS_dVa = -1j * V[yb_row] * np.conj(V[yb_col]) * yb_conj
        dS_dVa[is_diag] += 1j * V[yb_row[is_diag]] * np.conj(I[yb_row[is_diag]])

        # dS/dVm[i,k] = V[i] * conj(Vnorm[k]) * conj(Y[i,k])
        # diagonal: add Vnorm[i] * conj(I[i])
        dS_dVm = V[yb_row] * np.conj(Vnorm[yb_col]) * yb_conj
        dS_dVm[is_diag] += Vnorm[yb_row[is_diag]] * np.conj(I[yb_row[is_diag]])

        nb = self.nb
        eq_vals = np.concatenate([
            dS_dVa.real,             # dP/dVa  (n_Y)
            dS_dVm.real,             # dP/dVm  (n_Y)
            np.full(ng, -1.0),       # dP/dPg  (ng)
            np.full(nb, -1.0),       # dP/dPs  (nb)  — slack diagonal
            dS_dVa.imag,             # dQ/dVa  (n_Y)
            dS_dVm.imag,             # dQ/dVm  (n_Y)
            np.full(ng, -1.0),       # dQ/dQg  (ng)
            np.full(nb, -1.0),       # dQ/dQs  (nb)  — slack diagonal
        ])

        if self.n_lim == 0:
            return eq_vals

        ineq_vals = self._ineq_jac_vals(V, Vnorm)
        return np.concatenate([eq_vals, ineq_vals])

    def _ineq_jac_vals(self, V, Vnorm):
        """Analytic sparse Jacobian values for |Sf|² and |St|² constraints."""
        n_lim = self.n_lim
        fb, tb = self.fb_lim, self.tb_lim
        Yff, Yft = self.Yff_lim, self.Yft_lim
        Ytf, Ytt = self.Ytf_lim, self.Ytt_lim

        Vf, Vt = V[fb], V[tb]
        Vnf, Vnt = Vnorm[fb], Vnorm[tb]
        Vmf2 = np.abs(Vf)**2
        Vmt2 = np.abs(Vt)**2

        If = Yff * Vf + Yft * Vt
        It = Ytf * Vf + Ytt * Vt
        Sf = Vf * np.conj(If)
        St = Vt * np.conj(It)

        # d|Sf|²/dx = 2 * Re(Sf * conj(dSf/dx))
        # dSf/dVa_fb = j*(Sf - Vmf²*conj(Yff))
        # dSf/dVa_tb = -j * Vf * conj(Yft * Vt)
        # dSf/dVm_fb = Vnf*conj(If) + Vf*conj(Yff*Vnf)
        # dSf/dVm_tb = Vf * conj(Yft * Vnt)
        dSf_dVa_fb = 1j * (Sf - Vmf2 * np.conj(Yff))
        dSf_dVa_tb = -1j * Vf * np.conj(Yft * Vt)
        dSf_dVm_fb = Vnf * np.conj(If) + Vf * np.conj(Yff * Vnf)
        dSf_dVm_tb = Vf * np.conj(Yft * Vnt)

        d_Sf2_dVa_fb = 2 * (Sf * np.conj(dSf_dVa_fb)).real
        d_Sf2_dVa_tb = 2 * (Sf * np.conj(dSf_dVa_tb)).real
        d_Sf2_dVm_fb = 2 * (Sf * np.conj(dSf_dVm_fb)).real
        d_Sf2_dVm_tb = 2 * (Sf * np.conj(dSf_dVm_tb)).real

        # dSt/dVa_fb = -j * Vt * conj(Ytf * Vf)
        # dSt/dVa_tb = j*(St - Vmt²*conj(Ytt))
        # dSt/dVm_fb = Vt * conj(Ytf * Vnf)
        # dSt/dVm_tb = Vnt*conj(It) + Vt*conj(Ytt*Vnt)
        dSt_dVa_fb = -1j * Vt * np.conj(Ytf * Vf)
        dSt_dVa_tb = 1j * (St - Vmt2 * np.conj(Ytt))
        dSt_dVm_fb = Vt * np.conj(Ytf * Vnf)
        dSt_dVm_tb = Vnt * np.conj(It) + Vt * np.conj(Ytt * Vnt)

        d_St2_dVa_fb = 2 * (St * np.conj(dSt_dVa_fb)).real
        d_St2_dVa_tb = 2 * (St * np.conj(dSt_dVa_tb)).real
        d_St2_dVm_fb = 2 * (St * np.conj(dSt_dVm_fb)).real
        d_St2_dVm_tb = 2 * (St * np.conj(dSt_dVm_tb)).real

        # Order: for each branch k → [Va_fb, Vm_fb, Va_tb, Vm_tb]
        vals_Sf = np.zeros(4 * n_lim)
        vals_Sf[0::4] = d_Sf2_dVa_fb
        vals_Sf[1::4] = d_Sf2_dVm_fb
        vals_Sf[2::4] = d_Sf2_dVa_tb
        vals_Sf[3::4] = d_Sf2_dVm_tb

        vals_St = np.zeros(4 * n_lim)
        vals_St[0::4] = d_St2_dVa_fb
        vals_St[1::4] = d_St2_dVm_fb
        vals_St[2::4] = d_St2_dVa_tb
        vals_St[3::4] = d_St2_dVm_tb

        return np.concatenate([vals_Sf, vals_St])


# ======================================================================
# Main solver class
# ======================================================================

class ACOPFSolverBuiltin:
    """Lightweight AC-OPF solver.

    Build once from a ClearCase, solve repeatedly with different loads.

    Args:
        backend: ``'auto'`` (default) tries cyipopt first and falls back to SLSQP;
                 ``'ipopt'`` always uses cyipopt (ImportError if not installed);
                 ``'slsqp'`` always uses scipy SLSQP.
    """

    # ====== Initialization ======

    def __init__(
        self,
        case,
        baseMVA: float = None,
        v_min: float = 0.95,
        v_max: float = 1.05,
        q_factor: float = 0.75,
        max_iter: int = 300,
        tol: float = 1e-6,
        verbose: bool = False,
        backend: str = 'auto',
        penalty: float = 1e6,
        penalty_p: Optional[float] = None,
        penalty_q: Optional[float] = None,
    ):
        global _CYIPOPT_WARNED

        self.case = case
        self.baseMVA = baseMVA or getattr(case, 'baseMVA', 100.0)
        self.v_min = v_min
        self.v_max = v_max
        self.q_factor = q_factor
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        # penalty_p / penalty_q allow different weights for P vs Q slack.
        # If only `penalty` is given, both default to it (backward compatible).
        self.penalty_p: float = float(penalty_p) if penalty_p is not None else float(penalty)
        self.penalty_q: float = float(penalty_q) if penalty_q is not None else self.penalty_p

        # Resolve backend
        if backend == 'auto':
            if HAS_CYIPOPT:
                self._use_ipopt = True
            else:
                self._use_ipopt = False
                if not _CYIPOPT_WARNED:
                    warnings.warn(
                        "cyipopt not found — AC-OPF falling back to scipy SLSQP which is "
                        "~50-100x slower (Case118 ≈ 4-12s vs ~0.3s). "
                        "Install with: conda install -c conda-forge cyipopt",
                        RuntimeWarning, stacklevel=2,
                    )
                    _CYIPOPT_WARNED = True
        elif backend == 'ipopt':
            if not HAS_CYIPOPT:
                raise ImportError(
                    "cyipopt is required for backend='ipopt'. "
                    "Install with: conda install -c conda-forge cyipopt"
                )
            self._use_ipopt = True
        elif backend == 'slsqp':
            self._use_ipopt = False
        else:
            raise ValueError(f"backend must be 'auto', 'ipopt', or 'slsqp', got '{backend}'")

        if not getattr(case, 'init_flag', False):
            case.init()

        self._parse_case()
        self._build_admittance()

        # Cache last solution for warm-starting
        self._last_x: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _parse_case(self):
        case = self.case
        nodes, units, lines = case.nodes, case.units, case.lines
        nb, ng, nl = len(nodes), len(units), len(lines)  # bus / gen / branch counts
        self.n_bus, self.n_gen, self.n_branch = nb, ng, nl

        self.ref_bus = getattr(case, 'slack_bus', 0)   # slack (reference) bus index
        self.gen_bus = units['#bus_id'].values.astype(int)  # bus index of each generator

        bMVA = self.baseMVA                            # system base power (MVA)
        self.Pg_max = units['p_max'].values / bMVA     # gen active power upper bound (p.u.)
        self.Pg_min = units['p_min'].values / bMVA     # gen active power lower bound (p.u.)

        if 'Qmax' in units.columns and 'Qmin' in units.columns:
            # Explicit Q limits from case data take strict priority.
            # Extension point: for inverter-based resources with a circular
            # capability curve, caller can pre-compute Qmax/Qmin per operating
            # point (e.g. Qmax = sqrt(Smax^2 - Pg^2)) and pass them in the
            # units DataFrame before constructing the solver.
            self.Qg_max = units['Qmax'].values / bMVA
            self.Qg_min = units['Qmin'].values / bMVA
        else:
            # Fallback: symmetric Q envelope scaled from Pg_max via q_factor.
            # This is a rough proxy for synchronous generators; it is not
            # appropriate for inverters where Q capability depends on Pg.
            self.Qg_max = self.Pg_max * self.q_factor   # gen Q upper bound (p.u.)
            self.Qg_min = -self.Pg_max * self.q_factor   # gen Q lower bound (p.u.)

        # TC(P) = tc_a·P² + tc_b·P;  gradient = 2·tc_a·P + tc_b = MC(P) = mc_b·P + mc_c
        # Mapping from MC coefficients: tc_a = mc_b/2, tc_b = mc_c  (exact when mc_a=0)
        self.tc_a = units['mc_b'].values / 2     # TC quadratic coeff (¥/MW²)
        self.tc_b = units['mc_c'].values          # TC linear coeff (¥/MW)

        if 'Vmax' in nodes.columns and 'Vmin' in nodes.columns:
            self.Vmax = nodes['Vmax'].values.astype(float).copy()  # bus voltage upper limit (p.u.)
            self.Vmin = nodes['Vmin'].values.astype(float).copy()  # bus voltage lower limit (p.u.)
        else:
            self.Vmax = np.full(nb, self.v_max)
            self.Vmin = np.full(nb, self.v_min)
        if self.v_min != 0.95 or self.v_max != 1.05:
            self.Vmin[:] = self.v_min
            self.Vmax[:] = self.v_max

        self.bus_Gs = nodes['Gs'].values.astype(float) / bMVA if 'Gs' in nodes.columns else np.zeros(nb)  # bus shunt conductance (p.u.)
        self.bus_Bs = nodes['Bs'].values.astype(float) / bMVA if 'Bs' in nodes.columns else np.zeros(nb)  # bus shunt susceptance (p.u.)

        self.from_bus = lines['#from'].values.astype(int)      # branch from-bus index
        self.to_bus = lines['#to'].values.astype(int)          # branch to-bus index
        self.branch_x = lines['x'].values.astype(float)        # branch reactance (p.u.)
        self.branch_r = lines['r'].values.astype(float) if 'r' in lines.columns else np.zeros(nl)       # branch resistance (p.u.)
        self.branch_b = lines['b'].values.astype(float) if 'b' in lines.columns else np.zeros(nl)       # branch charging susceptance (p.u.)
        self.branch_tap = lines['ratio'].values.astype(float) if 'ratio' in lines.columns else np.zeros(nl)  # transformer tap ratio
        self.branch_shift = lines['angle'].values.astype(float) if 'angle' in lines.columns else np.zeros(nl)  # phase shift angle (deg)

        cap = lines['cap'].values.astype(float).copy()
        cap[cap >= 1e5] = 0
        cap[cap < 0] = 0
        self.Sf_max_pu = cap / bMVA                   # branch apparent power limit (p.u.)
        self.line_limited = self.Sf_max_pu > 0         # mask: branches with finite thermal limit

        self.Cg = np.zeros((nb, ng))                   # generator-bus incidence matrix (nb × ng)
        self.Cg[self.gen_bus, np.arange(ng)] = 1.0

        self.nodes_units_map = case.get_nodes_units_map()  # (n_bus × n_gen) mapping matrix

    # ------------------------------------------------------------------
    def _build_admittance(self):
        self.Ybus, self.Yf, self.Yt = _build_ybus(    # bus / from-branch / to-branch admittance matrices
            self.n_bus, self.from_bus, self.to_bus,
            self.branch_r, self.branch_x, self.branch_b,
            self.branch_tap, self.branch_shift,
        )
        Ysh = self.bus_Gs + 1j * self.bus_Bs           # bus shunt admittance
        self.Ybus = self.Ybus + sp.diags(Ysh, format='csc')

        # Dense conjugate for SLSQP Jacobian
        self._Ybus_conj_dense = self.Ybus.conj().toarray()

        # Canonical COO for IPOPT sparse Jacobian
        Ycsr = self.Ybus.tocsr()
        Ycoo = Ycsr.tocoo()
        self._yb_coo_row = Ycoo.row.copy()             # Ybus COO row indices
        self._yb_coo_col = Ycoo.col.copy()             # Ybus COO column indices
        self._yb_coo_conj = Ycoo.data.conj().copy()    # conjugate of Ybus COO data

        # Pre-extract Yf/Yt element values for limited branches
        self._precompute_lim_branches()

    def _precompute_lim_branches(self):
        limited_idx = np.where(self.line_limited)[0]
        self.n_lim = len(limited_idx)

        if self.n_lim == 0:
            self.fb_lim = np.array([], dtype=int)
            self.tb_lim = np.array([], dtype=int)
            self.Yff_lim = np.array([], dtype=complex)
            self.Yft_lim = np.array([], dtype=complex)
            self.Ytf_lim = np.array([], dtype=complex)
            self.Ytt_lim = np.array([], dtype=complex)
            self.Smax2_lim = np.array([])
            return

        fb_lim = self.from_bus[limited_idx]
        tb_lim = self.to_bus[limited_idx]

        # Extract Yf[k, fb_k] etc. from sparse Yf/Yt (rows = limited_idx)
        Yf_lim = self.Yf[limited_idx].tocsr()
        Yt_lim = self.Yt[limited_idx].tocsr()

        idx = np.arange(self.n_lim)
        Yff = np.asarray(Yf_lim[idx, fb_lim]).ravel()
        Yft = np.asarray(Yf_lim[idx, tb_lim]).ravel()
        Ytf = np.asarray(Yt_lim[idx, fb_lim]).ravel()
        Ytt = np.asarray(Yt_lim[idx, tb_lim]).ravel()

        self.fb_lim = fb_lim                      # from-bus indices of limited branches
        self.tb_lim = tb_lim                      # to-bus indices of limited branches
        self.Yff_lim = Yff                        # Yff of limited branches (complex)
        self.Yft_lim = Yft                        # Yft of limited branches (complex)
        self.Ytf_lim = Ytf                        # Ytf of limited branches (complex)
        self.Ytt_lim = Ytt                        # Ytt of limited branches (complex)
        self.Smax2_lim = self.Sf_max_pu[limited_idx] ** 2  # (Sf_max)² (p.u.²)
        self.lim_idx = limited_idx                # original branch indices of limited branches

    # ------------------------------------------------------------------
    def _dc_warm_start(self, Pd, Pg_min, Pg_max, commitment):
        nb, ng = self.n_bus, self.n_gen
        bMVA = self.baseMVA
        try:
            from powerzoo.envs.grid.cal_dcopf_trans import solve_ed_opf_detailed
            dc_result = solve_ed_opf_detailed(
                self.case, Pd * bMVA,
                commitment=commitment, solver_type='scipy', rebuild=False,
            )
            if dc_result['success']:
                Pg0 = np.clip(dc_result['unit_power_mw'] / bMVA, Pg_min, Pg_max)
                return Pg0, np.zeros(nb)
        except Exception:
            pass
        total_Pd = Pd.sum()
        total_cap = Pg_max.sum()
        frac = np.clip(total_Pd / total_cap, 0, 1) if total_cap > 1e-8 else 0.0
        Pg0 = Pg_min + (Pg_max - Pg_min) * frac
        return np.clip(Pg0, Pg_min, Pg_max), np.zeros(nb)

    # ====== Solver Interface ======

    def reset_warm_start(self) -> None:
        """Discard the cached warm-start point.

        Call this at the start of each RL episode (``env.reset()``) to prevent
        the solver from being seeded with a stale solution from the previous
        episode, which can cause convergence to a wrong local optimum.
        """
        self._last_x = None

    def solve(
        self,
        node_net_load_mw: np.ndarray,
        commitment: Optional[np.ndarray] = None,
        gen_p_max_mw: Optional[np.ndarray] = None,
        gen_p_min_mw: Optional[np.ndarray] = None,
        gen_q_max_mvar: Optional[np.ndarray] = None,
        gen_q_min_mvar: Optional[np.ndarray] = None,
        clear_warm_start: bool = False,
    ) -> Dict[str, Any]:
        """Solve AC-OPF for the given load.

        Args:
            node_net_load_mw: Per-bus net active load (MW).
            commitment: Binary array (ng,); 0 = unit offline.
            gen_p_max_mw: Per-generator active power upper bound (MW).
                Overrides the value from ``_parse_case`` for this call only.
                Useful for time-varying renewable output limits.
            gen_p_min_mw: Per-generator active power lower bound (MW).
            gen_q_max_mvar: Per-generator reactive power upper bound (MVAr).
            gen_q_min_mvar: Per-generator reactive power lower bound (MVAr).
            clear_warm_start: If ``True``, discard the cached warm-start point
                before solving (equivalent to calling ``reset_warm_start()``
                first).  Useful when the operating condition has changed
                drastically between calls.
        """
        if clear_warm_start:
            self._last_x = None

        nb, ng, nl = self.n_bus, self.n_gen, self.n_branch
        bMVA = self.baseMVA

        Pd = node_net_load_mw / bMVA
        Qd = np.zeros(nb)
        if 'Qd' in self.case.nodes.columns:
            Pd_orig = self.case.nodes['Pd'].values / bMVA
            Qd_orig = self.case.nodes['Qd'].values / bMVA
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(np.abs(Pd_orig) > 1e-8, Pd / Pd_orig, 0.0)
            Qd = Qd_orig * ratio

        Pg_min, Pg_max = self.Pg_min.copy(), self.Pg_max.copy()
        Qg_min, Qg_max = self.Qg_min.copy(), self.Qg_max.copy()

        # Dynamic per-step generator bound overrides (e.g. wind/solar curtailment)
        if gen_p_max_mw is not None:
            Pg_max[:] = np.asarray(gen_p_max_mw, dtype=float) / bMVA
        if gen_p_min_mw is not None:
            Pg_min[:] = np.asarray(gen_p_min_mw, dtype=float) / bMVA
        if gen_q_max_mvar is not None:
            Qg_max[:] = np.asarray(gen_q_max_mvar, dtype=float) / bMVA
        if gen_q_min_mvar is not None:
            Qg_min[:] = np.asarray(gen_q_min_mvar, dtype=float) / bMVA

        if commitment is not None:
            off = np.asarray(commitment, dtype=float) == 0
            Pg_min[off] = Pg_max[off] = 0.0
            Qg_min[off] = Qg_max[off] = 0.0

        # Variable layout: x = [Va(nb), Vm(nb), Pg(ng), Qg(ng), Ps(nb), Qs(nb)]
        nx_base = 2 * nb + 2 * ng
        nx = nx_base + 2 * nb          # 4*nb + 2*ng  (includes slack variables)
        iVa = slice(0, nb)
        iVm = slice(nb, 2*nb)
        iPg = slice(2*nb, 2*nb+ng)
        iQg = slice(2*nb+ng, nx_base)
        iPs = slice(nx_base, nx_base+nb)          # P-balance slack per bus
        iQs = slice(nx_base+nb, nx)               # Q-balance slack per bus

        lb = np.concatenate([np.full(nb, -np.pi), self.Vmin, Pg_min, Qg_min,
                             np.full(nb, -np.inf), np.full(nb, -np.inf)])
        ub = np.concatenate([np.full(nb, np.pi), self.Vmax, Pg_max, Qg_max,
                             np.full(nb, np.inf), np.full(nb, np.inf)])
        lb[self.ref_bus] = 0.0
        ub[self.ref_bus] = 0.0

        # Initial point — slacks initialised to 0 (assume feasibility)
        use_warm = (self._last_x is not None
                    and len(self._last_x) == nx
                    and (nb < 50 or self._use_ipopt))
        if use_warm:
            x0 = self._last_x.copy()
            x0[iVa] = np.clip(x0[iVa], -np.pi, np.pi)
            x0[iVm] = np.clip(x0[iVm], self.Vmin, self.Vmax)
            x0[iPg] = np.clip(x0[iPg], Pg_min, Pg_max)
            x0[iQg] = np.clip(x0[iQg], Qg_min, Qg_max)
            x0[iPs] = 0.0      # reset slacks at warm start
            x0[iQs] = 0.0
        else:
            Pg0, Va0 = self._dc_warm_start(Pd, Pg_min, Pg_max, commitment)
            x0 = np.concatenate([Va0, np.ones(nb), Pg0,
                                  np.clip(np.zeros(ng), Qg_min, Qg_max),
                                  np.zeros(nb), np.zeros(nb)])

        # Dispatch to solver backend
        _ipopt_info = None
        if self._use_ipopt:
            x, success, _ipopt_info = self._solve_ipopt(
                Pd, Qd, Pg_min, Pg_max, Qg_min, Qg_max, x0, lb, ub)
        else:
            x, success = self._solve_slsqp(
                Pd, Qd, Pg_min, Pg_max, Qg_min, Qg_max, x0, lb, ub,
                iVa, iVm, iPg, iQg, iPs, iQs, use_warm)

        if not success:
            return self._failure_result()

        # Cache for warm-start
        self._last_x = x.copy()

        Va, Vm = x[iVa], x[iVm]
        Pg, Qg = x[iPg], x[iQg]
        Ps, Qs = x[iPs], x[iQs]                   # per-bus slack values (p.u.)
        unit_power_mw = Pg * bMVA
        q_gen = Qg * bMVA

        V = Vm * np.exp(1j * Va)
        Sf = np.asarray(V[self.from_bus] * np.conj(self.Yf @ V)).ravel()
        line_flow_mw = Sf.real * bMVA
        line_flow_q_mvar = Sf.imag * bMVA

        # line_viol_mva: maximum single-branch MVA overload (from-end |Sf| vs cap).
        # Unlimited branches (Sf_max_pu == 0) are skipped.  MAX semantics match
        # the JAX reference (ac_opf.py) and the Python _trans_solve.ac_thermal_check
        # convention: cost_thermal uses sum; line_viol_mva is the worst-line diagnostic.
        sf_mva = np.abs(Sf) * bMVA
        effective_cap_mva = np.where(self.line_limited, self.Sf_max_pu * bMVA, np.inf)
        violations = np.maximum(0.0, sf_mva - effective_cap_mva)
        line_viol_mva = float(np.max(violations)) if violations.size > 0 else 0.0

        node_unit_power_mw = self.nodes_units_map @ unit_power_mw
        node_net_injection_mw = node_unit_power_mw - node_net_load_mw
        total_cost = float(np.sum(self.tc_a * unit_power_mw**2 + self.tc_b * unit_power_mw))

        # Slack magnitude: largest per-bus imbalance absorbed (MW)
        slack_violation = float(max(np.max(np.abs(Ps)), np.max(np.abs(Qs))) * bMVA)
        status = 'feasibility_restored' if slack_violation > 1.0 else 'optimal'

        # LMP: try exact KKT multipliers from IPOPT, fall back to heuristic.
        # Note: when slacks are large (infeasible region), LMP values reflect the
        # penalised dual prices and should be interpreted with caution.
        if _ipopt_info is not None:
            lmp, lmp_quality = self._lmp_from_mult_g(_ipopt_info, bMVA)
            if lmp is None:
                lmp = self._estimate_lmp(x, iPg, bMVA)
                lmp_quality = 'approximate'
        else:
            lmp = self._estimate_lmp(x, iPg, bMVA)
            lmp_quality = 'approximate'

        # When feasibility slacks are active, KKT duals include 1e6-scale
        # penalty terms that distort LMP.  Downgrade quality accordingly.
        if status == 'feasibility_restored' and lmp_quality == 'exact_kkt':
            lmp_quality = 'infeasible_penalized'

        backend_name = 'cyipopt_ipopt' if self._use_ipopt else 'scipy_slsqp'
        return {
            'unit_power_mw': unit_power_mw,
            'line_flow_mw': line_flow_mw,
            'line_flow_q_mvar': line_flow_q_mvar,
            'line_viol_mva': line_viol_mva,
            'node_net_injection_mw': node_net_injection_mw,
            'total_cost': total_cost,
            'slack_violation': slack_violation,
            'status': status,
            'success': True,
            'lmp': lmp,
            'vm_pu': Vm.copy(),
            'va_deg': np.rad2deg(Va),
            'q_gen': q_gen,
            'solver_backend': backend_name,
            'lmp_method': 'kkt_dual',
            'lmp_quality': lmp_quality,
            'lmp_available': True,
        }

    # ====== Internal Solver Backends ======

    def _solve_ipopt(self, Pd, Qd, Pg_min, Pg_max, Qg_min, Qg_max, x0, lb, ub):
        nb, ng = self.n_bus, self.n_gen
        nx = 4 * nb + 2 * ng           # includes Ps(nb) and Qs(nb) slack variables
        n_lim = self.n_lim
        n_con = 2 * nb + 2 * n_lim

        prob = _IPOPTProblem(
            nb=nb, ng=ng, Ybus=self.Ybus,
            yb_row=self._yb_coo_row, yb_col=self._yb_coo_col,
            yb_conj=self._yb_coo_conj,
            Cg=self.Cg, gen_bus=self.gen_bus,
            tc_a=self.tc_a, tc_b=self.tc_b, baseMVA=self.baseMVA,
            Pd=Pd, Qd=Qd,
            n_lim=n_lim, fb_lim=self.fb_lim, tb_lim=self.tb_lim,
            Yff_lim=self.Yff_lim, Yft_lim=self.Yft_lim,
            Ytf_lim=self.Ytf_lim, Ytt_lim=self.Ytt_lim,
            Smax2_lim=self.Smax2_lim,
            penalty_p=self.penalty_p,
            penalty_q=self.penalty_q,
        )

        # Constraint bounds: equality = 0; inequality: |Sf|² ≤ Smax²
        cl = np.zeros(n_con)
        cu = np.zeros(n_con)
        if n_lim > 0:
            cl[2*nb:] = -np.inf
            cu[2*nb:2*nb + n_lim] = self.Smax2_lim
            cu[2*nb + n_lim:] = self.Smax2_lim

        nlp = _cyipopt.Problem(
            n=nx, m=n_con,
            problem_obj=prob,
            lb=lb, ub=ub, cl=cl, cu=cu,
        )
        nlp.add_option('hessian_approximation', 'limited-memory')
        nlp.add_option('mu_strategy', 'adaptive')
        nlp.add_option('tol', self.tol)
        nlp.add_option('max_iter', self.max_iter)
        nlp.add_option('print_level', 5 if self.verbose else 0)
        nlp.add_option('sb', 'yes')    # suppress IPOPT banner

        x_opt, info = nlp.solve(x0)
        status = info['status']
        success = status in (0, 1)  # 0=optimal, 1=acceptable

        if not success and self.verbose:
            logger.warning("IPOPT status %d: %s", status, info.get('status_msg', ''))

        return x_opt, success, info

    # ------------------------------------------------------------------
    def _solve_slsqp(self, Pd, Qd, Pg_min, Pg_max, Qg_min, Qg_max, x0, lb, ub,
                     iVa, iVm, iPg, iQg, iPs, iQs, use_warm):
        nb, ng = self.n_bus, self.n_gen
        nx = 4 * nb + 2 * ng           # includes Ps(nb) and Qs(nb)
        penalty_p = self.penalty_p
        penalty_q = self.penalty_q
        # Replace ±inf slack bounds with finite scipy-compatible values
        lb_slsqp = np.where(np.isinf(lb), -1e10, lb)
        ub_slsqp = np.where(np.isinf(ub),  1e10, ub)
        bounds = list(zip(lb_slsqp, ub_slsqp))
        Cg = self.Cg
        tc_a, tc_b = self.tc_a, self.tc_b
        bMVA = self.baseMVA
        Ybus = self.Ybus
        Ybc = self._Ybus_conj_dense

        def objective(x):
            Pg_mw = x[iPg] * bMVA
            gen_cost = float(np.sum(tc_a * Pg_mw**2 + tc_b * Pg_mw))
            Ps_mw = x[iPs] * bMVA
            Qs_mw = x[iQs] * bMVA
            slack_cost = (penalty_p * float(np.sum(Ps_mw**2))
                          + penalty_q * float(np.sum(Qs_mw**2)))
            return gen_cost + slack_cost

        def grad_obj(x):
            g = np.zeros(nx)
            Pg_mw = x[iPg] * bMVA
            g[iPg] = (2 * tc_a * Pg_mw + tc_b) * bMVA
            Ps_mw = x[iPs] * bMVA
            Qs_mw = x[iQs] * bMVA
            g[iPs] = 2 * penalty_p * Ps_mw * bMVA
            g[iQs] = 2 * penalty_q * Qs_mw * bMVA
            return g

        def eq_con(x):
            V = x[iVm] * np.exp(1j * x[iVa])
            S = V * np.conj(Ybus @ V)
            return np.concatenate([S.real - Cg @ x[iPg] + Pd - x[iPs],
                                    S.imag - Cg @ x[iQg] + Qd - x[iQs]])

        def eq_jac(x):
            Va_, Vm_ = x[iVa], x[iVm]
            V = Vm_ * np.exp(1j * Va_)
            I = np.asarray(Ybus @ V).ravel()
            VVc = np.outer(V, V.conj())
            dS_dVa = -1j * (VVc * Ybc)
            np.fill_diagonal(dS_dVa, dS_dVa.diagonal() + 1j * V * I.conj())
            Vnorm = V / np.abs(V)
            VNc = np.outer(V, Vnorm.conj())
            dS_dVm = VNc * Ybc
            np.fill_diagonal(dS_dVm, dS_dVm.diagonal() + Vnorm * I.conj())
            J = np.zeros((2 * nb, nx))
            J[:nb, iVa] = dS_dVa.real
            J[:nb, iVm] = dS_dVm.real
            J[:nb, iPg] = -Cg
            J[:nb, iPs] = -np.eye(nb)   # dP_balance/dPs = -I
            J[nb:, iVa] = dS_dVa.imag
            J[nb:, iVm] = dS_dVm.imag
            J[nb:, iQg] = -Cg
            J[nb:, iQs] = -np.eye(nb)   # dQ_balance/dQs = -I
            return J

        n_lim = self.n_lim
        cons = [{'type': 'eq', 'fun': eq_con, 'jac': eq_jac}]
        if n_lim > 0:
            fb_lim, tb_lim = self.fb_lim, self.tb_lim
            Yf_lim = self.Yf[self.lim_idx]
            Yt_lim = self.Yt[self.lim_idx]
            Smax2_lim = self.Smax2_lim

            def ineq_con(x):
                V = x[iVm] * np.exp(1j * x[iVa])
                Sf = V[fb_lim] * np.conj(Yf_lim @ V)
                St = V[tb_lim] * np.conj(Yt_lim @ V)
                return np.concatenate([Smax2_lim - np.abs(Sf)**2,
                                       Smax2_lim - np.abs(St)**2])
            cons.append({'type': 'ineq', 'fun': ineq_con})

        feas_tol = 0.01

        def _run(x_init):
            r = minimize(objective, x_init, method='SLSQP', jac=grad_obj,
                         bounds=bounds, constraints=cons,
                         options={'maxiter': self.max_iter,
                                  'ftol': self.tol * 0.01,
                                  'disp': self.verbose})
            ev = np.max(np.abs(eq_con(r.x)))
            if ev > feas_tol:
                r2 = minimize(objective, r.x, method='SLSQP', jac=grad_obj,
                              bounds=bounds, constraints=cons,
                              options={'maxiter': self.max_iter,
                                       'ftol': 1e-14,
                                       'disp': self.verbose})
                ev2 = np.max(np.abs(eq_con(r2.x)))
                if ev2 < ev:
                    return r2, ev2
            return r, ev

        res, eq_viol = _run(x0)

        # If warm-start failed, try cold start
        if eq_viol > feas_tol and use_warm:
            Pg0, Va0 = self._dc_warm_start(Pd, Pg_min, Pg_max, None)
            x0_cold = np.concatenate([Va0, np.ones(nb), Pg0,
                                       np.clip(np.zeros(ng), Qg_min, Qg_max),
                                       np.zeros(nb), np.zeros(nb)])
            res, eq_viol = _run(x0_cold)

        # Bail on genuine numerical failure (NaN/Inf in solution vector)
        if not np.all(np.isfinite(res.x)):
            if self.verbose:
                logger.warning("SLSQP: non-finite values in solution")
            return None, False

        # Analytical projection: Ps/Qs are free and appear linearly in the
        # balance equalities, so any remaining constraint residual can always
        # be absorbed directly into the slack variables.  This guarantees
        # feasibility and ensures the SLSQP backend returns a result (with
        # possibly large slack_violation) instead of a hard failure.
        final_x = res.x.copy()
        if eq_viol > 1e-8:
            residual = eq_con(final_x)
            final_x[iPs] += residual[:nb]
            final_x[iQs] += residual[nb:]
            if self.verbose:
                logger.info("SLSQP: slack projection applied, max residual=%.2e", eq_viol)

        return final_x, True

    # ------------------------------------------------------------------
    def _lmp_from_mult_g(self, info, bMVA):
        """Extract exact nodal LMPs from IPOPT P-balance KKT multipliers.

        The P-balance constraints occupy the first ``nb`` slots of ``mult_g``.
        At optimality, the KKT condition for an interior generator g at bus b
        gives:   MC_g * baseMVA = mult_g[b]
        so:      LMP[b]  [¥/MW]  = mult_g[b] / baseMVA.

        Returns ``(lmp_array, 'exact_kkt')`` on success, or ``(None, None)``
        when multipliers are unavailable or contain non-finite values.
        """
        nb = self.n_bus
        try:
            mult_g = info.get('mult_g') if isinstance(info, dict) else getattr(info, 'mult_g', None)
            if mult_g is None or len(mult_g) < nb:
                return None, None
            lmp = np.asarray(mult_g[:nb], dtype=float) / bMVA
            if not np.all(np.isfinite(lmp)):
                return None, None
            return lmp, 'exact_kkt'
        except Exception:
            return None, None

    # ------------------------------------------------------------------
    def _estimate_lmp(self, x, iPg, bMVA):
        """KKT-based LMP estimate at the optimal point (heuristic fallback)."""
        nb, ng = self.n_bus, self.n_gen
        tc_a, tc_b = self.tc_a, self.tc_b
        Pg_mw = x[iPg] * bMVA
        mc = 2 * tc_a * Pg_mw + tc_b  # MC(P) = mc_b·P + mc_c (when mc_a=0)

        Pg_pu = x[iPg]
        interior = np.ones(ng, dtype=bool)
        tol = 1e-6
        interior[np.abs(Pg_pu - self.Pg_min) < tol] = False
        interior[np.abs(Pg_pu - self.Pg_max) < tol] = False

        lmp = np.full(nb, np.nan)
        for g in range(ng):
            b = self.gen_bus[g]
            if interior[g] and np.isnan(lmp[b]):
                lmp[b] = mc[g]
        for g in range(ng):
            b = self.gen_bus[g]
            if np.isnan(lmp[b]):
                lmp[b] = mc[g]

        for _ in range(nb):
            changed = False
            for k in range(self.n_branch):
                f, t = self.from_bus[k], self.to_bus[k]
                if np.isnan(lmp[f]) and not np.isnan(lmp[t]):
                    lmp[f] = lmp[t]; changed = True
                elif np.isnan(lmp[t]) and not np.isnan(lmp[f]):
                    lmp[t] = lmp[f]; changed = True
            if not changed:
                break

        if np.any(np.isnan(lmp)):
            avg = np.nanmean(lmp) if np.any(~np.isnan(lmp)) else np.mean(mc)
            lmp[np.isnan(lmp)] = avg
        return lmp

    # ------------------------------------------------------------------
    def _failure_result(self) -> Dict[str, Any]:
        backend_name = 'cyipopt_ipopt' if self._use_ipopt else 'scipy_slsqp'
        return {
            'unit_power_mw': np.zeros(self.n_gen),
            'line_flow_mw': np.zeros(self.n_branch),
            'line_flow_q_mvar': np.zeros(self.n_branch),
            'line_viol_mva': 0.0,
            'node_net_injection_mw': np.zeros(self.n_bus),
            'total_cost': np.inf,
            'slack_violation': np.inf,
            'status': 'failed',
            'success': False,
            'lmp': np.zeros(self.n_bus),
            'vm_pu': np.ones(self.n_bus),
            'va_deg': np.zeros(self.n_bus),
            'q_gen': np.zeros(self.n_gen),
            'solver_backend': backend_name,
            'lmp_method': 'none',
            'lmp_quality': 'unavailable',
            'lmp_available': False,
        }


# ======================================================================
# Module-level API
# ======================================================================

# Stores (case_ref, solver) tuples so the case object stays alive and its
# id() cannot be recycled by the GC.
_solver_cache: Dict[int, tuple] = {}


def clear_acopf_cache(case=None) -> None:
    """Invalidate the module-level AC-OPF solver cache.

    The cache key is ``id(case)``, which cannot detect in-place modifications
    to a case's topology or parameters.  Call this function (or pass
    ``rebuild=True`` to ``solve_acopf``) after any structural change to the
    case object.

    Args:
        case: If given, remove only the entry for this specific case object.
              If ``None`` (default), clear the entire cache.
    """
    global _solver_cache
    if case is None:
        _solver_cache.clear()
    else:
        _solver_cache.pop(id(case), None)


def solve_acopf(
    case,
    node_net_load_mw: np.ndarray,
    commitment: Optional[np.ndarray] = None,
    baseMVA: float = None,
    v_min: float = 0.95,
    v_max: float = 1.05,
    q_factor: float = 0.75,
    max_iter: int = 300,
    tol: float = 1e-6,
    rebuild: bool = False,
    verbose: bool = False,
    backend: str = 'auto',
    penalty: float = 1e6,
    penalty_p: Optional[float] = None,
    penalty_q: Optional[float] = None,
    gen_p_max_mw: Optional[np.ndarray] = None,
    gen_p_min_mw: Optional[np.ndarray] = None,
    gen_q_max_mvar: Optional[np.ndarray] = None,
    gen_q_min_mvar: Optional[np.ndarray] = None,
    clear_warm_start: bool = False,
) -> Dict[str, Any]:
    """Solve AC-OPF with the built-in PowerZoo backend.

    Args:
        backend: ``'auto'`` uses cyipopt if available, otherwise falls back to
            scipy SLSQP with a one-time warning.  SLSQP is ~50-100x slower and
            is not suitable for RL training loops; use ``backend='ipopt'`` to
            raise ``ImportError`` immediately when cyipopt is absent rather than
            silently falling back.
        rebuild: Force reconstruction of the cached solver.  Pass ``True``
            after any in-place modification of the case's topology or
            parameters; alternatively call ``clear_acopf_cache(case)``.
        penalty: Quadratic penalty coefficient (¥/MW²) applied to both P and Q
            slack variables when ``penalty_p`` / ``penalty_q`` are not given.
        penalty_p: Quadratic penalty on P-balance slack (¥/MW²).  Overrides
            ``penalty`` for active-power imbalance.
        penalty_q: Quadratic penalty on Q-balance slack (¥/MVAr²).  Overrides
            ``penalty`` for reactive-power imbalance.  Defaults to ``penalty_p``
            when not specified.
        gen_p_max_mw: Per-generator active power upper bound (MW) for this call
            only.  When supplied, overrides the values parsed from the case.
            Pass a length-ng array; useful for time-varying renewable output.
        gen_p_min_mw: Per-generator active power lower bound (MW).
        gen_q_max_mvar: Per-generator reactive power upper bound (MVAr).
        gen_q_min_mvar: Per-generator reactive power lower bound (MVAr).
    """
    case_id = id(case)
    entry = _solver_cache.get(case_id)
    if entry is None or entry[0] is not case or rebuild:
        solver = ACOPFSolverBuiltin(
            case, baseMVA=baseMVA, v_min=v_min, v_max=v_max,
            q_factor=q_factor, max_iter=max_iter, tol=tol,
            verbose=verbose, backend=backend,
            penalty=penalty, penalty_p=penalty_p, penalty_q=penalty_q,
        )
        _solver_cache[case_id] = (case, solver)
    else:
        solver = entry[1]
    return solver.solve(
        node_net_load_mw, commitment,
        gen_p_max_mw=gen_p_max_mw, gen_p_min_mw=gen_p_min_mw,
        gen_q_max_mvar=gen_q_max_mvar, gen_q_min_mvar=gen_q_min_mvar,
        clear_warm_start=clear_warm_start,
    )


def solve_acopf_detailed(
    case,
    node_net_load_mw: np.ndarray,
    commitment: Optional[np.ndarray] = None,
    baseMVA: float = None,
    vn_kv: float = None,
    v_min: float = 0.95,
    v_max: float = 1.05,
    q_factor: float = 0.75,
    max_iter: int = 300,
    tol: float = 1e-6,
    rebuild: bool = False,
    verbose: bool = False,
    backend: str = 'auto',
    penalty: float = 1e6,
    penalty_p: Optional[float] = None,
    penalty_q: Optional[float] = None,
    gen_p_max_mw: Optional[np.ndarray] = None,
    gen_p_min_mw: Optional[np.ndarray] = None,
    gen_q_max_mvar: Optional[np.ndarray] = None,
    gen_q_min_mvar: Optional[np.ndarray] = None,
    clear_warm_start: bool = False,
) -> Dict[str, Any]:
    """Compatibility alias using the historical cal_* detailed solver name."""
    _ = vn_kv
    return solve_acopf(
        case,
        node_net_load_mw,
        commitment=commitment,
        baseMVA=baseMVA,
        v_min=v_min,
        v_max=v_max,
        q_factor=q_factor,
        max_iter=max_iter,
        tol=tol,
        rebuild=rebuild,
        verbose=verbose,
        backend=backend,
        penalty=penalty,
        penalty_p=penalty_p,
        penalty_q=penalty_q,
        gen_p_max_mw=gen_p_max_mw,
        gen_p_min_mw=gen_p_min_mw,
        gen_q_max_mvar=gen_q_max_mvar,
        gen_q_min_mvar=gen_q_min_mvar,
        clear_warm_start=clear_warm_start,
    )


ACOPFSolver = ACOPFSolverBuiltin

__all__ = [
    'HAS_CYIPOPT',
    'ACOPFSolver',
    'ACOPFSolverBuiltin',
    'solve_acopf',
    'solve_acopf_detailed',
    'clear_acopf_cache',
]
