"""DC-OPF (PTDF-based linear OPF / economic dispatch) for transmission grid.

Supports three LP backends (auto-selected at runtime):
  1. **Gurobi**  — fastest; requires ``pip install powerzoo[gurobi]`` + license.
  2. **HiGHS**   — free, via ``scipy.optimize.linprog``; used by default when
                   Gurobi is not installed.  No extra install needed.
  3. **CVXPY**   — optional; ``pip install powerzoo[cvxpy]``; tries GLPK→ECOS→SCS.

The solver is selected via ``solver_type`` (default ``'auto'``).
``'auto'`` prefers Gurobi when available, otherwise falls back to HiGHS.

The model is built once and reused for multiple solves by updating variable bounds.

**System-balance feasibility**: all three backends include a global load-shedding
slack ``ls`` and curtailment slack ``cur`` in the power-balance equality so that
the LP is always feasible even under extreme commitment or renewable scenarios.
Both are non-negative and penalised at ``slack_penalty`` in the objective.
Their combined magnitude is reported in ``slack_violation``.

**RL safety**: failure branches return ``_INFEASIBLE_COST`` (1e9) instead of
``np.inf`` to keep gradients finite during training.

**Performance**: the SciPy backend caches constant LP matrices (``A_ub``,
``A_eq``, ``M_u``) per case.  The CVXPY backend uses ``cp.Parameter`` so the
problem is canonicalised once and only parameter values are updated per step.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

# Sentinel used when a line limit of 0 means "no limit".
_NO_LIMIT = 1e5
# Finite cost returned on infeasibility; avoids NaN / inf in RL training.
_INFEASIBLE_COST = 1e9

logger = logging.getLogger(__name__)

try:
    import gurobipy as gp
    from gurobipy import GRB

    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
    gp = None
    GRB = None
    logger.debug(
        "gurobipy not found — Gurobi solver disabled. "
        "PowerZoo will use the free HiGHS solver (scipy) automatically. "
        "To enable Gurobi: pip install powerzoo[gurobi]"
    )


# ---------------------------------------------------------------------------
# Module-level pure functions  (shared by all three backends)
# ---------------------------------------------------------------------------

def _eval_generation_cost(mc_a: np.ndarray, mc_b: np.ndarray,
                           mc_c: np.ndarray, p_mw: np.ndarray) -> float:
    """True cubic generation cost: TC(P) = (mc_a/3)·P³ + (mc_b/2)·P² + mc_c·P.

    The LP objective minimises only mc_c·P (linear approximation); this
    function evaluates the full post-hoc cost for reporting purposes.
    """
    return float(((mc_a / 3) * p_mw ** 3
                  + (mc_b / 2) * p_mw ** 2
                  + mc_c * p_mw).sum())


def _infeasible_result(n_lines: int, n_nodes: int, p_min_eff: np.ndarray,
                       status, backend: str, **extra) -> Dict[str, Any]:
    """Return a finite-cost failure dict so RL gradients stay finite.

    ``p_min_eff`` should already reflect commitment (physical lower bounds).
    Extra keyword arguments are merged in to support backend-specific fields
    (e.g. ``offer_cost`` for the piecewise solver).
    """
    return {
        'unit_power_mw': p_min_eff,
        'line_flow_mw': np.zeros(n_lines),
        'node_net_injection_mw': np.zeros(n_nodes),
        'total_cost': _INFEASIBLE_COST,
        'slack_violation': _INFEASIBLE_COST,
        'load_shedding_mw': 0.0,
        'curtailment_mw': 0.0,
        'status': status,
        'success': False,
        'lmp': np.zeros(n_nodes),
        'solver_backend': backend,
        'lmp_method': None,
        'lmp_quality': None,
        'lmp_available': False,
        **extra,
    }


def _warn_nonlinear_cost(backend: str, mc_a: np.ndarray, mc_b: np.ndarray) -> None:
    """Warn once when units have nonlinear cost coefficients the LP cannot optimise."""
    if os.environ.get("POWERZOO_SUPPRESS_NONLINEAR_ED_WARNING", "").strip() == "1":
        return
    n_nonlinear = int(((mc_a != 0) | (mc_b != 0)).sum())
    if n_nonlinear:
        logger.warning(
            "%s: %d unit(s) have mc_a/mc_b > 0 but the LP objective uses only mc_c·P. "
            "total_cost is evaluated post-hoc as (mc_a/3)·P³+(mc_b/2)·P²+mc_c·P "
            "but the dispatch is NOT cubic/quadratic-optimal. "
            "Set mc_a=mc_b=0 to silence this warning.",
            backend, n_nonlinear,
        )


# ---------------------------------------------------------------------------
# Abstract backend interface
# ---------------------------------------------------------------------------

class _EDBackend(ABC):
    """Common interface for all ED-OPF LP backends.

    Each concrete backend is constructed once per (case, slack_penalty) and
    reused across RL steps via the module-level ``_backend_cache``.
    """

    @abstractmethod
    def solve(self, node_net_load_mw: np.ndarray,
              commitment: Optional[np.ndarray]) -> Dict[str, Any]:
        """Solve the ED-OPF for one time step.

        Parameters
        ----------
        node_net_load_mw : ndarray (n_nodes,)
            Net load at each bus (load − renewables − storage net).
        commitment : ndarray (n_units,) or None
            Unit on/off status (0/1); ``None`` means all units are committed.

        Returns
        -------
        dict with keys: unit_power_mw, line_flow_mw, node_net_injection_mw,
        total_cost, slack_violation, load_shedding_mw, curtailment_mw,
        status, success, lmp, solver_backend, lmp_method, lmp_quality,
        lmp_available.
        """


# ---------------------------------------------------------------------------
# Gurobi backend
# ---------------------------------------------------------------------------

class _GurobiEDSolver(_EDBackend):
    """Gurobi LP backend for economic dispatch.

    Build once, solve multiple times by updating variable bounds.

    For multi-backend support (auto-selected), use the module-level function
    ``solve_ed_opf_detailed(solver_type='auto')`` instead.

    Attributes
    ----------
    case : power system case
    model : gurobipy.Model
    unit_power_mw_var : MVar (n_units,)
    node_net_load_var : MVar (n_nodes,)  — fixed per solve
    line_flow_var : MVar (n_lines,)
    line_slack_pos_var, line_slack_neg_var : MVar (n_lines,)
    sys_ls_var, sys_cur_var : scalar Var — system balance slack
    """

    # ====== Initialization ======

    def __init__(self, case, slack_penalty: float = 1e6, verbose: bool = False):
        """Build Gurobi model.

        Parameters
        ----------
        case : power system case
        slack_penalty : float
            Penalty for line and balance constraint slack in the objective.
        verbose : bool
            Show Gurobi solver output.

        Raises
        ------
        ImportError
            If gurobipy is not installed.
        """
        if not HAS_GUROBI:
            raise ImportError(
                "gurobipy is required for the Gurobi solver. "
                "Install via: pip install powerzoo[gurobi]"
            )

        self.case = case
        self.slack_penalty = slack_penalty
        self.verbose = verbose

        self.units = case.units
        self.nodes = case.nodes
        self.lines = case.lines
        self.n_units = len(self.units)
        self.n_nodes = len(self.nodes)
        self.n_lines = len(self.lines)

        self.PTDF = case.get_node_gsdf().values           # (n_lines, n_nodes)
        self.nodes_units_map = case.get_nodes_units_map() # (n_nodes, n_units)

        self.p_min = self.units['p_min'].values
        self.p_max = self.units['p_max'].values
        self.mc_a  = self.units['mc_a'].values
        self.mc_b  = self.units['mc_b'].values
        self.mc_c  = self.units['mc_c'].values

        _warn_nonlinear_cost('_GurobiEDSolver', self.mc_a, self.mc_b)

        self.line_floor = self.lines['floor'].values.copy()
        self.line_cap   = self.lines['cap'].values.copy()
        self.line_floor[self.line_floor == 0] = -_NO_LIMIT
        self.line_cap[self.line_cap == 0]     = _NO_LIMIT

        self.model = None
        self.unit_power_mw_var       = None
        self.node_net_load_var       = None
        self.node_net_injection_var  = None
        self.line_flow_var           = None
        self.line_slack_pos_var      = None
        self.line_slack_neg_var      = None
        self.system_balance_constr   = None
        self.line_flow_constr        = None
        self.node_net_inj_constr     = None
        self.sys_ls_var              = None
        self.sys_cur_var             = None

        self._build_model()

    def _build_model(self):
        """Construct the Gurobi LP model (called once at initialisation)."""
        self.model = gp.Model("ED_OPF")
        self.model.setParam('OutputFlag', int(self.verbose))
        self.model.setParam('LogToConsole', int(self.verbose))
        self.model.setParam('FeasibilityTol', 1e-6)
        self.model.setParam('OptimalityTol', 1e-6)

        # Decision variables
        self.unit_power_mw_var = self.model.addMVar(
            (self.n_units,), lb=self.p_min, ub=self.p_max, name='UnitPower')
        self.node_net_load_var = self.model.addMVar(
            (self.n_nodes,), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='NodeNetLoad')
        self.node_net_injection_var = self.model.addMVar(
            (self.n_nodes,), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='NodeNetInjection')
        self.line_flow_var = self.model.addMVar(
            (self.n_lines,), lb=self.line_floor, ub=self.line_cap, name='LineFlow')
        self.line_slack_pos_var = self.model.addMVar(
            (self.n_lines,), lb=0, ub=GRB.INFINITY, name='LineSlackPos')
        self.line_slack_neg_var = self.model.addMVar(
            (self.n_lines,), lb=0, ub=GRB.INFINITY, name='LineSlackNeg')
        # System balance slack: ls > 0 means load-shedding; cur > 0 means curtailment.
        self.sys_ls_var  = self.model.addVar(lb=0, ub=GRB.INFINITY, name='SysLS')
        self.sys_cur_var = self.model.addVar(lb=0, ub=GRB.INFINITY, name='SysCur')

        # Constraints
        self.system_balance_constr = self.model.addConstr(
            self.unit_power_mw_var.sum() + self.sys_ls_var - self.sys_cur_var
            == self.node_net_load_var.sum(),
            name='SystemBalance')
        self.node_net_inj_constr = self.model.addConstr(
            self.nodes_units_map @ self.unit_power_mw_var - self.node_net_load_var
            == self.node_net_injection_var,
            name='NodeNetInjection')
        self.line_flow_constr = self.model.addConstr(
            self.PTDF @ self.node_net_injection_var
            + self.line_slack_pos_var - self.line_slack_neg_var
            == self.line_flow_var,
            name='LineFlow')

        # Objective: generation cost + slack penalty
        gen_cost      = self.mc_c @ self.unit_power_mw_var
        slack_cost    = self.slack_penalty * (self.line_slack_pos_var.sum()
                                              + self.line_slack_neg_var.sum())
        balance_slack = self.slack_penalty * (self.sys_ls_var + self.sys_cur_var)
        self.model.setObjective(gen_cost + slack_cost + balance_slack, GRB.MINIMIZE)
        self.model.update()

    # ====== Solver Interface ======

    def solve(self, node_net_load_mw: np.ndarray,
              commitment: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Solve by updating variable bounds.

        Parameters
        ----------
        node_net_load_mw : ndarray (n_nodes,)
        commitment : ndarray (n_units,) or None
        """
        self.node_net_load_var.lb = node_net_load_mw
        self.node_net_load_var.ub = node_net_load_mw

        if commitment is not None:
            commitment = np.asarray(commitment)
            self.unit_power_mw_var.lb = self.p_min * commitment
            self.unit_power_mw_var.ub = self.p_max * commitment
        else:
            self.unit_power_mw_var.lb = self.p_min
            self.unit_power_mw_var.ub = self.p_max

        self.model.optimize()

        if self.model.status != GRB.OPTIMAL:
            logger.warning("OPF solver status = %s", self.model.status)
            if self.model.status == GRB.INFEASIBLE:
                logger.warning("Model is infeasible. Computing IIS...")
                self.model.computeIIS()
                import tempfile, os
                iis_path = os.path.join(tempfile.gettempdir(), "powerzoo_infeasible.ilp")
                self.model.write(iis_path)
                logger.info("IIS written to %s", iis_path)

        if self.model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            unit_power_mw        = self.unit_power_mw_var.X
            line_flow_mw         = self.line_flow_var.X
            node_net_injection_mw = self.node_net_injection_var.X
            total_cost = _eval_generation_cost(
                self.mc_a, self.mc_b, self.mc_c, unit_power_mw)
            slack_violation = (self.line_slack_pos_var.X.sum()
                               + self.line_slack_neg_var.X.sum()
                               + self.sys_ls_var.X + self.sys_cur_var.X)
            lmp = self.node_net_inj_constr.pi + self.system_balance_constr.pi
            return {
                'unit_power_mw': unit_power_mw,
                'line_flow_mw': line_flow_mw,
                'node_net_injection_mw': node_net_injection_mw,
                'total_cost': total_cost,
                'slack_violation': slack_violation,
                'load_shedding_mw': float(self.sys_ls_var.X),
                'curtailment_mw': float(self.sys_cur_var.X),
                'status': self.model.status,
                'success': True,
                'lmp': lmp,
                'solver_backend': 'gurobi',
                'lmp_method': 'nodal_dual',
                'lmp_quality': 'nodal',
                'lmp_available': True,
            }
        else:
            p_min_eff = self.p_min * commitment if commitment is not None else self.p_min.copy()
            return _infeasible_result(
                self.n_lines, self.n_nodes, p_min_eff,
                self.model.status, 'gurobi')

    def rebuild(self):
        """Rebuild Gurobi model (use when the system structure changes)."""
        if self.model is not None:
            self.model.dispose()
        self._build_model()


# ---------------------------------------------------------------------------
# SciPy / HiGHS backend
# ---------------------------------------------------------------------------

class _ScipyEDSolver(_EDBackend):
    """SciPy/HiGHS LP backend for economic dispatch.

    LP variables: ``x = [p (n_u), s_pos (n_l), s_neg (n_l), ls (1), cur (1)]``

    Constant matrices ``A_ub``, ``A_eq``, ``M_u`` are built once at
    construction and reused across RL steps; only ``b_ub``, ``b_eq``, and
    variable bounds change per ``solve()`` call.
    """

    def __init__(self, case, slack_penalty: float = 1e6, verbose: bool = False):
        if not getattr(case, 'init_flag', False):
            case.init()

        n_u = len(case.units)
        n_l = len(case.lines)
        n_n = len(case.nodes)
        self.n_u, self.n_l, self.n_n = n_u, n_l, n_n
        self.slack_penalty = slack_penalty
        self.verbose = verbose

        self.p_min = case.units['p_min'].values.copy()
        self.p_max = case.units['p_max'].values.copy()
        self.mc_a  = case.units['mc_a'].values.copy()
        self.mc_b  = case.units['mc_b'].values.copy()
        self.mc_c  = case.units['mc_c'].values.copy()

        _warn_nonlinear_cost('_ScipyEDSolver', self.mc_a, self.mc_b)

        line_floor = case.lines['floor'].values.copy()
        line_cap   = case.lines['cap'].values.copy()
        line_floor[line_floor == 0] = -_NO_LIMIT
        line_cap[line_cap == 0]     = _NO_LIMIT
        self.line_floor = line_floor
        self.line_cap   = line_cap

        PTDF      = case.get_node_gsdf().values    # (n_l, n_n)
        A_u       = case.get_nodes_units_map()     # (n_n, n_u)
        self.PTDF = PTDF
        self.A_u  = A_u
        self.M_u  = PTDF @ A_u                    # (n_l, n_u) — constant

        # Constant LP structure matrices
        I_l = np.eye(n_l)
        z1  = np.zeros((n_l, 1))
        # A_ub: (2*n_l, n_u + 2*n_l + 2) — ls/cur do not appear in line constraints
        self.A_ub = np.block([
            [ self.M_u,  I_l, -I_l, z1, z1],
            [-self.M_u, -I_l,  I_l, z1, z1],
        ])
        # A_eq: (1, n_u + 2*n_l + 2) — sum(p) + ls - cur = sum(load)
        self.A_eq = np.concatenate([
            np.ones(n_u), np.zeros(2 * n_l), [1.0, -1.0]
        ]).reshape(1, -1)

    def solve(self, node_net_load_mw: np.ndarray,
              commitment: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Solve via scipy.optimize.linprog (HiGHS backend).

        LP formulation
        --------------
        variables:  x = [p (n_u),  s_pos (n_l),  s_neg (n_l),  ls (1),  cur (1)]

          * p     — generator dispatch (MW)
          * s_pos — positive line-flow slack
          * s_neg — negative line-flow slack
          * ls    — load-shedding slack (ensures feasibility when capacity is short)
          * cur   — curtailment slack (ensures feasibility when generation > demand)

        objective:  min  mc_c @ p  +  penalty · (sum(s_pos) + sum(s_neg) + ls + cur)

        equality:   sum(p) + ls - cur  =  sum(node_net_load_mw)
        inequality: M_u @ p + s_pos - s_neg  <=  cap + c0
                   -M_u @ p - s_pos + s_neg  <= -floor - c0
        bounds:     p_min <= p <= p_max,  0 <= s_pos, s_neg, ls, cur

        where c0 = PTDF @ node_net_load_mw.
        """
        from scipy.optimize import linprog

        n_u, n_l, n_n = self.n_u, self.n_l, self.n_n

        p_min = self.p_min.copy()
        p_max = self.p_max.copy()
        if commitment is not None:
            commitment = np.asarray(commitment, dtype=float)
            p_min = p_min * commitment
            p_max = p_max * commitment

        c0  = self.PTDF @ node_net_load_mw   # (n_l,) per-step line-flow offset

        c = np.concatenate([
            self.mc_c,
            np.full(n_l, self.slack_penalty),
            np.full(n_l, self.slack_penalty),
            [self.slack_penalty, self.slack_penalty],
        ])
        b_ub = np.concatenate([
            self.line_cap + c0, -self.line_floor - c0])
        b_eq = np.array([node_net_load_mw.sum()])
        bounds = list(zip(p_min, p_max)) + [(0, None)] * (2 * n_l + 2)

        result = linprog(c, A_ub=self.A_ub, b_ub=b_ub,
                         A_eq=self.A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs',
                         options={'disp': self.verbose})

        if result.status == 0:
            x             = result.x
            unit_power_mw = x[:n_u]
            s_pos         = x[n_u:n_u + n_l]
            s_neg         = x[n_u + n_l:n_u + 2 * n_l]
            ls            = x[n_u + 2 * n_l]
            cur           = x[n_u + 2 * n_l + 1]

            node_net_injection_mw = self.A_u @ unit_power_mw - node_net_load_mw
            line_flow_mw          = self.M_u @ unit_power_mw - c0 + s_pos - s_neg
            total_cost = _eval_generation_cost(
                self.mc_a, self.mc_b, self.mc_c, unit_power_mw)

            # Nodal LMP: λ_sys + PTDF^T @ (μ_upper - μ_lower)
            lmp_scalar = (result.eqlin.marginals[0]
                          if hasattr(result, 'eqlin') and result.eqlin is not None
                          else 0.0)
            if hasattr(result, 'ineqlin') and result.ineqlin is not None:
                mu       = result.ineqlin.marginals  # (2*n_l,)
                lmp      = lmp_scalar + self.PTDF.T @ (mu[:n_l] - mu[n_l:])
            else:
                lmp = np.full(n_n, lmp_scalar)

            return {
                'unit_power_mw': unit_power_mw,
                'line_flow_mw': line_flow_mw,
                'node_net_injection_mw': node_net_injection_mw,
                'total_cost': total_cost,
                'slack_violation': float(s_pos.sum() + s_neg.sum() + ls + cur),
                'load_shedding_mw': float(ls),
                'curtailment_mw': float(cur),
                'status': 'optimal',
                'success': True,
                'lmp': lmp,
                'solver_backend': 'scipy',
                'lmp_method': 'nodal_dual_reconstruction',
                'lmp_quality': 'nodal',
                'lmp_available': True,
            }
        else:
            return _infeasible_result(
                n_l, n_n, p_min.copy(),
                f'scipy_status_{result.status}', 'scipy')


# ---------------------------------------------------------------------------
# CVXPY backend
# ---------------------------------------------------------------------------

class _CVXPYEDSolver(_EDBackend):
    """CVXPY LP backend for economic dispatch.

    The LP is canonicalised once at construction; subsequent solves update
    ``cp.Parameter`` values only, avoiding repeated canonicalisation overhead
    which is significant for RL-frequency ``step()`` calls.

    Tries solvers in order: GLPK → ECOS → SCS.

    Variables: ``p (n_u), s_pos (n_l), s_neg (n_l), ls (scalar), cur (scalar)``
    Parameters: ``c0_param (n_l), net_load_sum_param, p_lower_param (n_u), p_upper_param (n_u)``
    """

    def __init__(self, case, slack_penalty: float = 1e6, verbose: bool = False):
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError(
                "cvxpy is required for the CVXPY solver. "
                "Install via: pip install powerzoo[cvxpy]"
            )

        if not getattr(case, 'init_flag', False):
            case.init()

        n_u = len(case.units)
        n_l = len(case.lines)
        n_n = len(case.nodes)
        self.n_u, self.n_l, self.n_n = n_u, n_l, n_n
        self.slack_penalty = slack_penalty
        self.verbose = verbose

        self.mc_a        = case.units['mc_a'].values.copy()
        self.mc_b        = case.units['mc_b'].values.copy()
        self.mc_c        = case.units['mc_c'].values.copy()
        self.p_min_base  = case.units['p_min'].values.copy()
        self.p_max_base  = case.units['p_max'].values.copy()

        _warn_nonlinear_cost('_CVXPYEDSolver', self.mc_a, self.mc_b)

        line_floor = case.lines['floor'].values.copy()
        line_cap   = case.lines['cap'].values.copy()
        line_floor[line_floor == 0] = -_NO_LIMIT
        line_cap[line_cap == 0]     = _NO_LIMIT

        PTDF      = case.get_node_gsdf().values
        A_u       = case.get_nodes_units_map()
        M_u       = PTDF @ A_u
        self.PTDF = PTDF
        self.A_u  = A_u
        self.M_u  = M_u

        # Decision variables
        p     = cp.Variable(n_u, name='p')
        s_pos = cp.Variable(n_l, name='s_pos', nonneg=True)
        s_neg = cp.Variable(n_l, name='s_neg', nonneg=True)
        ls    = cp.Variable(name='ls',  nonneg=True)
        cur   = cp.Variable(name='cur', nonneg=True)
        self.p, self.s_pos, self.s_neg, self.ls, self.cur = p, s_pos, s_neg, ls, cur

        # Parameters updated each solve
        self.c0_param           = cp.Parameter(n_l)
        self.net_load_sum_param = cp.Parameter()
        self.p_lower_param      = cp.Parameter(n_u)
        self.p_upper_param      = cp.Parameter(n_u)

        flow_expr = M_u @ p + s_pos - s_neg

        # Store constraint references for dual extraction
        self.balance_constr = cp.sum(p) + ls - cur == self.net_load_sum_param
        # Two separate ≤ inequalities give unambiguous dual signs.
        self.upper_constr   = flow_expr <=  self.c0_param + line_cap
        self.lower_constr   = -flow_expr <= -(self.c0_param + line_floor)
        self.p_lower_constr = p >= self.p_lower_param
        self.p_upper_constr = p <= self.p_upper_param

        objective = cp.Minimize(
            self.mc_c @ p
            + slack_penalty * (cp.sum(s_pos) + cp.sum(s_neg) + ls + cur)
        )
        self.prob = cp.Problem(objective, [
            self.balance_constr,
            self.upper_constr, self.lower_constr,
            self.p_lower_constr, self.p_upper_constr,
        ])

    def solve(self, node_net_load_mw: np.ndarray,
              commitment: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Update parameters and solve; return result dict."""
        n_u, n_l, n_n = self.n_u, self.n_l, self.n_n

        p_min = self.p_min_base.copy()
        p_max = self.p_max_base.copy()
        if commitment is not None:
            c = np.asarray(commitment, dtype=float)
            p_min = p_min * c
            p_max = p_max * c

        self.c0_param.value           = self.PTDF @ node_net_load_mw
        self.net_load_sum_param.value = float(node_net_load_mw.sum())
        self.p_lower_param.value      = p_min
        self.p_upper_param.value      = p_max

        import cvxpy as cp
        solved = False
        for solver in [cp.GLPK, cp.ECOS, cp.SCS]:
            try:
                self.prob.solve(solver=solver, verbose=self.verbose)
                if self.prob.status in ('optimal', 'optimal_inaccurate'):
                    solved = True
                    break
            except Exception:
                continue

        if solved and self.p.value is not None:
            unit_power_mw         = self.p.value
            node_net_injection_mw = self.A_u @ unit_power_mw - node_net_load_mw
            line_flow_mw = (self.M_u @ unit_power_mw
                            - self.c0_param.value
                            + self.s_pos.value - self.s_neg.value)
            sys_slack  = float(self.ls.value + self.cur.value)
            total_cost = _eval_generation_cost(
                self.mc_a, self.mc_b, self.mc_c, unit_power_mw)

            # Nodal LMP: λ_sys + PTDF^T @ (μ_upper - μ_lower)
            lmp          = np.zeros(n_n)
            lmp_available = False
            try:
                lam_sys  = float(self.balance_constr.dual_value)
                mu_upper = np.asarray(self.upper_constr.dual_value, dtype=float)
                mu_lower = np.asarray(self.lower_constr.dual_value, dtype=float)
                if mu_upper.shape == (n_l,) and mu_lower.shape == (n_l,):
                    lmp           = lam_sys + self.PTDF.T @ (mu_upper - mu_lower)
                    lmp_available = True
            except Exception:
                pass

            return {
                'unit_power_mw': unit_power_mw,
                'line_flow_mw': line_flow_mw,
                'node_net_injection_mw': node_net_injection_mw,
                'total_cost': total_cost,
                'slack_violation': (float(self.s_pos.value.sum()
                                          + self.s_neg.value.sum()) + sys_slack),
                'load_shedding_mw': float(self.ls.value),
                'curtailment_mw': float(self.cur.value),
                'status': self.prob.status,
                'success': True,
                'lmp': lmp,
                'solver_backend': 'cvxpy',
                'lmp_method': 'nodal_dual' if lmp_available else 'uniform_scalar',
                'lmp_quality': 'nodal' if lmp_available else 'system',
                'lmp_available': lmp_available,
            }
        else:
            return _infeasible_result(
                n_l, n_n, p_min.copy(),
                self.prob.status if self.prob.status else 'failed',
                'cvxpy')


# ---------------------------------------------------------------------------
# Unified backend cache and registry
# ---------------------------------------------------------------------------

# Stores (case_ref, backend) tuples keyed by (id(case), backend_cls, slack_penalty).
# Holding case_ref prevents the case object from being garbage-collected so that
# id(case) cannot be recycled by the GC.
_backend_cache: Dict[tuple, tuple] = {}


def _get_or_create_backend(case, backend_cls, force_rebuild: bool = False,
                            **kwargs) -> _EDBackend:
    """Return a cached backend instance, creating or replacing it when needed.

    Parameters
    ----------
    case : power system case
    backend_cls : one of _GurobiEDSolver, _ScipyEDSolver, _CVXPYEDSolver
    force_rebuild : bool
        If True, always create a new instance (useful after structural changes).
    **kwargs : passed verbatim to ``backend_cls.__init__``.
    """
    key = (id(case), backend_cls, kwargs.get('slack_penalty'))
    entry = _backend_cache.get(key)
    if entry is None or entry[0] is not case or force_rebuild:
        backend = backend_cls(case, **kwargs)
        _backend_cache[key] = (case, backend)
    return _backend_cache[key][1]


# Maps solver_type strings to backend classes for ``solve_ed_opf_detailed``.
_BACKEND_REGISTRY: Dict[str, type] = {
    'gurobi': _GurobiEDSolver,
    'scipy':  _ScipyEDSolver,
    'cvxpy':  _CVXPYEDSolver,
}

# Backward-compatible alias — external code that imported EDOptimizer directly
# continues to work.
EDOptimizer = _GurobiEDSolver


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_ed_opf(case, node_net_load_mw: np.ndarray,
                 commitment: Optional[np.ndarray] = None,
                 slack_penalty: float = 1e6, rebuild: bool = False,
                 verbose: bool = False) -> np.ndarray:
    """Solve single-period economic dispatch OPF (Gurobi backend, convenience wrapper).

    Caches the solver instance for reuse across RL steps.

    Parameters
    ----------
    case : power system case
    node_net_load_mw : ndarray (n_nodes,)
    commitment : ndarray (n_units,), optional
    slack_penalty : float
    rebuild : bool
        Force model rebuild (e.g. after topology change).
    verbose : bool

    Returns
    -------
    unit_power_mw : ndarray (n_units,)
    """
    backend = _get_or_create_backend(
        case, _GurobiEDSolver, force_rebuild=rebuild,
        slack_penalty=slack_penalty, verbose=verbose)
    return backend.solve(node_net_load_mw, commitment)['unit_power_mw']


def solve_ed_opf_detailed(case, node_net_load_mw: np.ndarray,
                           commitment: Optional[np.ndarray] = None,
                           slack_penalty: float = 1e6, rebuild: bool = False,
                           verbose: bool = False,
                           solver_type: str = 'auto') -> Dict[str, Any]:
    """Solve ED-OPF and return detailed results.

    Supports multiple solvers:
        'gurobi'  — Gurobi LP (requires gurobipy + license, fastest)
        'scipy'   — scipy.optimize.linprog (HiGHS backend, free, fast)
        'cvxpy'   — CVXPY with auto-selected open-source backend (GLPK/SCS/ECOS)
        'auto'    — Use Gurobi if available, fall back to scipy

    Parameters
    ----------
    case : power system case
    node_net_load_mw : ndarray (n_nodes,)
    commitment : ndarray (n_units,), optional
    slack_penalty : float
        Penalty for line/balance constraint slack in the objective.
        **RL note**: ``total_cost`` equals generation cost only; slack penalty
        is *not* included.  Use ``slack_violation`` and ``total_cost`` as
        separate signals rather than summing them with the 1e6 weight.
    rebuild : bool
        Force backend rebuild (Gurobi only; no-op for others).
    verbose : bool
    solver_type : str

    Returns
    -------
    dict with unit_power_mw, line_flow_mw, total_cost, slack_violation, lmp, etc.
    ``total_cost`` = true cubic cost (mc_a/3)·P³+(mc_b/2)·P²+mc_c·P.
    On failure: ``success=False``, ``total_cost=_INFEASIBLE_COST`` (1e9).
    """
    if solver_type == 'auto':
        solver_type = 'gurobi' if HAS_GUROBI else 'scipy'

    if solver_type not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown solver_type '{solver_type}'. "
            f"Choose from: {list(_BACKEND_REGISTRY)}"
        )

    backend_cls = _BACKEND_REGISTRY[solver_type]
    backend = _get_or_create_backend(
        case, backend_cls, force_rebuild=rebuild,
        slack_penalty=slack_penalty, verbose=verbose)
    return backend.solve(node_net_load_mw, commitment)


# ---------------------------------------------------------------------------
# Piecewise-linear cost / offer-curve support
# ---------------------------------------------------------------------------

def make_cost_segments(case, n_segments: int = 5) -> Dict[str, np.ndarray]:
    """Build piecewise-linear offer segments from quadratic cost data.

    For each generator *i* with marginal cost
    ``MC(P) = mc_a·P² + mc_b·P + mc_c``, the operating range
    ``[p_min, p_max]`` is divided into *n_segments* equal-width blocks.
    The price of each block is the marginal cost evaluated at the block
    midpoint, guaranteeing monotonically non-decreasing prices (convex
    cost assumption).

    Returns
    -------
    dict with:
        seg_widths : ndarray (n_units, n_segments)  — MW width per block
        seg_prices : ndarray (n_units, n_segments)  — $/MWh price per block
    """
    if not getattr(case, 'init_flag', False):
        case.init()

    mc_a  = case.units['mc_a'].values
    mc_b  = case.units['mc_b'].values
    mc_c  = case.units['mc_c'].values
    p_min = case.units['p_min'].values
    p_max = case.units['p_max'].values

    n_u = len(case.units)
    K   = n_segments

    seg_widths = np.zeros((n_u, K))
    seg_prices = np.zeros((n_u, K))

    for i in range(n_u):
        width = (p_max[i] - p_min[i]) / K
        seg_widths[i, :] = width
        for k in range(K):
            p_mid = p_min[i] + (k + 0.5) * width
            seg_prices[i, k] = mc_a[i] * p_mid ** 2 + mc_b[i] * p_mid + mc_c[i]
        # Enforce monotonicity (holds for convex cost, but guard against numerics)
        for k in range(1, K):
            if seg_prices[i, k] < seg_prices[i, k - 1]:
                seg_prices[i, k] = seg_prices[i, k - 1] + 0.01

    return {'seg_widths': seg_widths, 'seg_prices': seg_prices}


def solve_piecewise_ed_opf(
    case,
    node_net_load_mw: np.ndarray,
    offer_segments: Dict[str, np.ndarray],
    commitment: Optional[np.ndarray] = None,
    slack_penalty: float = 1e6,
    verbose: bool = False,
    p_min_rt: Optional[np.ndarray] = None,
    p_max_rt: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Network-constrained SCED with piecewise-linear offer curves.

    Uses scipy/HiGHS LP.  Each generator's output range ``[p_min, p_max]``
    is decomposed into *K* price–quantity segments.  The LP naturally fills
    cheaper segments first (convex cost / increasing prices).

    Parameters
    ----------
    case : ClearCase
        Power system case.
    node_net_load_mw : ndarray (n_nodes,)
        Net load at each bus.
    offer_segments : dict
        ``seg_widths`` (n_units, K) and ``seg_prices`` (n_units, K).
    commitment : ndarray (n_units,), optional
        Unit commitment (0/1).
    slack_penalty : float
        Penalty for line/balance constraint slack.
        **RL note**: ``total_cost`` is the true cost (mc_a/3·P³+mc_b/2·P²+mc_c·P)
        independent of this penalty; ``offer_cost`` uses offer prices.
    verbose : bool
        Solver verbosity.

    Returns
    -------
    Same dictionary format as ``solve_ed_opf_detailed``, plus:
        ``offer_cost``       — cost at offer prices.
        ``offer_segments``   — the segments passed to this solve.
        ``cost_model``       — ``'piecewise'``.
        ``segment_dispatch`` — ndarray (n_units, K) incremental dispatch per segment.
    """
    from scipy.optimize import linprog

    if not getattr(case, 'init_flag', False):
        case.init()

    seg_widths = offer_segments['seg_widths']   # (n_u, K)
    seg_prices = offer_segments['seg_prices']   # (n_u, K)

    n_u, K = seg_widths.shape
    n_l    = len(case.lines)
    n_n    = len(case.nodes)
    n_seg  = n_u * K

    p_min = case.units['p_min'].values.copy()
    p_max = case.units['p_max'].values.copy()
    mc_a  = case.units['mc_a'].values.copy()
    mc_b  = case.units['mc_b'].values.copy()
    mc_c  = case.units['mc_c'].values.copy()

    line_floor = case.lines['floor'].values.copy()
    line_cap   = case.lines['cap'].values.copy()
    line_floor[line_floor == 0] = -_NO_LIMIT
    line_cap[line_cap == 0]     = _NO_LIMIT

    PTDF = case.get_node_gsdf().values   # (n_l, n_n)
    A_u  = case.get_nodes_units_map()    # (n_n, n_u)
    M_u  = PTDF @ A_u                   # (n_l, n_u)
    c0   = PTDF @ node_net_load_mw      # (n_l,)

    if commitment is not None:
        commitment = np.asarray(commitment, dtype=float)
        p_min      = p_min * commitment
        p_max      = p_max * commitment
        seg_widths = seg_widths * commitment[:, None]

    # Apply runtime ramp bounds (enforce intertemporal ramp coupling).
    # p_min_rt / p_max_rt must lie within the static [p_min, p_max] range;
    # they are clipped here to ensure LP feasibility.
    if p_min_rt is not None:
        p_min = np.clip(np.asarray(p_min_rt, dtype=float), p_min, p_max)
    if p_max_rt is not None:
        p_max_rt_clipped = np.clip(np.asarray(p_max_rt, dtype=float), p_min, p_max)
        new_range  = np.maximum(p_max_rt_clipped - p_min, 0.0)
        orig_range = seg_widths.sum(axis=1) + 1e-8
        seg_widths = seg_widths * (new_range / orig_range)[:, None]

    # LP variables: x = [delta (n_seg), s_pos (n_l), s_neg (n_l), ls (1), cur (1)]
    #   delta[i*K + k] — output increment of generator i in segment k
    #   p[i] = p_min[i] + sum_k delta[i*K + k]

    M_S = np.repeat(M_u, K, axis=1)    # (n_l, n_seg)

    c_obj = np.concatenate([
        seg_prices.ravel(),
        np.full(n_l, slack_penalty),
        np.full(n_l, slack_penalty),
        [slack_penalty, slack_penalty],
    ])

    I_l     = np.eye(n_l)
    n_total = n_seg + 2 * n_l + 2
    A_ub    = np.zeros((2 * n_l, n_total))
    A_ub[:n_l,  :n_seg]                    =  M_S
    A_ub[:n_l,  n_seg:n_seg + n_l]         =  I_l
    A_ub[:n_l,  n_seg + n_l:n_seg + 2*n_l] = -I_l
    A_ub[n_l:,  :n_seg]                    = -M_S
    A_ub[n_l:,  n_seg:n_seg + n_l]         = -I_l
    A_ub[n_l:,  n_seg + n_l:n_seg + 2*n_l] =  I_l

    flow_from_pmin = M_u @ p_min
    b_ub = np.concatenate([
        line_cap  + c0 - flow_from_pmin,
        -line_floor - c0 + flow_from_pmin,
    ])

    A_eq = np.zeros((1, n_total))
    A_eq[0, :n_seg]               = 1.0
    A_eq[0, n_seg + 2 * n_l]      = 1.0   # ls
    A_eq[0, n_seg + 2 * n_l + 1]  = -1.0  # cur
    b_eq = np.array([node_net_load_mw.sum() - p_min.sum()])

    bounds = ([(0.0, float(w)) for w in seg_widths.ravel()]
              + [(0, None)] * (2 * n_l + 2))

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs',
                     options={'disp': verbose})

    if result.status == 0:
        x     = result.x
        delta = x[:n_seg].reshape(n_u, K)
        s_pos = x[n_seg:n_seg + n_l]
        s_neg = x[n_seg + n_l:n_seg + 2 * n_l]
        ls    = x[n_seg + 2 * n_l]
        cur   = x[n_seg + 2 * n_l + 1]

        unit_power_mw         = p_min + delta.sum(axis=1)
        node_net_injection_mw = A_u @ unit_power_mw - node_net_load_mw
        line_flow_mw          = M_u @ unit_power_mw - c0 + s_pos - s_neg
        total_cost  = _eval_generation_cost(mc_a, mc_b, mc_c, unit_power_mw)
        offer_cost  = float((seg_prices * delta).sum())

        # LMP: λ_sys + PTDF^T @ (μ_upper - μ_lower)
        lmp_scalar = (result.eqlin.marginals[0]
                      if hasattr(result, 'eqlin') and result.eqlin is not None
                      else 0.0)
        if hasattr(result, 'ineqlin') and result.ineqlin is not None:
            mu   = result.ineqlin.marginals
            lmp  = lmp_scalar + PTDF.T @ (mu[:n_l] - mu[n_l:])
        else:
            lmp = np.full(n_n, lmp_scalar)

        return {
            'unit_power_mw': unit_power_mw,
            'line_flow_mw': line_flow_mw,
            'node_net_injection_mw': node_net_injection_mw,
            'total_cost': total_cost,
            'offer_cost': offer_cost,
            'slack_violation': float(s_pos.sum() + s_neg.sum() + ls + cur),
            'load_shedding_mw': float(ls),
            'curtailment_mw': float(cur),
            'status': 'optimal',
            'success': True,
            'lmp': lmp,
            'solver_backend': 'scipy_piecewise',
            'lmp_method': 'nodal_dual_reconstruction',
            'lmp_quality': 'nodal',
            'lmp_available': True,
            'cost_model': 'piecewise',
            'offer_segments': offer_segments,
            'segment_dispatch': delta,
        }
    else:
        return _infeasible_result(
            n_l, n_n, p_min.copy(),
            f'scipy_status_{result.status}', 'scipy_piecewise',
            offer_cost=_INFEASIBLE_COST,
            cost_model='piecewise',
            offer_segments=offer_segments,
            segment_dispatch=None,
        )


if __name__ == "__main__":
    # Quick test
    from powerzoo.case import load_case

    case = load_case(5)

    # Test with uniform load
    node_net_load_mw = np.array([100.0, 150.0, 200.0, 180.0, 120.0])

    print("=" * 80)
    print("Testing ED-OPF Solver")
    print("=" * 80)

    result = solve_ed_opf_detailed(case, node_net_load_mw, verbose=True)

    print(f"\nOptimization Status: {result['status']}")
    print(f"Total Cost: ${result['total_cost']:.2f}")
    print(f"Slack Violation: {result['slack_violation']:.6f}")

    print(f"\nUnit Power Output:")
    for i, p in enumerate(result['unit_power_mw']):
        print(
            f"  Unit {i + 1}: {p:.2f} MW (p_min={case.units.iloc[i]['p_min']:.1f}, p_max={case.units.iloc[i]['p_max']:.1f}, mc={case.units.iloc[i]['mc_c']:.1f})")

    print(f"\nLine Flow:")
    for i, flow in enumerate(result['line_flow_mw']):
        floor = case.lines.iloc[i]['floor']
        cap   = case.lines.iloc[i]['cap']
        print(f"  Line {i + 1}: {flow:.2f} MW (floor={floor:.1f}, cap={cap:.1f})")

    print(f"\nSystem Balance Check:")
    print(f"  Total Generation: {result['unit_power_mw'].sum():.2f} MW")
    print(f"  Total Net Load: {node_net_load_mw.sum():.2f} MW")
    print(f"  Difference: {abs(result['unit_power_mw'].sum() - node_net_load_mw.sum()):.6f} MW")
