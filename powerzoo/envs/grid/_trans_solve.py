"""Transmission grid solver paths and physical helper functions.

Extracted from TransGridEnv to keep trans.py focused on the RL/benchmark
interface layer.  Functions here take ``env`` (a TransGridEnv instance) as
their first argument.

Controlled side effects
-----------------------
* ``cal_pf`` writes ``env._power_imbalance_mw`` and
  ``env._slack_gen_violation_mw`` as side effects — required for
  backward-compatible direct calls from tests and market environments.
* ``run_acpf`` writes ``env._power_imbalance_mw`` *before* the NR call so
  that the value is available for preservation when NR fails (see
  ``TransGridEnv._run_power_flow_acpf``).

All other functions are free of env mutations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

from powerzoo.envs.grid.cal_dcopf_trans import solve_ed_opf_detailed
from powerzoo.envs.grid.cal_pf_trans import run_acpf as _nr_acpf

if TYPE_CHECKING:
    from powerzoo.envs.grid.trans import TransGridEnv

logger = logging.getLogger(__name__)


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class TransSolveResult:
    """Immutable snapshot of one solver-path execution.

    Passed back to ``TransGridEnv._apply_solve_result`` which writes every
    field to the corresponding ``self._*`` attribute atomically.
    """
    converged: bool
    unit_power_mw: Optional[np.ndarray]
    lines: Optional[pd.DataFrame]
    nodes: Optional[pd.DataFrame]
    is_safe: bool
    safety_info: dict[str, Any]
    opf_result: Optional[dict[str, Any]] = None
    pf_result: Optional[dict[str, Any]] = None
    power_imbalance_mw: float = 0.0
    slack_gen_violation_mw: float = 0.0


# ── Physical helper functions ────────────────────────────────────────────────

def get_default_node_load(env: TransGridEnv) -> np.ndarray:
    """Return per-load-point gross load for the current time step (n_loads,)."""
    return env._get_node_loads_p_current()


def get_slack_unit_mask(env: TransGridEnv) -> np.ndarray:
    """Boolean mask selecting generators connected to the slack bus."""
    unit_at_node = env.case.get_nodes_units_map()
    return unit_at_node[env.slack_bus_id, :].astype(bool)


def calculate_node_net_load(
    env: TransGridEnv,
    node_load_mw: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Net load at each node = gross load minus all registered DER injections.

    In AC modes all registered DER are assumed to operate at unity power
    factor (Q = 0), so only active-power net load is adjusted here.

    Args:
        node_load_mw: Optional gross load override.  Accepts either per-node
            arrays ``(n_nodes,)`` or per-load arrays ``(n_loads,)``; per-load
            arrays are mapped to nodes via the case load-incidence matrix.

    Returns:
        Net active-power load per node, shape ``(n_nodes,)``.
    """
    if node_load_mw is None:
        raw = get_default_node_load(env)
        if hasattr(env.case, 'loads') and len(env.case.loads) > 0:
            node_load_mw = (env.case.get_nodes_loads_map() @ raw).astype(float)
        else:
            node_load_mw = np.asarray(raw, dtype=float).copy()
    else:
        node_load_mw = np.asarray(node_load_mw, dtype=float).copy()
        n_nodes = len(env.case.nodes)
        if node_load_mw.shape[0] != n_nodes:
            n_loads = len(env.case.loads) if hasattr(env.case, 'loads') else n_nodes
            if (node_load_mw.shape[0] == n_loads
                    and hasattr(env.case, 'loads')
                    and len(env.case.loads) > 0):
                node_load_mw = env.case.get_nodes_loads_map() @ node_load_mw
            else:
                raise ValueError(
                    "node_load_mw must have shape (n_nodes,) or (n_loads,), "
                    f"got {node_load_mw.shape}"
                )

    if env.nodes_resources_map is not None and len(env.sub_resources) > 0:
        resource_power = np.array([
            float(np.atleast_1d(env.sub_resources[rid].current_p_mw)[0])
            for rid in env.sub_resources
        ])
        node_load_mw -= env.nodes_resources_map @ resource_power

    return node_load_mw


def compute_slack_gen_violation(
    env: TransGridEnv,
    slack_gen_mw: float,
    slack_unit_mask: Optional[np.ndarray] = None,
) -> float:
    """Slack-generator bound exceedance for an aggregate MW output.

    Args:
        slack_gen_mw: Aggregate active-power output of all units at the
            slack bus (MW).
        slack_unit_mask: Pre-computed boolean mask.  When supplied the
            internal ``get_nodes_units_map`` call is skipped.

    Returns:
        Bound exceedance in MW (>= 0).
    """
    if slack_unit_mask is None:
        slack_unit_mask = get_slack_unit_mask(env)
    if not slack_unit_mask.any():
        return 0.0
    slack_p_min = float(env.case.units['p_min'].values[slack_unit_mask].sum())
    slack_p_max = float(env.case.units['p_max'].values[slack_unit_mask].sum())
    return float(
        max(0.0, slack_gen_mw - slack_p_max)
        + max(0.0, slack_p_min - slack_gen_mw)
    )


def proportional_dispatch(env: TransGridEnv, total_load_mw: float) -> np.ndarray:
    """Distribute ``total_load_mw`` pro-rata by each unit's headroom (p_max - p_min)."""
    p_min = env.case.units['p_min'].values.astype(float)
    p_max = env.case.units['p_max'].values.astype(float)
    margin = p_max - p_min
    total_margin = margin.sum()
    if total_margin > 0:
        remaining = max(0.0, total_load_mw - p_min.sum())
        return p_min + margin * min(remaining / total_margin, 1.0)
    return p_min.copy()


def cal_pf(
    env: TransGridEnv,
    unit_power_mw,
    node_load_mw,
    df: bool = False,
):
    """DC power flow via PTDF matrix.

    Power imbalance is absorbed by the slack bus generator (not by load
    adjustment).  This preserves the physical meaning of ``node_load_mw``
    and allows bounds-checking the slack unit's effective output.

    Side effects
    ------------
    Updates ``env._power_imbalance_mw`` and ``env._slack_gen_violation_mw``.
    Callers that use this function as part of a solver path should capture
    those values into ``TransSolveResult`` afterwards.

    Returns
    -------
    ``(line_flow_mw, node_inj_mw)`` or ``(lines_df, nodes_df)`` when
    ``df=True``.
    """
    unit_at_node = env.case.get_nodes_units_map()
    node_power_mw = unit_at_node.dot(unit_power_mw)
    if env.nodes_resources_map is not None:
        resource_power = np.array([
            env.sub_resources[rid].current_p_mw for rid in env.sub_resources
        ])
        node_power_mw += env.nodes_resources_map.dot(resource_power)

    node_load_mw = node_load_mw.copy()
    total_gen = np.sum(node_power_mw)
    total_load = np.sum(node_load_mw)

    slack_imbalance = total_gen - total_load
    node_power_mw = node_power_mw.copy()
    node_power_mw[env.slack_bus_id] -= slack_imbalance

    env._power_imbalance_mw = abs(float(slack_imbalance))

    slack_unit_mask = get_slack_unit_mask(env)
    if slack_unit_mask.any():
        slack_gen = float(
            np.asarray(unit_power_mw, dtype=float)[slack_unit_mask].sum()
            - slack_imbalance
        )
        env._slack_gen_violation_mw = compute_slack_gen_violation(
            env, slack_gen, slack_unit_mask
        )
    else:
        env._slack_gen_violation_mw = 0.0

    node_inj_mw = node_power_mw - node_load_mw
    line_flow_mw = env.PTDF.dot(node_inj_mw)

    if df:
        lines = env.case.lines.copy()
        lines["line_flow_mw"] = line_flow_mw
        nodes = env.case.nodes.copy()
        nodes['node_inj_mw'] = node_inj_mw
        return lines, nodes
    return line_flow_mw, node_inj_mw


def ac_thermal_check(
    pf_from: np.ndarray,
    qf_from: np.ndarray,
    line_cap: np.ndarray,
    pf_to: Optional[np.ndarray] = None,
    qf_to: Optional[np.ndarray] = None,
    use_both_ends: bool = True,
):
    """AC thermal safety check on apparent power (MVA).

    Matches the JAX reference ``powerzoojax.envs.grid.power_flow.ac_thermal_check``.

    ``line_cap`` is the MVA thermal rating (MATPOWER ``rateA``).  Lines with
    ``line_cap == 0`` are treated as uncapped (unlimited).

    ``thermal_flow`` is ``max(|Sf|, |St|)`` when ``use_both_ends`` is True
    (ACPF, both ends solved), or ``|Sf|`` when False (ACOPF, from-end only).

    Args:
        pf_from: From-end active branch flow [MW].
        qf_from: From-end reactive branch flow [MVAr].
        line_cap: Thermal rating per line [MVA].
        pf_to: To-end active flow [MW] (ignored when ``use_both_ends`` is False).
        qf_to: To-end reactive flow [MVAr] (ignored when ``use_both_ends`` is False).
        use_both_ends: If True, thermal = max(|Sf|, |St|); else thermal = |Sf|.

    Returns:
        is_safe: True if no line exceeds its thermal cap.
        n_violations: Count of violated lines.
        cost_thermal: Sum of positive MVA overloads [MVA].
    """
    sf = np.sqrt(pf_from ** 2 + qf_from ** 2)
    if use_both_ends and pf_to is not None and qf_to is not None:
        st = np.sqrt(pf_to ** 2 + qf_to ** 2)
        thermal = np.maximum(sf, st)
    else:
        thermal = sf
    # Treat cap == 0 as uncapped (unlimited branch)
    effective_cap = np.where(line_cap > 0, line_cap, np.inf)
    over = np.maximum(thermal - effective_cap, 0.0)
    n_violations = int(np.sum(over > 0))
    cost_thermal = float(np.sum(over))
    is_safe = n_violations == 0
    return is_safe, n_violations, cost_thermal


def safety_check(env: TransGridEnv, line_flow_mw, with_info: bool = False):
    """Check whether line flows are within [floor, cap] bounds.

    Args:
        line_flow_mw: Either a ``pd.DataFrame`` with a ``'line_flow_mw'``
            column (returned by ``cal_pf`` with ``df=True``) or a plain
            ndarray of flows.
        with_info: When ``True`` also return a dict of unsafe-line details.

    Returns:
        ``(line_flow_safe, info)``  where ``info`` is ``None`` when
        ``with_info=False``.
    """
    if isinstance(line_flow_mw, pd.DataFrame):
        lines = line_flow_mw
    else:
        lines = env.case.lines.copy()
        lines["line_flow_mw"] = line_flow_mw

    line_flow_safe = (
        (lines["line_flow_mw"] <= lines["cap"])
        & (lines["line_flow_mw"] >= lines["floor"])
    )
    if with_info:
        unsafe = lines[~line_flow_safe]
        info = {
            "unsafe_line_ids":    unsafe["#id"].astype(int).tolist(),
            "unsafe_line_flows":  unsafe["line_flow_mw"].tolist(),
            "unsafe_line_caps":   unsafe["cap"].tolist(),
            "unsafe_line_floors": unsafe["floor"].tolist(),
        }
    else:
        info = None
    return line_flow_safe, info


# ── Four solver paths ────────────────────────────────────────────────────────

def run_dcopf(env: TransGridEnv, action: dict) -> TransSolveResult:
    """DC-OPF solver path.

    Net-load OPF assumption: the LP is solved against net load
    (gross_load - DER_injections).  Dispatchable DER are treated as
    fixed injections for this step.

    When ``unit_power_mw`` is present in *action* the OPF is bypassed and
    line flows are computed via DC power flow instead.

    Structure
    ---------
    1. reset step-local state
    2. prepare inputs (net load, commitment)
    3. solve (OPF or bypass)
    4. build lines / nodes / unit_power_mw
    5. finalize safety / violation diagnostics
    """
    # 1. reset step-local state
    power_imbalance_mw = 0.0
    slack_gen_violation_mw = 0.0

    # 2. prepare inputs
    node_net_load_mw = calculate_node_net_load(env)
    commitment = action.get('commitment')

    # 3. solve
    if 'unit_power_mw' in action:
        unit_power_mw = action['unit_power_mw']
        opf_result = None
    else:
        opf_result = solve_ed_opf_detailed(
            env.case,
            node_net_load_mw,
            commitment=commitment,
            verbose=False,
            solver_type=env.solver_type,
        )
        unit_power_mw = opf_result['unit_power_mw']

    # 4. build lines / nodes
    if opf_result is not None:
        lines = env.case.lines.copy()
        lines['line_flow_mw'] = opf_result['line_flow_mw']

        nodes = env.case.nodes.copy()
        nodes['node_inj_mw'] = opf_result['node_net_injection_mw']
        nodes['node_net_load_mw'] = node_net_load_mw
    else:
        node_load_mw = action.get('node_load_mw', get_default_node_load(env))
        lines, nodes = cal_pf(env, unit_power_mw, node_load_mw, df=True)
        power_imbalance_mw = env._power_imbalance_mw
        slack_gen_violation_mw = env._slack_gen_violation_mw

    # 5. finalize safety / violation
    line_flow_safe, safety_info = safety_check(env, lines, with_info=True)
    is_safe = bool(line_flow_safe.all())

    if opf_result is not None:
        slack_viol = opf_result.get('slack_violation', 0.0)
        if slack_viol > 1e-3:
            is_safe = False
            if safety_info is not None:
                safety_info['slack_violation'] = float(slack_viol)

    if opf_result is None and slack_gen_violation_mw > 1e-3:
        is_safe = False
        if safety_info is not None:
            safety_info['slack_gen_violation_mw'] = slack_gen_violation_mw

    return TransSolveResult(
        converged=True,
        unit_power_mw=unit_power_mw,
        lines=lines,
        nodes=nodes,
        is_safe=is_safe,
        safety_info=safety_info,
        opf_result=opf_result,
        pf_result=None,
        power_imbalance_mw=power_imbalance_mw,
        slack_gen_violation_mw=slack_gen_violation_mw,
    )


def run_acopf(env: TransGridEnv, action: dict) -> TransSolveResult:
    """AC-OPF solver path (NLP solve only; bypass is handled by the thin wrapper).

    Net-load OPF assumption: identical to ``run_dcopf``.

    Note: the ``unit_power_mw`` bypass (which delegates to ``run_acpf``) is
    handled inside ``TransGridEnv._run_power_flow_acopf`` so that instance-level
    monkeypatching of ``_run_power_flow_acpf`` in tests is preserved.  This
    function only covers the full AC-OPF (NLP) path.

    Structure
    ---------
    1. lazy-init ACOPF solver
    2. prepare inputs (net load, commitment)
    3. solve (NLP)
    4. build lines / nodes / unit_power_mw
    5. finalize safety / violation diagnostics
    """
    # 1. lazy-init AC-OPF solver
    if env._acopf_solver is None:
        if env._acopf_solver_type == 'pandapower':
            from powerzoo.envs.grid.cal_acopf_trans_pandapower import ACOPFSolver
            env._acopf_solver = ACOPFSolver(
                env.case,
                v_min=env._ac_v_min,
                v_max=env._ac_v_max,
                q_factor=env._ac_q_factor,
            )
        else:
            from powerzoo.envs.grid.cal_acopf_trans import ACOPFSolverBuiltin
            env._acopf_solver = ACOPFSolverBuiltin(
                env.case,
                v_min=env._ac_v_min,
                v_max=env._ac_v_max,
                q_factor=env._ac_q_factor,
                backend=env._ac_backend,
            )

    # 2. prepare inputs
    node_net_load_mw = calculate_node_net_load(env)
    commitment = action.get('commitment')

    # 3. solve
    opf_result = env._acopf_solver.solve(node_net_load_mw, commitment)
    unit_power_mw = opf_result['unit_power_mw']

    # 4. build lines / nodes
    lines = env.case.lines.copy()
    lines['line_flow_mw'] = opf_result['line_flow_mw']
    lines['line_flow_q_mvar'] = opf_result['line_flow_q_mvar']  # from-end reactive [MVAr]

    nodes = env.case.nodes.copy()
    nodes['node_inj_mw'] = opf_result['node_net_injection_mw']
    nodes['node_net_load_mw'] = node_net_load_mw
    if 'vm_pu' in opf_result:
        nodes['vm_pu'] = opf_result['vm_pu']
        nodes['va_deg'] = opf_result['va_deg']

    # 5. finalize safety / violation
    # AC thermal check: from-end |Sf| vs MVA cap (use_both_ends=False, ACOPF only returns from-end)
    cap = env.case.lines['cap'].values
    pf_from = opf_result['line_flow_mw']
    qf_from = opf_result['line_flow_q_mvar']
    is_thermal_safe, n_thermal_viol, _ = ac_thermal_check(
        pf_from, qf_from, cap, use_both_ends=False,
    )
    is_safe = is_thermal_safe

    # Build safety_info with MVA-based line violation details
    if not is_thermal_safe:
        sf = np.sqrt(pf_from ** 2 + qf_from ** 2)
        effective_cap = np.where(cap > 0, cap, np.inf)
        over = np.maximum(sf - effective_cap, 0.0)
        unsafe_mask = over > 0
        line_ids = (
            env.case.lines['#id'].values if '#id' in env.case.lines.columns
            else np.arange(len(cap))
        )
        floor_vals = (
            env.case.lines['floor'].values[unsafe_mask].tolist()
            if 'floor' in env.case.lines.columns else []
        )
        safety_info = {
            'unsafe_line_ids':    line_ids[unsafe_mask].astype(int).tolist(),
            'unsafe_line_flows':  sf[unsafe_mask].tolist(),        # MVA
            'unsafe_line_caps':   cap[unsafe_mask].tolist(),       # MVA
            'unsafe_line_floors': floor_vals,
            'line_viol_mva':      float(opf_result.get('line_viol_mva', 0.0)),
        }
    else:
        safety_info = {
            'unsafe_line_ids': [], 'unsafe_line_flows': [],
            'unsafe_line_caps': [], 'unsafe_line_floors': [],
            'line_viol_mva': float(opf_result.get('line_viol_mva', 0.0)),
        }

    slack_viol = opf_result.get('slack_violation', 0.0)
    if slack_viol > 1e-3:
        is_safe = False
        safety_info['slack_violation'] = float(slack_viol)

    return TransSolveResult(
        converged=True,
        unit_power_mw=unit_power_mw,
        lines=lines,
        nodes=nodes,
        is_safe=is_safe,
        safety_info=safety_info,
        opf_result=opf_result,
        pf_result=None,
        power_imbalance_mw=0.0,
        slack_gen_violation_mw=0.0,
    )


def run_dcpf(env: TransGridEnv, action: dict) -> TransSolveResult:
    """DCPF solver path — PTDF-based DC power flow (no optimisation).

    The agent must supply ``action['unit_power_mw']``; if omitted a simple
    proportional dispatch is used as fallback.

    Structure
    ---------
    1. reset step-local state
    2. prepare inputs (net load, dispatch or fallback)
    3. solve (cal_pf)
    4. build lines / nodes / unit_power_mw  [done inside cal_pf]
    5. finalize safety / violation diagnostics
    """
    # 1. reset step-local state
    power_imbalance_mw = 0.0
    slack_gen_violation_mw = 0.0

    # 2. prepare inputs
    node_net_load_mw = calculate_node_net_load(env)

    if 'unit_power_mw' in action:
        unit_power_mw = np.asarray(action['unit_power_mw'], dtype=float)
    else:
        unit_power_mw = proportional_dispatch(env, float(node_net_load_mw.sum()))

    # 3. solve (PTDF-based DC PF)
    node_load_mw = action.get('node_load_mw', get_default_node_load(env))
    lines, nodes = cal_pf(env, unit_power_mw, node_load_mw, df=True)

    # 4. capture side-effects written by cal_pf
    power_imbalance_mw = env._power_imbalance_mw
    slack_gen_violation_mw = env._slack_gen_violation_mw

    # 5. finalize safety / violation
    line_flow_safe, safety_info = safety_check(env, lines, with_info=True)
    is_safe = bool(line_flow_safe.all())

    if slack_gen_violation_mw > 1e-3:
        is_safe = False
        if safety_info is not None:
            safety_info['slack_gen_violation_mw'] = slack_gen_violation_mw

    return TransSolveResult(
        converged=True,
        unit_power_mw=unit_power_mw,
        lines=lines,
        nodes=nodes,
        is_safe=is_safe,
        safety_info=safety_info,
        opf_result=None,
        pf_result=None,
        power_imbalance_mw=power_imbalance_mw,
        slack_gen_violation_mw=slack_gen_violation_mw,
    )


def run_acpf(env: TransGridEnv, action: dict) -> TransSolveResult:
    """ACPF solver path — Newton-Raphson AC power flow (no optimisation).

    The agent must supply ``action['unit_power_mw']``; if omitted a simple
    proportional dispatch is used as fallback.

    Side effect
    -----------
    Writes ``env._power_imbalance_mw`` *before* calling NR so that the value
    is available for preservation in the thin wrapper's exception handler if
    the NR solve fails.

    Structure
    ---------
    1. reset step-local state
    2. prepare inputs (net load, dispatch or fallback)
    3. pre-NR: compute power_imbalance_mw and write to env (see above)
    4. solve (Newton-Raphson via _nr_acpf)
    5. build lines / nodes / unit_power_mw
    6. finalize safety / voltage / slack-violation diagnostics
    """
    # 1. reset step-local state
    slack_gen_violation_mw = 0.0

    # 2. prepare inputs
    node_net_load_mw = calculate_node_net_load(env, action.get('node_load_mw'))

    if 'unit_power_mw' in action:
        unit_power_mw = np.asarray(action['unit_power_mw'], dtype=float)
    else:
        unit_power_mw = proportional_dispatch(env, float(node_net_load_mw.sum()))

    # 3. pre-NR: compute imbalance now so it survives a NR failure
    power_imbalance_mw = abs(
        float(unit_power_mw.sum()) - float(node_net_load_mw.sum())
    )
    env._power_imbalance_mw = power_imbalance_mw  # preserved on NR failure

    # 4. solve
    pf_result = _nr_acpf(
        env.case, Pd_mw=node_net_load_mw, Pg_mw=unit_power_mw, verbose=False
    )

    # 5. build lines / nodes
    n_lines = len(env.case.lines)
    pf_from = pf_result['pf_from']
    qf_from = pf_result.get('qf_from', np.zeros(n_lines))  # defensive: some mocks omit Q
    pf_to   = pf_result.get('pf_to',   np.zeros(n_lines))
    qf_to   = pf_result.get('qf_to',   np.zeros(n_lines))

    lines = env.case.lines.copy()
    lines['line_flow_mw'] = pf_from
    lines['line_flow_q_mvar'] = qf_from  # from-end reactive flow [MVAr]

    unit_at_node = env.case.get_nodes_units_map()
    node_power_mw = unit_at_node.dot(unit_power_mw)
    nodes = env.case.nodes.copy()
    nodes['node_inj_mw'] = node_power_mw - node_net_load_mw
    nodes['node_net_load_mw'] = node_net_load_mw
    nodes['vm_pu'] = pf_result['vm']
    nodes['va_deg'] = pf_result['va_deg']

    # 6. finalize safety / voltage / slack-violation
    # AC thermal check: compare max(|Sf|, |St|) against MVA cap (use_both_ends=True)
    cap = env.case.lines['cap'].values
    is_thermal_safe, n_thermal_viol, _ = ac_thermal_check(
        pf_from, qf_from, cap, pf_to, qf_to, use_both_ends=True,
    )
    is_safe = is_thermal_safe

    # Build safety_info keyed by unsafe lines (MVA thermal magnitude)
    if not is_thermal_safe:
        sf = np.sqrt(pf_from ** 2 + qf_from ** 2)
        st = np.sqrt(pf_to ** 2 + qf_to ** 2)
        thermal = np.maximum(sf, st)
        effective_cap = np.where(cap > 0, cap, np.inf)
        over = np.maximum(thermal - effective_cap, 0.0)
        unsafe_mask = over > 0
        line_ids = (
            env.case.lines['#id'].values if '#id' in env.case.lines.columns
            else np.arange(len(cap))
        )
        floor_vals = (
            env.case.lines['floor'].values[unsafe_mask].tolist()
            if 'floor' in env.case.lines.columns else []
        )
        safety_info = {
            'unsafe_line_ids':    line_ids[unsafe_mask].astype(int).tolist(),
            'unsafe_line_flows':  thermal[unsafe_mask].tolist(),   # MVA
            'unsafe_line_caps':   cap[unsafe_mask].tolist(),       # MVA
            'unsafe_line_floors': floor_vals,
            'line_viol_mva':      float(np.max(over)),             # max single-line MVA overload
        }
    else:
        safety_info = {
            'unsafe_line_ids': [], 'unsafe_line_flows': [],
            'unsafe_line_caps': [], 'unsafe_line_floors': [],
            'line_viol_mva': 0.0,
        }

    vm = pf_result['vm']
    if bool(np.any(vm < env._ac_v_min) or np.any(vm > env._ac_v_max)):
        is_safe = False
        safety_info['voltage_violation'] = True
        safety_info['vm_min'] = float(vm.min())
        safety_info['vm_max'] = float(vm.max())

    actual_p_gen = pf_result.get('p_gen')
    if actual_p_gen is not None:
        slack_unit_mask = get_slack_unit_mask(env)
        if slack_unit_mask.any():
            actual_slack_gen = float(
                np.asarray(actual_p_gen, dtype=float)[slack_unit_mask].sum()
            )
            slack_gen_violation_mw = compute_slack_gen_violation(
                env, actual_slack_gen, slack_unit_mask
            )
            if slack_gen_violation_mw > 1e-3:
                is_safe = False
                if safety_info is not None:
                    safety_info['slack_gen_violation_mw'] = slack_gen_violation_mw

    return TransSolveResult(
        converged=bool(pf_result['converged']),
        unit_power_mw=unit_power_mw,
        lines=lines,
        nodes=nodes,
        is_safe=is_safe,
        safety_info=safety_info,
        opf_result=None,
        pf_result=pf_result,
        power_imbalance_mw=power_imbalance_mw,
        slack_gen_violation_mw=slack_gen_violation_mw,
    )
