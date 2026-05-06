"""Power flow mixin for DistGridEnv.

Contains physics methods: cal_pf, safety_check, get_total_loss,
and the aligned-resource helper that feeds cal_pf.

These are separated from the RL interface (dist.py) so that the
physics core can be read and tested independently.
"""
from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

from powerzoo.envs.grid.cal_pf_dist import (
    run_bfs_power_flow, calculate_line_losses
)


class _DistPFMixin:
    """Physics mixin: power flow, safety checks, loss calculation."""

    def _get_aligned_resource_power(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return resource P/Q vectors aligned with ``nodes_resources_map`` columns."""
        if self.nodes_resources_map is None or len(self.sub_resources) == 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

        n_resources = int(self.nodes_resources_map.shape[1])
        col_index = getattr(self, '_resource_col_index', {})
        has_valid_col_index = (
            len(col_index) == n_resources
            and set(col_index.keys()) == set(self.sub_resources.keys())
            and set(col_index.values()) == set(range(n_resources))
        )
        if not has_valid_col_index:
            raise RuntimeError(
                "nodes_resources_map column order is undefined because "
                "_resource_col_index is out of sync with sub_resources. "
                "Register resources via register_resource()/unregister_resource() "
                "instead of mutating these internals directly."
            )

        resource_p = np.zeros(n_resources, dtype=float)
        resource_q = np.zeros(n_resources, dtype=float)
        for rid, res in self.sub_resources.items():
            col_idx = int(col_index[rid])
            resource_p[col_idx] = float(np.atleast_1d(res.current_p_mw)[0])
            resource_q[col_idx] = float(np.atleast_1d(getattr(res, 'current_q_mvar', 0.0))[0])
        return resource_p, resource_q

    def cal_pf(self, p_load: np.ndarray = None, q_load: np.ndarray = None,
               df: bool = False) -> Tuple[Any, Any]:
        """Run Backward/Forward Sweep (BFS) power flow.

        Args:
            p_load: Gross active load at each node (MW) before resource
                subtraction. If None, uses the current case/time-series load.
            q_load: Gross reactive load at each node (MVAr), same semantics.
                If None, inferred from p_load via per-node power factor.
            df: If True, return DataFrames with detailed results.

        Returns:
            If df=False: (v_mag, p_flow_MW)
                v_mag:        (n_nodes,) voltage magnitudes in p.u.
                p_flow_MW:    (n_lines,) sending-end active power flows in MW
            If df=True: (nodes_df, lines_df)
                nodes_df columns: v_mag (p.u.), p_load_MW, q_load_MVAr
                lines_df columns: p_flow_MW, q_flow_MVAr, p_loss_MW, q_loss_MVAr

        Notes:
            Slack-bus exchange and convergence status are stored on the
            environment and exposed via ``state`` / ``info``.
        """
        p_load_mw = self._get_node_loads_p() if p_load is None else p_load.copy()
        q_load_mvar = self._get_node_loads_q() if q_load is None else q_load.copy()

        if self.nodes_resources_map is not None and len(self.sub_resources) > 0:
            resource_p, resource_q = self._get_aligned_resource_power()
            p_load_mw -= self.nodes_resources_map @ resource_p
            q_load_mvar -= self.nodes_resources_map @ resource_q

        p_load_pu = p_load_mw / self.baseMVA
        q_load_pu = q_load_mvar / self.baseMVA

        result = run_bfs_power_flow(
            topo=self.topo,
            p_load_pu=p_load_pu,
            q_load_pu=q_load_pu,
            v_slack=self.v_slack,
            slack_bus_id=self.slack_bus_id,
            max_iter=self.max_iter,
            tol=self.tol,
            v_sq_init=self._last_v_sq,
        )

        self._converged = result['converged']
        if self._converged and 'v_sq' in result:
            self._last_v_sq = result['v_sq'].copy()
        self._iterations = result['iterations']
        self._is_diverged = bool(result.get('is_diverged', False))
        self._voltage_collapse = bool(result.get('voltage_collapse', False))
        self._p_slack_mw = float(result.get('p_slack', 0.0)) * self.baseMVA
        self._q_slack_mvar = float(result.get('q_slack', 0.0)) * self.baseMVA

        if not result['converged']:
            reasons = []
            if self._voltage_collapse:
                reasons.append("severe low-voltage collapse")
            if self._is_diverged:
                reasons.append("iteration tolerance not reached")
            failure_reason = f" ({'; '.join(reasons)})" if reasons else ""
            warnings.warn(
                f"BFS power flow did not produce a valid operating point after "
                f"{self._iterations} iterations{failure_reason}",
                RuntimeWarning, stacklevel=2,
            )

        p_loss_pu, q_loss_pu = calculate_line_losses(
            self.topo, result['p_branch'], result['q_branch'], result['v_sq']
        )

        if df:
            nodes = self.case.nodes.copy()
            nodes['v_mag'] = result['v_mag']
            nodes['p_load_MW'] = p_load_mw
            nodes['q_load_MVAr'] = q_load_mvar

            lines = self.case.lines.loc[self.topo.active_line_indices].copy()
            lines['p_flow_MW'] = result['p_branch'] * self.baseMVA
            lines['q_flow_MVAr'] = result['q_branch'] * self.baseMVA
            lines['p_loss_MW'] = p_loss_pu * self.baseMVA
            lines['q_loss_MVAr'] = q_loss_pu * self.baseMVA
            return nodes, lines

        return result['v_mag'], result['p_branch'] * self.baseMVA

    def safety_check(self, nodes_result: Any = None, lines_result: Any = None,
                     v_min: float = None, v_max: float = None,
                     with_info: bool = False) -> Tuple[bool, Optional[Dict]]:
        """Check voltage and line capacity limits.

        Handles power-flow divergence explicitly: when the solver did not
        converge, a low-voltage collapse was detected, or the voltage vector
        contains NaN/Inf, the system is unconditionally unsafe and all nodes
        are counted as violating.

        Args:
            nodes_result: Node results DataFrame from ``cal_pf(df=True)``.
                Must be a DataFrame; passing a raw ndarray raises TypeError.
            lines_result: Line results DataFrame from ``cal_pf(df=True)``.
                Must be a DataFrame; passing a raw ndarray raises TypeError.
            v_min: Minimum voltage limit (p.u.)
            v_max: Maximum voltage limit (p.u.)
            with_info: If True, return detailed info dict.

        Returns:
            is_safe: True if all constraints satisfied.
            info: Violation details (if with_info=True), else None.

        Raises:
            TypeError: If nodes_result or lines_result is not a DataFrame.
        """
        if v_min is None:
            v_min = self.v_min
        if v_max is None:
            v_max = self.v_max

        if nodes_result is None or lines_result is None:
            nodes_result, lines_result = self.cal_pf(df=True)

        if not isinstance(nodes_result, pd.DataFrame):
            raise TypeError(
                f"nodes_result must be a pandas DataFrame (from cal_pf(df=True)), "
                f"got {type(nodes_result).__name__}."
            )
        if not isinstance(lines_result, pd.DataFrame):
            raise TypeError(
                f"lines_result must be a pandas DataFrame (from cal_pf(df=True)), "
                f"got {type(lines_result).__name__}.  Passing a raw ndarray silently "
                f"skips all thermal limit checks."
            )

        converged = getattr(self, '_converged', True)
        is_diverged = bool(getattr(self, '_is_diverged', False))
        voltage_collapse = bool(getattr(self, '_voltage_collapse', False))
        v_mag = nodes_result['v_mag'].values if isinstance(nodes_result, pd.DataFrame) else nodes_result
        has_bad_values = np.any(~np.isfinite(v_mag))

        if not converged or has_bad_values or voltage_collapse:
            all_nodes = list(range(len(v_mag)))
            all_lines = list(range(self.n_lines))
            if with_info:
                return False, {
                    'v_min_actual': float(np.nanmin(v_mag)) if not has_bad_values else float('nan'),
                    'v_max_actual': float(np.nanmax(v_mag)) if not has_bad_values else float('nan'),
                    'v_violation_nodes': all_nodes,
                    'line_violation_ids': all_lines,
                    'converged': converged,
                    'is_diverged': is_diverged,
                    'voltage_collapse': voltage_collapse,
                    'iterations': getattr(self, '_iterations', 0),
                }
            return False, None

        # Normal path: finite voltages, solver converged
        v_violation = (v_mag < v_min) | (v_mag > v_max)
        v_safe = ~np.any(v_violation)

        lines = self.case.lines
        if 'cap' in lines.columns and isinstance(lines_result, pd.DataFrame) \
                and 'p_flow_MW' in lines_result.columns:
            p_flow = lines_result['p_flow_MW'].values
            q_flow = (lines_result['q_flow_MVAr'].values
                      if 'q_flow_MVAr' in lines_result.columns
                      else np.zeros_like(p_flow))
            s_flow = np.sqrt(p_flow ** 2 + q_flow ** 2)
            line_cap = lines.loc[self.active_line_indices, 'cap'].values
            valid_cap = line_cap > 0
            line_violation = np.zeros(len(lines_result), dtype=bool)
            line_violation[valid_cap] = s_flow[valid_cap] > line_cap[valid_cap]
            line_safe = ~np.any(line_violation)
        else:
            line_violation = np.zeros(self.n_lines, dtype=bool)
            line_safe = True

        is_safe = v_safe and line_safe

        if with_info:
            return is_safe, {
                'v_min_actual': float(np.min(v_mag)),
                'v_max_actual': float(np.max(v_mag)),
                'v_violation_nodes': np.where(v_violation)[0].tolist(),
                'line_violation_ids': np.where(line_violation)[0].tolist(),
                'converged': converged,
                'is_diverged': is_diverged,
                'voltage_collapse': voltage_collapse,
                'iterations': getattr(self, '_iterations', 0),
            }
        return is_safe, None

    def get_total_loss(self, lines_result: Any = None) -> Tuple[float, float]:
        """Calculate total network losses.

        On solver divergence, returns a finite sentinel equal to baseMVA (MW)
        rather than 0 (reward hacking) or Inf (gradient explosion).
        Normal distribution-network losses are well below baseMVA.

        Args:
            lines_result: Line results DataFrame from ``cal_pf(df=True)``.
                Must be a DataFrame; passing a raw ndarray raises TypeError.
                When None, runs a fresh power flow internally.

        Returns:
            p_loss_total: Total active power loss in MW (≥ 0).
            q_loss_total: Total reactive power loss in MVAr (≥ 0).

        Raises:
            TypeError: If lines_result is not a DataFrame.
        """
        if getattr(self, '_pf_failed', False):
            sentinel = float(self.baseMVA)
            return sentinel, sentinel

        if lines_result is None:
            _, lines_result = self.cal_pf(df=True)

        if not isinstance(lines_result, pd.DataFrame):
            raise TypeError(
                f"lines_result must be a pandas DataFrame (from cal_pf(df=True)), "
                f"got {type(lines_result).__name__}."
            )

        p = float(np.nansum(lines_result['p_loss_MW'].values))
        q = float(np.nansum(lines_result['q_loss_MVAr'].values))
        if not np.isfinite(p) or not np.isfinite(q):
            sentinel = float(self.baseMVA)
            return sentinel, sentinel
        return max(p, 0.0), max(q, 0.0)
