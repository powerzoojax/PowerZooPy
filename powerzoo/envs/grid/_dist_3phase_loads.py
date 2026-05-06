"""Three-phase load assembly mixin for DistGrid3PhaseEnv.

File layout:
    dist_3phase.py          — RL interface: __init__, spaces, obs, reward, info
    _dist_3phase_physics.py — Physics integration: topology, cal_pf, safety, VUF
    _dist_3phase_loads.py   — Load data: per-phase load matrices and helpers (this file)
    cal_pf_dist_3phase.py   — Solver kernel: ThreePhaseTopology, BFS core (do not edit here)
"""
from typing import Tuple

import numpy as np


class _Dist3PhaseLoadsMixin:
    """Load data assembly mixin: per-phase load matrices and scaling."""

    def _get_case_phase_load_matrix(self, col: str) -> np.ndarray:
        """Return (n_nodes, 3) static per-phase load array in MW.

        Args:
            col: Base column name -- 'Pd' for active power, 'Qd' for reactive.
                 Per-phase columns are expected as '{col}_A', '{col}_B', '{col}_C'
                 in case.nodes (values assumed in p.u. on baseMVA base).

        Returns:
            Array of shape (n_nodes, 3) in MW; zero where phase column is absent.
            Cached after first call per column.
        """
        if col == 'Pd' and self._cached_case_phase_p is not None:
            return self._cached_case_phase_p
        if col == 'Qd' and self._cached_case_phase_q is not None:
            return self._cached_case_phase_q

        nodes = self.case.nodes
        result = np.zeros((self.n_nodes, 3))
        for j, ph in enumerate('ABC'):
            full_col = f'{col}_{ph}'
            if full_col in nodes.columns:
                result[:, j] = nodes[full_col].fillna(0).values
        result = result * self.baseMVA  # p.u. -> MW

        if col == 'Pd':
            self._cached_case_phase_p = result
        elif col == 'Qd':
            self._cached_case_phase_q = result
        return result

    def _get_case_node_loads_p(self) -> np.ndarray:
        """Per-node static total P load (MW), summed across A/B/C phases."""
        return self._get_case_phase_load_matrix('Pd').sum(axis=1)

    def _get_case_node_loads_q(self) -> np.ndarray:
        """Per-node static total Q load (MVAr), summed across A/B/C phases."""
        return self._get_case_phase_load_matrix('Qd').sum(axis=1)

    def _get_node_loads_p(self) -> np.ndarray:
        """Total active power load per node (MW) for the current time step.

        Uses the precomputed time-varying load matrix when available.
        Falls back to feeder reference values (cached) otherwise.
        """
        if self._node_loads_p is not None:
            idx = self._get_current_time_index()
            if 0 <= idx < len(self._node_loads_p):
                return self._get_loads_map().dot(self._node_loads_p[idx])
        if self._cached_case_p is None:
            self._ensure_case_init()
            loads_map = self._get_loads_map()
            if self._node_loads_p is not None and self.max_load_ratio > 0:
                ref_per_load = self._node_loads_p.max(axis=0) / self.max_load_ratio
                self._cached_case_p = loads_map.dot(ref_per_load)
            elif hasattr(self.case, 'loads') and 'd_max' in self.case.loads.columns:
                self._cached_case_p = loads_map.dot(
                    self.case.loads['d_max'].values * self.baseMVA
                )
            else:
                self._cached_case_p = self._get_case_phase_load_matrix('Pd').sum(axis=1)
        return self._cached_case_p

    def _get_node_loads_q(self) -> np.ndarray:
        """Total reactive power load per node (MVAr) for the current time step.

        For three-phase cases, reactive power is stored as Qd_A/B/C rather than
        a single Qd column. This override sums the per-phase case data to build
        the total Q reference used by _get_3phase_loads() for scaling.

        Falls back to the per-phase case sum when no time-varying reactive
        series is present.
        """
        if self._node_loads_q is not None:
            idx = self._get_current_time_index()
            if 0 <= idx < len(self._node_loads_q):
                current_q = np.asarray(self._node_loads_q[idx], dtype=float)
                if current_q.shape[0] == self.n_nodes:
                    return current_q.copy()
                return self._get_loads_map().dot(current_q)
        if self._cached_case_q is None:
            self._cached_case_q = self._get_case_phase_load_matrix('Qd').sum(axis=1)
        return self._cached_case_q

    def _get_3phase_loads(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get three-phase loads for all nodes (excluding reference bus).

        Per-phase loads scale with the time-varying total while preserving the
        static case phase ratios: P_ph(t) = P_total(t) * (P_ph_static / P_total_static).

        Returns:
            P_3ph_pu: (3*(n_nodes-1),) active power demands in p.u., ordered
                      [Pd_A1, Pd_B1, Pd_C1, Pd_A2, Pd_B2, Pd_C2, ...]
            Q_3ph_pu: (3*(n_nodes-1),) reactive power demands in p.u., same order
        """
        case_p = self._get_case_phase_load_matrix('Pd')
        case_q = self._get_case_phase_load_matrix('Qd')

        # Fall back to equal-split single-phase if no per-phase columns exist
        if case_p.sum() == 0:
            nodes = self.case.nodes
            non_ref_mask = self._non_ref_mask()
            Pd = nodes['Pd'].fillna(0).values[non_ref_mask] / 3.0 if 'Pd' in nodes.columns else \
                np.zeros(self.n_nodes - 1)
            Qd = nodes['Qd'].fillna(0).values[non_ref_mask] / 3.0 if 'Qd' in nodes.columns else \
                np.zeros(self.n_nodes - 1)
            return np.repeat(Pd, 3) / self.baseMVA, np.repeat(Qd, 3) / self.baseMVA

        # Scale phases by time-varying node total while preserving phase shares
        p_total_current = self._get_node_loads_p()         # (n_nodes,) MW
        p_total_static = case_p.sum(axis=1)                # (n_nodes,) MW static total

        # Avoid divide-by-zero for nodes with no static load (generation-only buses)
        safe_p_static = np.where(p_total_static > 0, p_total_static, 1.0)
        p_scale = np.where(p_total_static > 0,
                           p_total_current / safe_p_static,
                           0.0)                            # (n_nodes,)

        # Scale Q independently so that Q is not zeroed when net P ≈ 0
        # (e.g. DER offsets P but Q demand persists).
        q_total_current = self._get_node_loads_q()         # (n_nodes,) MVAr
        q_total_static = case_q.sum(axis=1)                # (n_nodes,) MVAr static total
        safe_q_static = np.where(q_total_static > 0, q_total_static, 1.0)
        q_scale = np.where(q_total_static > 0,
                           q_total_current / safe_q_static,
                           0.0)                            # (n_nodes,)

        p_scaled = case_p * p_scale[:, None]  # (n_nodes, 3)
        q_scaled = case_q * q_scale[:, None]  # (n_nodes, 3)

        # Exclude reference bus
        non_ref_mask = self._non_ref_mask()
        P_3ph_pu = p_scaled[non_ref_mask].flatten() / self.baseMVA
        Q_3ph_pu = q_scaled[non_ref_mask].flatten() / self.baseMVA

        return P_3ph_pu, Q_3ph_pu
