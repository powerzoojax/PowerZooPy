"""Load data mixin for DistGridEnv.

Contains node-load accessors: static case loads, time-series loads,
and reactive-power scaling.

These are separated from the RL interface (dist.py) so that load-data
logic can be read and tested independently.
"""
import numpy as np


class _DistLoadsMixin:
    """Load data mixin: node load vectors for each time step."""

    def _get_loads_map(self) -> np.ndarray:
        """Cached node-to-loads incidence matrix."""
        if self._nodes_loads_map is None:
            self._nodes_loads_map = self.case.get_nodes_loads_map()
        return self._nodes_loads_map

    def _get_node_loads(self, col: str) -> np.ndarray:
        """Static case nodal loads for a given column (e.g. 'Pd' or 'Qd')."""
        self._ensure_case_init()
        if hasattr(self.case, 'loads') and col in self.case.loads.columns:
            return self._get_loads_map().dot(self.case.loads[col].values)
        elif col in self.case.nodes.columns:
            return self.case.nodes[col].values.copy()
        return np.zeros(self.n_nodes)

    def _get_case_node_loads_p(self) -> np.ndarray:
        """Per-node static active load (MW) from case data."""
        return self._get_node_loads('Pd')

    def _get_case_node_loads_q(self) -> np.ndarray:
        """Per-node static reactive load (MVAr) from case data."""
        return self._get_node_loads('Qd')

    def _get_node_loads_p(self) -> np.ndarray:
        """Active power loads at each node (MW) for the current time step.

        Uses the precomputed time-varying load matrix when available.
        Falls back to static case values when no time series is present.
        """
        if self._node_loads_p is not None:
            idx = self._get_current_time_index()
            if 0 <= idx < len(self._node_loads_p):
                return self._get_loads_map().dot(self._node_loads_p[idx])
        if self._cached_case_p is None:
            self._cached_case_p = self._get_node_loads('Pd')
        return self._cached_case_p

    def _get_node_loads_q(self) -> np.ndarray:
        """Reactive power loads at each node (MVAr) for the current time step.

        Priority order:
        1. Explicit reactive time series (``self._node_loads_q``), e.g. from
           a ``load.reactive_mvar`` signal.
        2. Per-node constant-power-factor scaling from the active time series:
               Q_i(t) = Q_base_i × (P_i(t) / P_base_i)
           Buses with P_base_i = 0 get Q_i = 0.
        3. Static case Q values when no active time series is available.

        Note: P_i(t) is gross demand before resource injections are subtracted;
        deductions happen later in cal_pf() via nodes_resources_map.
        """
        if self._node_loads_q is not None:
            idx = self._get_current_time_index()
            if 0 <= idx < len(self._node_loads_q):
                current_q = np.asarray(self._node_loads_q[idx], dtype=float)
                if current_q.shape[0] == self.n_nodes:
                    return current_q.copy()
                return self._get_loads_map().dot(current_q)

        if self._cached_case_p is None:
            self._cached_case_p = self._get_node_loads('Pd')
        if self._cached_case_q is None:
            self._cached_case_q = self._get_node_loads('Qd')
        base_p = self._cached_case_p
        base_q = self._cached_case_q

        if self._node_loads_p is not None:
            idx = self._get_current_time_index()
            if 0 <= idx < len(self._node_loads_p):
                current_p = self._get_loads_map().dot(self._node_loads_p[idx])
                ratio = np.divide(
                    current_p,
                    base_p,
                    out=np.zeros_like(current_p, dtype=float),
                    where=base_p > 0,
                )
                return base_q * ratio
        return base_q
