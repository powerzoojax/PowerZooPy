"""Three-phase physics integration mixin for DistGrid3PhaseEnv.

File layout:
    dist_3phase.py          — RL interface: __init__, spaces, obs, reward, info
    _dist_3phase_physics.py — Physics integration: topology, cal_pf, safety, VUF (this file)
    _dist_3phase_loads.py   — Load data: per-phase load matrices and helpers
    cal_pf_dist_3phase.py   — Solver kernel: ThreePhaseTopology, BFS core (do not edit here)
"""
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

from powerzoo.envs.grid.cal_pf_dist_3phase import (
    build_3phase_topology, run_3phase_bfs_power_flow,
    calculate_3phase_losses, get_phase_results,
)


class _Dist3PhasePhysicsMixin:
    """Physics integration mixin: topology, power flow, safety, VUF."""

    # ====== Topology ======

    def _build_topology(self) -> None:
        """Build the three-phase radial topology from case data.

        The topology layer itself is phase-agnostic: it expands one radial tree
        into ``A/B/C`` state triplets via a Kronecker product. Any physically
        missing phase must therefore already be encoded in the upstream line
        impedance blocks (for example, Case123 uses sparse/zero-padded ``3x3``
        line configurations). Physical node IDs are passed through so
        ``self.topo3ph`` can expose explicit solver-vector index mappings.
        """
        self._ensure_case_init()

        n_nodes = len(self.case.nodes)
        lines = self.case.lines

        active_lines = (lines[lines['status'] == 1].copy()
                        if 'status' in lines.columns else lines.copy())

        active_line_indices = active_lines.index.tolist()
        n_lines = len(active_lines)

        from_nodes = active_lines['#from'].values.astype(int)
        to_nodes = active_lines['#to'].values.astype(int)

        Z_3ph_pu = self._build_impedance_matrices(active_lines)

        self.topo3ph = build_3phase_topology(
            n_nodes=n_nodes,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            Z_3ph_pu=Z_3ph_pu,
            ref_bus=self.slack_bus_id,
            v_ref_mag=self.v_ref_mag,
            node_ids=self.case.nodes.index.to_numpy(),
        )

        self.n_nodes = n_nodes
        self.n_lines = n_lines
        self.active_line_indices = active_line_indices

    def _build_impedance_matrices(self, lines: pd.DataFrame) -> np.ndarray:
        """Build ``(n_lines, 3, 3)`` impedance matrices from ``line_config``.

        This function does not infer phase availability from the branch type.
        Whatever ``3x3`` block appears in ``line_config`` is passed directly to
        the three-phase BFS core.

        Model boundary:
            The current three-phase BFS path uses only the series impedance
            block. Branch metadata such as ``ratio`` / ``angle`` and shunt
            charging entries in ``line_config`` are ignored here.
        """
        if not hasattr(self.case, 'line_config'):
            raise AttributeError("Case must have 'line_config' for three-phase power flow")

        n_lines = len(lines)
        config_ids = lines['config_name'].values
        lengths = lines['length'].values

        configs = self.case.line_config.loc[config_ids]

        Z_unit = np.zeros((n_lines, 3, 3), dtype=np.complex128)

        for i in range(3):
            for j in range(3):
                Z_unit[:, i, j] = configs[f'Z{i+1}{j+1}'].values

        return (Z_unit * lengths[:, None, None]) / self.Zbase

    # ====== Topology masks (cached) ======

    def _non_ref_mask(self) -> np.ndarray:
        """Boolean mask for all load buses (True) vs. the reference bus (False).

        The reference (slack) bus is excluded from the BIBC/BCBV voltage vector
        because its voltage is the known boundary condition (v_ref_mag at 0 deg).
        Returns a boolean array of length n_nodes.  Cached after first call.
        """
        if self._cached_non_ref_mask is None:
            mask = np.ones(self.n_nodes, dtype=bool)
            mask[self.slack_bus_id] = False
            self._cached_non_ref_mask = mask
            self._cached_non_ref_mask.flags.writeable = False
        return self._cached_non_ref_mask

    def _line_phase_mask(self) -> np.ndarray:
        """Boolean mask of energized phases for each active line.

        The mask is inferred from the diagonal entries of ``topo3ph.Z_3ph``.
        A phase is considered present when its self-impedance is non-zero.
        """
        if self._cached_line_phase_mask is None:
            diag = np.abs(np.diagonal(self.topo3ph.Z_3ph, axis1=1, axis2=2))
            mask = diag > 1e-12
            self._cached_line_phase_mask = mask
            self._cached_line_phase_mask.flags.writeable = False
        return self._cached_line_phase_mask

    def _line_phase_counts(self) -> np.ndarray:
        """Return the energized phase count for each active line."""
        if self._cached_line_phase_counts is None:
            counts = self._line_phase_mask().sum(axis=1).astype(float)
            counts = np.maximum(counts, 1.0)
            self._cached_line_phase_counts = counts
            self._cached_line_phase_counts.flags.writeable = False
        return self._cached_line_phase_counts

    def _get_ref_bus_label(self):
        """Index label of the reference (slack) bus in case.nodes."""
        return self.case.nodes.index[self.slack_bus_id]

    # ====== Power Flow ======

    def cal_pf(self, p_load: np.ndarray = None, q_load: np.ndarray = None,
               df: bool = False) -> Tuple[Any, Any]:
        """Run three-phase power flow using BIBC/BCBV method.

        Args:
            p_load: Active power load (n_nodes-1, 3) or (3*(n_nodes-1),) in MW.
                    If None, uses case data.
            q_load: Reactive power load (n_nodes-1, 3) or (3*(n_nodes-1),) in MVAr.
                    If None, uses case data.
            df: If True, return DataFrames with detailed results.

        Returns:
            If df=False: (V_mag, P_branch_MW)
            If df=True: (nodes_df, lines_df)

        Convergence contract:
            When the BFS solver does not converge, this method still returns the
            last iterate so callers can inspect the failed solve, but those
            arrays/DataFrames are not a valid physical solution. RL-facing code
            must check ``self._converged`` (or ``info['pf_converged']`` from
            ``step()``) before consuming the outputs.

        Core-vector metadata:
            ``self.topo3ph`` stores the explicit mapping between physical node
            IDs and the ``3*n_lines`` solver vectors used by the core BFS path.
            The flattened order is node-major ``ABC``:
            ``[node1_A, node1_B, node1_C, node2_A, ...]``.
            Any power assigned to a phase that is absent on the bus's parent
            branch is clamped to zero inside the core solver.

        Resource injection conventions:

            Route A (default — scalar per resource):
                Each registered resource produces one scalar MW value
                (``current_p_mw[0]``) and carries a ``phase`` attribute that
                determines which phases receive the injection via ``_PHASE_ALLOC``.
                To achieve *unbalanced* per-phase control, register three
                independent single-phase resources (``phase='A'``, ``'B'``,
                ``'C'``) at the same node.

            Route B (three-phase inverter with independent phase control):
                When ``resource.phase == 'ABC'`` **and** ``current_p_mw`` has
                exactly 3 elements, they are interpreted as ``[p_A, p_B, p_C]``
                MW injections applied independently to each phase.  ``current_q_mvar``
                may likewise be a length-3 array; a scalar is broadcast to all
                three phases.  This route models a three-phase inverter with a
                joint capacity constraint but independent phase-level setpoints.
                For ``phase='ABC'`` resources, both ``current_p_mw`` and
                ``current_q_mvar`` must therefore have length 1 or 3; other
                lengths raise ``ValueError`` so action-packing bugs are not
                silently truncated.
                If an upstream policy emits a native 6D continuous action
                ``[p_A, p_B, p_C, q_A, q_B, q_C]``, a Wrapper or ``PowerEnv``
                adapter can pass it through by writing the first three entries
                to ``current_p_mw`` and the last three to ``current_q_mvar``.
        """
        P_3ph_pu, Q_3ph_pu = self._get_3phase_loads()

        if p_load is not None:
            P_3ph_pu = (p_load.flatten() if p_load.ndim > 1 else p_load) / self.baseMVA
        if q_load is not None:
            Q_3ph_pu = (q_load.flatten() if q_load.ndim > 1 else q_load) / self.baseMVA

        # Subtract resource injections from net load (resources reduce net load).
        # Each resource's ``phase`` attribute ('A','B','C','AB','ABC', etc.)
        # determines how its power is distributed across the three phases.
        # 3-phase load vector ordering: [A1,B1,C1, A2,B2,C2, ...] for non-ref nodes.
        #
        # Inject resources: Route A (scalar + phase_alloc) or Route B (3-element array).
        # See cal_pf() docstring for full convention description.
        if self.nodes_resources_map is not None and len(self.sub_resources) > 0:
            non_ref_mask = self._non_ref_mask()
            p_inj_phase = np.zeros((self.n_nodes, 3))
            q_inj_phase = np.zeros((self.n_nodes, 3))

            resource_ids = list(self.sub_resources.keys())
            for j, rid in enumerate(resource_ids):
                res = self.sub_resources[rid]
                phase_str = getattr(res, 'phase', 'ABC').upper()
                # Flatten row/column vectors from wrappers/policies so Route-B
                # length validation sees the true action dimensionality.
                p_raw = np.asarray(res.current_p_mw, dtype=float).flatten()
                q_raw = np.asarray(getattr(res, 'current_q_mvar', 0.0), dtype=float).flatten()
                node_weights = self.nodes_resources_map[:, j]  # (n_nodes,)

                if phase_str == 'ABC' and len(p_raw) not in (1, 3):
                    raise ValueError(
                        f"Resource '{rid}' with phase='ABC' must provide "
                        f"current_p_mw with length 1 or 3, got length {len(p_raw)}."
                    )
                if phase_str == 'ABC' and len(q_raw) not in (1, 3):
                    raise ValueError(
                        f"Resource '{rid}' with phase='ABC' must provide "
                        f"current_q_mvar with length 1 or 3, got length {len(q_raw)}."
                    )

                if phase_str == 'ABC' and len(p_raw) == 3:
                    # Route B: independent per-phase injection [p_A, p_B, p_C].
                    q_3 = q_raw if len(q_raw) == 3 else np.full(3, float(q_raw[0]))
                    p_inj_phase += np.outer(node_weights, p_raw)
                    q_inj_phase += np.outer(node_weights, q_3)
                else:
                    # Route A: scalar output distributed via _PHASE_ALLOC.
                    p_mw = float(p_raw[0])
                    q_mvar = float(q_raw[0])
                    alloc = self._PHASE_ALLOC.get(phase_str, self._PHASE_ALLOC['ABC'])
                    p_inj_phase += np.outer(node_weights * p_mw, alloc)
                    q_inj_phase += np.outer(node_weights * q_mvar, alloc)

            p_inj_3ph = p_inj_phase[non_ref_mask].flatten() / self.baseMVA
            q_inj_3ph = q_inj_phase[non_ref_mask].flatten() / self.baseMVA

            P_3ph_pu = P_3ph_pu - p_inj_3ph
            Q_3ph_pu = Q_3ph_pu - q_inj_3ph

        result = run_3phase_bfs_power_flow(
            topo3ph=self.topo3ph,
            P_3ph_pu=P_3ph_pu,
            Q_3ph_pu=Q_3ph_pu,
            v_ref_mag=self.v_ref_mag,
            max_iter=self.max_iter,
            tol=self.tol
        )

        self._converged = result['converged']
        self._iterations = result['iterations']

        if not result['converged']:
            warnings.warn(
                f"Three-phase BFS power flow status={result['convergence_status']}: "
                f"{result['convergence_message']} Check self._converged or "
                f"info['pf_converged'] before using the returned voltages/flows.",
                RuntimeWarning, stacklevel=2,
            )

        if df:
            P_loss_pu, Q_loss_pu = calculate_3phase_losses(self.topo3ph, result)
            nodes_df = self._build_nodes_df(result, P_3ph_pu, Q_3ph_pu)
            lines_df = self._build_lines_df(result, P_loss_pu, Q_loss_pu)
            return nodes_df, lines_df

        return result['V_mag'], result['P_branch'] * self.baseMVA

    # ====== DataFrame builders ======

    def _build_nodes_df(self, result: Dict, P_3ph_pu: np.ndarray, Q_3ph_pu: np.ndarray) -> pd.DataFrame:
        """Build nodes DataFrame with three-phase power flow results."""
        nodes = self.case.nodes.copy()
        res = get_phase_results(result, self.n_nodes - 1)

        v_cols = [f'V_{ph}' for ph in 'ABC']
        ang_cols = [f'angle_{ph}' for ph in 'ABC']

        non_ref_idx = nodes.index[self._non_ref_mask()]

        nodes.loc[non_ref_idx, v_cols] = np.column_stack([res[c] for c in v_cols])
        nodes.loc[non_ref_idx, ang_cols] = np.column_stack([res[c] for c in ang_cols])

        ref_idx = (self.slack_bus_id if self.slack_bus_id in nodes.index
                   else nodes.index[self.slack_bus_id])
        nodes.loc[ref_idx, v_cols] = self.v_ref_mag
        # Derive reference-bus angles from the complex reference voltage so
        # that they stay consistent even with non-standard V_ref_3ph.
        ref_angles = np.angle(self.topo3ph.V_ref_3ph, deg=True)
        nodes.loc[ref_idx, ang_cols] = ref_angles

        # Per-phase and total load per node (MW / MVAr); ref bus has no load.
        # Per-phase columns (p_load_A/B/C_MW, q_load_A/B/C_MVAr) are stored
        # for use by obs() to expose unbalance sources to the RL agent.
        # Total columns (p_load_MW, q_load_MVAr) are kept for build_info / loss tracking.
        non_ref_pos = np.where(self._non_ref_mask())[0]
        p_phase_mat = np.zeros((self.n_nodes, 3), dtype=float)
        q_phase_mat = np.zeros((self.n_nodes, 3), dtype=float)
        p_phase_mat[non_ref_pos] = P_3ph_pu.reshape(self.n_nodes - 1, 3) * self.baseMVA
        q_phase_mat[non_ref_pos] = Q_3ph_pu.reshape(self.n_nodes - 1, 3) * self.baseMVA
        for j, ph in enumerate('ABC'):
            nodes[f'p_load_{ph}_MW'] = p_phase_mat[:, j]
            nodes[f'q_load_{ph}_MVAr'] = q_phase_mat[:, j]
        nodes['p_load_MW'] = p_phase_mat.sum(axis=1)
        nodes['q_load_MVAr'] = q_phase_mat.sum(axis=1)

        # Average voltage magnitude across three phases (used by obs() and build_info())
        nodes['v_mag'] = nodes[v_cols].mean(axis=1)

        return nodes

    def _build_lines_df(self, result: Dict, P_loss_pu: np.ndarray, Q_loss_pu: np.ndarray) -> pd.DataFrame:
        """Build lines DataFrame with three-phase results."""
        lines = self.case.lines.loc[self.active_line_indices].copy()
        n_lines = len(lines)
        res = get_phase_results(result, n_lines)

        p_cols = [f'P_{ph}_MW' for ph in 'ABC']
        q_cols = [f'Q_{ph}_MVAr' for ph in 'ABC']
        ploss_cols = [f'p_loss_{ph}_MW' for ph in 'ABC']
        qloss_cols = [f'q_loss_{ph}_MVAr' for ph in 'ABC']

        p_flow_mat = np.column_stack([res[f'P_{ph}'] for ph in 'ABC']) * self.baseMVA
        q_flow_mat = np.column_stack([res[f'Q_{ph}'] for ph in 'ABC']) * self.baseMVA
        p_loss_mat = P_loss_pu.reshape(n_lines, 3) * self.baseMVA
        q_loss_mat = Q_loss_pu.reshape(n_lines, 3) * self.baseMVA

        lines[p_cols] = p_flow_mat
        lines[q_cols] = q_flow_mat
        lines[ploss_cols] = p_loss_mat
        lines[qloss_cols] = q_loss_mat

        lines['p_flow_MW'] = p_flow_mat.sum(axis=1)
        lines['q_flow_MVAr'] = q_flow_mat.sum(axis=1)
        lines['p_loss_MW'] = p_loss_mat.sum(axis=1)
        lines['q_loss_MVAr'] = q_loss_mat.sum(axis=1)

        return lines

    # ====== VUF and Safety ======

    def calculate_vuf(self, nodes_df: Any) -> Tuple[np.ndarray, float]:
        """Calculate Voltage Unbalance Factor (VUF) for all nodes.

        VUF = |V_neg| / |V_pos| * 100%  (Fortescue / symmetrical components)

        Args:
            nodes_df: DataFrame (or dict-like) with per-phase voltage magnitudes
                      (columns V_A, V_B, V_C) and angles **in degrees**
                      (columns angle_A, angle_B, angle_C).  The angle unit
                      matches the contract of ``run_3phase_bfs_power_flow``
                      which uses ``np.angle(…, deg=True)``.

        Returns:
            vuf_percent: (n_nodes,) VUF values in percent.
            max_vuf: scalar maximum VUF across the network.

        Raises:
            TypeError: if *nodes_df* is not a DataFrame and cannot be used.
        """
        if not isinstance(nodes_df, pd.DataFrame):
            raise TypeError(
                f"calculate_vuf expects a pandas DataFrame, got {type(nodes_df).__name__}")

        cols_v = [f'V_{ph}' for ph in 'ABC']
        cols_ang = [f'angle_{ph}' for ph in 'ABC']
        for c in cols_v + cols_ang:
            if c not in nodes_df.columns:
                raise KeyError(f"Required column '{c}' missing from nodes_df")

        V_mag = nodes_df[cols_v].values
        # Angles from the BFS solver are in degrees; convert to radians.
        V_ang = np.deg2rad(nodes_df[cols_ang].values)

        V_complex = V_mag * np.exp(1j * V_ang)

        alpha = np.exp(1j * 2 * np.pi / 3)
        alpha_sq = alpha ** 2

        Va, Vb, Vc = V_complex[:, 0], V_complex[:, 1], V_complex[:, 2]

        V_pos = (Va + alpha * Vb + alpha_sq * Vc) / 3.0
        V_neg = (Va + alpha_sq * Vb + alpha * Vc) / 3.0

        v_pos_mag = np.abs(V_pos)
        v_neg_mag = np.abs(V_neg)

        vuf = np.zeros_like(v_pos_mag)
        mask = v_pos_mag > 1e-6
        vuf[mask] = (v_neg_mag[mask] / v_pos_mag[mask]) * 100.0

        # vuf is initialised to zeros; only nodes with v_pos_mag > 1e-6 are updated.
        # Use np.max (not nanmax) since vuf contains no NaN values.
        # If mask is entirely False (degenerate case: all positive-sequence voltages
        # near zero), vuf stays all-zero and max_vuf returns 0.0 safely.
        max_vuf = float(np.max(vuf)) if len(vuf) > 0 else 0.0
        return vuf, max_vuf

    def safety_check(self, nodes_result: Any = None, lines_result: Any = None,
                     v_min: float = None, v_max: float = None,
                     with_info: bool = False) -> Tuple[bool, Optional[Dict]]:
        """Check per-phase voltage limits, VUF limit, and thermal limits.

        Handles power-flow divergence explicitly: when the solver did not
        converge or the voltage matrix contains NaN/Inf, the system is
        unconditionally unsafe and all nodes/lines are counted as violating.

        Compared with the single-phase base class, this method additionally:
        - checks each phase voltage (V_A, V_B, V_C) independently,
        - checks VUF against ``self.vuf_max`` (default 2 %),
        - checks branch thermal limits phase-by-phase using
          ``sqrt(P_ph^2 + Q_ph^2)``,
        - reports ``vuf_violation: True/False`` and ``max_vuf_percent``.

        Thermal-capacity interpretation:
            ``case.lines['cap']`` is a single aggregate apparent-power limit per
            line. When a line energises fewer than three phases, this method
            splits that aggregate limit evenly across the energized phases
            before checking phase-wise overloads. This avoids both
            under-detecting severe phase imbalance on 3-phase lines and
            over-penalising single-phase laterals by blindly dividing by 3.

        Args:
            nodes_result: Node results **DataFrame** with V_A/V_B/V_C and
                angle_A/angle_B/angle_C columns (i.e. ``cal_pf(df=True)``
                output).  Passing a numpy array raises ``TypeError`` because
                per-phase voltage magnitudes and angles are required for VUF
                calculation and cannot be recovered from a flat array.
            lines_result: Line results **DataFrame** (``cal_pf(df=True)``
                output).  Passing a numpy array raises ``TypeError``.
            v_min: Minimum voltage limit (p.u.). Defaults to self.v_min.
            v_max: Maximum voltage limit (p.u.). Defaults to self.v_max.
            with_info: If True, return detailed info dict.

        Returns:
            is_safe: True if all constraints satisfied (voltage + VUF + thermal).
            info: Violation details (if with_info=True), else None.

        Raises:
            TypeError: If *nodes_result* or *lines_result* (after the internal
                ``cal_pf`` fallback) is not a ``pd.DataFrame``.  Three-phase
                safety checks depend on per-phase columns that are only present
                in DataFrame output; silently skipping them would return a
                false-safe result.
        """
        if v_min is None:
            v_min = self.v_min
        if v_max is None:
            v_max = self.v_max

        if nodes_result is None or lines_result is None:
            nodes_result, lines_result = self.cal_pf(df=True)

        # Strict type enforcement: three-phase checks require per-phase columns
        # (V_A/B/C, angle_A/B/C) that only exist in DataFrame output.
        # A numpy array input would cause all voltage/VUF checks to be silently
        # skipped, returning a false-safe result — which is worse than crashing.
        if not isinstance(nodes_result, pd.DataFrame):
            raise TypeError(
                "safety_check requires nodes_result to be a pd.DataFrame "
                "(use cal_pf(df=True)). "
                f"Got {type(nodes_result).__name__}. "
                "Three-phase voltage and VUF checks depend on per-phase columns "
                "(V_A, V_B, V_C, angle_A, angle_B, angle_C) that are absent in "
                "numpy array output."
            )
        if not isinstance(lines_result, pd.DataFrame):
            raise TypeError(
                "safety_check requires lines_result to be a pd.DataFrame "
                "(use cal_pf(df=True)). "
                f"Got {type(lines_result).__name__}."
            )

        # --- Early exit: power flow divergence / NaN guard ---
        converged = getattr(self, '_converged', True)
        cols_check = [c for c in [f'V_{ph}' for ph in 'ABC'] if c in nodes_result.columns]
        has_bad_values = False
        if cols_check:
            v_matrix = nodes_result[cols_check].values
            has_bad_values = np.any(~np.isfinite(v_matrix))

        if not converged or has_bad_values:
            all_nodes = list(range(len(nodes_result)))
            all_lines = list(range(self.n_lines))
            if with_info:
                return False, {
                    'v_min_actual': float('nan'),
                    'v_max_actual': float('nan'),
                    'v_violation_nodes': all_nodes,
                    'line_violation_ids': all_lines,
                    'max_vuf_percent': 100.0,
                    'vuf_violation': True,
                    'vuf_violation_nodes': all_nodes,
                    'converged': converged,
                    'iterations': getattr(self, '_iterations', 0),
                }
            return False, None

        # ---- Per-phase voltage check ----
        v_safe = True
        v_violation_nodes: List[int] = []
        all_voltages = np.array([1.0])

        if cols_check:
            v_matrix = nodes_result[cols_check].values
            valid_mask = ~np.isnan(v_matrix)

            if np.any(valid_mask):
                violation_mask = (v_matrix < v_min) | (v_matrix > v_max)
                violation_mask &= valid_mask

                node_violation = np.any(violation_mask, axis=1)
                if np.any(node_violation):
                    v_safe = False
                    v_violation_nodes = np.where(node_violation)[0].tolist()

                all_voltages = v_matrix[valid_mask]

        # ---- VUF check ----
        vuf_safe = True
        max_vuf = 0.0
        vuf_violation_nodes: List[int] = []
        if isinstance(nodes_result, pd.DataFrame):
            vuf_arr, max_vuf = self.calculate_vuf(nodes_result)
            if max_vuf > self.vuf_max:
                vuf_safe = False
                vuf_violation_nodes = np.where(vuf_arr > self.vuf_max)[0].tolist()

        # ---- Branch thermal limit ----
        line_safe = True
        line_violation_ids: List[int] = []
        if isinstance(lines_result, pd.DataFrame) and 'p_flow_MW' in lines_result.columns:
            active_lines = self.case.lines.loc[self.active_line_indices]
            if 'cap' in active_lines.columns:
                line_cap = active_lines['cap'].values.astype(float)
                valid_cap = line_cap > 0
                if np.any(valid_cap):
                    phase_p_cols = [f'P_{ph}_MW' for ph in 'ABC']
                    phase_q_cols = [f'Q_{ph}_MVAr' for ph in 'ABC']

                    if all(col in lines_result.columns for col in phase_p_cols):
                        p_phase = lines_result[phase_p_cols].to_numpy(dtype=float)
                        if all(col in lines_result.columns for col in phase_q_cols):
                            q_phase = lines_result[phase_q_cols].to_numpy(dtype=float)
                        else:
                            q_phase = np.zeros_like(p_phase)
                        s_phase = np.sqrt(p_phase ** 2 + q_phase ** 2)
                        s_phase[~np.isfinite(s_phase)] = np.inf

                        phase_cap = line_cap / self._line_phase_counts()
                        phase_mask = self._line_phase_mask()
                        # Approximate a phase-level thermal limit by evenly
                        # splitting the aggregate apparent-power cap across the
                        # energized phases; this is a benchmark simplification,
                        # not a conductor-ampacity calculation.
                        valid_phase = valid_cap[:, None] & phase_mask
                        violation_phase = valid_phase & (s_phase > phase_cap[:, None])
                        violation = np.any(violation_phase, axis=1)
                    else:
                        p_flow = lines_result['p_flow_MW'].values
                        q_flow = (lines_result['q_flow_MVAr'].values
                                  if 'q_flow_MVAr' in lines_result.columns
                                  else np.zeros_like(p_flow))
                        s_flow = np.sqrt(p_flow ** 2 + q_flow ** 2)
                        violation = np.zeros(len(lines_result), dtype=bool)
                        violation[valid_cap] = s_flow[valid_cap] > line_cap[valid_cap]
                    if np.any(violation):
                        line_safe = False
                        line_violation_ids = np.where(violation)[0].tolist()

        is_safe = v_safe and vuf_safe and line_safe

        if not with_info:
            return is_safe, None

        return is_safe, {
            'v_min_actual': float(np.min(all_voltages)),
            'v_max_actual': float(np.max(all_voltages)),
            'v_violation_nodes': v_violation_nodes,
            'line_violation_ids': line_violation_ids,
            'max_vuf_percent': float(max_vuf),
            'vuf_violation': not vuf_safe,
            'vuf_violation_nodes': vuf_violation_nodes,
            'converged': getattr(self, '_converged', True),
            'iterations': getattr(self, '_iterations', 0),
        }
