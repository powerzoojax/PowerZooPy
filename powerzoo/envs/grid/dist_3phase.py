"""Three-phase radial distribution environment built on the BFS core.

The underlying solver uses the Kron-expanded BIBC/BCBV formulation from
``cal_pf_dist_3phase``. Every non-reference bus therefore carries an ``A/B/C``
state triplet; true single-/two-phase laterals are not modelled as a reduced
topology here and must already be encoded upstream in the line ``3x3``
impedance blocks.

The core solver supports full mutual coupling inside those series ``3x3``
impedance blocks, but it does not currently apply branch shunt charging
(``B``), off-nominal transformer tap ratios, or phase shifts from branch
metadata such as ``ratio`` / ``angle``.

File layout:
    dist_3phase.py          — RL interface: __init__, spaces, obs, reward, info (this file)
    _dist_3phase_physics.py — Physics integration: topology, cal_pf, safety, VUF
    _dist_3phase_loads.py   — Load data: per-phase load matrices and helpers
    cal_pf_dist_3phase.py   — Solver kernel: ThreePhaseTopology, BFS core (do not edit here)

Naming conventions:
    ref_bus      — reference (slack/swing) bus index (0-based); alias: slack_bus_id
    v_ref_mag    — reference bus voltage magnitude (p.u.)
    V_A/V_B/V_C  — per-phase voltage magnitudes (p.u.) at each node
    P_A_MW etc.  — per-phase sending-end active power flows
    p_loss_MW    — total three-phase branch active loss, including mutual terms
"""
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:
    from powerzoo.data import DataLoader

import numpy as np

from powerzoo.case.CaseBase import ClearCase
from powerzoo.envs.grid.dist import DistGridEnv
from powerzoo.envs.grid._dist_3phase_physics import _Dist3PhasePhysicsMixin
from powerzoo.envs.grid._dist_3phase_loads import _Dist3PhaseLoadsMixin


class DistGrid3PhaseEnv(_Dist3PhasePhysicsMixin, _Dist3PhaseLoadsMixin, DistGridEnv):
    """Three-phase distribution grid environment using BIBC/BCBV power flow.

    Default case is Case123; handles three-phase unbalanced power flow.
    The solver stores a full three-state ``A/B/C`` vector at every non-slack
    bus, so branch phase availability must already be reflected in the case's
    ``3x3`` impedance data rather than inferred from a separate missing-phase
    topology.
    Solver vectors use node-major ``ABC`` order
    ``[node1_A, node1_B, node1_C, node2_A, ...]``; the explicit node mapping is
    available via ``self.topo3ph.non_ref_node_ids`` and
    ``self.topo3ph.node_id_to_matrix_index``.

    Resource phase injection:
        Resources carry a ``phase`` attribute ('A', 'B', 'C', or 'ABC').
        Power is injected only into the connected phase(s), enabling the
        RL agent to learn phase-balancing strategies.

    Non-convergence contract:
        ``cal_pf()`` still returns the last BFS iterate for debugging, but those
        voltages/flows/losses are diagnostic only. RL-facing callers must check
        ``self._converged`` / ``info['pf_converged']`` and treat a False value
        as a power-flow failure rather than a valid operating point.
    """

    # ====== Initialization ======

    def __init__(self, case: ClearCase = None, solver: Any = None,
                 delta_t_minutes: float = 30.0,
                 data_loader: Optional['DataLoader'] = None,
                 start_date: str = '2024-01-01',
                 end_date: str = '2024-01-31',
                 load_columns: Optional[List[str]] = None,
                 max_load_ratio: float = 0.9,
                 min_load_ratio: Optional[float] = None,
                 time_series: Any = None,
                 max_iter: int = 100, tol: float = 1e-6,
                 max_episode_steps: Optional[int] = None,
                 randomize_start_time: bool = False,
                 v_min: float = 0.90, v_max: float = 1.10,
                 vuf_max: float = 2.0,
                 difficulty: Optional[str] = None,
                 violation_penalty_weight: float = 0.0,
                 loss_penalty_weight: float = 0.1,
                 vuf_dense_penalty_weight: float = 0.0):
        """Initialize three-phase distribution grid environment.

        Args:
            case: ClearCase instance (default: Case123).
            solver: Optional external solver.
            delta_t_minutes: Time step length in minutes (default: 30).
            data_loader: DataLoader for external time-series data.
            start_date: Start date for data loading.
            end_date: End date for data loading.
            load_columns: Columns to load from DataLoader.
            max_load_ratio: Peak load as fraction of case capacity (default: 0.9).
            min_load_ratio: Minimum load ratio (optional).
            time_series: Custom time-series (numpy array or DataFrame).
            max_iter: Maximum iterations for power flow (default: 100).
            tol: Convergence tolerance (default: 1e-6).
            max_episode_steps: Max steps per episode before truncation.
            randomize_start_time: Randomize intra-day start offset on reset.
            v_min: Minimum voltage limit (p.u., default: 0.90).
            v_max: Maximum voltage limit (p.u., default: 1.10).
            vuf_max: Maximum voltage unbalance factor (%, default: 2.0).
                     IEEE Std 1159 / EN 50160 typical limit is 2%.
            difficulty: Preset - 'easy', 'medium', or 'hard'. Overrides v_min/v_max.
            violation_penalty_weight: Weight for soft-penalty mode (default: 0.0).
            loss_penalty_weight: Weight on active-loss reward shaping
                     ``-loss_penalty_weight * p_loss_MW`` (default: 0.1).
            vuf_dense_penalty_weight: Weight for dense VUF penalty (default: 0.0).
                     When > 0, adds
                     ``-vuf_dense_penalty_weight * max(max_vuf_percent - 0.75 * vuf_max, 0) / 100``
                     to the reward at every step. This keeps a deadband over the
                     benign low-VUF region (default: 1.5 % when ``vuf_max=2.0``),
                     so the dense shaping only activates near the safety boundary.
                     Operates independently of ``violation_penalty_weight``.
        """
        import warnings
        if case is None:
            from powerzoo.case.distribution import Case123
            case = Case123()
        if getattr(case, "PHASE", "1") != "3":
            warnings.warn(
                f"Case '{type(case).__name__}' is not a three-phase case "
                f"(PHASE='{getattr(case, 'PHASE', '1')}'). "
                f"DistGrid3PhaseEnv expects PHASE='3'.",
                UserWarning,
                stacklevel=2,
            )

        # Zbase must be available before super().__init__() calls _build_topology(),
        # which in turn calls _build_impedance_matrices().
        _baseMVA = getattr(case, 'baseMVA', 10.0)   # system base power (MVA)
        _baseKV = getattr(case, 'baseKV', 4.16)       # system base voltage (kV)
        self.Zbase = (_baseKV ** 2) / _baseMVA         # base impedance (Ω) = kV²/MVA

        # Three-phase specific caches (populated lazily)
        self._cached_non_ref_mask: Optional[np.ndarray] = None
        self._cached_line_phase_mask: Optional[np.ndarray] = None
        self._cached_line_phase_counts: Optional[np.ndarray] = None
        self._cached_case_phase_p: Optional[np.ndarray] = None
        self._cached_case_phase_q: Optional[np.ndarray] = None

        # VUF safety threshold (percentage)
        self.vuf_max: float = vuf_max

        # Dense VUF penalty weight (0 = disabled)
        self.vuf_dense_penalty_weight: float = vuf_dense_penalty_weight

        # Phase allocation lookup: maps phase string → (3,) weight vector.
        # Used by cal_pf to distribute resource injections to A/B/C phases.
        # Weights sum to 1: total power is *split* across connected phases,
        # not replicated.  E.g. a 3 kW 'ABC' resource → 1 kW per phase.
        self._PHASE_ALLOC = {
            'A':   np.array([1.0, 0.0, 0.0]),
            'B':   np.array([0.0, 1.0, 0.0]),
            'C':   np.array([0.0, 0.0, 1.0]),
            'AB':  np.array([0.5, 0.5, 0.0]),
            'AC':  np.array([0.5, 0.0, 0.5]),
            'BC':  np.array([0.0, 0.5, 0.5]),
            'ABC': np.array([1/3, 1/3, 1/3]),
        }

        super().__init__(
            case=case, solver=solver,
            delta_t_minutes=delta_t_minutes,
            data_loader=data_loader,
            start_date=start_date,
            end_date=end_date,
            load_columns=load_columns,
            max_load_ratio=max_load_ratio,
            min_load_ratio=min_load_ratio,
            time_series=time_series,
            max_iter=max_iter,
            tol=tol,
            max_episode_steps=max_episode_steps,
            randomize_start_time=randomize_start_time,
            v_slack=1.05,
            v_min=v_min,
            v_max=v_max,
            difficulty=difficulty,
            violation_penalty_weight=violation_penalty_weight,
            loss_penalty_weight=loss_penalty_weight,
        )

    def _get_load_scaling_capacity(self) -> float:
        """Use feeder d_max x baseMVA (MW) as the demand scaling reference.

        Distribution feeders should scale loads relative to the feeder's own
        native demand capacity, not the connected slack-bus generator's p_max
        (which is typically an unconstrained large value for distribution cases).

        Note: called during super().__init__(), so baseMVA is read from case
        directly rather than from self.baseMVA (not yet assigned).
        """
        baseMVA = getattr(self.case, 'baseMVA', 10.0)
        if (hasattr(self.case, 'loads')
                and hasattr(self.case.loads, 'columns')
                and 'd_max' in self.case.loads.columns):
            return float(self.case.loads['d_max'].values.sum()) * baseMVA
        return super()._get_load_scaling_capacity()

    # ====== Observation / Action Spaces (three-phase override) ======

    def _build_spaces(self) -> None:
        """Define observation and action spaces with per-phase features.

        Observation vector layout (all normalised):
            V_A (n_nodes,)      — phase-A voltage: (v - 1) / 0.1
            V_B (n_nodes,)      — phase-B voltage: (v - 1) / 0.1
            V_C (n_nodes,)      — phase-C voltage: (v - 1) / 0.1
            P_A (n_lines,)      — phase-A line active flow / baseMVA
            P_B (n_lines,)      — phase-B line active flow / baseMVA
            P_C (n_lines,)      — phase-C line active flow / baseMVA
            Q_A (n_lines,)      — phase-A line reactive flow / baseMVA
            Q_B (n_lines,)      — phase-B line reactive flow / baseMVA
            Q_C (n_lines,)      — phase-C line reactive flow / baseMVA
            p_load_A (n_nodes,) — phase-A node active load / baseMVA
            p_load_B (n_nodes,) — phase-B node active load / baseMVA
            p_load_C (n_nodes,) — phase-C node active load / baseMVA
            q_load_A (n_nodes,) — phase-A node reactive load / baseMVA
            q_load_B (n_nodes,) — phase-B node reactive load / baseMVA
            q_load_C (n_nodes,) — phase-C node reactive load / baseMVA
            [time_sin, time_cos]  (2,)

        Per-phase load exposure (vs. single-phase base class total load)
        lets the agent directly observe the unbalance source, improving
        Markov observability in three-phase unbalance control tasks.

        Compared with the single-phase base class, the voltage vector is
        expanded from 1×n_nodes to 3×n_nodes, and line flows from 2×n_lines
        to 6×n_lines, so the RL agent can observe per-phase unbalance.

        Normalisation note:
            Per-phase power features keep the system-level ``baseMVA``
            denominator rather than ``baseMVA / 3``. This preserves the same
            per-unit observation contract as ``DistGridEnv``; users who want
            larger policy-side feature magnitudes can still apply an external
            observation wrapper.

        Observation bounds:
            Bounds are set to ``[-20.0, 20.0]`` to match the hard clip applied
            in ``obs()``. This keeps the space strictly consistent with the
            values the policy will actually receive, which benefits RL
            algorithms that use space bounds for normalisation (e.g. PPO with
            observation normalisation, SAC's replay-buffer bounds check).
            The ±20 limit is lossless for all valid operating points:
            voltage deviations up to ±2 p.u. and flows up to 20 × baseMVA are
            far outside the physical range of any distribution feeder.
        """
        from gymnasium import spaces as _spaces

        n = self.n_nodes
        nl = self.n_lines
        obs_dim = 3 * n + 6 * nl + 6 * n + 2

        self.observation_space = _spaces.Box(
            low=-20.0, high=20.0,
            shape=(obs_dim,), dtype=np.float32,
        )

        node_ids = list(range(n))
        line_ids = list(range(nl))

        self.obs_names: List[str] = (
            [f'node_{i}_V_A_norm' for i in node_ids]
            + [f'node_{i}_V_B_norm' for i in node_ids]
            + [f'node_{i}_V_C_norm' for i in node_ids]
            + [f'line_{j}_P_A_norm' for j in line_ids]
            + [f'line_{j}_P_B_norm' for j in line_ids]
            + [f'line_{j}_P_C_norm' for j in line_ids]
            + [f'line_{j}_Q_A_norm' for j in line_ids]
            + [f'line_{j}_Q_B_norm' for j in line_ids]
            + [f'line_{j}_Q_C_norm' for j in line_ids]
            + [f'node_{i}_p_load_A_norm' for i in node_ids]
            + [f'node_{i}_p_load_B_norm' for i in node_ids]
            + [f'node_{i}_p_load_C_norm' for i in node_ids]
            + [f'node_{i}_q_load_A_norm' for i in node_ids]
            + [f'node_{i}_q_load_B_norm' for i in node_ids]
            + [f'node_{i}_q_load_C_norm' for i in node_ids]
            + ['time_sin', 'time_cos']
        )

        # Action space: built from currently registered resources (same logic as
        # single-phase parent, duplicated here to keep _build_spaces() self-contained).
        if self.sub_resources:
            lows  = [np.float32(res.grid_action_bounds()[0]) for res in self.sub_resources.values()]
            highs = [np.float32(res.grid_action_bounds()[1]) for res in self.sub_resources.values()]
            self.action_space = _spaces.Box(
                low=np.array(lows, dtype=np.float32),
                high=np.array(highs, dtype=np.float32),
                dtype=np.float32,
            )
            self.action_names: List[str] = list(self.sub_resources.keys())
        else:
            self.action_space = _spaces.Box(
                low=np.array([], dtype=np.float32),
                high=np.array([], dtype=np.float32),
                shape=(0,), dtype=np.float32,
            )
            self.action_names: List[str] = []

    def obs(self, state: Any = None) -> np.ndarray:
        """Return flat float32 observation, clipped to [-20.0, 20.0].

        When ``state`` is provided it is used as source instead of the live
        cache. On PF failure a penalty observation (v_min voltages, zero flows)
        is returned. Observation layout and clip rationale: see ``_build_spaces()``.
        """
        nodes = state['nodes'] if state is not None and 'nodes' in state else self._nodes
        lines = state['lines'] if state is not None and 'lines' in state else self._lines
        time_step = state['time_step'] if state is not None and 'time_step' in state else self.time_step

        # Pre-reset edge case: nodes is None on the very first call before any step.
        # _prev_nodes is used only here — never as a PF-divergence fallback.
        if nodes is None:
            nodes = self._prev_nodes

        # When the power flow solver diverged, construct a penalty observation
        # (v_min voltages, zero line flows) rather than returning _prev_nodes.
        #
        # Reusing the previous healthy state would break the Markov contract:
        # the agent would receive a catastrophic reward / termination signal
        # paired with an apparently normal post-action observation.
        #
        # Only apply when computing the live observation (state is None); if a
        # caller re-computes a historical observation from an explicit state
        # dict, that state must be trusted as-is.
        # Double-guard: _pf_failed is normally set immediately after cal_pf(),
        # but also check _converged directly in case they drift out of sync.
        if state is None and (self._pf_failed or not getattr(self, '_converged', True)):
            nodes = None
            lines = None

        parts: List[np.ndarray] = []

        # 1–3. Per-phase voltages (normalised: (v - 1) / 0.1)
        for ph in ('V_A', 'V_B', 'V_C'):
            if nodes is not None and ph in nodes.columns:
                v = nodes[ph].values.astype(np.float32)
            else:
                v = np.full(self.n_nodes, self.v_min, dtype=np.float32)
            np.nan_to_num(v, copy=False, nan=self.v_min, posinf=self.v_max, neginf=self.v_min)
            parts.append((v - 1.0) / 0.1)

        # 4–9. Per-phase line flows (P and Q, normalised by baseMVA)
        for col_base in ('P_{ph}_MW', 'Q_{ph}_MVAr'):
            for ph in 'ABC':
                col = col_base.format(ph=ph)
                if lines is not None and col in lines.columns:
                    f = lines[col].values.astype(np.float32) / self.baseMVA
                else:
                    f = np.zeros(self.n_lines, dtype=np.float32)
                np.nan_to_num(f, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                parts.append(f)

        # 10–15. Per-phase node loads (P_A/B/C then Q_A/B/C, normalised by baseMVA).
        # Fallback when per-phase columns are absent (e.g. before first power flow):
        # P phases use equal-split of total load; Q phases default to zero.
        for ph in 'ABC':
            col = f'p_load_{ph}_MW'
            if nodes is not None and col in nodes.columns:
                f = nodes[col].values.astype(np.float32) / self.baseMVA
            else:
                f = self._get_node_loads_p().astype(np.float32) / (self.baseMVA * 3)
            np.nan_to_num(f, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(f)
        for ph in 'ABC':
            col = f'q_load_{ph}_MVAr'
            if nodes is not None and col in nodes.columns:
                f = nodes[col].values.astype(np.float32) / self.baseMVA
            else:
                f = np.zeros(self.n_nodes, dtype=np.float32)
            np.nan_to_num(f, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(f)

        # 12. Time encoding
        phase = 2.0 * np.pi * time_step / max(self.steps_per_day, 1)
        parts.append(np.array([np.sin(phase), np.cos(phase)], dtype=np.float32))

        # Hard-clip to prevent gradient explosion from extreme-but-finite values
        # (e.g. near-diverged BFS may produce voltages of thousands of p.u. that
        # nan_to_num cannot catch).  Threshold ±20 corresponds to ±3 p.u. voltage
        # on the normalised scale and to 20 × baseMVA on the flow scale — far
        # beyond any physically valid operating point.
        obs_array = np.concatenate(parts)
        return np.clip(obs_array, -20.0, 20.0)

    # ====== Reward and Info ======

    def _compute_reward(self, state):
        """Compute reward for three-phase systems, extending the base with VUF terms.

        Two additional penalty terms beyond the base-class loss + sparse-violation:

        1. VUF violation count (``vuf_violation_nodes``) is folded into
           ``violation_penalty`` when ``violation_penalty_weight > 0``.

        2. Dense VUF penalty: when ``vuf_dense_penalty_weight > 0``, adds
           ``-weight * max(max_vuf_percent - 0.75 * vuf_max, 0) / 100`` each step.
           The deadband keeps shaping inactive in the low-VUF regime so the
           policy only attends to unbalance near the safety boundary.

        NaN/Inf p_loss_MW is mapped to ``baseMVA`` sentinel before shaping.
        ``_ep_violations`` is always incremented to track constraint violations
        regardless of whether soft-penalty mode is active.
        """
        p_loss_mw = float(state.get('p_loss_MW', self.baseMVA))
        if not np.isfinite(p_loss_mw):
            p_loss_mw = float(self.baseMVA)

        loss_penalty = -self.loss_penalty_weight * p_loss_mw
        self._ep_loss_mw += p_loss_mw
        components = {'loss_penalty': loss_penalty}
        reward = loss_penalty

        # Count all three violation types unconditionally so _ep_violations is
        # accurate in both soft-penalty (MDP) and pure-CMDP (weight=0) modes.
        safety_info = state.get('safety_info') or {}
        n_v = len(safety_info.get('v_violation_nodes', []))
        n_l = len(safety_info.get('line_violation_ids', []))
        n_vuf = len(safety_info.get('vuf_violation_nodes', []))
        self._ep_violations += int(n_v + n_l + n_vuf)

        if self.violation_penalty_weight > 0.0:
            violation_penalty = -self.violation_penalty_weight * float(n_v + n_l + n_vuf)
            components['violation_penalty'] = violation_penalty
            reward += violation_penalty

        if self.vuf_dense_penalty_weight > 0.0:
            max_vuf = float(safety_info.get('max_vuf_percent', 0.0))
            vuf_dense_excess = max(0.0, max_vuf - max(0.0, 0.75 * float(self.vuf_max)))
            vuf_dense_penalty = -self.vuf_dense_penalty_weight * vuf_dense_excess / 100.0
            components['vuf_dense_penalty'] = vuf_dense_penalty
            reward += vuf_dense_penalty

        state['reward_components'] = components
        return reward

    def constraint_names(self) -> Tuple[str, ...]:
        """Return the fixed benchmark cost-channel order."""
        return ('voltage_violation', 'thermal_overload', 'vuf_violation', 'resource')

    def build_info(self, state):
        """Build info dict with three-phase CMDP cost fields.

        Extends the base class by adding ``cost_vuf_violation`` — the count
        of nodes whose VUF exceeds ``self.vuf_max``.  This is included in
        ``cost_sum`` so that Safe RL algorithms can constrain VUF.

        ``pf_converged`` is exposed here as well, even though ``GridEnv.step()``
        overwrites the same key afterwards, so direct ``build_info(state)``
        callers can still track divergence without going through ``step()``.

        ``p_loss_MW`` and ``q_loss_MVAr`` use ``state.get()`` with safe
        fallbacks (``baseMVA`` and ``0.0`` respectively) consistent with
        ``_compute_reward``, preventing ``KeyError`` when an external wrapper
        returns a truncated state dict.
        """
        safety_info = state['safety_info'] or {}
        n_v = len(safety_info.get('v_violation_nodes', []))
        n_l = len(safety_info.get('line_violation_ids', []))
        n_vuf = len(safety_info.get('vuf_violation_nodes', []))
        return self.attach_constraint_costs({
            'is_safe': state['is_safe'],
            'safety_info': safety_info,
            'pf_converged': bool(getattr(self, '_converged', True)),
            'p_loss_MW': state.get('p_loss_MW', self.baseMVA),
            'q_loss_MVAr': state.get('q_loss_MVAr', 0.0),
            # CMDP cost fields
            'cost_voltage_violation': float(n_v),
            'cost_thermal_overload': float(n_l),
            'cost_vuf_violation': float(n_vuf),
            'cost_resource': 0.0,
            'cost_sum': float(n_v + n_l + n_vuf),
            'goal_met': bool(state['is_safe']),
            'resource_status': {rid: res.status() for rid, res in self.sub_resources.items()},
            'reward_components': state.get('reward_components', {}),
        })
