"""Distribution Grid Environment

Env class for radial distribution networks. Uses Backward/Forward Sweep (BFS)
power flow via cal_pf_trans module (DistFlow equations).

File layout:
    dist.py        — RL interface: __init__, spaces, obs, reset, reward, info
    _dist_pf.py    — Physics: cal_pf, safety_check, get_total_loss
    _dist_loads.py — Load data: _get_node_loads_p/q and helpers
"""
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd

from powerzoo.case.CaseBase import ClearCase
from powerzoo.envs.grid import GridEnv
from powerzoo.envs.grid.cal_pf_dist import build_radial_topology
from powerzoo.envs.grid._dist_pf import _DistPFMixin
from powerzoo.envs.grid._dist_loads import _DistLoadsMixin

if TYPE_CHECKING:
    from powerzoo.data import DataLoader

from gymnasium import spaces as _spaces


class DistGridEnv(_DistPFMixin, _DistLoadsMixin, GridEnv):
    """Distribution grid environment using BFS (Backward/Forward Sweep) power flow.

    Default case is Case33bw; handles both active (P) and reactive (Q) power.
    The physical core is a **single-phase balanced radial** DistFlow solver:
    explicit voltage-angle states, phase coupling, and unbalance are outside
    this benchmark surface.

    Naming conventions (consistent with MATPOWER):
        baseMVA, baseKV  — system base values
        slack_bus_id     — slack/reference bus index (0-based); alias: ref_bus
        v_slack          — slack bus voltage setpoint (p.u.); alias: v_ref_mag
        p_flow_MW        — sending-end (from-bus) active power flow on each branch
        p_loss_MW        — active power loss (I²R) on each branch
        p_slack_MW       — total slack-bus active-power injection into the feeder
        q_slack_MVAr     — total slack-bus reactive-power injection into the feeder
        is_diverged      — BFS failed to satisfy the iteration tolerance before
                           hitting ``max_iter`` (distinct from voltage collapse)

    Resource control mode:
        All non-slack buses are treated as PQ buses by the BFS solver.
        When a resource is registered via ``register_resource()``, it operates
        in PQ control mode: its P/Q setpoints are subtracted from the nodal
        load before the BFS solve.

    Action space: only ``gymnasium.spaces.Box`` (continuous) is supported.
    """

    # Difficulty presets
    _DIFFICULTY_PRESETS = {
        'easy':   dict(v_min=0.88, v_max=1.12, description='33-bus, relaxed voltage limits'),
        'medium': dict(v_min=0.90, v_max=1.10, description='33-bus, standard voltage limits'),
        'hard':   dict(v_min=0.93, v_max=1.07, description='33-bus, tight voltage limits'),
    }
    _PF_FAILURE_VOLTAGE_DROP_PU = 0.2

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
                 v_slack: float = 1.0,
                 v_min: float = 0.90, v_max: float = 1.10,
                 allow_mesh_pruning: bool = True,
                 difficulty: Optional[str] = None,
                 violation_penalty_weight: float = 0.0,
                 v_dev_penalty_weight: float = 0.0,
                 loss_penalty_weight: float = 0.1):
        """Initialize distribution grid environment

        Args:
            case: ClearCase instance (default: Case33bw)
            solver: Optional external solver
            delta_t_minutes: Time step length in minutes (default: 30).
            data_loader: DataLoader for external time-series data.
            start_date: Start date for data loading.
            end_date: End date for data loading.
            load_columns: Columns to load from DataLoader. Distribution envs
                also honour an optional ``load.reactive_mvar`` signal when
                provided.
            max_load_ratio: Peak load as fraction of case capacity (default: 0.9).
            min_load_ratio: Minimum load ratio (optional).
            time_series: Custom time-series (numpy array or DataFrame).
            max_iter: Maximum iterations for power flow (default: 100)
            tol: Convergence tolerance (default: 1e-6)
            max_episode_steps: Max steps per episode before truncation.
            randomize_start_time: Randomize intra-day start offset on reset.
            v_slack: Slack bus voltage setpoint in p.u. (default: 1.0).
            v_min: Minimum voltage limit (p.u., default: 0.90).
            v_max: Maximum voltage limit (p.u., default: 1.10).
            allow_mesh_pruning: If True (default), extra lines in a non-radial
                input are pruned to the BFS first-visit spanning tree with a
                warning. If False, env initialization fails fast on mesh input.
            difficulty: Preset - 'easy', 'medium', or 'hard'. Overrides v_min/v_max.
            violation_penalty_weight: Weight for soft-penalty mode. When > 0,
                ``cost_voltage_violation`` and ``cost_thermal_overload`` are added
                as negative reward terms (standard RL mode). When 0 (default),
                violations are exposed only via info cost fields (CMDP mode).
            v_dev_penalty_weight: Weight for voltage-deviation penalty. When > 0,
                adds ``-v_dev_penalty_weight * mean((v - 1.0)²)`` to reward.
                MSE penalises large deviations quadratically with a natural soft
                deadband. Useful for Volt-VAR / voltage regulation tasks.
                When 0 (default), no voltage-deviation term is added.
            loss_penalty_weight: Weight on active-loss reward shaping.
                Default reward uses ``-0.1 * p_loss_MW``.
        """
        if difficulty is not None:
            if difficulty not in self._DIFFICULTY_PRESETS:
                raise ValueError(
                    f"difficulty must be one of {list(self._DIFFICULTY_PRESETS)}, "
                    f"got '{difficulty}'"
                )
            preset = self._DIFFICULTY_PRESETS[difficulty]
            v_min = preset['v_min']
            v_max = preset['v_max']

        if case is None:
            from powerzoo.case.distribution import Case33bw
            case = Case33bw()
        if getattr(case, "GRID_TYPE", "") == "transmission":
            warnings.warn(
                f"Case '{type(case).__name__}' is a transmission case. "
                f"DistGridEnv expects a distribution case.",
                UserWarning,
                stacklevel=2,
            )
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
            max_episode_steps=max_episode_steps,
            randomize_start_time=randomize_start_time,
        )

        self.difficulty = difficulty
        self.v_min = v_min
        self.v_max = v_max
        self.violation_penalty_weight = violation_penalty_weight
        self.v_dev_penalty_weight = v_dev_penalty_weight
        self.loss_penalty_weight = loss_penalty_weight

        self.baseMVA = getattr(case, 'baseMVA', 100.0)
        self.baseKV = getattr(case, 'baseKV', 12.66)

        # Episode metric accumulators
        self._ep_violations: int = 0
        self._ep_loss_mw: float = 0.0

        self.max_iter = max_iter
        self.tol = tol
        self.allow_mesh_pruning = allow_mesh_pruning

        self.slack_bus_id = getattr(case, 'slack_bus', 0)
        self.v_slack = v_slack

        self._build_topology()

        # Cached power flow results
        self._nodes = None
        self._lines = None
        self._is_safe = None
        self._safety_info = None
        self._p_loss = 0.0
        self._q_loss = 0.0
        self._p_slack_mw = 0.0
        self._q_slack_mvar = 0.0
        self._is_diverged = False
        self._voltage_collapse = False
        self._prev_nodes = None
        self._prev_lines = None
        self._pf_failed = False
        self._last_v_sq: Optional[np.ndarray] = None

        # Static load caches (populated lazily, cleared by _invalidate_caches)
        self._nodes_loads_map: Optional[np.ndarray] = None
        self._cached_case_p: Optional[np.ndarray] = None
        self._cached_case_q: Optional[np.ndarray] = None

        self._build_spaces()

    # ====== Load scaling ======

    def _get_load_scaling_capacity(self) -> float:
        """Use nominal feeder load (sum of loads d_max) for scaling.

        Distribution cases typically have only a slack-bus generator whose
        p_max is a virtual capacity ceiling, not a meaningful scaling reference.
        Using ``loads['d_max'].sum()`` keeps scaled demand proportional to the
        case's design point.
        """
        if hasattr(self.case, 'loads') and 'd_max' in self.case.loads.columns:
            total = float(self.case.loads['d_max'].sum())
            if total > 0:
                return total
        return super()._get_load_scaling_capacity()

    # ====== Cross-convention aliases ======

    @property
    def ref_bus(self) -> int:
        """Alias for slack_bus_id (three-phase convention compatibility)."""
        return self.slack_bus_id

    @property
    def v_ref_mag(self) -> float:
        """Alias for v_slack in p.u. (three-phase convention compatibility)."""
        return self.v_slack

    # ====== Topology ======

    def _build_topology(self) -> None:
        """Build radial network topology from case data."""
        self._ensure_case_init()

        n_nodes = len(self.case.nodes)
        lines = self.case.lines

        if 'status' in lines.columns:
            active_lines = lines[lines['status'] == 1].copy()
        else:
            active_lines = lines.copy()

        active_line_indices = active_lines.index.tolist()
        from_nodes = active_lines['#from'].values.astype(int)
        to_nodes = active_lines['#to'].values.astype(int)
        r_pu = active_lines['r'].values.copy()
        x_pu = active_lines['x'].values.copy()

        self.topo = build_radial_topology(
            n_nodes=n_nodes,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            r_pu=r_pu,
            x_pu=x_pu,
            slack_bus_id=self.slack_bus_id,
            active_line_indices=active_line_indices,
            allow_mesh_pruning=self.allow_mesh_pruning,
        )

        self.n_nodes = self.topo.n_nodes
        self.n_lines = self.topo.n_lines
        self.active_line_indices = self.topo.active_line_indices

    # ====== Spaces & Observation ======

    def _build_spaces(self) -> None:
        """Rebuild observation_space, action_space, obs_names, action_names.

        Called automatically after any resource register/unregister, and once
        during ``__init__``.

        Observation vector layout:
            node voltages (n_nodes,)        — normalised: (v - 1) / 0.1
            line active flows (n_lines,)    — normalised by baseMVA
            line reactive flows (n_lines,)  — normalised by baseMVA
            node active loads (n_nodes,)    — normalised by baseMVA
            node reactive loads (n_nodes,)  — normalised by baseMVA
            DER states (sum of grid_obs dims across sub_resources)
            [time_sin, time_cos]  (2,)
        """
        n_nodes = self.n_nodes
        n_lines = self.n_lines

        # Observation space
        der_obs_dim = sum(len(res.grid_obs()) for res in self.sub_resources.values())
        self.observation_space = _spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(3 * n_nodes + 2 * n_lines + der_obs_dim + 2,), dtype=np.float32,
        )
        self.obs_names = (
            [f'node_{i}_v_norm' for i in range(n_nodes)]
            + [f'line_{i}_p_norm' for i in range(n_lines)]
            + [f'line_{i}_q_norm' for i in range(n_lines)]
            + [f'node_{i}_p_load_norm' for i in range(n_nodes)]
            + [f'node_{i}_q_load_norm' for i in range(n_nodes)]
            + [n for rid, res in self.sub_resources.items() for n in res.grid_obs_names(rid)]
            + ['time_sin', 'time_cos']
        )

        # Action space
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

    def update_action_space(self) -> None:
        """Rebuild action_space to reflect currently registered resources.

        This is an alias for ``_build_spaces()`` kept for backward compatibility.
        """
        self._build_spaces()

    def obs(self, state: Any = None) -> np.ndarray:
        """Return flat float32 observation array.

        When ``state`` is provided, its nodes/lines/time_step are used instead
        of the live cache (useful for replaying a past step).

        On PF failure the agent receives a penalty observation (below-normal
        voltages, zero flows) rather than the pre-divergence state, so that
        the catastrophic reward is correctly paired with an out-of-band
        observation. See ``_obs_should_use_failure_fallback()`` for trigger
        conditions.

        Observation layout and normalisation: see ``_build_spaces()``.
        """
        nodes = state['nodes'] if state is not None and 'nodes' in state else self._nodes
        lines = state['lines'] if state is not None and 'lines' in state else self._lines
        time_step = state['time_step'] if state is not None and 'time_step' in state else self.time_step
        day_id = state['day_id'] if state is not None and 'day_id' in state else self.day_id

        # Pre-reset edge case: nodes is None on the very first call before reset.
        if nodes is None:
            nodes = self._prev_nodes

        # PF failure: expose a penalty observation (v_min - 0.2 voltages, zero
        # flows) so the catastrophic reward is paired with an out-of-band state.
        if self._obs_should_use_failure_fallback(state, nodes, lines):
            nodes = None
            lines = None

        parts = []

        failure_v = max(self.v_min - self._PF_FAILURE_VOLTAGE_DROP_PU, 0.0)

        # 1. Node voltages — normalised: (v - 1) / 0.1
        if nodes is not None and 'v_mag' in nodes.columns:
            v = nodes['v_mag'].values.astype(np.float32)
        else:
            v = np.full(self.n_nodes, failure_v, dtype=np.float32)
        np.nan_to_num(v, copy=False, nan=failure_v, posinf=self.v_max, neginf=failure_v)
        parts.append((v - 1.0) / 0.1)

        # 2. Line active flows — normalised by baseMVA
        if lines is not None and 'p_flow_MW' in lines.columns:
            p_flow = lines['p_flow_MW'].values.astype(np.float32) / self.baseMVA
        else:
            p_flow = np.zeros(self.n_lines, dtype=np.float32)
        np.nan_to_num(p_flow, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        parts.append(p_flow)

        # 3. Line reactive flows — normalised by baseMVA
        if lines is not None and 'q_flow_MVAr' in lines.columns:
            q_flow = lines['q_flow_MVAr'].values.astype(np.float32) / self.baseMVA
        else:
            q_flow = np.zeros(self.n_lines, dtype=np.float32)
        np.nan_to_num(q_flow, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        parts.append(q_flow)

        # 4. Node active loads — normalised by baseMVA
        if nodes is not None and 'p_load_MW' in nodes.columns:
            p_load = nodes['p_load_MW'].values.astype(np.float32) / self.baseMVA
        else:
            p_load = self._get_node_loads_p().astype(np.float32) / self.baseMVA
        np.nan_to_num(p_load, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        parts.append(p_load)

        # 5. Node reactive loads — normalised by baseMVA
        if nodes is not None and 'q_load_MVAr' in nodes.columns:
            q_load = nodes['q_load_MVAr'].values.astype(np.float32) / self.baseMVA
        else:
            q_load = self._get_node_loads_q().astype(np.float32) / self.baseMVA
        np.nan_to_num(q_load, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        parts.append(q_load)

        # 6. DER states (battery SOC / bounds; renewable available CF)
        for res in self.sub_resources.values():
            parts.append(res.grid_obs())

        # 7. Time encoding
        phase = self._get_obs_time_phase(time_step, day_id=day_id)
        parts.append(np.array([np.sin(phase), np.cos(phase)], dtype=np.float32))

        return np.concatenate(parts)

    def _get_episode_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'total_violations': self._ep_violations,
            'total_loss_mw': self._ep_loss_mw,
        }

    # ====== RL Interface ======

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict] = None,
              day_id: Optional[int] = None) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset distribution grid and run initial power flow."""
        super().reset(seed=seed, options=options, day_id=day_id)
        self._ep_violations = 0
        self._ep_loss_mw = 0.0
        self._last_v_sq = None
        self._run_power_flow({})
        state = self._get_state()
        info = self.build_info(state)
        return state, info

    def _run_power_flow(self, action):
        """Run BFS power flow with action parameters.

        Saves the current nodes/lines as _prev_nodes/_prev_lines before
        overwriting. These caches are used only for missing-state / pre-reset
        edge cases; PF divergence does not reuse the previous solved state.

        Returns:
            bool: True if power flow converged, False otherwise.
        """
        if self._nodes is not None:
            self._prev_nodes = self._nodes
            self._prev_lines = self._lines

        p_load = action.get('p_load')
        q_load = action.get('q_load')

        self._nodes, self._lines = self.cal_pf(p_load, q_load, df=True)
        self._pf_failed = not self._converged

        self._is_safe, self._safety_info = self.safety_check(
            self._nodes, self._lines, with_info=True
        )

        self._p_loss, self._q_loss = self.get_total_loss(self._lines)

        return self._converged

    def _get_state(self):
        """Get current state from cached power flow results."""
        return {
            'nodes': self._nodes,
            'lines': self._lines,
            'is_safe': self._is_safe,
            'safety_info': self._safety_info,
            'p_loss_MW': self._p_loss,
            'q_loss_MVAr': self._q_loss,
            'p_slack_MW': self._p_slack_mw,
            'q_slack_MVAr': self._q_slack_mvar,
            'is_diverged': self._is_diverged,
            'voltage_collapse': self._voltage_collapse,
            'day_id': self.day_id,
            'time_step': self.time_step,
        }

    def _compute_reward(self, state):
        """Compute scalar reward and populate state['reward_components'].

        Primary reward is loss-based; safety violations are exposed through
        build_info() as cost fields for CMDP formulations.

        Components:
            loss_penalty:      -loss_penalty_weight * total_active_loss_MW
                               (always present)
            violation_penalty: -violation_penalty_weight * total_violation_count
                               (only when violation_penalty_weight > 0)
            v_dev_penalty:     -v_dev_penalty_weight * mean((v - 1.0)²)
                               (only when v_dev_penalty_weight > 0)
        """
        loss_penalty = -self.loss_penalty_weight * state['p_loss_MW']
        self._ep_loss_mw += state['p_loss_MW']

        components = {'loss_penalty': loss_penalty}
        reward = loss_penalty

        safety_info = state.get('safety_info') or {}
        n_v = len(safety_info.get('v_violation_nodes', []))
        n_l = len(safety_info.get('line_violation_ids', []))
        self._ep_violations += int(n_v + n_l)

        if self.violation_penalty_weight > 0.0:
            violation_penalty = -self.violation_penalty_weight * float(n_v + n_l)
            components['violation_penalty'] = violation_penalty
            reward += violation_penalty

        if self.v_dev_penalty_weight > 0.0:
            nodes = state.get('nodes')
            if nodes is not None and 'v_mag' in nodes.columns:
                v_mag = nodes['v_mag'].values
                v_finite = v_mag[np.isfinite(v_mag)]
                v_dev = float(np.mean((v_finite - 1.0) ** 2)) if len(v_finite) > 0 else 0.0
            else:
                v_dev = 0.0
            v_dev_penalty = -self.v_dev_penalty_weight * v_dev
            components['v_dev_penalty'] = v_dev_penalty
            reward += v_dev_penalty

        state['reward_components'] = components
        return reward

    def constraint_names(self) -> Tuple[str, ...]:
        """Return the fixed benchmark cost-channel order."""
        return ('voltage_violation', 'thermal_overload', 'resource')

    def build_info(self, state):
        """Build info dict with power flow, safety details, and CMDP cost fields.

        Note: cost_voltage_violation and cost_thermal_overload are **counts**
        of violating nodes/lines (integers), unlike TransGridEnv which uses
        continuous violation magnitudes.
        """
        safety_info = state['safety_info'] or {}
        n_v = len(safety_info.get('v_violation_nodes', []))
        n_l = len(safety_info.get('line_violation_ids', []))
        return self.attach_constraint_costs({
            'is_safe': state['is_safe'],
            'safety_info': safety_info,
            'p_loss_MW': state['p_loss_MW'],
            'q_loss_MVAr': state['q_loss_MVAr'],
            'p_slack_MW': state.get('p_slack_MW', 0.0),
            'q_slack_MVAr': state.get('q_slack_MVAr', 0.0),
            'is_diverged': bool(state.get('is_diverged', False)),
            'voltage_collapse': bool(state.get('voltage_collapse', False)),
            'cost_voltage_violation': float(n_v),
            'cost_thermal_overload': float(n_l),
            'cost_resource': 0.0,
            'cost_sum': float(n_v + n_l),
            'goal_met': bool(state['is_safe']),
            'resource_status': {rid: res.status() for rid, res in self.sub_resources.items()},
            'reward_components': state.get('reward_components', {}),
        })

    # ====== Observation helpers ======

    def _obs_should_use_failure_fallback(
        self,
        state: Any,
        nodes: Optional[pd.DataFrame],
        lines: Optional[pd.DataFrame],
    ) -> bool:
        """Return True when obs() should expose the PF-failure penalty state."""
        if state is None:
            return bool(getattr(self, '_pf_failed', False))

        if isinstance(state, dict):
            safety_info = state.get('safety_info') or {}
            if safety_info.get('converged') is False:
                return True

        if isinstance(nodes, pd.DataFrame) and 'v_mag' in nodes.columns:
            if np.any(~np.isfinite(nodes['v_mag'].values)):
                return True

        if isinstance(lines, pd.DataFrame):
            for col in ('p_flow_MW', 'q_flow_MVAr'):
                if col in lines.columns and np.any(~np.isfinite(lines[col].values)):
                    return True

        return False

    def _get_obs_time_phase(self, time_step: int, day_id: Optional[int] = None) -> float:
        """Daily phase for time_sin/time_cos, using real clock time when available."""
        phase_day_id = self.day_id if day_id is None else day_id
        if phase_day_id is not None:
            try:
                current_dt = self._get_datetime_from_day_and_step(phase_day_id, time_step)
                if isinstance(current_dt, pd.Timestamp) and current_dt.tzinfo is not None:
                    current_dt = current_dt.tz_localize(None)
                minutes = (
                    current_dt.hour * 60.0
                    + current_dt.minute
                    + current_dt.second / 60.0
                    + current_dt.microsecond / 60_000_000.0
                )
                return 2.0 * np.pi * minutes / (24.0 * 60.0)
            except Exception:
                pass
        return 2.0 * np.pi * time_step / max(self.steps_per_day, 1)

    # ====== Render ======

    def render(self, mode: str = 'human'):
        """Render the distribution grid state.

        Produces a two-panel figure (radial network + voltage profile chart).
        See :func:`powerzoo.envs.grid._render.render_dist_grid` for details.

        Args:
            mode: ``'human'`` (interactive) or ``'rgb_array'`` (returns
                  a ``(H, W, 3)`` uint8 ndarray without opening a window).
        """
        from powerzoo.envs.grid._render import render_dist_grid
        return render_dist_grid(self, mode)


if __name__ == "__main__":
    env = DistGridEnv()
    print(f"Nodes: {env.n_nodes}, Lines: {env.n_lines}")

    nodes_df, lines_df = env.cal_pf(df=True)
    print(f"Converged: {env._converged} in {env._iterations} iterations")
    print(f"Voltage range: {nodes_df['v_mag'].min():.4f} - {nodes_df['v_mag'].max():.4f} p.u.")

    p_loss, q_loss = env.get_total_loss(lines_df)
    print(f"Total loss: {p_loss:.4f} MW, {q_loss:.4f} MVAr")
