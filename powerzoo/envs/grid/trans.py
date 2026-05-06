from typing import Any, Dict, Optional, List, Tuple
import dataclasses
import logging
import warnings

import numpy as np
from gymnasium import spaces as _spaces

from powerzoo.case.CaseBase import ClearCase
from powerzoo.envs.grid.base import GridEnv
from powerzoo.case import load_case
from powerzoo.data import DataLoader
import powerzoo.envs.grid._trans_solve as trans_solve
from powerzoo.envs.grid._trans_solve import TransSolveResult

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ACConfig:
    """AC power-flow solver parameters.  Only active when ``physics='ac'``.

    Can be passed as ``ac_config=ACConfig(...)`` instead of the individual
    ``ac_v_min`` / ``ac_v_max`` / … keyword arguments.

    Attributes:
        v_min:    Bus voltage lower bound (p.u.). Default 0.95.
        v_max:    Bus voltage upper bound (p.u.). Default 1.05.
        q_factor: |Q_max / P_max| ratio for generators. Default 0.75.
        backend:  NLP backend — ``'auto'`` (cyipopt if available, else SLSQP),
                  ``'ipopt'``, or ``'slsqp'``. Default ``'auto'``.
        solver:   AC-OPF implementation — ``'builtin'`` (default) or
                  ``'pandapower'`` (requires pandapower).
    """
    v_min: float = 0.95
    v_max: float = 1.05
    q_factor: float = 0.75
    backend: str = 'auto'
    solver: str = 'builtin'


class TransGridEnv(GridEnv):
    """Transmission grid environment.

    Two orthogonal parameters control the solver behaviour:

    * ``physics`` — ``'dc'`` (linearised, P only) or ``'ac'`` (full AC with
      voltage and reactive power).
    * ``solver_mode`` — ``'opf'`` (environment runs OPF internally; RL agent
      provides bids / commitments) or ``'pf'`` (environment evaluates power
      flow only; RL agent provides unit dispatch directly).

    The four resulting modes are DCOPF, ACOPF, DCPF, and ACPF.

    Default case is Case5.
    """

    # Difficulty presets: (max_load_ratio, delta_t_minutes, description)
    _DIFFICULTY_PRESETS = {
        'easy':   dict(max_load_ratio=0.7, delta_t_minutes=60.0,
                       description='Case5, 1-h steps, relaxed loading (70% capacity)'),
        'medium': dict(max_load_ratio=0.9, delta_t_minutes=30.0,
                       description='Case5, 30-min steps, standard loading (90% capacity)'),
        'hard':   dict(max_load_ratio=0.95, delta_t_minutes=15.0,
                       description='Case5, 15-min steps, tight loading (95% capacity)'),
    }

    def __init__(self, case: ClearCase = None, solver: Any = None,
                 delta_t_minutes: float = 30.0,
                 data_loader: Optional[DataLoader] = None,
                 start_date: str = '2024-01-01',
                 end_date: str = '2024-01-31',
                 load_columns: Optional[List[str]] = None,
                 max_load_ratio: float = 0.9,
                 min_load_ratio: Optional[float] = None,
                 time_series: Any = None,
                 max_episode_steps: Optional[int] = None,
                 randomize_start_time: bool = False,
                 physics: str = 'dc',
                 solver_mode: str = 'opf',
                 solver_type: str = 'auto',
                 normalize_actions: bool = True,
                 ac_config: ACConfig = None,
                 difficulty: Optional[str] = None,
                 reward_scale: float = 0.01,
                 control_der: bool = False):
        """Initialize transmission grid environment.

        Two orthogonal parameters control the solver behaviour:

        * ``physics`` — physical model: ``'ac'`` (full AC equations) or
          ``'dc'`` (linearised DC approximation).
        * ``solver_mode`` — solver role: ``'opf'`` (environment runs optimal
          power flow internally; RL agent provides bids / commitments) or
          ``'pf'`` (environment only evaluates power flow; RL agent provides
          unit dispatch directly via ``action['unit_power_mw']``).

        The four combinations are:

        =======  ===========  ==========  =======================================
        physics  solver_mode  Solver      RL use-case
        =======  ===========  ==========  =======================================
        dc       opf          DCOPF       Agent learns bidding / UC strategy
        ac       opf          ACOPF       Agent learns bidding (with voltage)
        dc       pf           DCPF        Agent learns dispatch (P only)
        ac       pf           ACPF (NR)   Agent learns dispatch (P + V + Q)
        =======  ===========  ==========  =======================================

        Args:
            case: Power system case (default: Case5)
            solver: Power flow solver
            delta_t_minutes: Time step in minutes (default: 30)
            data_loader: DataLoader instance
            start_date: Start date for data loading
            end_date: End date for data loading
            load_columns: List of columns to load
            max_load_ratio: Maximum load as ratio of total capacity
            min_load_ratio: Minimum load as ratio of total capacity (optional)
            time_series: Custom time-series data
            max_episode_steps: Max steps before truncation (default: one full day)
            physics: Physical model — ``'dc'`` (default) or ``'ac'``.
            solver_mode: Solver role — ``'opf'`` (default, environment optimises)
                or ``'pf'`` (agent provides dispatch, environment evaluates).
            solver_type: OPF LP solver - 'auto', 'gurobi', 'scipy', 'cvxpy'.
            normalize_actions: Whether to normalise actions to [-1, 1].
            ac_config: AC solver parameters as an :class:`ACConfig` object.
                Defaults to ``ACConfig()`` (standard voltage limits, built-in solver).
                See :class:`ACConfig` for all configurable fields.
            difficulty: Preset difficulty level - 'easy', 'medium', or 'hard'.
                        When set, overrides ``delta_t_minutes`` and ``max_load_ratio``.
            reward_scale: Multiplicative scaling factor applied to generation cost in the
                          reward signal (``economic_cost = -reward_scale * gen_cost``).
                          Default 0.01 is calibrated for Case5 (typical cost ~100-1000 $/step).
                          For larger systems (e.g. Case118) where cost can reach 10⁵-10⁶,
                          pass a smaller value such as 0.0001 to keep rewards in a
                          well-conditioned range for policy gradient algorithms.
            control_der: When ``True``, flatten DER control dimensions into the
                         environment's ``action_space``.  In PF modes the flat
                         action becomes
                         ``[unit_power_mw (n_units), der_actions (n_resources)]``.
                         In OPF modes with registered DER, the flat action
                         contains only ``der_actions`` so ndarray agents cannot
                         silently bypass the OPF by injecting
                         ``unit_power_mw``. Agents can then control DER through
                         a single flat Box.
                         DER states are always visible in ``observation_space``
                         regardless of this flag (when resources are registered).
                         Default ``False`` (DER run as non-dispatchable injections).
        """
        # ① Validate physics / solver_mode and apply difficulty preset
        delta_t_minutes, max_load_ratio = self._apply_difficulty_preset(
            difficulty, delta_t_minutes, max_load_ratio
        )

        # ② Case loading and distribution-case guard
        if case is None:
            case = load_case(5)
        if getattr(case, 'GRID_TYPE', '') == 'distribution':
            warnings.warn(
                f"Case '{type(case).__name__}' is a distribution case. "
                "TransGridEnv expects a transmission case.",
                UserWarning,
                stacklevel=2,
            )

        # ③ Base environment initialisation
        super().__init__(case=case, solver=solver, delta_t_minutes=delta_t_minutes,
                         data_loader=data_loader, start_date=start_date,
                         end_date=end_date, load_columns=load_columns,
                         max_load_ratio=max_load_ratio, min_load_ratio=min_load_ratio,
                         time_series=time_series, max_episode_steps=max_episode_steps,
                         randomize_start_time=randomize_start_time)

        # ④ Grid topology (requires case to be finalised by super().__init__)
        self.PTDF = self.case.get_node_gsdf().values  # (n_lines × n_nodes) PTDF matrix
        self.slack_bus_id = getattr(case, 'slack_bus', 0)
        self.difficulty = difficulty

        # ⑤ Solver and control configuration (validates physics / solver_mode)
        if ac_config is None:
            ac_config = ACConfig()
        self._init_solver_config(
            physics=physics, solver_mode=solver_mode, solver_type=solver_type,
            normalize_actions=normalize_actions, reward_scale=reward_scale,
            control_der=control_der, ac=ac_config,
        )

        # ⑥ Per-episode mutable state
        self._reset_episode_state()

        # ⑦ Observation / action spaces
        self._build_spaces()

        logger.info(
            "TransGridEnv: case=%s, reward_scale=%g (adjust for large cases)",
            type(self.case).__name__, self.reward_scale,
        )

    # ── Init helpers ──────────────────────────────────────────────────────────

    @classmethod
    def _apply_difficulty_preset(
        cls,
        difficulty: Optional[str],
        delta_t_minutes: float,
        max_load_ratio: float,
    ) -> tuple:
        """Apply difficulty preset, returning (delta_t_minutes, max_load_ratio).

        When ``difficulty`` is ``None``, the input values are returned unchanged.
        """
        if difficulty is None:
            return delta_t_minutes, max_load_ratio
        if difficulty not in cls._DIFFICULTY_PRESETS:
            raise ValueError(
                f"difficulty must be one of {list(cls._DIFFICULTY_PRESETS)}, "
                f"got '{difficulty}'"
            )
        preset = cls._DIFFICULTY_PRESETS[difficulty]
        return preset['delta_t_minutes'], preset['max_load_ratio']

    def _init_solver_config(
        self,
        *,
        physics: str,
        solver_mode: str,
        solver_type: str,
        normalize_actions: bool,
        reward_scale: float,
        control_der: bool,
        ac: ACConfig,
    ) -> None:
        """Validate and store solver / control configuration on ``self``."""
        if physics not in ('dc', 'ac'):
            raise ValueError(f"physics must be 'dc' or 'ac', got '{physics}'")
        if solver_mode not in ('pf', 'opf'):
            raise ValueError(f"solver_mode must be 'pf' or 'opf', got '{solver_mode}'")
        if ac.solver not in ('builtin', 'pandapower'):
            raise ValueError(
                f"acopf_solver must be 'builtin' or 'pandapower', got '{ac.solver}'"
            )

        self.physics = physics
        self.solver_mode = solver_mode
        self.solver_type = solver_type
        self.normalize_actions = normalize_actions
        self.reward_scale = reward_scale
        self.control_der = control_der

        # AC-specific parameters (unused in DC modes)
        self._ac_v_min = ac.v_min
        self._ac_v_max = ac.v_max
        self._ac_q_factor = ac.q_factor
        self._ac_backend = ac.backend
        self._acopf_solver_type = ac.solver
        self._acopf_solver = None  # lazy-init on first AC-OPF call

    def _reset_episode_state(self) -> None:
        """Initialise / clear all per-episode mutable state.

        Called from ``__init__`` and at the start of each ``reset()``.
        """
        # Power-flow result cache (overwritten by _apply_solve_result each step)
        self._lines = None
        self._nodes = None
        self._is_safe = None
        self._safety_info = None
        self._unit_power_mw = None
        self._opf_result = None
        self._pf_result = None

        # Episode metric accumulators (summed over steps, reported in info)
        self._ep_violations: int = 0
        self._ep_cost: float = 0.0

        # Power-balance tracking (non-zero in PF modes when dispatch ≠ load)
        self._power_imbalance_mw: float = 0.0
        # Slack generator bound exceedance (non-zero when slack forced outside [p_min, p_max])
        self._slack_gen_violation_mw: float = 0.0

    # ── State management ──────────────────────────────────────────────────────

    def _set_failed_state(self) -> None:
        """Reset cached results to a safe failed state (used after solver exceptions)."""
        self._opf_result = None
        self._pf_result = None
        self._unit_power_mw = None
        self._lines = None
        self._nodes = None
        self._is_safe = False
        self._power_imbalance_mw = 0.0
        self._slack_gen_violation_mw = 0.0
        self._safety_info = {'unsafe_line_ids': [], 'unsafe_line_flows': [],
                             'unsafe_line_caps': [], 'unsafe_line_floors': []}

    def _apply_solve_result(self, result: TransSolveResult) -> None:
        """Write all fields from a solver result to the corresponding self.* attributes."""
        self._unit_power_mw          = result.unit_power_mw
        self._lines                  = result.lines
        self._nodes                  = result.nodes
        self._opf_result             = result.opf_result
        self._pf_result              = result.pf_result
        self._is_safe                = result.is_safe
        self._safety_info            = result.safety_info
        self._power_imbalance_mw     = result.power_imbalance_mw
        self._slack_gen_violation_mw = result.slack_gen_violation_mw

    # ── Observation & action spaces ───────────────────────────────────────────

    def _build_spaces(self) -> None:
        """Define observation_space, action_space, obs_names, action_names.

        Observation vector layout (all normalised to ~[-1, 1] / [0, 1]):
            [line_flows / line_caps]       shape (n_lines,)
            [node_net_load_mw / p_max_sum] shape (n_loads,)
            [unit_power_mw / p_max]        shape (n_units,)
            [DER states]                   shape (n_der_obs,)  ← per registered resource
            [time_sin, time_cos]           shape (2,)

        DER obs slices come from each resource's ``grid_obs()`` / ``grid_obs_names()``.
        DER action bounds come from each resource's ``grid_action_bounds()`` / ``grid_action_from_normalized()``.

        Action vector layout:
            ``control_der=False`` (default):
                [unit_power_mw]  shape (n_units,)
            ``control_der=True``:
                ``solver_mode='pf'``:   [unit_power_mw, der_actions]
                ``solver_mode='opf'``:  [der_actions]  (OPF owns unit dispatch)
                Use ``_parse_flat_action()`` to split.

        Also called on resource register/unregister to keep spaces consistent.
        """
        n_lines = len(self.case.lines)
        n_loads = len(self.case.loads) if hasattr(self.case, 'loads') else len(self.case.nodes)
        n_units = len(self.case.units)

        # DER observation dimensions (depends on currently registered resources)
        der_obs_dim = sum(len(res.grid_obs()) for res in self.sub_resources.values())
        obs_dim = n_lines + n_loads + n_units + der_obs_dim + 2

        self.observation_space = _spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        p_min = self.case.units['p_min'].values.astype(np.float32)
        p_max = self.case.units['p_max'].values.astype(np.float32)
        self._action_phys_low = p_min.copy()
        self._action_phys_high = p_max.copy()

        # OPF+control_der: OPF owns unit dispatch, flat action carries only DER dims.
        include_unit_dispatch = not (self.solver_mode == 'opf' and self.control_der and bool(self.sub_resources))
        include_der = self.control_der and bool(self.sub_resources)

        lows, highs = [], []
        if include_unit_dispatch:
            if self.normalize_actions:
                lows.append(-np.ones(n_units, dtype=np.float32))
                highs.append(np.ones(n_units, dtype=np.float32))
            else:
                lows.append(p_min)
                highs.append(p_max)
        if include_der:
            der_low, der_high = self._der_action_bounds()
            lows.append(der_low)
            highs.append(der_high)
        self.action_space = _spaces.Box(
            low=np.concatenate(lows),
            high=np.concatenate(highs),
            dtype=np.float32)

        # Human-readable labels (for debugging / logging)
        line_ids = self.case.lines['#id'].astype(int).tolist() if '#id' in self.case.lines.columns else list(range(n_lines))
        load_ids = self.case.loads['#id'].astype(int).tolist() if (hasattr(self.case, 'loads') and '#id' in self.case.loads.columns) else list(range(n_loads))
        unit_ids = self.case.units['#id'].astype(int).tolist() if '#id' in self.case.units.columns else list(range(n_units))

        self.obs_names: List[str] = (
            [f'line_{i}_flow_norm' for i in line_ids]
            + [f'load_{i}_norm' for i in load_ids]
            + [f'unit_{i}_p_norm' for i in unit_ids]
            + [n for rid, res in self.sub_resources.items() for n in res.grid_obs_names(rid)]
            + ['time_sin', 'time_cos']
        )
        self.action_names: List[str] = (
            ([f'unit_{i}_p_mw' for i in unit_ids] if include_unit_dispatch else [])
            + ([f'{rid}_action' for rid in self.sub_resources] if include_der else [])
        )

    def _der_action_bounds(self, normalized: Optional[bool] = None):
        """Return (low, high) float32 arrays for DER flat-action dimensions.

        Defaults to ``self.normalize_actions``; pass ``True``/``False`` to override.
        """
        if normalized is None:
            normalized = self.normalize_actions

        lows, highs = [], []
        for res in self.sub_resources.values():
            if normalized:
                lows.append(-1.0)
                highs.append(1.0)
            else:
                low, high = res.grid_action_bounds()
                lows.append(low)
                highs.append(high)
        return np.array(lows, dtype=np.float32), np.array(highs, dtype=np.float32)

    def _parse_flat_action(self, action: np.ndarray) -> dict:
        """Split flat action array into ``unit_power_mw`` and per-resource actions.

        Used by ``step()`` when ``control_der=True`` and the agent submits a
        single ndarray.  In PF modes the leading ``n_units`` slice carries
        generator dispatch; in OPF modes the flat action intentionally omits
        that slice so ndarray control cannot silently bypass the OPF solver.
        DER entries are converted immediately into the child resource's
        physical action units and wrapped into the resource's dict interface.

        Returns a dict suitable for the base ``step()`` interface.
        """
        action = np.asarray(action, dtype=np.float32)
        include_unit_dispatch = not (
            self.solver_mode == 'opf' and self.control_der and bool(self.sub_resources)
        )
        n_units = len(self.case.units) if include_unit_dispatch else 0
        expected_dim = n_units + len(self.sub_resources)
        if action.shape[0] != expected_dim:
            raise ValueError(
                f"Flat action has length {action.shape[0]}, expected {expected_dim} "
                f"for solver_mode='{self.solver_mode}', control_der={self.control_der}, "
                f"n_resources={len(self.sub_resources)}"
            )

        result: dict = {}
        if n_units > 0:
            result['unit_power_mw'] = np.asarray(action[:n_units], dtype=np.float32)
        der_low, der_high = self._der_action_bounds(normalized=False)
        for i, (rid, res) in enumerate(self.sub_resources.items()):
            raw = float(action[n_units + i])
            if self.normalize_actions:
                physical = res.grid_action_from_normalized(float(np.clip(raw, -1.0, 1.0)))
            else:
                physical = float(np.clip(raw, float(der_low[i]), float(der_high[i])))
            action_names = getattr(res, 'action_names', None)
            if isinstance(action_names, list) and len(action_names) == 1:
                result[rid] = {action_names[0]: physical}
            else:
                result[rid] = physical
        return result

    def obs(self, state: Any = None) -> np.ndarray:
        """Convert current (or provided) state to a flat float32 observation array.

        Args:
            state: Internal state dict (as returned by ``_get_state``).
                   If None, the most recently cached state is used.

        Returns:
            numpy.ndarray of shape ``observation_space.shape`` and dtype float32.
        """
        if state is None:
            state = self._get_state()

        parts = []

        # 1. Line flows (normalised by capacity)
        # AC mode: use apparent power |S| = sqrt(P²+Q²) [MVA] / cap [MVA]
        # DC mode: use directed active power P [MW] / cap [MW] (Q≈0 approximation)
        if self._lines is not None and 'line_flow_mw' in self._lines.columns:
            caps = self.case.lines['cap'].values.astype(np.float32)
            caps = np.where(caps > 0, caps, 1.0)
            if self.physics == 'ac' and 'line_flow_q_mvar' in self._lines.columns:
                sf = np.sqrt(
                    self._lines['line_flow_mw'].values ** 2
                    + self._lines['line_flow_q_mvar'].values ** 2
                ).astype(np.float32)
                parts.append(sf / caps)
            else:
                parts.append((self._lines['line_flow_mw'].values / caps).astype(np.float32))
        else:
            parts.append(np.zeros(len(self.case.lines), dtype=np.float32))

        # 2. Node net loads (normalised by total p_max)
        p_max_sum = float(self.case.units['p_max'].sum()) or 1.0
        loads = self._get_default_node_load().astype(np.float32)
        parts.append(loads / p_max_sum)

        # 3. Unit power outputs (normalised by p_max)
        if self._unit_power_mw is not None:
            p_max = self.case.units['p_max'].values.astype(np.float32)
            p_max = np.where(p_max > 0, p_max, 1.0)
            parts.append((self._unit_power_mw / p_max).astype(np.float32))
        else:
            parts.append(np.zeros(len(self.case.units), dtype=np.float32))

        # 4. Per-resource observation slices (via grid_obs()), in registration order.
        for res in self.sub_resources.values():
            parts.append(res.grid_obs())

        # 5. Time encoding (cyclic, period = steps_per_day)
        phase = 2.0 * np.pi * self.time_step / max(self.steps_per_day, 1)
        parts.append(np.array([np.sin(phase), np.cos(phase)], dtype=np.float32))

        return np.concatenate(parts)

    # ── RL interface: step / reset ────────────────────────────────────────────

    def step(self, action):
        """Parse flat ndarray action when control_der=True, then delegate to base step().

        Flat layout and splitting logic are defined in ``_parse_flat_action()``.
        Dict actions and ``control_der=False`` are forwarded unchanged.
        """
        if self.control_der and isinstance(action, np.ndarray):
            action = self._parse_flat_action(action)
        return super().step(action)

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict] = None,
              day_id: Optional[int] = None) -> tuple:
        """Reset transmission grid and run initial power flow."""
        self._reset_episode_state()
        # Clear ACOPF warm-start to avoid cross-episode local-optima leakage
        if hasattr(self, '_acopf_solver') and hasattr(self._acopf_solver, 'reset_warm_start'):
            self._acopf_solver.reset_warm_start()
        super().reset(seed=seed, options=options, day_id=day_id)
        # Run initial power flow with default values
        self._run_power_flow({})
        # Update and return state
        state = self._get_state()
        info = self.build_info(state)
        return state, info

    @property
    def pf_mode(self) -> str:
        """Backward-compatible accessor.  Returns ``'ac'`` or ``'dc'``."""
        return self.physics

    # ── Solver dispatcher ─────────────────────────────────────────────────────

    def _run_power_flow(self, action) -> bool:
        """Dispatcher: route to one of four solver paths.

        * ``solver_mode='opf'`` + ``physics='dc'`` → DCOPF
        * ``solver_mode='opf'`` + ``physics='ac'`` → ACOPF
        * ``solver_mode='pf'``  + ``physics='dc'`` → DCPF
        * ``solver_mode='pf'``  + ``physics='ac'`` → ACPF (Newton-Raphson)

        Returns True on success, False if the solver fails.
        """
        # Denormalize unit_power_mw: [-1, 1] → [p_min, p_max]
        if (self.normalize_actions
                and 'unit_power_mw' in action
                and self._action_phys_low is not None):
            norm_dispatch = np.asarray(action['unit_power_mw'], dtype=np.float32)
            lo = self._action_phys_low
            hi = self._action_phys_high
            action = dict(action)
            action['unit_power_mw'] = (lo + hi) / 2 + norm_dispatch * (hi - lo) / 2

        if self.solver_mode == 'opf':
            if self.physics == 'ac':
                return self._run_power_flow_acopf(action)
            return self._run_power_flow_dcopf(action)
        else:  # solver_mode == 'pf'
            if self.physics == 'ac':
                return self._run_power_flow_acpf(action)
            return self._run_power_flow_dcpf(action)

    def _run_power_flow_dcopf(self, action) -> bool:
        """DC-OPF path — thin wrapper delegating to :func:`trans_solve.run_dcopf`.

        See :func:`powerzoo.envs.grid._trans_solve.run_dcopf` for the full
        docstring and five-section structure.

        Returns:
            True on success, False if the solver raises an exception.
        """
        try:
            result = trans_solve.run_dcopf(self, action)
            self._apply_solve_result(result)
            return True
        except Exception as e:
            logger.warning("DC-OPF failed: %s", e)
            self._set_failed_state()
            return False

    def _run_power_flow_acopf(self, action) -> bool:
        """AC-OPF path — thin wrapper delegating to :func:`trans_solve.run_acopf`.

        When ``unit_power_mw`` is present in *action* the OPF is bypassed and
        full AC power flow (Newton-Raphson) is run via
        :meth:`_run_power_flow_acpf`.  The bypass is handled here (not inside
        ``run_acopf``) so that monkeypatching ``_run_power_flow_acpf`` on the
        instance still intercepts it correctly in tests.

        Returns:
            True on success, False if the solver raises an exception.
        """
        if 'unit_power_mw' in action:
            return self._run_power_flow_acpf(action)

        self._power_imbalance_mw = 0.0
        try:
            result = trans_solve.run_acopf(self, action)
            self._apply_solve_result(result)
            return bool(result.converged)
        except Exception as e:
            logger.warning("AC-OPF failed: %s", e)
            self._set_failed_state()
            return False

    def _run_power_flow_dcpf(self, action) -> bool:
        """DCPF path — thin wrapper delegating to :func:`trans_solve.run_dcpf`.

        Returns:
            True on success, False on exception.
        """
        try:
            result = trans_solve.run_dcpf(self, action)
            self._apply_solve_result(result)
            return True
        except Exception as e:
            logger.warning("DCPF failed: %s", e)
            self._set_failed_state()
            return False

    def _run_power_flow_acpf(self, action) -> bool:
        """ACPF path — thin wrapper delegating to :func:`trans_solve.run_acpf`.

        ``_power_imbalance_mw`` is reset to 0 before the call.
        :func:`trans_solve.run_acpf` writes the pre-NR value back to
        ``env._power_imbalance_mw`` before calling the NR solver, so if NR
        raises the value is preserved here across ``_set_failed_state()``.

        Returns:
            True when NR converges, False on non-convergence or exception.
        """
        self._power_imbalance_mw = 0.0  # reset; run_acpf overwrites before NR
        try:
            result = trans_solve.run_acpf(self, action)
            self._apply_solve_result(result)
            return bool(result.converged)
        except Exception as e:
            logger.warning("ACPF failed: %s", e)
            saved_imbalance = self._power_imbalance_mw  # written by run_acpf before NR
            self._set_failed_state()
            self._power_imbalance_mw = saved_imbalance
            return False

    # ── State, reward & info ──────────────────────────────────────────────────

    def _get_episode_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Episode-level KPIs appended to info['episode']['metrics']."""
        return {
            'total_line_violations': self._ep_violations,
            'total_opf_cost': self._ep_cost,
        }

    def _get_state(self):
        """Get current state from cached power flow results."""
        state = {
            'lines': self._lines,
            'nodes': self._nodes,
            'is_safe': self._is_safe,
            'safety_info': self._safety_info,
            'time_step': self.time_step,
            'unit_power_mw': self._unit_power_mw,
            'physics': self.physics,
            'solver_mode': self.solver_mode,
            'pf_mode': self.pf_mode,  # backward compat
        }

        # Add OPF results if available (solver_mode='opf')
        if self._opf_result is not None:
            state['opf_cost'] = self._opf_result['total_cost']
            state['opf_slack'] = self._opf_result.get('slack_violation', 0.0)
            state['lmp'] = self._opf_result['lmp']  # Locational Marginal Price ($/MWh)
            state['solver_backend'] = self._opf_result.get('solver_backend')
            state['lmp_method'] = self._opf_result.get('lmp_method')
            state['lmp_quality'] = self._opf_result.get('lmp_quality')
            state['lmp_available'] = self._opf_result.get('lmp_available', False)
            # AC-mode extras
            if 'vm_pu' in self._opf_result:
                state['vm_pu'] = self._opf_result['vm_pu']
                state['va_deg'] = self._opf_result['va_deg']
                state['q_gen'] = self._opf_result['q_gen']

        # Add ACPF results if available (solver_mode='pf', physics='ac')
        if self._pf_result is not None:
            state['pf_converged'] = self._pf_result['converged']
            state['pf_iterations'] = self._pf_result['iterations']
            state['vm_pu'] = self._pf_result['vm']
            state['va_deg'] = self._pf_result['va_deg']
            state['q_gen'] = self._pf_result.get('q_gen')
            state['p_loss'] = self._pf_result.get('p_loss')

        return state

    def _compute_reward(self, state):
        """Compute scalar reward and populate state['reward_components'].

        The scalar reward carries **only** the economic objective.
        Safety diagnostics are tracked in ``reward_components`` for
        transparency but flow exclusively through the CMDP cost channel
        (``info['cost_sum']``, ``info['cost_thermal_overload']``, etc.)
        and termination signals — never through the reward scalar.

        Components recorded in ``state['reward_components']``:
            safety_diagnostic:  -10 * n_line_violations  (NOT in reward scalar)
            economic_cost:      -reward_scale * generation_cost
            der_econ_total:     reward_scale * sum(res.econ_components())
                                Resources return signed values (negative = cost).
        """
        components = {
            'safety_diagnostic': 0.0,
            'economic_cost': 0.0,
            'der_econ_total': 0.0,
        }

        if not state['is_safe']:
            n_violations = len(state['safety_info']['unsafe_line_ids'])
            components['safety_diagnostic'] = -10.0 * n_violations
            self._ep_violations += n_violations

        if 'opf_cost' in state:
            gen_cost = state['opf_cost']
        elif self._unit_power_mw is not None:
            # PF mode: total cost = integral of MC(p) = mc_a/3·p³ + mc_b/2·p² + mc_c·p
            p = self._unit_power_mw
            mc_a = self.case.units['mc_a'].values
            mc_b = self.case.units['mc_b'].values
            mc_c = self.case.units['mc_c'].values
            gen_cost = float(((mc_a / 3) * p**3 + (mc_b / 2) * p**2 + mc_c * p).sum())
        else:
            gen_cost = 0.0

        if gen_cost > 0:
            components['economic_cost'] = -self.reward_scale * gen_cost
            self._ep_cost += gen_cost

        # ── DER economic costs ──────────────────────────────────────────
        if self.sub_resources:
            dt_hours = self.delta_t_minutes / 60.0
            total_econ_raw = 0.0
            for res in self.sub_resources.values():
                for component in res.econ_components(dt_hours).values():
                    total_econ_raw += component
            if total_econ_raw != 0.0:
                components['der_econ_total'] = self.reward_scale * total_econ_raw

        state['reward_components'] = components

        return (components['economic_cost']
                + components['der_econ_total'])

    def constraint_names(self) -> Tuple[str, ...]:
        """Return the fixed benchmark cost-channel order."""
        return ('thermal_overload', 'voltage_violation', 'power_balance', 'resource')

    def build_info(self, state):
        """Build info dict with power flow, OPF, safety details, and reward components.

        Mandatory schema (F4 fix — all tasks must expose these keys):
            pf_converged            bool   - set by base GridEnv.step()
            is_safe                 bool
            cost_sum                float  - total safety cost (= cost return value);
                                            sum of thermal + voltage + power_balance costs
            cost_voltage_violation  float  - sum of voltage bound exceedances (p.u.)
            cost_thermal_overload   float  - sum of line overloads (MW)
            cost_load_shedding      float  - curtailed load (MW)
            cost_power_balance      float  - |slack bus imbalance| (MW); non-zero only
                                            in PF modes when the agent's total dispatch
                                            does not match system load.  In OPF modes
                                            the solver enforces balance so this is 0.
            cost_exception          float  - 1.0 if PF failed, else 0.0 (set by base)
            goal_met                bool   - True when no constraint violations
            episode_step            int    - set by PowerEnv.step() / base.step()

        DER-specific fields (present when sub_resources are registered):
            total_curtailment_mw    float  - total curtailed power this step (MW);
                                            computed from status()['curtailed_p_mw']
                                            across all resources (0 when absent).
            der_econ_total          float  - reward-scaled aggregate DER economic
                                            contribution (curtailment, cycle cost, etc.);
                                            sourced from econ_components() on each resource.
            slack_gen_violation_mw  float  - slack generator bound exceedance (MW);
                                            non-zero only in PF/bypass modes when the
                                            system imbalance forces the slack gen out of
                                            [p_min, p_max].
        """
        safety_info = state.get('safety_info') or {}

        # Compute cost_thermal_overload: sum of flow exceedances above cap.
        # AC mode: thermal = |S| = sqrt(P²+Q²) [MVA] vs cap [MVA].
        # DC mode: thermal = |P| [MW] vs cap [MW] (Q≈0 approximation).
        cost_thermal_overload = 0.0
        if self._lines is not None and 'line_flow_mw' in self._lines.columns:
            caps = self.case.lines['cap'].values
            if self.physics == 'ac' and 'line_flow_q_mvar' in self._lines.columns:
                sf = np.sqrt(
                    self._lines['line_flow_mw'].values ** 2
                    + self._lines['line_flow_q_mvar'].values ** 2
                )
                # Unlimited branches (cap == 0) cannot be overloaded
                effective_cap = np.where(caps > 0, caps, np.inf)
                cost_thermal_overload = float(np.sum(np.maximum(0.0, sf - effective_cap)))
            else:
                flows = self._lines['line_flow_mw'].values
                cost_thermal_overload = float(np.sum(np.maximum(0.0, np.abs(flows) - caps)))

        # Voltage violation cost (DC mode has no voltages → 0; AC mode: sum of |V-Vlim| excess)
        cost_voltage_violation = 0.0
        if self._nodes is not None and 'vm_pu' in self._nodes.columns:
            vm = self._nodes['vm_pu'].values
            cost_voltage_violation = float(
                np.sum(np.maximum(0.0, vm - self._ac_v_max))
                + np.sum(np.maximum(0.0, self._ac_v_min - vm))
            )

        # Power balance violation: slack bus imbalance in PF modes
        cost_power_balance = self._power_imbalance_mw
        cost_sum = cost_thermal_overload + cost_voltage_violation + cost_power_balance

        # DER economic metrics
        rc = state.get('reward_components', {})
        total_curtailment_mw = 0.0
        if self.sub_resources:
            for res in self.sub_resources.values():
                s = res.status()
                total_curtailment_mw += max(0.0, s.get('curtailed_p_mw', 0.0))

        # pf_converged / cost_exception injected by base GridEnv.step()
        info = {
            # Mandatory standard fields
            'is_safe': bool(state.get('is_safe', True)),
            'cost_sum': cost_sum,
            'cost_voltage_violation': cost_voltage_violation,
            'cost_thermal_overload': cost_thermal_overload,
            'cost_load_shedding': 0.0,
            'cost_power_balance': cost_power_balance,
            'cost_resource': 0.0,
            'goal_met': bool(state.get('is_safe', True)),
            # Extended fields
            'physics': state.get('physics', self.physics),
            'solver_mode': state.get('solver_mode', self.solver_mode),
            'pf_mode': state.get('pf_mode', self.pf_mode),  # backward compat
            'safety_info': safety_info,
            'resource_status': {rid: res.status() for rid, res in self.sub_resources.items()},
            'reward_components': rc,
            # DER economic diagnostics
            'total_curtailment_mw': total_curtailment_mw,
            'der_econ_total': rc.get('der_econ_total', 0.0),
            'slack_gen_violation_mw': self._slack_gen_violation_mw,
        }

        # line_viol_mva: MVA overload diagnostic from ACOPF solver (matches JAX info key)
        safety_info_inner = state.get('safety_info') or {}
        info['line_viol_mva'] = float(safety_info_inner.get('line_viol_mva', 0.0))

        if 'opf_cost' in state:
            info['opf_cost'] = state['opf_cost']
            info['opf_slack'] = state['opf_slack']

        if self._unit_power_mw is not None:
            info['unit_power_mw'] = self._unit_power_mw
            info['total_generation_mw'] = float(self._unit_power_mw.sum())

        for key in ('lmp', 'solver_backend', 'lmp_method', 'lmp_quality'):
            if key in state:
                info[key] = state[key]
        if 'lmp_available' in state:
            info['lmp_available'] = bool(state['lmp_available'])

        return self.attach_constraint_costs(info)

    # ── Thin wrappers (names kept for test compatibility) ─────────────────────

    def cal_pf(self, unit_power_mw, node_load_mw, df=False):
        """Thin wrapper — see :func:`trans_solve.cal_pf`.

        Side effects: updates ``self._power_imbalance_mw`` and
        ``self._slack_gen_violation_mw``.
        """
        return trans_solve.cal_pf(self, unit_power_mw, node_load_mw, df=df)

    def safety_check(self, line_flow_mw, with_info=False):
        """Thin wrapper — see :func:`trans_solve.safety_check`."""
        return trans_solve.safety_check(self, line_flow_mw, with_info=with_info)

    def _proportional_dispatch(self, total_load_mw: float) -> np.ndarray:
        """Thin wrapper — see :func:`trans_solve.proportional_dispatch`."""
        return trans_solve.proportional_dispatch(self, total_load_mw)

    def _get_slack_unit_mask(self) -> np.ndarray:
        """Thin wrapper — see :func:`trans_solve.get_slack_unit_mask`."""
        return trans_solve.get_slack_unit_mask(self)

    def _compute_slack_gen_violation(self, slack_gen_mw: float,
                                     slack_unit_mask: Optional[np.ndarray] = None) -> float:
        """Thin wrapper — see :func:`trans_solve.compute_slack_gen_violation`."""
        return trans_solve.compute_slack_gen_violation(self, slack_gen_mw, slack_unit_mask)

    def _calculate_node_net_load(self, node_load_mw: Optional[np.ndarray] = None) -> np.ndarray:
        """Thin wrapper — see :func:`trans_solve.calculate_node_net_load`."""
        return trans_solve.calculate_node_net_load(self, node_load_mw)

    def _get_default_node_load(self):
        """Thin wrapper — see :func:`trans_solve.get_default_node_load`."""
        return trans_solve.get_default_node_load(self)

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, mode: str = 'human'):
        """Render the transmission grid state.

        Produces a two-panel figure (network topology + unit dispatch chart).
        See :func:`powerzoo.envs.grid._render.render_trans_grid` for details.

        Args:
            mode: ``'human'`` (interactive) or ``'rgb_array'`` (returns
                  a ``(H, W, 3)`` uint8 ndarray without opening a window).
        """
        from powerzoo.envs.grid._render import render_trans_grid
        return render_trans_grid(self, mode)
