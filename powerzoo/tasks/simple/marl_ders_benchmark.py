"""DERs Main-Config Benchmark Task (PowerZoo Python reference)

Matches the PowerZooJax DERs main configuration for fair speed comparison:

    Grid      : Case141  (141-bus Caracas distribution, single-phase balanced)
    Horizon   : 48 steps × 30 min = 24 hours
    Agents    : 12 heterogeneous
                  4 × Battery  (buses 9, 55, 17, 122) — action [P, Q]
                  4 × Solar PV (buses 6, 73, 72, 82) — action [curtail, q]
                  4 × FlexLoad (buses 41, 70, 135, 24) — action [curtail, shift_out]
    Per-agent action dim : 2 (all types)
    Obs mode  : ders_local — 12-dim, type-specific real device state
                  Battery [7-11]: [soc, p_norm, q_norm, soc_headroom, soc_floor]
                  PV      [7-11]: [available_cf, p_norm, q_norm, curt_norm, 0]
                  FlexLoad[7-11]: [curt_n, sout_n, sin_n, buf_fill, buf_en]
    Voltage   : v_min=0.94, v_max=1.06 p.u.

Residual mismatches vs. PowerZooJax (documented, not masked):
1. Obs dim: 12 (ders_local) vs. JAX 15-dim; shared-context fields differ
   (Python: price + episode_progress; JAX: neighbor voltages + global stats).
   Within the 5-dim device slot the semantics are aligned.
2. Reward: base env reward is loss-based (network loss), while the cross-
   backend bridge adds the JAX benchmark's voltage-penalty shaping from the
   exact episode rollout state.  The per-agent CMDP channel still exposes
   voltage violation as info["cost_voltage_violation"].
3. No JAX JIT / vmap: this env runs on CPU with Python control flow.
4. Load profile shape: JAX load_profiles_p is (max_steps, n_all_buses) [p.u.];
   Python grid._node_loads_p is (T, n_load_buses) absolute [MW].  Values are
   equivalent but arrays differ in scope.  Users supplying real traces should
   provide (n_steps, n_load_buses) MW arrays to inject_load_profiles().

These gaps are acceptable for task-setting capability alignment.  The
purpose is to verify that both sides can instantiate the SAME scenario
(grid + agents + horizon) as a prerequisite for fair speed comparison.

Profile injection
-----------------
Use :func:`inject_pv_profiles` / :func:`inject_load_profiles` to override
PV capacity-factor and grid load arrays after ``create_env()`` — this
mirrors PowerZooJax's ``make_ders_params_with_profiles()`` interface.

Convenience factory :func:`make_ders_benchmark_env_with_profiles` accepts all
three profile types (``pv_profiles``, ``load_profiles_p``, ``load_profiles_q``)
and can be called with any combination (all optional).
"""

from typing import Any, Dict, Optional

from powerzoo.tasks.base import ConstraintSpec, MultiAgentTask
from powerzoo.tasks.observation import make_observation_config


# ── Canonical bus placements (mirrors powerzoojax.tasks.ders constants) ───
DERS_BATTERY_BUSES = [9, 55, 17, 122]
DERS_PV_BUSES      = [6, 73, 72, 82]
DERS_FLEXLOAD_BUSES = [41, 70, 135, 24]

DERS_V_MIN = 0.94
DERS_V_MAX = 1.06


def _battery_resource(i: int, bus_id: int) -> Dict[str, Any]:
    return {
        'type': 'battery',
        'name': f'bat_{i}',
        'bus_id': bus_id,
        'power_mw': 0.10,
        'capacity_mwh': 0.30,
        's_rated_mva': 0.15,
        'enable_q_control': True,
        'soc_min': 0.1,
        'soc_max': 0.9,
        'initial_soc': 0.5,
        'efficiency': 0.95,
    }


def _solar_resource(i: int, bus_id: int) -> Dict[str, Any]:
    return {
        'type': 'solar',
        'name': f'pv_{i}',
        'bus_id': bus_id,
        'capacity_mw': 0.20,
        's_rated_mva': 0.22,
        'enable_q_control': True,
        'normalize_actions': True,
    }


def _flexload_resource(i: int, bus_id: int) -> Dict[str, Any]:
    return {
        'type': 'flexload',
        'name': f'fl_{i}',
        'bus_id': bus_id,
        'curtail_cap_mw': 0.10,
        'shift_cap_mw': 0.10,
        'shift_horizon': 4,
        'action_scale': 'unit',
    }


class MARLDERBenchmarkTask(MultiAgentTask):
    """12-agent DERs benchmark task on Case141.

    Matches PowerZooJax ``make_ders_marl_env()`` / ``ders-medium`` preset for
    scenario-level comparison (grid + agents + horizon + voltage focus).
    """

    name = "marl_ders_benchmark"
    description = (
        "12-agent heterogeneous DERs benchmark (4 Battery + 4 PV + 4 FlexLoad) "
        "on Case141, 48×30min episode.  Matches PowerZooJax DERs main config."
    )
    difficulty = "middle"
    agent_mode = "multi"
    training_contract = "cmdp_env_plus_mdp_fallback"

    _TIGHTNESS_PRESETS: Dict[str, Any] = {
        'loose':    {'v_min': 0.92, 'v_max': 1.08, 'cost_threshold': 2.0},
        'standard': {'v_min': 0.94, 'v_max': 1.06, 'cost_threshold': 0.5},
        'strict':   {'v_min': 0.96, 'v_max': 1.04, 'cost_threshold': 0.1},
    }

    eval_protocol: Dict[str, Any] = {
        "n_episodes": 100,
        "seed_start": 42,
        "split": "test",
        "constraint_names": ["voltage_violation", "thermal_overload", "resource"],
        "cost_thresholds": [0.25, 0.125, 0.125],
        "cost_threshold": 0.5,
        "metrics": [
            "mean_reward", "std_reward",
            "constraint_violation_rate",
            "mean_episode_cost", "cost_violation_rate",
            "mean_voltage_violation_steps",
        ],
    }

    # Default voltage_penalty matches PowerZooJax DERs dress-rehearsal value
    # (``benchmarks/ders/configs/train_ippo.json::voltage_penalty=4.0``).
    # Cross-backend records are only fair when both backends use the same
    # weight; the cross-backend driver passes this explicitly via
    # ``task_mapping.py::ders.powerzoo_factory_kwargs``.  Pre-2026-04 default
    # was 8.0 which left the SB3 reward 2x larger than the JAX side.
    DEFAULT_VOLTAGE_PENALTY: float = 4.0

    def __init__(
        self,
        case: str = 'Case141',
        split: Optional[str] = 'train',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        delta_t_minutes: int = 30,
        max_steps: int = 48,
        observation_mode: str = 'ders_local',
        constraint_tightness: str = 'standard',
        voltage_penalty: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(constraint_tightness=constraint_tightness, **kwargs)

        from powerzoo.tasks.simple.marl_der_arbitrage import MARLDERArbitrageTask
        SPLIT_DATES = MARLDERArbitrageTask.SPLIT_DATES

        if start_date is None and end_date is None:
            if split not in SPLIT_DATES:
                raise ValueError(f"split must be one of {list(SPLIT_DATES)}, got '{split}'")
            start_date, end_date = SPLIT_DATES[split]
        elif start_date is None or end_date is None:
            raise ValueError("Provide both start_date and end_date, or neither.")

        self._case = case
        self._split = split
        self._start_date = start_date
        self._end_date = end_date
        self._delta_t_minutes = delta_t_minutes
        self._max_steps = max_steps
        self._observation_mode = observation_mode
        self._v_min = self._tightness_param('v_min', default=DERS_V_MIN)
        self._v_max = self._tightness_param('v_max', default=DERS_V_MAX)
        self._voltage_penalty: float = (
            float(voltage_penalty)
            if voltage_penalty is not None
            else self.DEFAULT_VOLTAGE_PENALTY
        )

    def constraint_spec(self) -> ConstraintSpec:
        return ConstraintSpec(
            selected_names=("voltage_violation", "thermal_overload", "resource"),
            thresholds=(0.25, 0.125, 0.125),
            fallback_weights=(4.0, 1.0, 1.0),
        )

    def get_scenario_config(self) -> Dict[str, Any]:
        resources = (
            [_battery_resource(i, bus) for i, bus in enumerate(DERS_BATTERY_BUSES)]
            + [_solar_resource(i, bus) for i, bus in enumerate(DERS_PV_BUSES)]
            + [_flexload_resource(i, bus) for i, bus in enumerate(DERS_FLEXLOAD_BUSES)]
        )
        return {
            'name': f'{self.name}_scenario',
            'description': self.description,
            'grid': {
                'type': 'distribution',
                'case': self._case,
                'start_date': self._start_date,
                'end_date': self._end_date,
                'delta_t_minutes': self._delta_t_minutes,
                'v_min': self._v_min,
                'v_max': self._v_max,
            },
            'resources': resources,
            'reward': {
                'type': 'network_loss',
                'loss_penalty_weight': 0.1,
                'v_min': self._v_min,
                'v_max': self._v_max,
                'voltage_penalty': self._voltage_penalty,
            },
            'episode': {'max_steps': self._max_steps},
        }

    def get_agents_config(self) -> Dict[str, Any]:
        return {
            'agent_type': 'resource',
            # Accept all three resource types — this is the key fix
            'resource_filter': ['battery', 'solar', 'flexload'],
            'reward_type': 'shared',
            'observation': make_observation_config(
                mode=self._observation_mode,
                supported_modes=('ders_local', 'local_plus_voltage', 'global', 'local'),
                global_features=('total_load_mw', 'voltage_summary'),
                local_features=(
                    'time_of_day', 'hour_norm', 'price_signal', 'is_peak',
                    'is_offpeak', 'episode_progress', 'local_bus_voltage',
                    'ders_state_0', 'ders_state_1', 'ders_state_2',
                    'ders_state_3', 'ders_state_4',
                ),
                forecast_features=(),
                forecast_horizon_steps=0,
            ),
            'action': {'type': 'continuous', 'mode': 'direct'},
            'constraints': {
                'v_min': self._v_min,
                'v_max': self._v_max,
                'penalty_weight': self._voltage_penalty,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Profile injection helpers (mirrors JAX make_ders_params_with_profiles)
# ─────────────────────────────────────────────────────────────────────────────

def inject_pv_profiles(env: Any, pv_profiles: Any) -> None:
    """Inject external PV capacity-factor profiles into an existing benchmark env.

    Replaces each PV resource's ``_available_cf`` array so that
    ``resource.available_cf`` returns the injected values during rollout.
    This mirrors PowerZooJax's ``make_ders_params_with_profiles()`` interface.

    Parameters
    ----------
    env : TaskResourceMultiAgentEnv
        Environment returned by ``MARLDERBenchmarkTask().create_env()``.
    pv_profiles : array-like, shape ``(n_steps,)`` or ``(n_steps, n_pv)``
        Capacity factor profiles for the PV agents, in the order they appear
        in ``env.possible_agents``.  Values are clipped to ``[0, 1]``.

    Raises
    ------
    ValueError
        If the number of profile columns does not match the number of PV agents.
    """
    import numpy as np
    pv_agents = [a for a in env.possible_agents if 'pv' in a]
    profiles = np.asarray(pv_profiles, dtype=np.float32)
    if profiles.ndim == 1:
        profiles = profiles[:, np.newaxis]
    n_pv = len(pv_agents)
    if profiles.shape[1] != n_pv:
        raise ValueError(
            f"pv_profiles has {profiles.shape[1]} column(s) but env has "
            f"{n_pv} PV agent(s): {pv_agents}"
        )
    n_profile_steps = profiles.shape[0]
    for col_idx, agent in enumerate(pv_agents):
        resource = env._resources[agent]
        col = np.clip(profiles[:, col_idx], 0.0, 1.0)
        # _available_cf covers the full time-series range (all days in the split),
        # which is typically much longer than one episode.  Tile the injected
        # profile to fill the entire array so that any episode start position
        # (day_id * steps_per_day + time_step) returns the correct value.
        current_len = len(resource._available_cf) if resource._available_cf is not None else n_profile_steps
        if current_len <= n_profile_steps:
            resource._available_cf = col[:current_len]
        else:
            reps = -(-current_len // n_profile_steps)   # ceiling division
            tiled = np.tile(col, reps)[:current_len]
            resource._available_cf = tiled


def inject_load_profiles(
    env: Any,
    load_profiles_p: Any,
    load_profiles_q: Any = None,
) -> None:
    """Inject external active (and optionally reactive) load profiles.

    Replaces ``grid._node_loads_p`` (and optionally ``grid._node_loads_q``)
    so that subsequent episodes draw loads from the injected traces.  This
    is the Python-side equivalent of supplying ``load_profiles_p`` /
    ``load_profiles_q`` to JAX's ``make_ders_params_with_profiles()``.

    The injected profile is tiled to cover the full underlying time-series
    length so that episodes starting at any ``day_id`` still get valid data.

    Parameters
    ----------
    env : TaskResourceMultiAgentEnv
        Environment returned by ``MARLDERBenchmarkTask().create_env()``.
    load_profiles_p : array-like, shape ``(n_steps,)`` or ``(n_steps, n_load_buses)``
        Active load in MW.

        * **1-D** ``(n_steps,)`` — total MW per step, distributed proportionally
          across load buses using fractions derived from the existing
          ``grid._node_loads_p`` matrix (median load-bus fractions).
        * **2-D** ``(n_steps, n_load_buses)`` — per-bus absolute load in MW.
          ``n_load_buses`` must equal ``grid._node_loads_p.shape[1]``.

    load_profiles_q : array-like or None, shape ``(n_steps,)`` or ``(n_steps, n_load_buses)``
        Reactive load in MVAr.  Same shape rules as ``load_profiles_p``.
        ``None`` (default) leaves the existing power-factor-derived Q unchanged.

    Raises
    ------
    ValueError
        If the column count of a 2-D input does not match the grid's load bus
        count, or if a 1-D input is passed when the grid has no existing load
        matrix to derive bus fractions from.

    Notes
    -----
    Grid access: ``env.grid._node_loads_p`` / ``env.grid._node_loads_q``.
    For Q: if the grid currently has no explicit Q series (``_node_loads_q``
    is ``None``), the injected Q becomes a new per-bus series; subsequent
    steps will use it instead of the power-factor fallback.
    """
    import numpy as np

    grid = env.grid

    # ── helpers ──────────────────────────────────────────────────────────────

    def _tile_to_len(arr2d: np.ndarray, target_len: int) -> np.ndarray:
        """Tile row-wise until arr2d has at least target_len rows."""
        n = len(arr2d)
        if target_len <= n:
            return arr2d[:target_len]
        reps = -(-target_len // n)           # ceiling division
        return np.tile(arr2d, (reps, 1))[:target_len]

    def _bus_fractions(ref_matrix: np.ndarray) -> np.ndarray:
        """Median per-bus load fractions from reference matrix (n_load_buses,)."""
        row_totals = ref_matrix.sum(axis=1, keepdims=True)
        safe = np.where(row_totals > 0, row_totals, 1.0)
        fracs = ref_matrix / safe
        avg = np.median(fracs, axis=0)       # robust to zero-load hours
        s = avg.sum()
        return avg / s if s > 0 else np.ones(ref_matrix.shape[1]) / ref_matrix.shape[1]

    def _process(
        profile: Any,
        ref_array: Any,
        label: str,
    ) -> np.ndarray:
        arr = np.asarray(profile, dtype=np.float32)
        if arr.ndim == 1:
            if ref_array is None:
                raise ValueError(
                    f"{label}: cannot auto-distribute a 1-D profile when "
                    "grid._node_loads_p is None.  Pass a 2-D array instead."
                )
            fracs = _bus_fractions(ref_array.astype(np.float64))
            arr2d = arr[:, np.newaxis] * fracs[np.newaxis, :]
        elif arr.ndim == 2:
            arr2d = arr
            if ref_array is not None:
                expected = ref_array.shape[1]
                if arr2d.shape[1] != expected:
                    raise ValueError(
                        f"{label} has {arr2d.shape[1]} column(s) but grid has "
                        f"{expected} load bus(es). "
                        f"Pass shape (n_steps, {expected}) or shape (n_steps,) "
                        "for proportional auto-distribution."
                    )
        else:
            raise ValueError(
                f"{label} must be 1-D or 2-D, got shape {arr.shape}"
            )
        # Tile to the full stored time-series length
        target = len(ref_array) if ref_array is not None else len(arr2d)
        return _tile_to_len(arr2d, target).astype(np.float32)

    # ── inject P ─────────────────────────────────────────────────────────────
    grid._node_loads_p = _process(load_profiles_p, grid._node_loads_p, "load_profiles_p")

    # ── inject Q (optional) ──────────────────────────────────────────────────
    if load_profiles_q is not None:
        # Use existing Q matrix as shape reference; fall back to injected P shape
        q_ref = grid._node_loads_q if grid._node_loads_q is not None else grid._node_loads_p
        grid._node_loads_q = _process(load_profiles_q, q_ref, "load_profiles_q")


def make_ders_benchmark_env_with_profiles(
    pv_profiles: Any = None,
    load_profiles_p: Any = None,
    load_profiles_q: Any = None,
    split: str = 'train',
    **kwargs: Any,
) -> Any:
    """Create a DERs benchmark env with pre-loaded external profiles.

    Convenience wrapper that mirrors JAX's ``make_ders_params_with_profiles()``.
    All profile arguments are optional; pass any combination.

    Parameters
    ----------
    pv_profiles : array-like or None, shape ``(n_steps, 4)``
        PV capacity-factor profiles for the 4 PV agents.  Values in ``[0, 1]``.
        ``None`` → synthetic bell-curve profile (JAX default behaviour).
    load_profiles_p : array-like or None, shape ``(n_steps,)`` or ``(n_steps, n_load_buses)``
        Active load in MW.  ``None`` → flat base-load from case data.
    load_profiles_q : array-like or None, shape ``(n_steps,)`` or ``(n_steps, n_load_buses)``
        Reactive load in MVAr.  ``None`` → power-factor-derived from P.
        Ignored unless ``load_profiles_p`` is also provided.
    split : str
        Data split — ``'train'``, ``'val'``, or ``'test'``.
    **kwargs
        Forwarded to :class:`MARLDERBenchmarkTask`.

    Returns
    -------
    TaskResourceMultiAgentEnv
        12-agent env with the requested profiles pre-injected.

    Raises
    ------
    ValueError
        If ``load_profiles_q`` is supplied without ``load_profiles_p``.

    Example::

        import numpy as np
        from powerzoo.tasks.simple.marl_ders_benchmark import (
            make_ders_benchmark_env_with_profiles,
        )

        # Mirror JAX: same pv + load traces on both sides
        env = make_ders_benchmark_env_with_profiles(
            pv_profiles=np.ones((48, 4)),
            load_profiles_p=np.full(48, 5.0),    # 5 MW aggregate
            split='test',
        )
        obs, _ = env.reset(seed=0)
    """
    if load_profiles_q is not None and load_profiles_p is None:
        raise ValueError(
            "load_profiles_q requires load_profiles_p to also be provided."
        )
    task = MARLDERBenchmarkTask(split=split, **kwargs)
    env = task.create_env()
    if pv_profiles is not None:
        inject_pv_profiles(env, pv_profiles)
    if load_profiles_p is not None:
        inject_load_profiles(env, load_profiles_p, load_profiles_q)
    return env
