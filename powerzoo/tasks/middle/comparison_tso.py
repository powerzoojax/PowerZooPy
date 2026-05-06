"""TSO comparison task — PowerZoo side.

Single-agent Gymnasium env wrapping Case118 UC for cross-backend comparison
with PowerZooJax.

Contract (matches JAX ``TSO_COMPARISON_SCHEMA``):
    case_id          : Case118  (54 generators, 118 buses)
    n_agents         : 1  (centralized)
    max_steps        : 48
    delta_t_minutes  : 30
    load_source      : real GB demand + wind + solar from
                       GB_Forecast_Actual_Demand_2023_2025_30min and
                       GB_Gen_by_Type_2016_2025_30min, sliced by
                       (split, episode_start).  Identical formula on
                       PowerZooJax side; locked by
                       tests/benchmarks/test_tso_comparison_parity.py.
    enable_uc        : True
    enable_reserve   : True  (reserve_margin_frac = 0.05)
    action_space     : Box(108,) in [-1, 1]
    action_layout    : [commit_intent(54) | dispatch(54)]
    action_semantics : commit_intent > 0 → unit ON
    info keys        : gen_cost, startup_cost, no_load_cost, reserve_shortfall, cost

Accepted implementation gaps vs JAX side (irreducible, must be documented in
paper supplementary, do NOT silently bridge):
    obs_shape   : JAX (406,) vs Python (249,) — different state encoding
    solver      : JAX DC-OPF vs Python score-based allocation
    reserve_cost: JAX CMDP cost vs Python info-only
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium
from gymnasium import spaces

from powerzoo.tasks.base import ConstraintSpec
from powerzoo.tasks.adapters.uc import TaskUCMultiAgentEnv
from powerzoo.tasks.middle.marl_uc import MARLUCTask


# ---------------------------------------------------------------------------
# GB net-load trace — must agree byte-for-byte with PowerZooJax sibling
# ``powerzoojax.tasks.tso.make_comparison_tso_load_trace``.
# ---------------------------------------------------------------------------

# Mirror of PowerZooJax ``powerzoojax.data.splits``.  Cross-backend split
# alignment is enforced by ``tests/benchmarks/test_split_alignment.py`` for
# Ausgrid; the GB windows below are checked by
# ``tests/benchmarks/test_tso_comparison_parity.py``.
_GB_SPLIT_DATES: Dict[str, Tuple[str, str]] = {
    "train": ("2025-04-01", "2025-12-31"),
    "iid":   ("2026-01-01", "2026-03-31"),
}

_SPLIT_SETTINGS: Dict[str, Tuple[str, float, float]] = {
    # split -> (GB data window, load_scale, line_rating_scale)
    "train": ("train", 1.0, 1.0),
    "iid": ("iid", 1.0, 1.0),
    "load_stress": ("iid", 1.15, 1.0),
    "line_tightening": ("iid", 1.0, 0.85),
}


def _make_gb_load_trace(
    split: str = "train",
    episode_start_idx: int = 0,
    n_steps: int = 48,
    wind_frac: float = 0.7,
    solar_frac: float = 0.5,
    data_loader: Optional[Any] = None,
) -> np.ndarray:
    """Real GB net-load trace, peak-normalised, identical to PowerZooJax.

    Shape: ``(n_steps,)`` float32 in ``[0, 1]``.
    """
    if split not in _SPLIT_SETTINGS:
        raise ValueError(
            f"comparison_tso load trace supports split in {sorted(_SPLIT_SETTINGS)}, "
            f"got {split!r}.  Extend both sides together."
        )
    gb_split, load_scale, _line_rating_scale = _SPLIT_SETTINGS[split]
    start, end = _GB_SPLIT_DATES[gb_split]

    if data_loader is None:
        from powerzoo.data.data_loader import DataLoader
        data_loader = DataLoader()

    from powerzoo.data import signals as S
    df = data_loader.load_signals(
        [S.LOAD_ACTUAL_MW, S.WIND_AVAILABLE_MW, S.SOLAR_AVAILABLE_MW],
        start_date=start,
        end_date=end,
        resample="30min",
    )
    raw = np.column_stack([
        df[S.LOAD_ACTUAL_MW].to_numpy(dtype=np.float64),
        df[S.WIND_AVAILABLE_MW].to_numpy(dtype=np.float64),
        df[S.SOLAR_AVAILABLE_MW].to_numpy(dtype=np.float64),
    ])
    t0 = int(episode_start_idx)
    t1 = t0 + int(n_steps)
    if raw.shape[0] < t1:
        raise RuntimeError(
            f"GB parquet returned {raw.shape[0]} rows but episode_start_idx "
            f"({t0}) + n_steps ({n_steps}) requires {t1} (split={split!r}, "
            f"window={start}..{end})."
        )
    demand = raw[t0:t1, 0].astype(np.float32)
    wind = raw[t0:t1, 1].astype(np.float32)
    solar = raw[t0:t1, 2].astype(np.float32)

    # Identical formula to PowerZooJax `make_comparison_tso_load_trace`
    # (which mirrors the per-series peak-normalisation used by
    # `make_tso_net_load_profiles`).  Clamp net-load to [0.05, 1.0] so a
    # degenerate window (wind oversupply) cannot collapse the trace.
    gross = demand / max(float(demand.max()), 1.0)
    wind_n = wind / max(float(wind.max()), 1.0)
    solar_n = solar / max(float(solar.max()), 1.0)
    net_norm = np.clip(
        gross - wind_frac * wind_n - solar_frac * solar_n, 0.05, 1.0
    )
    return (net_norm * float(load_scale)).astype(np.float32)


def _comparison_tso_synthetic_trace(
    n_steps: int = 48,
    delta_t_hours: float = 0.5,
) -> np.ndarray:
    """Sin-based deterministic trace (test/CI helper only).

    Reserved for unit tests that must run without GB parquet on disk.
    NOT used by the cross-backend comparison env.
    """
    t = np.arange(n_steps, dtype=np.float64) * delta_t_hours
    phase = 2.0 * np.pi * (t - 3.0) / 24.0
    return np.clip(0.75 + 0.25 * np.sin(phase), 0.10, 1.00).astype(np.float32)


# ---------------------------------------------------------------------------
# Centralized env
# ---------------------------------------------------------------------------

class CentralizedComparisonTSOEnv(gymnasium.Env):
    """Single-agent Gymnasium env wrapping TaskUCMultiAgentEnv on Case118.

    Centralizes 54 MARL agents into one agent with Box(108,) action:
        action[:54]  = commit_intent  (> 0 → ON)
        action[54:]  = dispatch score (scaled to [0, 1] internally)

    UC parameters are loaded from Case118 columns:
        startup_cost      ← init_start_up_cost
        ramp_rate (MW/s)  ← ramp_up * p_max * delta_t_hours
        no_load_cost      ← init_no_load_cost (per step, while unit is ON)
        shutdown_cost     = 0  (JAX has no_load_cost instead)
        min_up/down_time  ← from Case118 directly

    Load is injected via monkeypatch of grid._get_node_loads_p_current.
    """

    metadata = {"render_modes": []}

    # Same as PowerZooJax `make_tso_case118_params(reward_scale=1e-4)` in
    # ``powerzoojax/tasks/tso.py``: the SB3-facing reward signal is
    # ``-reward_scale * (gen_cost + startup_cost + no_load_cost)``.
    # Keeping the same scale on both backends means a hyperparameter set
    # tuned on one side ports without retuning, AND the SB3 reward
    # ep_rew_mean is meaningfully comparable cross-backend.  The
    # underlying ``info["gen_cost"] / startup_cost / no_load_cost"]`` are
    # always reported in raw USD on both sides, so the
    # ``total_operating_cost`` paper-table metric is unaffected by this
    # scaling choice.
    REWARD_SCALE: float = 1e-4

    def constraint_names(self) -> Tuple[str, ...]:
        """Return the benchmark TSO CMDP channels exposed by this env."""
        return ('thermal_overload', 'reserve_shortfall')

    def __init__(self, task: "CentralizedComparisonTSOTask"):
        super().__init__()

        self._task = task
        self._delta_t_hours: float = task._delta_t_minutes / 60.0
        self._max_steps: int = task._max_steps
        self._reserve_margin_frac: float = task._reserve_margin_frac
        self._split: str = getattr(task, "_split", "train") or "train"
        self._episode_start_idx: int = int(getattr(task, "_episode_start_idx", 0))
        if self._split not in _SPLIT_SETTINGS:
            raise ValueError(
                f"comparison_tso supports split in {sorted(_SPLIT_SETTINGS)}, "
                f"got {self._split!r}. Extend both sides together."
            )
        _gb_split, _load_scale, self._line_rating_scale = _SPLIT_SETTINGS[self._split]

        # Build inner MARL env
        self._inner = TaskUCMultiAgentEnv(task)
        self._n_units: int = self._inner.n_units  # 54

        # Override UC params from Case118 columns
        units = self._inner.units
        self._apply_line_rating_scale()
        self._inner.startup_cost = units["init_start_up_cost"].values.astype(np.float32)
        self._inner.shutdown_cost = np.zeros(self._n_units, dtype=np.float32)
        # ramp_up is a per-unit fraction per hour → MW/step = ramp_up * p_max * dt_h
        self._inner.ramp_rate = (
            units["ramp_up"].values * units["p_max"].values * self._delta_t_hours
        ).astype(np.float32)
        # no_load_cost: cost incurred each step a unit is ON
        self._no_load_cost_per_step = units["init_no_load_cost"].values.astype(np.float32)

        # Inject GB real load (re-injected after each reset).
        self._inject_gb_loads()

        # Action space: Box(108,) in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2 * self._n_units,), dtype=np.float32,
        )

        # Observation space: same dim as first inner agent
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._inner._obs_dim,), dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # GB real load injection (mirrors PowerZooJax cross-backend contract)
    # ------------------------------------------------------------------

    def _apply_line_rating_scale(self) -> None:
        scale = float(self._line_rating_scale)
        if abs(scale - 1.0) <= 1e-6:
            return
        lines = self._inner.case.lines.copy()
        for col in ("cap", "rateA"):
            if col in lines.columns:
                lines[col] = lines[col].astype(float) * scale
        if "floor" in lines.columns:
            lines["floor"] = lines["floor"].astype(float) * scale
        self._inner.case.lines = lines

    def _inject_gb_loads(self) -> None:
        grid = self._inner.grid
        d_max = self._inner.case.loads["d_max"].values.astype(np.float64)
        trace = _make_gb_load_trace(
            split=self._split,
            episode_start_idx=self._episode_start_idx,
            n_steps=self._max_steps,
        )
        load_matrix = (trace[:, None] * d_max[None, :]).astype(np.float32)

        def _gb_node_loads() -> np.ndarray:
            t = int(getattr(grid, "time_step", 0))
            t = min(t, len(load_matrix) - 1)
            return load_matrix[t]

        grid._get_node_loads_p_current = _gb_node_loads

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        obs_dict, _ = self._inner.reset(seed=seed, options=options)
        # Re-inject after reset (base env may recreate grid internals).
        self._inject_gb_loads()
        obs = obs_dict[self._inner.possible_agents[0]]
        return obs.astype(np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        commit_intent = action[: self._n_units]
        dispatch = action[self._n_units :]

        # commit_intent > 0 → on_off signal = 1.0 (≥ 0.5 threshold in inner env)
        on_off = (commit_intent > 0.0).astype(np.float32)
        # dispatch ∈ [-1,1] → score ∈ [0,1]
        score = (dispatch + 1.0) * 0.5

        action_dict = {
            f"unit_{i}": np.array([score[i], on_off[i]], dtype=np.float32)
            for i in range(self._n_units)
        }

        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self._inner.step(action_dict)

        # Pick first agent's obs and info as representative
        first_agent = self._inner.possible_agents[0]
        obs = obs_dict[first_agent].astype(np.float32)
        inner_info = info_dict[first_agent]

        # No-load cost: incurred for every ON unit this step
        no_load_total = float(
            np.sum(self._no_load_cost_per_step * self._inner._committed)
        )

        # Reserve shortfall: committed capacity vs required (load × (1 + margin))
        total_committed_cap = float(
            np.sum(self._inner.p_max * self._inner._committed)
        )
        total_load = float(
            np.sum(self._inner.grid._get_node_loads_p_current())
        )
        reserve_shortfall = max(
            0.0,
            total_load * (1.0 + self._reserve_margin_frac) - total_committed_cap,
        )

        # Reward is computed from the per-step USD cost components using
        # the SAME scale as PowerZooJax (`reward_scale=1e-4`), NOT from
        # the inner MARL env's reward (which uses /1000 scaling and a
        # different cost-component sum, leaving cross-backend SB3
        # ep_rew_mean values 10x apart).  Locked by
        # tests/benchmarks/test_tso_reward_parity.py.
        gen_cost_step = float(inner_info.get("gen_cost", 0.0))
        startup_cost_step = float(inner_info.get("startup_cost", 0.0))
        operating_cost_step = gen_cost_step + startup_cost_step + no_load_total
        reward = -self.REWARD_SCALE * operating_cost_step

        info = {
            "gen_cost":          gen_cost_step,
            "startup_cost":      startup_cost_step,
            "no_load_cost":      no_load_total,
            "shutdown_cost":     0.0,
            "operating_cost":    operating_cost_step,
            "cost_thermal_overload": float(
                inner_info.get("costs", {}).get("thermal_overload", inner_info.get("cost", 0.0))
            ),
            "reserve_shortfall": reserve_shortfall,
            "committed":         self._inner._committed.copy(),
        }
        info["cost_reserve_shortfall"] = reserve_shortfall
        info["constraint_names"] = self.constraint_names()
        info["constraint_costs"] = np.asarray(
            [info["cost_thermal_overload"], info["cost_reserve_shortfall"]],
            dtype=np.float32,
        )
        info["cost_sum"] = float(info["constraint_costs"].sum())

        terminated = bool(term_dict.get("__all__", False))
        truncated = bool(trunc_dict.get("__all__", False))
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class CentralizedComparisonTSOTask(MARLUCTask):
    """TSO comparison task: Case118, 54 units, 1 centralized agent, 48 steps.

    Matches the JAX comparison contract (TSO_COMPARISON_SCHEMA).
    Does NOT affect the default marl_uc task behaviour.
    """

    name = "comparison_tso_centralized"
    description = (
        "Centralized TSO UC comparison task — Case118, 48 steps, real GB net-load."
    )
    difficulty = "middle"
    training_contract = "cmdp_env_plus_scalar_safe_projection"
    eval_protocol: Dict[str, Any] = {
        "n_episodes": 100,
        "seed_start": 42,
        "split": "train",
        "constraint_names": ["thermal_overload", "reserve_shortfall"],
        "cost_thresholds": [0.0, 5.0],
        "cost_threshold": 5.0,
    }

    def __init__(
        self,
        *,
        split: str = "train",
        episode_start_idx: int = 0,
        **kwargs,
    ):
        if split not in _SPLIT_SETTINGS:
            raise ValueError(
                f"comparison_tso supports split in {sorted(_SPLIT_SETTINGS)}, "
                f"got {split!r}. Extend both sides together."
            )
        gb_split, _load_scale, _line_rating_scale = _SPLIT_SETTINGS[split]
        start_date, end_date = _GB_SPLIT_DATES[gb_split]

        # Inner MARL env requires a date range.  We still override its load
        # source via _inject_gb_loads, but passing the same GB window avoids a
        # misleading empty-data warning during smoke checks.
        kwargs.setdefault("case", "Case118")
        kwargs.setdefault("max_steps", 48)
        kwargs.setdefault("delta_t_minutes", 30)
        kwargs.pop("split", None)  # never forward to MARLUCTask; we own it
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            split=None,
            **kwargs,
        )
        self._reserve_margin_frac: float = 0.05
        self._split: str = split
        self._episode_start_idx: int = int(episode_start_idx)

    def constraint_spec(self) -> ConstraintSpec:
        return ConstraintSpec(
            selected_names=("thermal_overload", "reserve_shortfall"),
            thresholds=(0.0, 5.0),
            fallback_weights=(1.0, 1.0),
        )

    def create_env(self) -> CentralizedComparisonTSOEnv:
        from powerzoo.wrappers.safe_rl_wrapper import TaskCMDPWrapper

        env = CentralizedComparisonTSOEnv(self)
        return TaskCMDPWrapper(env, constraint_spec=self.constraint_spec())
