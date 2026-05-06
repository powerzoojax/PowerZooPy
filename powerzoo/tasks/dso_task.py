"""DSO (Distribution System Operator) task for Python PowerZoo.

Mirrors the PowerZooJax DSO task surface for cross-implementation comparison.

Key alignments with PowerZooJax:
- Same physical setup: Case33bw + 6× FlexLoad at buses [6, 14, 18, 22, 28, 33]
- Same data source: Ausgrid zone-substation load shapes (15min → 30min)
- Same feeder-wise load semantics: 3 feeder segments on case33bw are driven by
  distinct Ausgrid feeder-shape profiles rather than by one scalar total-load series
- Same reward: ``-loss_penalty_weight * p_loss_MW``
- Same benchmark CMDP selection: ``selected_constraint_names = ("voltage_violation",)``
- Same 3 baseline helpers: no-control, TOU, droop
- Same metric key names as ``compute_dso_metrics()`` in PowerZooJax
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
import pandas as pd

from powerzoo.tasks.base import ConstraintSpec


# ---------------------------------------------------------------------------
# Constants (match PowerZooJax DSO_FLEXLOAD_CONFIG exactly)
# ---------------------------------------------------------------------------

DSO_FLEXLOAD_CONFIG: List[Dict[str, Any]] = [
    {"name": "fl_a1", "bus_id": 6,  "curtail_cap_mw": 0.15, "shift_cap_mw": 0.15},
    {"name": "fl_a2", "bus_id": 14, "curtail_cap_mw": 0.10, "shift_cap_mw": 0.10},
    {"name": "fl_a3", "bus_id": 18, "curtail_cap_mw": 0.10, "shift_cap_mw": 0.10},
    {"name": "fl_b1", "bus_id": 22, "curtail_cap_mw": 0.08, "shift_cap_mw": 0.08},
    {"name": "fl_c1", "bus_id": 28, "curtail_cap_mw": 0.12, "shift_cap_mw": 0.12},
    {"name": "fl_c2", "bus_id": 33, "curtail_cap_mw": 0.10, "shift_cap_mw": 0.10},
]

DSO_V_MIN: float = 0.94
DSO_V_MAX: float = 1.06

# Match PowerZooJax feeder segmentation exactly.
DSO_FEEDER_BUS_MAP: Dict[str, List[int]] = {
    "feeder_A": list(range(2, 19)),
    "feeder_B": list(range(19, 23)),
    "feeder_C": list(range(23, 34)),
}

# Match PowerZooJax Ausgrid split dates and held-out substation pools.
_AUSGRID_FEEDER_POOLS: Dict[str, Dict[str, List[str]]] = {
    "feeder_A": {
        "train": [
            "Broadmeadow 132_11kV",
            "Charlestown 132_11kV",
            "Jesmond 132_11kV",
        ],
        "zone_holdout": [
            "Mayfield West 132_11kV",
            "Kotara 33_11kV",
        ],
    },
    "feeder_B": {
        "train": [
            "Burwood 132_11kV",
            "Homebush Bay 132_11kV",
            "Strathfield South 132_11kV",
        ],
        "zone_holdout": [
            "Flemington 132_11kV",
            "Lidcombe 33_11kV",
        ],
    },
    "feeder_C": {
        "train": [
            "Cronulla 132_11kV",
            "Caringbah 33_11kV",
            "Miranda 33_11kV",
        ],
        "zone_holdout": [
            "Kurnell South 132_11kV",
            "Jannali 33_11kV",
        ],
    },
}

# Single source of truth for these constants is
# ``powerzoojax/data/splits.py``.  The values are mirrored here (rather than
# cross-imported) to keep PowerZoo independently runnable, and a guardrail
# test (``tests/benchmarks/test_split_alignment.py``) asserts they stay in
# sync with PowerZooJax on every cross-backend run.
#
# - ``train`` and ``iid`` deliberately share the same window AND substation
#   pool.  Their separation is achieved at the calendar-day level via
#   ``filter_ausgrid_role_days`` below (every 4th local-Sydney day = ``iid``,
#   the rest = ``train``).
# - ``summer_ood`` is the high-temperature season immediately after the
#   training window.
# - ``zone_holdout`` shares the train window but uses a disjoint substation
#   pool from ``_AUSGRID_FEEDER_POOLS[*]['zone_holdout']``.
_AUSGRID_SPLIT_DATES: Dict[str, Tuple[str, str]] = {
    "train":        ("2024-05-01", "2024-11-30"),
    "iid":          ("2024-05-01", "2024-11-30"),
    "summer_ood":   ("2024-12-01", "2025-02-28"),
    "zone_holdout": ("2024-05-01", "2024-11-30"),
}

# Day-level partition stride mirrors PowerZooJax
# ``powerzoojax.data.ausgrid_utils._IID_DAY_STRIDE``.  Changing it here
# without changing the JAX side will make the cross-backend split
# guardrail fail.
_IID_DAY_STRIDE = 4
_IID_DAY_OFFSET = 0

DSO_CONSTRAINT_SPEC = ConstraintSpec(
    selected_names=("voltage_violation",),
    thresholds=(5.0,),
    fallback_weights=(1.0,),
)


def _normalize_bus_load_scale_overrides(
    bus_load_scale_overrides: Optional[Dict[int, float]],
) -> Optional[Dict[int, float]]:
    """Coerce YAML / JSON bus-scale overrides to ``{int: float}``."""
    if bus_load_scale_overrides is None:
        return None

    normalized: Dict[int, float] = {}
    for raw_bus_id, raw_scale in bus_load_scale_overrides.items():
        bus_id = int(raw_bus_id)
        scale = float(raw_scale)
        if scale <= 0.0:
            raise ValueError(
                f"bus_load_scale_overrides[{bus_id}] must be > 0, got {scale}"
            )
        normalized[bus_id] = scale
    return normalized


def filter_ausgrid_role_days(
    df: "pd.DataFrame",
    role: str,
    *,
    time_col: str = "datetime",
    local_tz: str = "Australia/Sydney",
) -> "pd.DataFrame":
    """Filter rows to the day subset associated with ``role``.

    ``train`` and ``iid`` share the same window + substation pool; this
    function separates them at the calendar-day level.  Every 4th local
    Sydney day is ``iid``, the rest is ``train``.  Other roles pass
    through unchanged.
    """
    if role not in {"train", "iid"}:
        return df
    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not found in DataFrame")
    if df.empty:
        return df.copy()

    out = df.copy()
    ts = pd.to_datetime(out[time_col], utc=True)
    local_days = ts.dt.tz_convert(local_tz).dt.strftime("%Y-%m-%d")
    day_codes, _ = pd.factorize(local_days, sort=True)
    iid_mask = (day_codes % _IID_DAY_STRIDE) == _IID_DAY_OFFSET

    if role == "iid":
        if not iid_mask.any():
            return out
        return out.loc[iid_mask].reset_index(drop=True)

    train_mask = ~iid_mask
    if not train_mask.any():
        return out
    return out.loc[train_mask].reset_index(drop=True)


def load_feeder_shape(
    data_loader,
    feeder: str,
    role: str = "train",
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    resample: str = "30min",
) -> np.ndarray:
    """Load one mean-normalised Ausgrid feeder shape for a named split."""
    from powerzoo.data.signals import LOAD_ACTUAL_MW

    if role not in _AUSGRID_SPLIT_DATES:
        raise KeyError(
            f"Unknown DSO split '{role}'. Available: {sorted(_AUSGRID_SPLIT_DATES)}"
        )

    pool_role = "zone_holdout" if role == "zone_holdout" else "train"
    substations = _AUSGRID_FEEDER_POOLS[feeder][pool_role]
    split_start, split_end = _AUSGRID_SPLIT_DATES[role]
    start = start_date or split_start
    end = end_date or split_end

    profiles = []
    for sub in substations:
        df = data_loader.load_signals(
            [LOAD_ACTUAL_MW],
            source="ausgrid",
            region=sub,
            start_date=start,
            end_date=end,
            resample=resample,
        )
        # train / iid share the same window + substation pool; the actual
        # split happens here at the calendar-day level (mirrors PowerZooJax).
        if "datetime" in df.columns and role in {"train", "iid"}:
            df = filter_ausgrid_role_days(df, role)
        profiles.append(df[LOAD_ACTUAL_MW].to_numpy(dtype=np.float32).reshape(-1))

    min_len = min(len(p) for p in profiles)
    stacked = np.stack([p[:min_len] for p in profiles], axis=0)
    avg = stacked.mean(axis=0)
    shape = avg / max(float(avg.mean()), 1e-8)
    return shape.astype(np.float32)


def load_dso_feeder_shapes(
    data_loader=None,
    role: str = "train",
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    resample: str = "30min",
) -> Dict[str, np.ndarray]:
    """Load all three canonical DSO feeder shapes for a named Ausgrid split."""
    if data_loader is None:
        from powerzoo.data.data_loader import DataLoader
        data_loader = DataLoader()

    return {
        feeder: load_feeder_shape(
            data_loader,
            feeder,
            role=role,
            start_date=start_date,
            end_date=end_date,
            resample=resample,
        )
        for feeder in DSO_FEEDER_BUS_MAP
    }


def make_dso_load_matrices(
    case,
    feeder_shapes: Dict[str, np.ndarray],
    *,
    max_steps: int = 48,
    episode_start: int = 0,
    load_scale: float = 1.0,
    bus_load_scale_overrides: Optional[Dict[int, float]] = None,
    preserve_feeder_totals: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map feeder shapes to per-load active/reactive load matrices in MW/MVAr."""
    if load_scale <= 0.0:
        raise ValueError(f"load_scale must be > 0, got {load_scale}")

    load_bus_ids = case.loads["bus_id"].to_numpy(dtype=np.int32)
    base_p = case.loads["Pd"].to_numpy(dtype=np.float32) * float(load_scale)
    base_q = case.loads["Qd"].to_numpy(dtype=np.float32) * float(load_scale)
    target_p = base_p.copy()
    target_q = base_q.copy()
    n_loads = len(load_bus_ids)

    bus_load_scale_overrides = _normalize_bus_load_scale_overrides(
        bus_load_scale_overrides
    )
    if bus_load_scale_overrides:
        base_p = base_p.copy()
        base_q = base_q.copy()
        for bus_id, scale in bus_load_scale_overrides.items():
            mask = load_bus_ids == int(bus_id)
            if np.any(mask):
                base_p[mask] *= float(scale)
                base_q[mask] *= float(scale)

        if preserve_feeder_totals:
            for bus_ids in DSO_FEEDER_BUS_MAP.values():
                feeder_mask = np.isin(load_bus_ids, np.asarray(bus_ids, dtype=np.int32))
                if not np.any(feeder_mask):
                    continue
                p_before = float(np.sum(target_p[feeder_mask]))
                q_before = float(np.sum(target_q[feeder_mask]))
                p_after = float(np.sum(base_p[feeder_mask]))
                q_after = float(np.sum(base_q[feeder_mask]))
                if p_after > 1e-8:
                    base_p[feeder_mask] *= p_before / p_after
                if q_after > 1e-8:
                    base_q[feeder_mask] *= q_before / q_after

    shape_matrix = np.ones((max_steps, n_loads), dtype=np.float32)

    for feeder_name, bus_ids in DSO_FEEDER_BUS_MAP.items():
        raw_shape = feeder_shapes.get(feeder_name)
        if raw_shape is None:
            continue
        indices = (np.arange(max_steps) + episode_start) % len(raw_shape)
        shape_window = raw_shape[indices]
        mask = np.isin(load_bus_ids, np.asarray(bus_ids, dtype=np.int32))
        shape_matrix[:, mask] = shape_window[:, None]

    load_p = shape_matrix * base_p[None, :]
    load_q = shape_matrix * base_q[None, :]
    return load_p.astype(np.float32), load_q.astype(np.float32)


def _apply_explicit_dso_loads(
    base_env,
    load_p_mw: np.ndarray,
    load_q_mvar: np.ndarray,
    *,
    start_date: str,
    delta_t_minutes: float,
) -> None:
    """Override PowerEnv's scalar-load cache with explicit feeder-wise matrices."""
    grid = base_env.grid
    grid._node_loads_p = np.asarray(load_p_mw, dtype=np.float32)
    grid._node_loads_q = np.asarray(load_q_mvar, dtype=np.float32)

    periods = load_p_mw.shape[0]
    freq = f"{int(delta_t_minutes)}min"
    index = pd.date_range(start=start_date, periods=periods, freq=freq, tz="UTC")
    grid._time_series_data = pd.DataFrame(
        {
            "load.actual_mw": load_p_mw.sum(axis=1),
            "load.reactive_mvar": load_q_mvar.sum(axis=1),
        },
        index=index,
    )
    grid._regular_time_index = True
    grid.n_days = int(np.ceil(periods / max(int(grid.steps_per_day), 1)))
    grid.start_date = pd.Timestamp(start_date)
    grid.end_date = grid.start_date + pd.Timedelta(minutes=delta_t_minutes * max(periods - 1, 0))


# ---------------------------------------------------------------------------
# DSO CMDP wrapper
# ---------------------------------------------------------------------------

class DSOCostWrapper(gym.Wrapper):
    """Backward-compatible alias for the generic task-level CMDP wrapper.

    Kept only so older imports do not break immediately; the preferred surface
    is :class:`powerzoo.wrappers.safe_rl_wrapper.TaskCMDPWrapper`.
    """

    def __new__(cls, env):
        from powerzoo.wrappers.safe_rl_wrapper import TaskCMDPWrapper
        return TaskCMDPWrapper(env, constraint_spec=DSO_CONSTRAINT_SPEC)


class _DSOEpisodeWindowWrapper(gym.Wrapper):
    """Re-apply one explicit DSO load window on every reset."""

    def __init__(
        self,
        env: gym.Env,
        *,
        base_env,
        feeder_shapes: Dict[str, np.ndarray],
        max_steps: int,
        start_date: str,
        delta_t_minutes: float,
        episode_starts: List[int],
        sampling: str,
        load_scale: float,
        bus_load_scale_overrides: Optional[Dict[int, float]],
        preserve_feeder_totals: bool,
        seed: int = 0,
    ) -> None:
        super().__init__(env)
        if not episode_starts:
            raise ValueError("episode_starts must be non-empty")
        if sampling not in {"cycle", "random"}:
            raise ValueError(
                f"sampling must be 'cycle' or 'random', got {sampling!r}"
            )
        self._base_env = base_env
        self._feeder_shapes = feeder_shapes
        self._max_steps = int(max_steps)
        self._start_date = start_date
        self._delta_t_minutes = float(delta_t_minutes)
        self._episode_starts = tuple(int(x) for x in episode_starts)
        self._sampling = sampling
        self._load_scale = float(load_scale)
        self._bus_load_scale_overrides = _normalize_bus_load_scale_overrides(
            bus_load_scale_overrides
        )
        self._preserve_feeder_totals = bool(preserve_feeder_totals)
        self._rng = np.random.default_rng(int(seed))
        self._cycle_idx = 0
        self.last_episode_start = int(self._episode_starts[0])

    def _apply_window(self, episode_start: int) -> None:
        load_p_mw, load_q_mvar = make_dso_load_matrices(
            self._base_env.grid.case,
            self._feeder_shapes,
            max_steps=self._max_steps,
            episode_start=int(episode_start),
            load_scale=self._load_scale,
            bus_load_scale_overrides=self._bus_load_scale_overrides,
            preserve_feeder_totals=self._preserve_feeder_totals,
        )
        _apply_explicit_dso_loads(
            self._base_env,
            load_p_mw,
            load_q_mvar,
            start_date=self._start_date,
            delta_t_minutes=self._delta_t_minutes,
        )
        self.last_episode_start = int(episode_start)

    def _sample_episode_start(self) -> int:
        if self._sampling == "random":
            return int(self._rng.choice(np.asarray(self._episode_starts, dtype=np.int32)))
        episode_start = int(self._episode_starts[self._cycle_idx % len(self._episode_starts)])
        self._cycle_idx += 1
        return episode_start

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        opts = dict(options or {})
        requested_start = opts.pop("episode_start", None)
        episode_start = (
            int(requested_start)
            if requested_start is not None
            else self._sample_episode_start()
        )
        self._apply_window(episode_start)
        obs, info = self.env.reset(seed=seed, options=opts)
        info = dict(info)
        info["episode_start"] = int(episode_start)
        return obs, info


# ---------------------------------------------------------------------------
# make_dso_env / make_dso_1flex_env
# ---------------------------------------------------------------------------

def make_dso_env(
    flexload_config: Optional[List[Dict[str, Any]]] = None,
    split: str = "train",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_loader=None,
    feeder_shapes: Optional[Dict[str, np.ndarray]] = None,
    episode_start: int = 0,
    delta_t_minutes: float = 30.0,
    max_steps: int = 48,
    max_load_ratio: float = 0.9,
    shift_horizon: int = 4,
    curtail_cost_per_mwh: float = 50.0,
    shift_cost_per_mwh: float = 10.0,
    loss_penalty_weight: float = 0.1,
    load_scale: float = 1.0,
    bus_load_scale_overrides: Optional[Dict[int, float]] = None,
    preserve_feeder_totals: bool = False,
    reset_episode_starts: Optional[List[int]] = None,
    reset_sampling: str = "cycle",
    reset_seed: int = 0,
    v_slack: float = 1.0,
    v_min: float = DSO_V_MIN,
    v_max: float = DSO_V_MAX,
) -> gym.Env:
    """One-call factory for the PowerZoo DSO environment.

    Creates ``PowerEnv(Case33bw + N×FlexLoad)`` wrapped with
    ``FlattenWrapper`` and a generic task-level CMDP selector.

    Returned env:
    - ``action_space``: ``Box(2*N,)`` — N=6 by default → ``Box(12,)``
    - full env vector stays in ``info["constraint_costs"]``
    - task CMDP selection is ``info["selected_constraint_costs"]`` with
      ``selected_constraint_names = ("voltage_violation",)``
    - reward = ``-loss_penalty_weight * p_loss_MW``

    Args:
        flexload_config: List of dicts with keys name/bus_id/curtail_cap_mw/
                         shift_cap_mw.  None → DSO_FLEXLOAD_CONFIG (6-device).
        split: Data split to use (``"train"``, ``"iid"``,
               ``"summer_ood"``, ``"zone_holdout"``).
        start_date / end_date: Optional override for the chosen Ausgrid split window.
        data_loader: Optional ``DataLoader`` instance for real Ausgrid data.
        feeder_shapes: Optional explicit feeder-shape dict.  When provided,
            bypasses the loader path and is mapped directly onto case33bw.
        episode_start: Offset into the feeder-shape arrays.
        delta_t_minutes: Time-step duration (default 30 min = 48 steps/day).
        max_steps: Episode length in steps (default 48 = one day).
        max_load_ratio: Retained for backward compatibility with ``PowerEnv``'s
            scalar load initialisation.  Explicit DSO load matrices override it.
        shift_horizon: Shift-horizon H for each FlexLoad (steps).
        curtail_cost_per_mwh: Cost coefficient for curtailment (£/MWh).
        shift_cost_per_mwh: Cost coefficient for demand shift (£/MWh).
        loss_penalty_weight: Weight of the network-loss reward term.
        load_scale: Uniform multiplier on the case's static P/Q loads.
        bus_load_scale_overrides: Optional per-bus spatial load multipliers.
        preserve_feeder_totals: Re-normalise adjusted loads within each feeder.
        reset_episode_starts: Optional fixed bank of episode starts to sample
            from on every reset. Useful for matching the canonical reset-bank
            training protocol.
        reset_sampling: ``"cycle"`` or ``"random"`` when
            ``reset_episode_starts`` is provided.
        reset_seed: RNG seed for ``reset_sampling="random"``.
        v_slack: Slack-bus voltage setpoint (p.u.).
        v_min / v_max: Voltage limits (p.u.).
    """
    from powerzoo.envs.power_env import PowerEnv
    from powerzoo.wrappers.flatten import FlattenWrapper
    from powerzoo.tasks.rewards import get_reward_function

    if flexload_config is None:
        flexload_config = DSO_FLEXLOAD_CONFIG

    if split not in _AUSGRID_SPLIT_DATES:
        raise KeyError(
            f"Unknown DSO split '{split}'. Available: {sorted(_AUSGRID_SPLIT_DATES)}"
        )
    split_start, split_end = _AUSGRID_SPLIT_DATES[split]
    start_date = start_date or split_start
    end_date = end_date or split_end

    resources = [
        {
            "type": "flexload",
            "name": cfg["name"],
            "bus_id": cfg["bus_id"],
            "curtail_cap_mw": cfg["curtail_cap_mw"],
            "shift_cap_mw": cfg["shift_cap_mw"],
            "shift_horizon": shift_horizon,
            "baseline_mw": cfg["curtail_cap_mw"] * 2.0,
            "curtail_cost_per_mwh": curtail_cost_per_mwh,
            "shift_cost_per_mwh": shift_cost_per_mwh,
        }
        for cfg in flexload_config
    ]

    scenario_config = {
        "name": "dso_flexload_scenario",
        "grid": {
            "type": "distribution",
            "case": "Case33bw",
            "start_date": start_date,
            "end_date": end_date,
            "delta_t_minutes": delta_t_minutes,
            "max_load_ratio": max_load_ratio,
            "v_slack": v_slack,
            "v_min": v_min,
            "v_max": v_max,
        },
        "resources": resources,
        "episode": {"max_steps": max_steps},
    }

    reward_fn = get_reward_function({
        "type": "network_loss",
        "loss_penalty_weight": loss_penalty_weight,
    })

    base_env = PowerEnv(scenario_config, reward_fn=reward_fn)
    if feeder_shapes is None:
        feeder_shapes = load_dso_feeder_shapes(
            data_loader=data_loader,
            role=split,
            start_date=start_date,
            end_date=end_date,
            resample=f"{int(delta_t_minutes)}min",
        )
    load_p_mw, load_q_mvar = make_dso_load_matrices(
        base_env.grid.case,
        feeder_shapes,
        max_steps=max_steps,
        episode_start=episode_start,
        load_scale=load_scale,
        bus_load_scale_overrides=bus_load_scale_overrides,
        preserve_feeder_totals=preserve_feeder_totals,
    )
    _apply_explicit_dso_loads(
        base_env,
        load_p_mw,
        load_q_mvar,
        start_date=start_date,
        delta_t_minutes=delta_t_minutes,
    )
    flat_env = FlattenWrapper(base_env, obs_keys=["grid", "resources", "time"])
    from powerzoo.wrappers.safe_rl_wrapper import TaskCMDPWrapper
    wrapped = TaskCMDPWrapper(flat_env, constraint_spec=DSO_CONSTRAINT_SPEC)
    if reset_episode_starts:
        wrapped = _DSOEpisodeWindowWrapper(
            wrapped,
            base_env=base_env,
            feeder_shapes=feeder_shapes,
            max_steps=max_steps,
            start_date=start_date,
            delta_t_minutes=delta_t_minutes,
            episode_starts=[int(x) for x in reset_episode_starts],
            sampling=reset_sampling,
            load_scale=load_scale,
            bus_load_scale_overrides=bus_load_scale_overrides,
            preserve_feeder_totals=preserve_feeder_totals,
            seed=reset_seed,
        )
    return wrapped


def make_dso_1flex_env(
    bus_id: int = 18,
    curtail_cap_mw: float = 0.10,
    shift_cap_mw: float = 0.10,
    **kwargs,
) -> gym.Env:
    """1-device DSO variant (mirrors PowerZooJax make_dso_1flex_params).

    action_space = Box(2,)  (curtail fraction, shift-out fraction)
    """
    config = [{
        "name": "fl_0",
        "bus_id": bus_id,
        "curtail_cap_mw": curtail_cap_mw,
        "shift_cap_mw": shift_cap_mw,
    }]
    return make_dso_env(flexload_config=config, **kwargs)


# ---------------------------------------------------------------------------
# Baseline rollout helpers
# ---------------------------------------------------------------------------

def _env_max_steps(env: gym.Env) -> int:
    """Read max_steps from the underlying PowerEnv, falling back to 48."""
    base: Any = env
    while hasattr(base, "env"):
        base = base.env
    # PowerEnv stores this as max_steps_per_episode; _clock.max_steps is the same value
    return int(
        getattr(base, "max_steps_per_episode", None)
        or getattr(getattr(base, "_clock", None), "max_steps", None)
        or 48
    )


def rollout_dso(
    env: gym.Env,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    n_steps: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, List]:
    """Run one episode with *policy_fn* and collect per-step diagnostics.

    Args:
        env: A fully-wrapped DSO environment (action_space is Box).
        policy_fn: Callable ``obs → action`` (flat numpy array).
        n_steps: Maximum episode steps.  ``None`` (default) reads
            ``env.max_steps_per_episode`` — mirrors JAX ``params.max_steps``.
        seed: RNG seed for ``env.reset()``.

    Returns:
        Dict with lists of per-step values:
        ``rewards``, ``losses`` (p_loss_MW), ``violations``,
        ``curtailed`` (MW), ``shifted`` (shift-out MW), ``shift_in`` (MW).
    """
    if n_steps is None:
        n_steps = _env_max_steps(env)

    obs, _ = env.reset(seed=seed)
    results: Dict[str, List] = {
        "rewards": [],
        "losses": [],
        "violations": [],
        "voltage_violations": [],
        "thermal_violations": [],
        "curtailed": [],
        "shifted": [],
        "shift_in": [],
    }

    for _ in range(n_steps):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        results["rewards"].append(float(reward))
        results["losses"].append(float(info.get("p_loss_MW", 0.0)))
        voltage_violations = float(info.get("cost_voltage_violation", 0.0))
        thermal_violations = float(info.get("cost_thermal_overload", 0.0))
        results["violations"].append(int(voltage_violations + thermal_violations))
        results["voltage_violations"].append(float(voltage_violations))
        results["thermal_violations"].append(float(thermal_violations))

        res = info.get("resources", {})
        results["curtailed"].append(
            sum(float(v.get("curtailed_mw", 0.0)) for v in res.values())
        )
        results["shifted"].append(
            sum(float(v.get("shift_out_mw", 0.0)) for v in res.values())
        )
        results["shift_in"].append(
            sum(float(v.get("shift_in_mw", 0.0)) for v in res.values())
        )

        if terminated or truncated:
            break

    return results


def dso_no_control_rollout(env: gym.Env, seed: int = 0) -> Dict[str, List]:
    """No-control baseline: zero action for all devices every step."""
    action_dim = env.action_space.shape[0]
    return rollout_dso(env, lambda obs: np.zeros(action_dim, dtype=np.float32), seed=seed)


def dso_tou_heuristic_rollout(
    env: gym.Env,
    peak_start: int = 16,
    peak_end: int = 21,
    seed: int = 0,
) -> Dict[str, List]:
    """Time-of-Use heuristic: 80% curtail + 50% shift during peak hours.

    Peak hours are identified by intra-day step index (0-based) and mirror the
    PowerZooJax helper defaults.
    """
    action_dim = env.action_space.shape[0]
    n_devices = action_dim // 2
    peak_action = np.array([0.8, 0.5] * n_devices, dtype=np.float32)
    zero_action = np.zeros(action_dim, dtype=np.float32)
    step_counter = [0]  # mutable cell so closure can mutate it

    # Derive steps_per_day from the underlying PowerEnv clock, falling back to
    # delta_t_minutes, then to 48.  Mirrors JAX's params.steps_per_day usage.
    base: Any = env
    while hasattr(base, "env"):
        base = base.env
    steps_per_day: int = int(
        getattr(getattr(base, "_clock", None), "steps_per_day", None)
        or getattr(getattr(base, "grid", None), "steps_per_day", None)
        or max(int(1440 / getattr(getattr(base, "grid", None), "delta_t_minutes", 30.0)), 1)
    )

    def policy(obs: np.ndarray) -> np.ndarray:
        t = step_counter[0] % steps_per_day
        step_counter[0] += 1
        return peak_action if peak_start <= t < peak_end else zero_action

    return rollout_dso(env, policy, seed=seed)


def dso_droop_heuristic_rollout(
    env: gym.Env,
    v_low: float = 0.96,
    seed: int = 0,
) -> Dict[str, List]:
    """Voltage-droop heuristic: curtail proportionally when mean voltage is low.

    Reads node voltages from the grid's DataFrame after each step
    (``base.grid.nodes['v_mag']``).  Falls back to zero action if voltage
    data are unavailable.
    """
    action_dim = env.action_space.shape[0]
    n_devices = action_dim // 2

    # Unwrap to reach the PowerEnv (to access grid.nodes)
    base: Any = env
    while hasattr(base, "env"):
        base = base.env  # peel FlattenWrapper / TaskCMDPWrapper

    def policy(obs: np.ndarray) -> np.ndarray:
        action = np.zeros(action_dim, dtype=np.float32)
        try:
            nodes = base.grid.nodes  # DistGridEnv stores post-step DataFrame here
            if nodes is not None and "v_mag" in nodes.columns:
                v = nodes["v_mag"].values
                v_mean = float(np.mean(v))
                curtail = float(np.clip((v_low - v_mean) / 0.04, 0.0, 1.0))
                shift = float(np.clip((v_low - v_mean) / 0.06, 0.0, 0.5))
                for i in range(n_devices):
                    action[2 * i] = curtail
                    action[2 * i + 1] = shift
        except Exception:  # noqa: BLE001
            pass
        return action

    return rollout_dso(env, policy, seed=seed)


# ---------------------------------------------------------------------------
# compute_dso_metrics — same key names as PowerZooJax
# ---------------------------------------------------------------------------

def compute_dso_metrics(
    rollout_results: Dict[str, List],
    baseline_results: Optional[Dict[str, List]] = None,
    dt_hours: float = 0.5,
) -> Dict[str, Any]:
    """Summarise a rollout into scalar benchmark metrics.

    Key names are intentionally identical to PowerZooJax's
    ``compute_dso_metrics`` so results tables can be compared directly.

    Args:
        rollout_results: Output of one of the ``*_rollout`` helpers above.
        baseline_results: Optional no-control rollout for relative metrics.
        dt_hours: Step duration in hours (default 0.5 = 30 min).

    Returns:
        Dict with scalar values for:
        ``total_reward``, ``total_loss_mwh``, ``mean_loss_mw``,
        ``total_violations``, ``total_curtailed_mwh``,
        ``total_shifted_mwh``, ``total_shift_in_mwh``,
        ``served_flex_ratio`` (= shift_in / shift_out, buffer clearance),
        ``network_loss_reduction_pct``, ``peak_shaving_pct``.
    """
    rewards = rollout_results["rewards"]
    losses = rollout_results["losses"]
    violations = rollout_results["violations"]
    voltage_violations = rollout_results.get("voltage_violations", violations)
    thermal_violations = rollout_results.get(
        "thermal_violations",
        [0.0 for _ in range(len(violations))],
    )
    curtailed = rollout_results["curtailed"]
    shifted = rollout_results["shifted"]
    shift_in = rollout_results["shift_in"]

    total_curtailed_mwh = sum(curtailed) * dt_hours
    total_shifted_mwh = sum(shifted) * dt_hours
    total_shift_in_mwh = sum(shift_in) * dt_hours

    # served_flex_ratio = shift_in / shift_out (buffer clearance).
    # Matches PowerZooJax compute_dso_metrics(): shift_in.sum() / max(shifted.sum(), 1e-8).
    # Physical meaning: fraction of shifted-out demand that was actually re-served.
    _shift_out_total = sum(shifted)
    served_flex_ratio = (
        float(sum(shift_in) / max(_shift_out_total, 1e-8))
        if _shift_out_total > 0 else 0.0
    )

    # Relative metrics require a baseline
    network_loss_reduction_pct: Optional[float] = None
    peak_shaving_pct: Optional[float] = None

    if baseline_results is not None:
        base_loss = sum(baseline_results["losses"]) * dt_hours
        ctrl_loss = sum(losses) * dt_hours
        if base_loss > 1e-9:
            network_loss_reduction_pct = float(
                (base_loss - ctrl_loss) / base_loss * 100.0
            )
        else:
            network_loss_reduction_pct = 0.0

        base_peak = max(baseline_results["losses"]) if baseline_results["losses"] else 0.0
        ctrl_peak = max(losses) if losses else 0.0
        if base_peak > 1e-9:
            peak_shaving_pct = float((base_peak - ctrl_peak) / base_peak * 100.0)
        else:
            peak_shaving_pct = 0.0

    return {
        "total_reward": float(sum(rewards)),
        "total_loss_mwh": float(sum(losses) * dt_hours),
        "mean_loss_mw": float(sum(losses) / len(losses)) if losses else 0.0,
        "total_voltage_violations": float(sum(voltage_violations)),
        "total_thermal_overloads": float(sum(thermal_violations)),
        "total_violations": int(sum(violations)),
        "voltage_violation_count_per_step": (
            float(sum(voltage_violations) / max(len(violations), 1))
        ),
        "thermal_overload_count_per_step": (
            float(sum(thermal_violations) / max(len(violations), 1))
        ),
        "total_curtailed_mwh": float(total_curtailed_mwh),
        "total_shifted_mwh": float(total_shifted_mwh),
        "total_shift_in_mwh": float(total_shift_in_mwh),
        "served_flex_ratio": float(served_flex_ratio),
        "network_loss_reduction_pct": network_loss_reduction_pct,
        "peak_shaving_pct": peak_shaving_pct,
    }
