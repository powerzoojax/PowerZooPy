"""GenCos Market Bidding Task — competitive rolling market on case5.

Benchmark setup matching powerzoojax GenCos preset:
- case5, 5 agents (one per generator), 48 × 30-min steps
- Action: Box(3) ∈ [-1,1] (3-segment monotone markup)
- Reward: dispatch profit = LMP * P * dt - TC * dt
- Ramp coupling between steps

Usage::

    from powerzoo.tasks import make_task_env, list_tasks

    env = make_task_env('gencos_bidding', split='train')   # GB real-data train split
    env = make_task_env('gencos_bidding', split='iid')     # GB real-data iid split

    env.reset()
    obs, rw, term, trunc, info = env.step(
        {ag: env.action_spaces[ag].sample() for ag in env.agents}
    )
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from powerzoo.tasks.base import ConstraintSpec, MultiAgentTask


_GB_SPLIT_DATES: Dict[str, Tuple[str, str]] = {
    "train": ("2025-04-01", "2025-12-31"),
    "iid": ("2026-01-01", "2026-03-31"),
}

_SPLIT_SETTINGS: Dict[str, Tuple[str, Optional[str]]] = {
    "train": ("train", None),
    "iid": ("iid", None),
    # Match PowerZooJax's current GenCos task implementation:
    # OOD loads are derived from the IID window, then stressed.
    "demand_shift": ("iid", "demand_shift"),
    "renewable_shock": ("iid", "renewable_shock"),
}


def load_gencos_profiles(
    case,
    *,
    split: str = "train",
    ood_axis: Optional[str] = None,
    data_loader=None,
    resample: str = "30min",
) -> np.ndarray:
    """Load GB demand and map it onto case5 nodal loads.

    This mirrors the PowerZooJax GenCos benchmark contract:
    - real GB demand trace
    - case5 nodal scaling
    - OOD stress applied *after* case scaling
    - total load projected back into the feasible dispatch band
    """
    from powerzoo.data.data_loader import DataLoader
    from powerzoo.data.signals import LOAD_ACTUAL_MW

    if split not in _GB_SPLIT_DATES:
        raise ValueError(
            f"Unknown GenCos split {split!r}; expected one of {sorted(_GB_SPLIT_DATES)}"
        )

    if data_loader is None:
        data_loader = DataLoader()

    start_date, end_date = _GB_SPLIT_DATES[split]
    df = data_loader.load_signals(
        [LOAD_ACTUAL_MW],
        source="gb",
        start_date=start_date,
        end_date=end_date,
        resample=resample,
    )
    if LOAD_ACTUAL_MW not in df.columns:
        raise RuntimeError(
            f"GB loader returned columns {list(df.columns)} without {LOAD_ACTUAL_MW!r}"
        )
    demand_mw = df[LOAD_ACTUAL_MW].to_numpy(dtype=np.float32).reshape(-1)
    if demand_mw.size == 0:
        raise RuntimeError(
            f"GB demand loader returned no rows for split={split!r} "
            f"window={start_date}..{end_date}"
        )

    d_min = np.zeros(len(case.nodes), dtype=np.float32)
    d_max = case.nodes["Pd"].to_numpy(dtype=np.float32)

    denom = float(demand_mw.max() - demand_mw.min()) + 1e-8
    demand_norm = (demand_mw - demand_mw.min()) / denom
    profiles = d_min[None, :] + demand_norm[:, None] * (d_max - d_min)[None, :]
    profiles = np.clip(profiles, d_min[None, :], d_max[None, :])

    if ood_axis == "demand_shift":
        profiles = profiles * np.float32(1.10)
    elif ood_axis == "renewable_shock":
        profiles = profiles * np.float32(1.05)
    elif ood_axis is not None:
        raise ValueError(
            f"Unknown ood_axis={ood_axis!r}; expected one of ('demand_shift', 'renewable_shock')"
        )

    min_total_load = float(case.units["p_min"].to_numpy(dtype=np.float32).sum()) + 1.0
    max_total_load = float(case.units["p_max"].to_numpy(dtype=np.float32).sum()) - 1.0
    total_load = profiles.sum(axis=1, keepdims=True)

    base_weights = np.clip(d_max, 0.0, None)
    base_weight_sum = float(base_weights.sum())
    if base_weight_sum <= 0.0:
        raise RuntimeError("GenCos case has no positive load buses for profile projection.")
    base_weights = base_weights / base_weight_sum
    row_weights = np.where(
        total_load > 1e-6,
        profiles / (total_load + 1e-8),
        base_weights[None, :],
    )
    profiles = np.where(total_load < min_total_load, row_weights * min_total_load, profiles)
    total_load = profiles.sum(axis=1, keepdims=True)
    scale_down = np.minimum(1.0, max_total_load / (total_load + 1e-8))
    profiles = profiles * scale_down
    return profiles.astype(np.float32)


class GenCosTask(MultiAgentTask):
    """GenCos competitive market bidding task (case5, 5 agents, 48 steps).

    Wraps ``GenCosMARLEnv`` via the task framework's ``custom`` adapter path.
    The env is created directly (not via a PowerEnv scenario) because the
    market semantics differ fundamentally from grid-OPF tasks.

    Task metadata:
        name:        'gencos_bidding'
        difficulty:  'simple'
        agent_mode:  'multi'
    """

    name = "gencos_bidding"
    description = "GenCos competitive market bidding — case5, 5 agents, 48×30min"
    difficulty = "simple"
    agent_mode = "multi"
    training_contract = "cmdp_env_plus_mdp_fallback"
    SPLIT_DATES: Dict[str, Tuple[str, str]] = {
        "train": _GB_SPLIT_DATES["train"],
        "iid": _GB_SPLIT_DATES["iid"],
        "demand_shift": _GB_SPLIT_DATES["iid"],
        "renewable_shock": _GB_SPLIT_DATES["iid"],
    }

    _TIGHTNESS_PRESETS: Dict[str, Any] = {
        'loose':    {'max_markup': 3.0, 'cost_threshold': 50.0},
        'standard': {'max_markup': 2.0, 'cost_threshold': 20.0},
        'strict':   {'max_markup': 1.0, 'cost_threshold':  5.0},
    }
    eval_protocol: Dict[str, Any] = {
        "n_episodes": 100,
        "seed_start": 42,
        "split": "iid",
        "constraint_names": ["thermal_overload"],
        "cost_thresholds": [0.0],
        "cost_threshold": 0.0,
    }

    def __init__(
        self,
        case_id: int = 5,
        n_segments: int = 3,
        max_steps: int = 48,
        delta_t_hours: float = 0.5,
        lmp_scale: float = 100.0,
        lmp_history_len: int = 4,
        ramp_rate_fraction: float = 0.5,
        max_markup: float = 2.0,
        split: Optional[str] = 'train',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        data_loader=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if start_date is None and end_date is None:
            if split not in self.SPLIT_DATES:
                raise ValueError(
                    f"split must be one of {list(self.SPLIT_DATES)}, got {split!r}"
                )
            start_date, end_date = self.SPLIT_DATES[split]
        elif start_date is None or end_date is None:
            raise ValueError("Provide both start_date and end_date, or neither.")
        load_split, ood_axis = _SPLIT_SETTINGS.get(split, (None, None))
        if load_split is None:
            raise ValueError(
                f"Unknown GenCos split {split!r}; expected one of {sorted(_SPLIT_SETTINGS)}"
            )
        self._case_id = case_id
        self._n_segments = n_segments
        self._max_steps = max_steps
        self._delta_t_hours = delta_t_hours
        self._lmp_scale = lmp_scale
        self._lmp_history_len = lmp_history_len
        self._ramp_rate_fraction = ramp_rate_fraction
        self._max_markup = float(max_markup)
        self._split = split
        self._start_date = start_date
        self._end_date = end_date
        self._load_split = load_split
        self._ood_axis = ood_axis
        self._data_loader = data_loader

    def constraint_spec(self) -> ConstraintSpec:
        return ConstraintSpec(
            selected_names=("thermal_overload",),
            thresholds=(0.0,),
            fallback_weights=(1.0,),
        )

    def get_scenario_config(self) -> Dict[str, Any]:
        return {
            'name': 'gencos_bidding_scenario',
            'description': self.description,
            'grid': {
                'type': 'transmission',
                'case': f'Case{self._case_id}',
                'start_date': self._start_date,
                'end_date': self._end_date,
            },
            'resources': [],
            'reward': {'type': 'dispatch_profit'},
            'episode': {'max_steps': self._max_steps},
        }

    def get_agents_config(self) -> Dict[str, Any]:
        return {
            'agent_type': 'custom',
            'env_class': _GenCosAdapter,
            'n_segments': self._n_segments,
            'max_steps': self._max_steps,
            'delta_t_hours': self._delta_t_hours,
            'lmp_scale': self._lmp_scale,
            'lmp_history_len': self._lmp_history_len,
            'ramp_rate_fraction': self._ramp_rate_fraction,
            'max_markup': self._max_markup,
        }


class _GenCosAdapter:
    """Thin adapter: accepts a GenCosTask and returns a GenCosMARLEnv.

    Used by the ``custom`` adapter path in
    ``powerzoo.tasks.adapters.base._create_specialized_env``.
    Forwards observation_spaces, action_spaces, agents, reset, and step
    so it behaves as a GenCosMARLEnv from the caller's perspective.
    """

    def __init__(self, task: GenCosTask):
        from powerzoo.case import load_case
        from powerzoo.envs.market.gencos_marl import make_gencos_env

        cfg = task.get_agents_config()
        case = load_case(task._case_id)
        profiles = load_gencos_profiles(
            case,
            split=task._load_split,
            ood_axis=task._ood_axis,
            data_loader=task._data_loader,
        )
        self._env = make_gencos_env(
            case=case,
            load_profiles=profiles,
            n_segments=cfg['n_segments'],
            max_markup=cfg['max_markup'],
            max_steps=cfg['max_steps'],
            delta_t_hours=cfg['delta_t_hours'],
            lmp_scale=cfg['lmp_scale'],
            lmp_history_len=cfg['lmp_history_len'],
            ramp_rate_fraction=cfg['ramp_rate_fraction'],
            data_source="gb_real",
            benchmark_split=task._split,
            ood_axis=task._ood_axis,
            profile_window=(task._start_date, task._end_date),
        )

    # Delegate key attributes / methods to the wrapped env
    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def step(self, actions):
        return self._env.step(actions)
