"""Task 1: Multi-Agent OPF Control

Multi-agent optimal power flow (OPF) control task.

Problem:
- In the IEEE 5-bus transmission network, 5 generators must coordinate output.
- Each generator is an independent agent that outputs a power allocation score.
- Goal: minimize generation cost while meeting grid security constraints.

Agent design:
- Agent: each generator (unit_0, unit_1, ...)
- Observation: global (loads, line flows) + local (unit parameters)
- Action: power allocation score in [0, 1]
- Reward: cost per MWh (shared reward, cooperative setting)

Data splits (non-overlapping, fixed for benchmark reproducibility)
------------------------------------------------------------------
Aligned with bundled time-series coverage (``GB_Forecast_Actual_Demand_2023_2025_30min``).

  train : 2023-07-05 ~ 2024-12-31
  val   : 2025-01-01 ~ 2025-06-30
  test  : 2025-07-01 ~ 2025-12-15  ← official benchmark split

Usage:
    >>> from powerzoo.tasks import make_task_env
    >>>
    >>> train_env = make_task_env('marl_opf', split='train')
    >>> test_env  = make_task_env('marl_opf', split='test')
"""

from typing import Dict, Any, Optional

from powerzoo.tasks.base import MultiAgentTask
from powerzoo.tasks.observation import make_observation_config


class MARLOPFTask(MultiAgentTask):
    """Multi-Agent OPF Control Task

    Multi-agent economic dispatch on the IEEE 5-bus system.
    Each generator learns to coordinate its output as an independent agent.
    """

    # Task metadata
    name = "marl_opf"
    description = "Multi-Agent Optimal Power Flow control on IEEE 5-bus system"
    difficulty = "simple"
    agent_mode = "multi"

    # ---------- Fixed benchmark data splits (do not change) ----------
    SPLIT_DATES = {
        'train': ('2023-07-05', '2024-12-31'),
        'val':   ('2025-01-01', '2025-06-30'),
        'test':  ('2025-07-01', '2025-12-15'),
    }

    # Standardized evaluation protocol — must match powerzoo/benchmarks/baselines.json
    eval_protocol: Dict[str, Any] = {
        "n_episodes": 100,
        "seed_start": 42,
        "split": "test",
        # cost_threshold: max cumulative episode cost (MW-steps) for CMDP budget.
        # IEEE 5-bus, 48 steps: allow up to 10 MW-steps total violation per episode.
        "cost_threshold": 10.0,
        "metrics": ["mean_reward", "std_reward", "normalized_score",
                    "constraint_violation_rate",
                    "mean_episode_cost", "cost_violation_rate"],
    }

    # Constraint tightness presets.
    # Higher max_load_ratio → grid operates closer to capacity → more violations.
    _TIGHTNESS_PRESETS: Dict[str, Any] = {
        'loose':    {'max_load_ratio': 0.70, 'cost_threshold': 30.0},
        'standard': {'max_load_ratio': 0.90, 'cost_threshold': 10.0},
        'strict':   {'max_load_ratio': 0.98, 'cost_threshold':  3.0},
    }

    def __init__(self,
                 case: str = 'Case5',
                 split: Optional[str] = 'train',
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 delta_t_minutes: int = 30,
                 max_load_ratio: float = None,
                 max_steps: int = 48,
                 action_mode: str = 'score',
                 observation_mode: str = 'global',
                 forecast_horizon_steps: int = 4,
                 constraint_tightness: str = 'standard',
                 **kwargs):
        """Initialize the MARL OPF task.

        Args:
            case:          Grid case ('Case5', 'Case118', etc.)
            split:         Data split — 'train', 'val', or 'test'.  Sets the
                           date range for episode sampling.  Ignored when
                           ``start_date``/``end_date`` are given explicitly.
            start_date:    Explicit start date (overrides ``split``).
            end_date:      Explicit end date (overrides ``split``).
            delta_t_minutes: Time step in minutes.
            max_load_ratio:  Maximum load ratio.
            max_steps:     Max steps per episode (default 48 = 1 day @ 30 min).
            action_mode:   'score' for softmax allocation, 'direct' for MW.
            observation_mode: One of 'global', 'local', 'local_plus_forecast'.
            forecast_horizon_steps: Forecast horizon used in local_plus_forecast mode.
            **kwargs:      Other override parameters passed to Task base.
        """
        super().__init__(constraint_tightness=constraint_tightness, **kwargs)

        # Resolve date range from split or explicit dates
        if start_date is None and end_date is None:
            if split not in self.SPLIT_DATES:
                raise ValueError(
                    f"split must be one of {list(self.SPLIT_DATES)}, got '{split}'"
                )
            start_date, end_date = self.SPLIT_DATES[split]
        elif start_date is None or end_date is None:
            raise ValueError("Provide both start_date and end_date, or neither.")

        self._case = case
        self._split = split
        self._start_date = start_date
        self._end_date = end_date
        self._delta_t_minutes = delta_t_minutes
        # max_load_ratio: explicit arg > tightness preset > default 0.9
        self._max_load_ratio = (
            max_load_ratio
            if max_load_ratio is not None
            else self._tightness_param('max_load_ratio', default=0.9)
        )
        self._max_steps = max_steps
        self._action_mode = action_mode
        self._observation_mode = observation_mode
        self._forecast_horizon_steps = forecast_horizon_steps

    def get_scenario_config(self) -> Dict[str, Any]:
        """Return scenario configuration for PowerEnv."""
        return {
            'name': f'{self.name}_scenario',
            'description': self.description,

            'grid': {
                'type': 'transmission',
                'case': self._case,
                'start_date': self._start_date,
                'end_date': self._end_date,
                'delta_t_minutes': self._delta_t_minutes,
                'max_load_ratio': self._max_load_ratio,
            },

            'resources': [],  # OPF task does not require extra resources

            'reward': {
                'type': 'economic_dispatch',
                'cost_weight': 1.0,
            },

            'episode': {
                'max_steps': self._max_steps,
            },
        }

    def get_agents_config(self) -> Dict[str, Any]:
        """Return multi-agent configuration."""
        return {
            'agent_type': 'unit',          # each generator is an agent
            'action_mode': self._action_mode,
            'reward_type': 'shared',       # cooperative reward

            'observation': make_observation_config(
                mode=self._observation_mode,
                supported_modes=('global', 'local', 'local_plus_forecast'),
                global_features=('total_load_mw', 'line_flows', 'time_features'),
                local_features=('bus_load', 'adjacent_line_flows', 'unit_idx', 'p_min', 'p_max', 'cost_coeffs'),
                forecast_features=('future_total_load',),
                forecast_horizon_steps=self._forecast_horizon_steps,
            ),

            'action': {
                'type': 'continuous',
                'mode': self._action_mode,
            },
        }


class MARLOPFTask7Days(MARLOPFTask):
    """7-day MARL OPF task variant.

    Useful for longer training and evaluation windows (one week per episode).
    """

    name = "marl_opf_7d"
    description = "Multi-Agent OPF control - 7-day episode"

    def __init__(self, **kwargs):
        kwargs.setdefault('max_steps', 336)  # 7 * 48
        super().__init__(**kwargs)


class MARLOPFTaskCase118(MARLOPFTask):
    """MARL OPF task for the IEEE 118-bus system."""

    name = "marl_opf_118"
    description = "Multi-Agent OPF control on IEEE 118-bus system"
    difficulty = "middle"

    def __init__(self, **kwargs):
        kwargs.setdefault('case', 'Case118')
        super().__init__(**kwargs)
