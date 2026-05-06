"""Task 6: Multi-Agent OPF on IEEE 118-Bus System (Complex)

Large-scale OPF control on the IEEE 118-bus transmission network with 54 generators.
This task is significantly harder than the 5-bus version due to:
- 54 cooperative agents (generators)
- 186 transmission lines
- Complex power flow coupling
- Richer constraint landscape

Problem:
- 54 generators must coordinate output to meet demand while minimising total
  generation cost and respecting all line-flow limits.
- Each generator is an independent agent outputting a power allocation score.

Agent design (same protocol as Task 1):
- Action:      score ∈ [0, 1]  → softmax allocation of net load
- Observation: global (total load, line flows, time) + local (unit parameters)
- Reward:      negative generation cost (shared, cooperative)

Data splits (non-overlapping, fixed for benchmark reproducibility)
------------------------------------------------------------------
Aligned with bundled time-series coverage (``GB_Forecast_Actual_Demand_2023_2025_30min``).

  train : 2023-07-05 ~ 2024-12-31
  val   : 2025-01-01 ~ 2025-06-30
  test  : 2025-07-01 ~ 2025-12-15

Usage::

    from powerzoo.tasks import make_task_env

    train_env = make_task_env('opf_118', split='train')
    test_env  = make_task_env('opf_118', split='test')
"""

from typing import Dict, Any, Optional

from powerzoo.tasks.base import MultiAgentTask
from powerzoo.tasks.observation import make_observation_config


class OPF118Task(MultiAgentTask):
    """Multi-Agent OPF Control on the IEEE 118-bus system.

    A complex-difficulty benchmark for large-scale cooperative dispatch.
    """

    name = "opf_118"
    description = "Multi-Agent OPF control on IEEE 118-bus transmission system"
    difficulty = "complex"
    agent_mode = "multi"

    # Fixed benchmark data splits
    SPLIT_DATES = {
        'train': ('2023-07-05', '2024-12-31'),
        'val':   ('2025-01-01', '2025-06-30'),
        'test':  ('2025-07-01', '2025-12-15'),
    }

    eval_protocol: Dict[str, Any] = {
        "n_episodes": 50,
        "seed_start": 42,
        "split": "test",
        # cost_threshold: IEEE 118-bus is ~23× larger than 5-bus; scale threshold.
        "cost_threshold": 50.0,
        "metrics": ["mean_reward", "std_reward", "normalized_score",
                    "constraint_violation_rate",
                    "mean_episode_cost", "cost_violation_rate"],
    }

    # Constraint tightness presets.
    # 118-bus default load is lower (0.85) to avoid numerical issues in DCOPF.
    _TIGHTNESS_PRESETS: Dict[str, Any] = {
        'loose':    {'max_load_ratio': 0.70, 'cost_threshold': 150.0},
        'standard': {'max_load_ratio': 0.85, 'cost_threshold':  50.0},
        'strict':   {'max_load_ratio': 0.95, 'cost_threshold':  15.0},
    }

    def __init__(self,
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
        """Initialize the 118-bus OPF task.

        Args:
            split:               Data split — 'train', 'val', or 'test'.
            start_date:          Explicit start date (overrides ``split``).
            end_date:            Explicit end date (overrides ``split``).
            delta_t_minutes:     Time step in minutes.  Default 30.
            max_load_ratio:      Maximum load as fraction of total capacity.
                                 Defaults to tightness preset (0.85 for standard).
            max_steps:           Max steps per episode.  Default 48 (1 day).
            action_mode:         'score' (softmax allocation) or 'direct' (MW).
            constraint_tightness: One of 'loose', 'standard', 'strict'.
            **kwargs:            Passed to Task base.
        """
        super().__init__(constraint_tightness=constraint_tightness, **kwargs)

        if start_date is None and end_date is None:
            if split not in self.SPLIT_DATES:
                raise ValueError(
                    f"split must be one of {list(self.SPLIT_DATES)}, got '{split}'"
                )
            start_date, end_date = self.SPLIT_DATES[split]
        elif start_date is None or end_date is None:
            raise ValueError("Provide both start_date and end_date, or neither.")

        self._split = split
        self._start_date = start_date
        self._end_date = end_date
        self._delta_t_minutes = delta_t_minutes
        self._max_load_ratio = (
            max_load_ratio if max_load_ratio is not None
            else self._tightness_param('max_load_ratio', default=0.85)
        )
        self._max_steps = max_steps
        self._action_mode = action_mode
        self._observation_mode = observation_mode
        self._forecast_horizon_steps = forecast_horizon_steps

    def get_scenario_config(self) -> Dict[str, Any]:
        return {
            'name': f'{self.name}_scenario',
            'description': self.description,

            'grid': {
                'type': 'transmission',
                'case': 'Case118',
                'start_date': self._start_date,
                'end_date': self._end_date,
                'delta_t_minutes': self._delta_t_minutes,
                'max_load_ratio': self._max_load_ratio,
            },

            'resources': [],

            'reward': {
                'type': 'economic_dispatch',
                'cost_weight': 1.0,
            },

            'episode': {
                'max_steps': self._max_steps,
            },
        }

    def get_agents_config(self) -> Dict[str, Any]:
        return {
            'agent_type': 'unit',
            'action_mode': self._action_mode,
            'reward_type': 'shared',

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


class OPF118Task7Days(OPF118Task):
    """7-day variant of the 118-bus OPF task (one week per episode)."""

    name = "opf_118_7d"
    description = "Multi-Agent OPF control on IEEE 118-bus system — 7-day episode"

    def __init__(self, **kwargs):
        kwargs.setdefault('max_steps', 336)  # 7 * 48
        super().__init__(**kwargs)
