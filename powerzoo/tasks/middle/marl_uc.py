"""Task 4: Unit Commitment (UC) — Middle Difficulty

Multi-agent Unit Commitment on the IEEE 5-bus transmission system.
Each generator agent decides **whether to commit** (on/off) AND how much
power to generate (via a score).

Problem
-------
- 5 generators must schedule their commitment status and dispatch level over a
  24-hour horizon (48 × 30-minute steps).
- Objective: minimise total cost = generation cost + startup cost + shutdown cost.
- Constraints: ramp rates, minimum up/down times, line-flow limits (cost signal).

Compared to Task 1 (pure OPF), this task introduces:
- Binary commitment variables (discrete action component)
- Intertemporal coupling (min-up/down, ramp rates)
- Startup and shutdown cost penalties in reward

Agent design
------------
- Agent:       each generator (unit_0 … unit_4)
- Action:      [score ∈ [0,1], on_off ∈ [0,1]]  (on_off ≥ 0.5 → commit)
- Observation: global (loads, line flows, time) + local (unit params) + commitment vector
- Reward:      -(generation_cost + startup_cost + shutdown_cost) / 1000   (economic only)
- Cost signal: constraint violations → ``info['cost']``

UC parameters (used when case.units does not have UC columns)
-------------------------------------------------------------
    startup_cost  = 500  $/start
    shutdown_cost = 200  $/stop
    ramp_rate     = 999  MW/step  (no ramp limit by default)
    min_up_time   = 1    step
    min_down_time = 1    step

Data splits (aligned with ``GB_Forecast_Actual_Demand_2023_2025_30min``)
------------------------------------------------------------------------
  train : 2023-07-05 ~ 2024-12-31
  val   : 2025-01-01 ~ 2025-06-30
  test  : 2025-07-01 ~ 2025-12-15

Usage::

    from powerzoo.tasks import make_task_env

    env = make_task_env('marl_uc', split='train')
    obs, infos = env.reset(seed=0)
    while True:
        actions = {a: env.action_space[a].sample() for a in env.get_agent_ids()}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        if terminateds['__all__']:
            break
"""

from typing import Dict, Any, Optional

from powerzoo.tasks.base import MultiAgentTask
from powerzoo.tasks.observation import make_observation_config


class MARLUCTask(MultiAgentTask):
    """Multi-Agent Unit Commitment Task on the IEEE 5-bus system."""

    name = "marl_uc"
    description = "Multi-Agent Unit Commitment on IEEE 5-bus system"
    difficulty = "middle"
    agent_mode = "multi"

    SPLIT_DATES = {
        'train': ('2023-07-05', '2024-12-31'),
        'val':   ('2025-01-01', '2025-06-30'),
        'test':  ('2025-07-01', '2025-12-15'),
    }

    eval_protocol: Dict[str, Any] = {
        "n_episodes": 100,
        "seed_start": 42,
        "split": "test",
        # cost_threshold: max cumulative episode cost (MW-steps) for CMDP budget.
        # IEEE 5-bus, 48 steps: same threshold as Task 1.
        "cost_threshold": 10.0,
        "metrics": ["mean_reward", "std_reward", "total_startup_cost",
                    "commitment_rate", "constraint_violation_rate",
                    "mean_episode_cost", "cost_violation_rate"],
    }

    # Constraint tightness presets.
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
                 observation_mode: str = 'global',
                 forecast_horizon_steps: int = 4,
                 constraint_tightness: str = 'standard',
                 **kwargs):
        """Initialise the MARL UC task.

        Args:
            case:                Grid case. Default 'Case5'.
            split:               Data split ('train', 'val', 'test').
            start_date:          Explicit start date (overrides split).
            end_date:            Explicit end date (overrides split).
            delta_t_minutes:     Time step in minutes. Default 30.
            max_load_ratio:      Max load as fraction of capacity.
                                 Defaults to tightness preset (0.9 for standard).
            max_steps:           Steps per episode. Default 48.
            observation_mode:    One of 'global', 'local', 'local_plus_forecast'.
            forecast_horizon_steps: Forecast horizon used in local_plus_forecast mode.
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

        self._case = case
        self._split = split
        self._start_date = start_date
        self._end_date = end_date
        self._delta_t_minutes = delta_t_minutes
        self._max_load_ratio = (
            max_load_ratio if max_load_ratio is not None
            else self._tightness_param('max_load_ratio', default=0.9)
        )
        self._max_steps = max_steps
        self._observation_mode = observation_mode
        self._forecast_horizon_steps = forecast_horizon_steps

    def get_scenario_config(self) -> Dict[str, Any]:
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

            'resources': [],

            # Reward is purely economic (gen + startup + shutdown costs).
            # Safety violations appear in info['cost'] only.
            'reward': {
                'type': 'unit_commitment',
                'cost_weight': 1.0,
            },

            'episode': {
                'max_steps': self._max_steps,
            },
        }

    def get_agents_config(self) -> Dict[str, Any]:
        return {
            'agent_type': 'unit',
            'task_type': 'unit_commitment',   # signals UC adapter to be used
            'action_mode': 'uc',              # [score, on_off] 2-vector
            'reward_type': 'shared',

            'observation': make_observation_config(
                mode=self._observation_mode,
                supported_modes=('global', 'local', 'local_plus_forecast'),
                global_features=('total_load_mw', 'line_flows', 'time_features'),
                local_features=('bus_load', 'adjacent_line_flows', 'unit_idx', 'p_min', 'p_max', 'cost_coeffs', 'commitment'),
                forecast_features=('future_total_load',),
                forecast_horizon_steps=self._forecast_horizon_steps,
            ),

            'action': {
                'type': 'continuous',
                'mode': 'uc',
                'dims': 2,  # [score, on_off]
            },
        }

    def create_env(self):
        """Create UC-specific multi-agent env."""
        from powerzoo.tasks.adapters.uc import TaskUCMultiAgentEnv
        return TaskUCMultiAgentEnv(self)
