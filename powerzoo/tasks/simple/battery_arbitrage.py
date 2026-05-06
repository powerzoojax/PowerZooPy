"""Battery Energy Arbitrage Task

Problem:
- A single battery storage unit is connected to an IEEE 33-bus distribution grid.
- The RL agent decides how much power to charge or discharge each time step.
- Goal: buy electricity when price is low (off-peak), sell when high (peak),
  while keeping SOC within safe limits.

Grid: Case33bw (distribution, radial, 33 buses)
  Battery is sited at bus 6.
  Default scale: 0.5 MWh / 0.2 MW (distribution-scale DER).

Data splits (non-overlapping, fixed for benchmark reproducibility)
------------------------------------------------------------------
  train : 2023-07-05 ~ 2024-12-31
  val   : 2025-01-01 ~ 2025-06-30
  test  : 2025-07-01 ~ 2025-12-15
"""

from typing import Any, Dict, Optional

from powerzoo.tasks.base import SingleAgentTask


class BatteryArbitrageTask(SingleAgentTask):
    """Single-agent battery energy arbitrage on a distribution grid.

    The agent controls one continuous action: charge (negative) or
    discharge (positive) power in MW.  Reward combines peak/off-peak
    arbitrage profit and a soft SOC-deviation penalty.
    """

    name = "battery_arbitrage"
    description = "Battery energy arbitrage - buy low, sell high"
    difficulty = "simple"

    SPLIT_DATES = {
        'train': ('2023-07-05', '2024-12-31'),
        'val':   ('2025-01-01', '2025-06-30'),
        'test':  ('2025-07-01', '2025-12-15'),
    }

    eval_protocol: Dict[str, Any] = {
        "n_episodes": 100,
        "seed_start": 42,
        "split": "test",
        "cost_threshold": 1.0,
        "metrics": [
            "mean_reward", "std_reward", "normalized_score",
            "mean_episode_cost", "cost_violation_rate",
        ],
    }

    def __init__(
        self,
        case: str = 'Case33bw',
        split: Optional[str] = 'train',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        delta_t_minutes: int = 30,
        max_load_ratio: float = 0.85,
        max_steps: int = 48,
        # Battery parameters
        bus_id: int = 6,
        capacity_mwh: float = 0.5,
        power_mw: float = 0.2,
        efficiency: float = 0.95,
        initial_soc: float = 0.5,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        # Reward parameters
        arbitrage_weight: float = 1.0,
        soc_penalty_weight: float = 0.1,
        target_soc: float = 0.5,
        **kwargs,
    ):
        """Initialize the battery arbitrage task.

        Args:
            case:              Grid case (default Case33bw).
            split:             Data split -- 'train', 'val', or 'test'.
            start_date:        Explicit start date (overrides split).
            end_date:          Explicit end date (overrides split).
            delta_t_minutes:   Time step in minutes (default 30).
            max_load_ratio:    Max load ratio for the grid.
            max_steps:         Steps per episode (default 48 = 1 day @ 30 min).
            bus_id:            Bus where the battery connects (default 6).
            capacity_mwh:      Battery capacity in MWh (default 0.5).
            power_mw:          Max charge/discharge power in MW (default 0.2).
            efficiency:        Round-trip efficiency (default 0.95).
            initial_soc:       Initial state of charge (default 0.5).
            soc_min:           Minimum SOC (default 0.1).
            soc_max:           Maximum SOC (default 0.9).
            arbitrage_weight:  Reward weight for arbitrage profit.
            soc_penalty_weight: Penalty weight for SOC deviation.
            target_soc:        Target SOC for penalty calculation.
        """
        super().__init__(**kwargs)

        # Resolve split dates
        if start_date is None and end_date is None:
            if split not in self.SPLIT_DATES:
                raise ValueError(
                    f"split must be one of {list(self.SPLIT_DATES)}, got '{split}'"
                )
            start_date, end_date = self.SPLIT_DATES[split]
        elif start_date is None or end_date is None:
            raise ValueError("Provide both start_date and end_date, or neither.")

        self._case = case
        self._start_date = start_date
        self._end_date = end_date
        self._delta_t_minutes = delta_t_minutes
        self._max_load_ratio = max_load_ratio
        self._max_steps = max_steps

        # Battery config
        self._bus_id = bus_id
        self._capacity_mwh = capacity_mwh
        self._power_mw = power_mw
        self._efficiency = efficiency
        self._initial_soc = initial_soc
        self._soc_min = soc_min
        self._soc_max = soc_max

        # Reward config
        self._arbitrage_weight = arbitrage_weight
        self._soc_penalty_weight = soc_penalty_weight
        self._target_soc = target_soc

    def get_scenario_config(self) -> Dict[str, Any]:
        return {
            'name': f'{self.name}_scenario',
            'description': self.description,
            'grid': {
                'type': 'distribution',
                'case': self._case,
                'start_date': self._start_date,
                'end_date': self._end_date,
                'delta_t_minutes': self._delta_t_minutes,
                'max_load_ratio': self._max_load_ratio,
            },
            'resources': [{
                'type': 'battery',
                'name': 'battery_0',
                'bus_id': self._bus_id,
                'capacity_mwh': self._capacity_mwh,
                'power_mw': self._power_mw,
                'efficiency': self._efficiency,
                'initial_soc': self._initial_soc,
                'soc_min': self._soc_min,
                'soc_max': self._soc_max,
            }],
            'reward': {
                'type': 'battery_arbitrage',
                'arbitrage_weight': self._arbitrage_weight,
                'soc_penalty_weight': self._soc_penalty_weight,
                'target_soc': self._target_soc,
            },
            'episode': {'max_steps': self._max_steps},
        }

    def get_agents_config(self) -> Dict[str, Any]:
        return {
            'agent_type': 'single',
            'obs_keys': ['grid', 'resources', 'time'],
            'resource_names': ['battery_0'],
        }
