"""Task 2: Multi-Agent DER Storage Arbitrage

Problem:
- Multiple battery storage systems are installed in an IEEE 33-bus distribution
  grid (Case33bw, 12.66 kV radial feeder, ~3.7 MW total load).
- Each battery is an independent agent that decides charge/discharge power.
- Goal: maximize arbitrage profit while maintaining SOC constraints and
  keeping node voltages within acceptable bounds.

Grid: Case33bw (distribution, radial, 33 buses)
  Batteries are sited at buses 6, 12, 18 (spread along the main feeder trunk).
  Battery scale: 500 kWh / 200 kW each — consistent with distribution DER.

Data splits (non-overlapping, fixed for benchmark reproducibility)
------------------------------------------------------------------
Aligned with bundled time-series coverage (``GB_Forecast_Actual_Demand_2023_2025_30min``).

  train : 2023-07-05 ~ 2024-12-31
  val   : 2025-01-01 ~ 2025-06-30
  test  : 2025-07-01 ~ 2025-12-15  ← official benchmark split
"""

from typing import Dict, Any, Optional

from powerzoo.tasks.base import MultiAgentTask
from powerzoo.tasks.observation import make_observation_config


class MARLDERArbitrageTask(MultiAgentTask):
    """Multi-Agent DER Storage Arbitrage Task."""

    name = "marl_der_arbitrage"
    description = "Multi-Agent DER Storage Arbitrage - MAPPO baseline (penalty-based)"
    difficulty = "simple"
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
        # cost_threshold: cumulative SOC violation (dimensionless) per episode.
        # 3 batteries × 48 steps: allow up to 0.5 total SOC violation per episode.
        "cost_threshold": 0.5,
        "metrics": ["mean_reward", "std_reward", "normalized_score",
                    "constraint_violation_rate",
                    "mean_episode_cost", "cost_violation_rate"],
    }

    # Constraint tightness presets.
    _TIGHTNESS_PRESETS: Dict[str, Any] = {
        'loose':    {'max_load_ratio': 0.70, 'soc_min': 0.05, 'soc_max': 0.95, 'cost_threshold': 2.0},
        'standard': {'max_load_ratio': 0.90, 'soc_min': 0.10, 'soc_max': 0.90, 'cost_threshold': 0.5},
        'strict':   {'max_load_ratio': 0.98, 'soc_min': 0.15, 'soc_max': 0.85, 'cost_threshold': 0.1},
    }

    # Bus IDs where batteries are sited along the Case33bw main feeder trunk.
    # Using buses 6, 12, 18 (evenly spread; all are load buses with reasonable
    # loading). Override via battery_bus_ids for custom placement.
    DEFAULT_BUS_IDS = [6, 12, 18, 24, 30]

    def __init__(self,
                 case: str = 'Case33bw',
                 split: Optional[str] = 'train',
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 delta_t_minutes: int = 30,
                 max_load_ratio: float = None,
                 max_steps: int = 48,
                 num_batteries: int = 3,
                 battery_capacity_mwh: float = 0.5,
                 battery_power_mw: float = 0.2,
                 soc_min: float = None,
                 soc_max: float = None,
                 initial_soc: float = 0.5,
                 reward_type: str = 'shared',
                 observation_mode: str = 'local_plus_forecast',
                 forecast_horizon_steps: int = 4,
                 battery_bus_ids: Optional[list] = None,
                 constraint_tightness: str = 'standard',
                 **kwargs):
        """Initialize the MARL DER Arbitrage task.

        Args:
            case:               Grid case. Default: 'Case33bw' (IEEE 33-bus
                                distribution network, 12.66 kV, ~3.7 MW load).
            split:              Data split — 'train', 'val', or 'test'.
            start_date:         Explicit start date (overrides split).
            end_date:           Explicit end date (overrides split).
            delta_t_minutes:    Time step in minutes.
            max_load_ratio:     Maximum load ratio.
            max_steps:          Max steps per episode (default 48 = 1 day @ 30 min).
            num_batteries:      Number of battery agents (default 3).
            battery_capacity_mwh: Capacity per battery in MWh (default 0.5 MWh =
                                500 kWh, distribution-scale DER).
            battery_power_mw:   Max charge/discharge power per battery in MW
                                (default 0.2 MW = 200 kW).
            soc_min:            Minimum SOC (0–1).
            soc_max:            Maximum SOC (0–1).
            initial_soc:        Initial SOC (0–1).
            reward_type:        'shared' (cooperative) or 'individual'.
            observation_mode:   One of 'global', 'local', 'local_plus_forecast'.
            forecast_horizon_steps: Forecast horizon used in local_plus_forecast mode.
            battery_bus_ids:    Bus IDs for battery placement. Defaults to
                                ``DEFAULT_BUS_IDS[:num_batteries]``.
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
        self._num_batteries = num_batteries
        self._battery_capacity_mwh = battery_capacity_mwh
        self._battery_power_mw = battery_power_mw
        self._soc_min = (
            soc_min if soc_min is not None
            else self._tightness_param('soc_min', default=0.1)
        )
        self._soc_max = (
            soc_max if soc_max is not None
            else self._tightness_param('soc_max', default=0.9)
        )
        self._initial_soc = initial_soc
        self._reward_type = reward_type
        self._observation_mode = observation_mode
        self._forecast_horizon_steps = forecast_horizon_steps
        self._battery_bus_ids = (
            battery_bus_ids if battery_bus_ids is not None
            else self.DEFAULT_BUS_IDS[:num_batteries]
        )

    def get_scenario_config(self) -> Dict[str, Any]:
        resources = [
            {
                'type': 'battery',
                'name': f'bat_{i}',
                'capacity_mwh': self._battery_capacity_mwh,
                'power_mw': self._battery_power_mw,
                'bus_id': self._battery_bus_ids[i],
                'efficiency': 0.95,
                'soc_min': self._soc_min,
                'soc_max': self._soc_max,
                'initial_soc': self._initial_soc,
            }
            for i in range(self._num_batteries)
        ]
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
            'resources': resources,
            'reward': {
                'type': 'battery_arbitrage',
                'peak_hours': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                'off_peak_hours': [0, 1, 2, 3, 4, 5, 6, 23],
                'arbitrage_weight': 1.0,
                'soc_penalty_weight': 0.5,
                'target_soc': 0.5,
            },
            'episode': {'max_steps': self._max_steps},
        }

    def get_agents_config(self) -> Dict[str, Any]:
        return {
            'agent_type': 'resource',
            'resource_filter': ['battery'],
            'reward_type': self._reward_type,
            'observation': make_observation_config(
                mode=self._observation_mode,
                supported_modes=('global', 'local', 'local_plus_forecast'),
                global_features=('total_load_mw', 'voltage_summary'),
                local_features=('soc', 'p_mw', 'time_features', 'price_signal', 'power_limits', 'capacity', 'target_soc'),
                forecast_features=('future_total_load', 'future_price_signal'),
                forecast_horizon_steps=self._forecast_horizon_steps,
            ),
            'action': {'type': 'continuous', 'mode': 'direct'},
            'constraints': {
                'soc_min': self._soc_min,
                'soc_max': self._soc_max,
                'penalty_weight': 0.5,
            },
        }


class MARLDERArbitrageTask7Days(MARLDERArbitrageTask):
    """7-day DER arbitrage variant (one week per episode)."""

    name = "marl_der_arbitrage_7d"
    description = "Multi-Agent DER Arbitrage - 7-day episode"

    def __init__(self, **kwargs):
        kwargs.setdefault('max_steps', 336)   # 7 × 48
        super().__init__(**kwargs)


class MARLDERArbitrageTask5Batteries(MARLDERArbitrageTask):
    """5-battery variant for more complex coordination."""

    name = "marl_der_arbitrage_5bat"
    description = "Multi-Agent DER Arbitrage - 5 batteries"

    def __init__(self, **kwargs):
        kwargs.setdefault('num_batteries', 5)
        super().__init__(**kwargs)
