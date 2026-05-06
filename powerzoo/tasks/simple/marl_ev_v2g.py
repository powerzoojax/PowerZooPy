"""Task 3: Multi-Agent Electric Vehicle V2G/G2V Control

Problem:
- Multiple EVs connected to a distribution grid (IEEE 33-bus).
- Each EV is an independent agent deciding charge/discharge power.
- Goal: maximize arbitrage profit while ensuring departure readiness.

Data splits (non-overlapping, fixed for benchmark reproducibility)
------------------------------------------------------------------
Aligned with bundled time-series coverage (``GB_Forecast_Actual_Demand_2023_2025_30min``).

  train : 2023-07-05 ~ 2024-12-31
  val   : 2025-01-01 ~ 2025-06-30
  test  : 2025-07-01 ~ 2025-12-15  ← official benchmark split
"""

from typing import Dict, Any, List, Optional

from powerzoo.tasks.base import MultiAgentTask
from powerzoo.tasks.observation import make_observation_config


class MARLEVTask(MultiAgentTask):
    """Multi-Agent Electric Vehicle V2G/G2V Task."""

    name = "marl_ev_v2g"
    description = "Multi-Agent EV V2G/G2V - MAPPO baseline (penalty-based)"
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
        # cost_threshold: cumulative EV constraint violation per episode.
        # 5 EVs × 168 steps: allow up to 5.0 total violation units per episode.
        "cost_threshold": 5.0,
        "metrics": ["mean_reward", "std_reward", "normalized_score",
                    "constraint_violation_rate", "departure_success_rate",
                    "mean_episode_cost", "cost_violation_rate"],
    }

    # Constraint tightness presets.
    _TIGHTNESS_PRESETS: Dict[str, Any] = {
        'loose':    {'max_load_ratio': 0.70, 'soc_departure_min': 0.6, 'cost_threshold': 20.0},
        'standard': {'max_load_ratio': 0.90, 'soc_departure_min': 0.8, 'cost_threshold':  5.0},
        'strict':   {'max_load_ratio': 0.98, 'soc_departure_min': 0.9, 'cost_threshold':  1.0},
    }

    def __init__(self,
                 case: str = 'Case33bw',
                 split: Optional[str] = 'train',
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 delta_t_minutes: int = 60,
                 max_load_ratio: float = None,
                 max_steps: int = 168,        # 7 × 24 = 1 week
                 num_evs: int = 5,
                 ev_capacity_kwh: float = 60.0,
                 ev_charge_power_kw: float = 7.0,
                 ev_discharge_power_kw: float = 7.0,
                 soc_min: float = 0.1,
                 soc_max: float = 0.95,
                 soc_departure_min: float = None,
                 initial_soc: float = 0.6,
                 commute_schedules: Optional[List[List[Dict]]] = None,
                 reward_type: str = 'shared',
                 observation_mode: str = 'local_plus_forecast',
                 forecast_horizon_steps: int = 6,
                 constraint_tightness: str = 'standard',
                 **kwargs):
        """
        Args:
            case:                 Grid case (IEEE 33-bus distribution).
            split:                Data split — 'train', 'val', or 'test'.
            start_date:           Explicit start date (overrides split).
            end_date:             Explicit end date (overrides split).
            delta_t_minutes:      Time step in minutes (60 for hourly).
            max_load_ratio:       Maximum load ratio.
            max_steps:            Max steps per episode (168 = 1 week).
            num_evs:              Number of EV agents.
            ev_capacity_kwh:      Battery capacity per EV (kWh).
            ev_charge_power_kw:   Max charging power per EV (kW).
            ev_discharge_power_kw: Max V2G discharge power per EV (kW).
            soc_min:              Minimum SOC (0–1).
            soc_max:              Maximum SOC (0–1).
            soc_departure_min:    Minimum SOC required at departure.
            initial_soc:          Initial SOC (0–1).
            commute_schedules:    Commute schedules per EV.
            reward_type:          'shared' or 'individual'.
            observation_mode:     One of 'global', 'local', 'local_plus_forecast'.
            forecast_horizon_steps: Forecast horizon used in local_plus_forecast mode.
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
        self._num_evs = num_evs
        self._ev_capacity_kwh = ev_capacity_kwh
        self._ev_charge_power_kw = ev_charge_power_kw
        self._ev_discharge_power_kw = ev_discharge_power_kw
        self._soc_min = soc_min
        self._soc_max = soc_max
        self._soc_departure_min = (
            soc_departure_min if soc_departure_min is not None
            else self._tightness_param('soc_departure_min', default=0.8)
        )
        self._initial_soc = initial_soc
        self._commute_schedules = commute_schedules
        self._reward_type = reward_type
        self._observation_mode = observation_mode
        self._forecast_horizon_steps = forecast_horizon_steps

    def _get_default_commute_schedules(self) -> List[List[Dict]]:
        """Generate default diverse commute schedules for EVs."""
        patterns = [
            [{'departure': 8.0,  'arrival': 9.0,  'energy_kWh': 8.0},
             {'departure': 18.0, 'arrival': 19.0, 'energy_kWh': 8.0}],
            [{'departure': 7.0,  'arrival': 8.0,  'energy_kWh': 10.0},
             {'departure': 17.0, 'arrival': 18.0, 'energy_kWh': 10.0}],
            [{'departure': 6.0,  'arrival': 7.0,  'energy_kWh': 6.0},
             {'departure': 19.0, 'arrival': 20.0, 'energy_kWh': 6.0}],
            [{'departure': 8.0,  'arrival': 9.0,  'energy_kWh': 5.0},
             {'departure': 12.0, 'arrival': 13.0, 'energy_kWh': 3.0},
             {'departure': 18.0, 'arrival': 19.0, 'energy_kWh': 5.0}],
            [{'departure': 10.0, 'arrival': 11.0, 'energy_kWh': 4.0},
             {'departure': 16.0, 'arrival': 17.0, 'energy_kWh': 4.0}],
        ]
        return [patterns[i % len(patterns)] for i in range(self._num_evs)]

    def get_scenario_config(self) -> Dict[str, Any]:
        schedules = self._commute_schedules or self._get_default_commute_schedules()
        bus_ids = [6, 10, 14, 18, 22, 25, 28, 30, 32, 33]
        resources = [
            {
                'type': 'vehicle',
                'name': f'ev_{i}',
                'capacity_kwh': self._ev_capacity_kwh,
                'charge_power_kw': self._ev_charge_power_kw,
                'discharge_power_kw': self._ev_discharge_power_kw,
                'bus_id': bus_ids[i % len(bus_ids)],
                'soc_min': self._soc_min,
                'soc_max': self._soc_max,
                'soc_departure_min': self._soc_departure_min,
                'initial_soc': self._initial_soc,
                'commute_schedule': schedules[i % len(schedules)],
            }
            for i in range(self._num_evs)
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
                'type': 'ev_arbitrage',
                'peak_hours': [9, 10, 11, 17, 18, 19, 20],
                'off_peak_hours': [0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 23],
                'arbitrage_weight': 100.0,
                'soc_penalty_weight': 1.0,
                'departure_penalty_weight': 5.0,
                'home_violation_penalty': 100.0,
            },
            'episode': {'max_steps': self._max_steps},
        }

    def get_agents_config(self) -> Dict[str, Any]:
        return {
            'agent_type': 'resource',
            'resource_filter': ['vehicle'],
            'reward_type': self._reward_type,
            'observation': make_observation_config(
                mode=self._observation_mode,
                supported_modes=('global', 'local', 'local_plus_forecast'),
                global_features=('total_load_mw', 'voltage_summary'),
                local_features=('soc', 'is_home', 'departure_ready', 'time_to_departure', 'time_features', 'price_signal', 'power_limits', 'soc_departure_min'),
                forecast_features=('future_total_load', 'future_price_signal', 'future_home_availability'),
                forecast_horizon_steps=self._forecast_horizon_steps,
            ),
            'action': {'type': 'continuous', 'mode': 'direct'},
            'constraints': {
                'soc_min': self._soc_min,
                'soc_max': self._soc_max,
                'soc_departure_min': self._soc_departure_min,
                'home_only_charging': True,
                'penalty_weight': 0.5,
            },
        }


class MARLEVTask1Day(MARLEVTask):
    """1-day EV task variant for quick testing."""

    name = "marl_ev_v2g_1d"
    description = "Multi-Agent EV V2G - 1-day episode"

    def __init__(self, **kwargs):
        kwargs.setdefault('max_steps', 24)
        super().__init__(**kwargs)


class MARLEVTask10EVs(MARLEVTask):
    """10-EV variant for more complex coordination."""

    name = "marl_ev_v2g_10ev"
    description = "Multi-Agent EV V2G - 10 EVs"

    def __init__(self, **kwargs):
        kwargs.setdefault('num_evs', 10)
        super().__init__(**kwargs)
