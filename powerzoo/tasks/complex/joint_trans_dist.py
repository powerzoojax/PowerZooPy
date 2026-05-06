"""Task 5: Joint Transmission-Distribution CMDP

Coordinated control of a transmission-side OPF (IEEE 5-bus) and a distribution-side
DER storage (IEEE 33-bus), coupled at a common boundary bus.

Problem
-------
Real power systems have two distinct but coupled layers:

  Transmission (high-voltage)         Distribution (medium-voltage)
  ─────────────────────────────       ────────────────────────────────
  5 generators, Case5                 3 battery agents, Case33bw
  Minimize generation cost            Maximize arbitrage profit
  Constraints: line-flow limits       Constraints: voltage bands, SOC

The coupling link: the distribution grid imports power P_dist from bus-3 of the
transmission grid.  Generators must account for this extra load; batteries should
arbitrage the real-time price signal that propagates from the wholesale market.

CMDP cost
---------
  cost = cost_thermal_trans          # transmission thermal overloads (MW)
       + cost_voltage_dist           # distribution voltage violations (p.u.)
       + cost_soc_dist               # battery SOC bound violations

A joint episode terminates only when *all* agents report done, so the 48-step
horizon is shared across both grids.

Agent design
------------
  transmission_unit_0 … unit_4 : score ∈ [0,1]  (softmax dispatch)
  distribution_bat_0  … bat_2   : power ∈ [-P,P]  (direct charge/discharge)

  Reward (shared across all agents):
    r = -generation_cost / 1000  +  arbitrage_profit * 100

Observation
-----------
  Global (shared): total_load_mw, trans_line_flows, dist_voltage_profile, price, time
  Local (per agent): own unit/battery parameters

Data splits (non-overlapping, fixed for benchmark reproducibility)
------------------------------------------------------------------
Aligned with bundled time-series coverage (``GB_Forecast_Actual_Demand_2023_2025_30min``).

  train : 2023-07-05 ~ 2024-12-31
  val   : 2025-01-01 ~ 2025-06-30
  test  : 2025-07-01 ~ 2025-12-15  ← official benchmark split

Usage::

    from powerzoo.tasks import make_task_env

    train_env = make_task_env('joint_trans_dist', split='train')
    obs, infos = train_env.reset(seed=0)
    actions = {a: train_env.action_space[a].sample()
               for a in train_env.get_agent_ids()}
    obs, rewards, terminateds, truncateds, infos = train_env.step(actions)
"""

from typing import Dict, Any, Optional, List

from powerzoo.tasks.base import MultiAgentTask


class JointTransDistTask(MultiAgentTask):
    """Joint Transmission+Distribution CMDP Task (complex difficulty).

    Agents on the transmission side do OPF dispatch while agents on the
    distribution side do DER storage arbitrage.  Both grids share a cost
    signal that covers thermal overloads (transmission) and voltage /
    SOC violations (distribution).
    """

    name = "joint_trans_dist"
    description = (
        "Joint Transmission-Distribution CMDP: OPF (5-bus) + DER Arbitrage (33-bus)"
    )
    difficulty = "complex"
    agent_mode = "multi"

    # ---------- Fixed benchmark data splits (aligned with bundled parquet range) ----------
    SPLIT_DATES = {
        'train': ('2023-07-05', '2024-12-31'),
        'val':   ('2025-01-01', '2025-06-30'),
        'test':  ('2025-07-01', '2025-12-15'),
    }

    eval_protocol: Dict[str, Any] = {
        "n_episodes": 50,
        "seed_start": 42,
        "split": "test",
        # cost_threshold: combined trans thermal + dist voltage + SOC violations.
        # Tighter than Task 1 alone because the distribution layer adds more
        # constraint surfaces.
        "cost_threshold": 15.0,
        "metrics": [
            "mean_reward", "std_reward", "normalized_score",
            "constraint_violation_rate",
            "mean_episode_cost", "cost_violation_rate",
            # joint-specific
            "trans_thermal_violation_rate",
            "dist_voltage_violation_rate",
            "bat_soc_violation_rate",
        ],
    }

    # Constraint tightness presets.
    _TIGHTNESS_PRESETS: Dict[str, Any] = {
        'loose':    {'trans_load_ratio': 0.70, 'dist_load_ratio': 0.70,
                     'soc_min': 0.05, 'soc_max': 0.95, 'cost_threshold': 50.0},
        'standard': {'trans_load_ratio': 0.90, 'dist_load_ratio': 0.90,
                     'soc_min': 0.10, 'soc_max': 0.90, 'cost_threshold': 15.0},
        'strict':   {'trans_load_ratio': 0.98, 'dist_load_ratio': 0.95,
                     'soc_min': 0.15, 'soc_max': 0.85, 'cost_threshold':  4.0},
    }

    def __init__(
        self,
        # ---- data split ----
        split: Optional[str] = 'train',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        # ---- transmission grid ----
        trans_case: str = 'Case5',
        trans_load_ratio: float = None,
        trans_action_mode: str = 'score',
        # ---- distribution grid ----
        dist_case: str = 'Case33bw',
        dist_load_ratio: float = None,
        num_batteries: int = 3,
        battery_capacity_mwh: float = 0.5,
        battery_power_mw: float = 0.2,
        soc_min: float = None,
        soc_max: float = None,
        initial_soc: float = 0.5,
        battery_bus_ids: Optional[List[int]] = None,
        # ---- shared ----
        delta_t_minutes: int = 30,
        max_steps: int = 48,
        reward_type: str = 'shared',
        constraint_tightness: str = 'standard',
        **kwargs,
    ):
        """Initialise the Joint Trans-Dist task.

        Args:
            split:               Data split — 'train', 'val', or 'test'.
            start_date:          Explicit start date (overrides split).
            end_date:            Explicit end date (overrides split).
            trans_case:          Transmission grid case. Default 'Case5'.
            trans_load_ratio:    Max load ratio for transmission.
                                 Defaults to tightness preset.
            trans_action_mode:   'score' (softmax) or 'direct' (MW).
            dist_case:           Distribution grid case. Default 'Case33bw'.
            dist_load_ratio:     Max load ratio for distribution.
                                 Defaults to tightness preset.
            num_batteries:       Number of distribution battery agents.
            battery_capacity_mwh: Capacity per battery (MWh).
            battery_power_mw:    Max charge/discharge power per battery (MW).
            soc_min:             Minimum SOC. Defaults to tightness preset.
            soc_max:             Maximum SOC. Defaults to tightness preset.
            initial_soc:         Initial SOC (0–1).
            battery_bus_ids:     Bus IDs for battery placement in Case33bw.
            delta_t_minutes:     Time step in minutes.
            max_steps:           Max steps per episode (default 48 = 1 day).
            reward_type:         'shared' (all agents share joint reward) or
                                 'individual' (per-grid sub-rewards).
            constraint_tightness: One of 'loose', 'standard', 'strict'.
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

        # Transmission
        self._trans_case = trans_case
        self._trans_load_ratio = (
            trans_load_ratio if trans_load_ratio is not None
            else self._tightness_param('trans_load_ratio', default=0.9)
        )
        self._trans_action_mode = trans_action_mode

        # Distribution
        self._dist_case = dist_case
        self._dist_load_ratio = (
            dist_load_ratio if dist_load_ratio is not None
            else self._tightness_param('dist_load_ratio', default=0.9)
        )
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
        self._battery_bus_ids = (
            battery_bus_ids if battery_bus_ids is not None
            else [6, 12, 18, 24, 30][:num_batteries]
        )

        # Shared
        self._delta_t_minutes = delta_t_minutes
        self._max_steps = max_steps
        self._reward_type = reward_type

    # ------------------------------------------------------------------
    # Task interface
    # ------------------------------------------------------------------

    def get_scenario_config(self) -> Dict[str, Any]:
        """Return joint scenario configuration.

        The config contains two sub-grids under ``'grids'``.  The coupling
        between the grids is described in ``'coupling'``.
        """
        batteries = [
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

            # ── Two sub-grids ────────────────────────────────────────────
            'grids': {
                'transmission': {
                    'type': 'transmission',
                    'case': self._trans_case,
                    'start_date': self._start_date,
                    'end_date': self._end_date,
                    'delta_t_minutes': self._delta_t_minutes,
                    'max_load_ratio': self._trans_load_ratio,
                },
                'distribution': {
                    'type': 'distribution',
                    'case': self._dist_case,
                    'start_date': self._start_date,
                    'end_date': self._end_date,
                    'delta_t_minutes': self._delta_t_minutes,
                    'max_load_ratio': self._dist_load_ratio,
                },
            },

            # ── Grid coupling (boundary bus) ─────────────────────────────
            # The distribution grid is connected to bus-3 of the transmission
            # grid.  At each time step the distribution grid's net load is
            # added to bus-3 of the transmission grid before the OPF solve.
            'coupling': {
                'trans_bus_id': 3,     # bus-3 in Case5
                'dist_slack_bus': 1,   # slack bus of Case33bw (bus-1)
            },

            # ── Resources (distribution side only) ──────────────────────
            'resources': batteries,

            # ── Joint reward ─────────────────────────────────────────────
            # Economic components:
            #   trans : -generation_cost / 1000
            #   dist  : +arbitrage_profit × 100
            # Constraint violations go to info['cost'] exclusively.
            'reward': {
                'type': 'joint_trans_dist',
                # Transmission: economic dispatch component
                'trans_cost_weight': 1.0,
                # Distribution: battery arbitrage component
                'dist_arbitrage_weight': 100.0,
                'peak_hours':     [9, 10, 11, 17, 18, 19, 20],
                'off_peak_hours': [0, 1, 2, 3, 4, 5, 6, 23],
            },

            'episode': {'max_steps': self._max_steps},
        }

    def get_agents_config(self) -> Dict[str, Any]:
        """Return joint multi-agent configuration."""
        return {
            # Two distinct agent groups
            'agent_groups': [
                {
                    'group_id': 'transmission',
                    'agent_type': 'unit',
                    'action_mode': self._trans_action_mode,
                    'observation': {
                        'global': ['total_load_mw', 'trans_line_flows',
                                   'dist_voltage_profile', 'price_signal',
                                   'time_features'],
                        'local':  ['unit_idx', 'p_min', 'p_max', 'cost_coeffs'],
                    },
                    'action': {
                        'type': 'continuous',
                        'mode': self._trans_action_mode,
                    },
                },
                {
                    'group_id': 'distribution',
                    'agent_type': 'resource',
                    'resource_filter': ['battery'],
                    'observation': {
                        'global': ['total_load_mw', 'trans_line_flows',
                                   'dist_voltage_profile', 'price_signal',
                                   'time_features'],
                        'local':  ['soc', 'p_mw', 'power_limits'],
                    },
                    'action': {'type': 'continuous', 'mode': 'direct'},
                    'constraints': {
                        'soc_min': self._soc_min,
                        'soc_max': self._soc_max,
                    },
                },
            ],

            'reward_type': self._reward_type,
        }

    def create_env(self):
        """Create the joint trans-dist multi-agent environment."""
        try:
            from powerzoo.tasks.adapters.joint import JointTransDistMultiAgentEnv
            return JointTransDistMultiAgentEnv(self)
        except ImportError:
            raise NotImplementedError(
                "JointTransDistMultiAgentEnv is not yet implemented.\n"
                "To use this task, implement "
                "powerzoo/tasks/adapters/joint.py::JointTransDistMultiAgentEnv.\n\n"
                "The adapter should:\n"
                "  1. Instantiate both PowerEnv grids from get_scenario_config()['grids'].\n"
                "  2. At each step: solve distribution power flow, add net load to\n"
                "     the coupling bus, then solve transmission OPF.\n"
                "  3. Aggregate rewards and costs, returning CMDP-standard infos:\n"
                "     info[agent] = {'cost': float, 'costs': {thermal, voltage, soc}}.\n"
                "See task5_joint_trans_dist.py docstring for full coupling spec."
            )


class JointTransDistTask7Days(JointTransDistTask):
    """7-day variant of the joint task (one week per episode)."""

    name = "joint_trans_dist_7d"
    description = "Joint Trans-Dist CMDP — 7-day episode"

    def __init__(self, **kwargs):
        kwargs.setdefault('max_steps', 336)  # 7 × 48
        super().__init__(**kwargs)
