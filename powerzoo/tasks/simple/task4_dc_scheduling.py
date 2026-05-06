"""Task 4: Data Center GPU Scheduling

Problem:
- A single AI data center is connected to an IEEE 33-bus distribution grid.
- The RL agent decides GPU allocation ratios (training / finetuning) and
  cooling setpoint each time step.
- Goal: minimise total power consumption while meeting task deadlines (SLA)
  with thermal safety handled via the cost channel.

Grid: Case33bw (distribution, radial, 33 buses)
  Data center is sited at bus 6.
  Default scale: 1000 x H100 GPUs (~1.1 MW peak IT load).

Data splits (non-overlapping, fixed for benchmark reproducibility)
------------------------------------------------------------------
  train : 2023-07-05 ~ 2024-12-31
  val   : 2025-01-01 ~ 2025-06-30
  test  : 2025-07-01 ~ 2025-12-15  ← official benchmark split (parquet max date)
"""

from typing import Any, Dict, Optional

from powerzoo.tasks.base import SingleAgentTask


class DCSchedulingTask(SingleAgentTask):
    """Data Center GPU Scheduling Task.

    Single-agent task: one datacenter on a distribution grid.  The agent
    controls GPU scheduling ratios and cooling setpoint (3-D continuous).
    Reward is objective-only (power, SLA, PUE); thermal safety goes to cost.
    """

    name = "dc_scheduling"
    description = "Data Center GPU Scheduling - energy-SLA objective with thermal safety cost"
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
        "cost_threshold": 5.0,
        "metrics": [
            "mean_reward", "std_reward", "normalized_score",
            "mean_episode_cost", "cost_violation_rate",
            "mean_sla_violations", "mean_pue",
        ],
    }

    _TIGHTNESS_PRESETS: Dict[str, Any] = {
        'loose':    {'max_load_ratio': 0.70, 't_critical': 38.0, 'cost_threshold': 10.0},
        'standard': {'max_load_ratio': 0.90, 't_critical': 35.0, 'cost_threshold': 5.0},
        'strict':   {'max_load_ratio': 0.98, 't_critical': 32.0, 'cost_threshold': 2.0},
    }

    def __init__(
        self,
        case: str = 'Case33bw',
        split: Optional[str] = 'train',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        delta_t_minutes: int = 30,
        max_load_ratio: Optional[float] = None,
        max_steps: int = 48,
        # Datacenter parameters
        bus_id: int = 6,
        n_gpus: int = 1000,
        gpu_idle_w: float = 55.0,
        gpu_active_w: float = 1100.0,
        p_base_mw: float = 0.5,
        infer_gpu_peak: int = 400,
        cop_ref: float = 5.0,
        cop_decay: float = 0.04,
        t_ref: float = 20.0,
        c_thermal: float = 500.0,
        ua_cooling: float = 200.0,
        h_wall: float = 5.0,
        t_set_min: float = 18.0,
        t_set_max: float = 27.0,
        t_critical: Optional[float] = None,
        p_aux_frac: float = 0.05,
        train_cfg: Optional[Dict[str, Any]] = None,
        finetune_cfg: Optional[Dict[str, Any]] = None,
        # Reward tuning
        power_weight: float = 1.0,
        sla_weight: float = 50.0,
        overtemp_weight: float = 20.0,
        pue_bonus_weight: float = 0.5,
        target_pue: float = 1.3,
        constraint_tightness: str = 'standard',
        **kwargs,
    ):
        """Initialize the DC Scheduling task.

        Args:
            case:            Grid case (default Case33bw).
            split:           Data split -- 'train', 'val', or 'test'.
            start_date:      Explicit start date (overrides split).
            end_date:        Explicit end date (overrides split).
            delta_t_minutes: Time step in minutes.
            max_load_ratio:  Maximum load ratio for the grid.
            max_steps:       Steps per episode (default 48 = 1 day @ 30 min).
            bus_id:          Bus where the datacenter connects (default 6).
            n_gpus:          Total GPU count (default 1000).
            gpu_idle_w:      Per-GPU idle power in watts (default 55, H100).
            gpu_active_w:    Per-GPU system-level active power in watts (default 1100, H100 node).
            p_base_mw:       Baseline non-GPU IT power in MW (default 0.5).
            infer_gpu_peak:  Peak inference GPU count (default 400).
            cop_ref:         COP at reference temperature (default 5.0).
            cop_decay:       COP decay rate per degree (default 0.04).
            t_ref:           Reference temperature for COP (default 20).
            c_thermal:       Thermal capacitance kWh/C (default 500).
            ua_cooling:      Cooling heat-transfer kW/C (default 200).
            h_wall:          Envelope heat-transfer kW/C (default 5).
            t_set_min:       Min cooling setpoint C (default 18).
            t_set_max:       Max cooling setpoint C (default 27).
            t_critical:      Over-temperature threshold C (uses tightness default).
            p_aux_frac:      Auxiliary power fraction (default 0.05).
            train_cfg:       Training task generation config overrides.
            finetune_cfg:    Finetuning task generation config overrides.
            power_weight:    Reward weight for power cost.
            sla_weight:      Reward penalty per SLA violation.
            overtemp_weight: Legacy arg kept for config compatibility; thermal
                             safety now goes to cost instead of reward.
            pue_bonus_weight: Reward bonus for low PUE.
            target_pue:      PUE target for bonus calculation.
            constraint_tightness: 'loose', 'standard', or 'strict'.
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

        # Datacenter config
        self._bus_id = bus_id
        self._n_gpus = n_gpus
        self._gpu_idle_w = gpu_idle_w
        self._gpu_active_w = gpu_active_w
        self._p_base_mw = p_base_mw
        self._infer_gpu_peak = infer_gpu_peak
        self._cop_ref = cop_ref
        self._cop_decay = cop_decay
        self._t_ref = t_ref
        self._c_thermal = c_thermal
        self._ua_cooling = ua_cooling
        self._h_wall = h_wall
        self._t_set_min = t_set_min
        self._t_set_max = t_set_max
        self._t_critical = (
            t_critical if t_critical is not None
            else self._tightness_param('t_critical', default=35.0)
        )
        self._p_aux_frac = p_aux_frac
        self._train_cfg = train_cfg
        self._finetune_cfg = finetune_cfg

        # Reward config
        self._power_weight = power_weight
        self._sla_weight = sla_weight
        self._overtemp_weight = overtemp_weight
        self._pue_bonus_weight = pue_bonus_weight
        self._target_pue = target_pue

    def get_scenario_config(self) -> Dict[str, Any]:
        dc_config: Dict[str, Any] = {
            'type': 'datacenter',
            'name': 'dc_0',
            'bus_id': self._bus_id,
            'n_gpus': self._n_gpus,
            'gpu_idle_w': self._gpu_idle_w,
            'gpu_active_w': self._gpu_active_w,
            'p_base_mw': self._p_base_mw,
            'infer_gpu_peak': self._infer_gpu_peak,
            'cop_ref': self._cop_ref,
            'cop_decay': self._cop_decay,
            't_ref': self._t_ref,
            'c_thermal': self._c_thermal,
            'ua_cooling': self._ua_cooling,
            'h_wall': self._h_wall,
            't_set_min': self._t_set_min,
            't_set_max': self._t_set_max,
            't_critical': self._t_critical,
            'p_aux_frac': self._p_aux_frac,
        }
        if self._train_cfg is not None:
            dc_config['train_cfg'] = self._train_cfg
        if self._finetune_cfg is not None:
            dc_config['finetune_cfg'] = self._finetune_cfg

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
            'resources': [dc_config],
            'reward': {
                'type': 'dc_scheduling',
                'power_weight': self._power_weight,
                'sla_weight': self._sla_weight,
                'pue_bonus_weight': self._pue_bonus_weight,
                'target_pue': self._target_pue,
            },
            'episode': {'max_steps': self._max_steps},
        }

    def get_agents_config(self) -> Dict[str, Any]:
        return {
            'agent_type': 'single',
            'obs_keys': ['grid', 'resources', 'time'],
            'resource_names': ['dc_0'],
        }


class DCSchedulingTask7Days(DCSchedulingTask):
    """7-day datacenter scheduling variant (one week per episode)."""

    name = "dc_scheduling_7d"
    description = "Data Center GPU Scheduling - 7-day episode"

    def __init__(self, **kwargs):
        kwargs.setdefault('max_steps', 336)  # 7 x 48
        super().__init__(**kwargs)
