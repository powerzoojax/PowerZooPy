"""DC Microgrid benchmark task definitions.

Registers two tasks:
    ``dc_microgrid``      — standard config (PPO / objective-only)
    ``dc_microgrid_safe`` — CMDP config with vector thresholds and scalar
                            compatibility budget 0.5

Both tasks create a :class:`~powerzoo.envs.microgrid.DCMicrogridEnv`
directly, bypassing PowerEnv (no external grid is involved).

Benchmark parity surface
------------------------
These tasks share the same task-setting semantics as the JAX-side
``DataCenterMicrogridEnv`` / ``dc-microgrid`` / ``dc-microgrid-safe`` presets:

- Self-contained behind-the-meter microgrid (no external grid).
- Default 288 steps × 5 min = 24 h episode.
- 5-D action: ``[train_sched, ft_sched, cooling_setpoint, battery_power, dg_power]``.
- Resources: DataCenter + Battery + PV + Diesel Generator.
- Reward / cost separated; cost channels: ``cost_sla``, ``cost_overtemp``,
  ``cost_power_deficit``; ``info["reward_vector"]`` = 3-D.
- Workload profile sources: ``google`` / ``azure`` / ``alibaba`` / ``synthetic``.
- OOD scenarios: ``workload_swap``, ``workload_shock``, ``renewable_drought``,
  ``cooling_stress``, ``dg_derating``, ``sla_tighten``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from powerzoo.tasks.base import ConstraintSpec, SingleAgentTask


class DCMicrogridTask(SingleAgentTask):
    """Data Center Microgrid — self-contained 288×5min benchmark task.

    Creates a :class:`~powerzoo.envs.microgrid.DCMicrogridEnv` directly.

    Profile loading
    ---------------
    Pass ``workload_source`` (``'google'``, ``'azure'``, ``'alibaba'``, or
    ``'synthetic'``) to load a workload profile at reset time.  If the
    parquet data is unavailable the env falls back to synthetic profiles
    (use ``strict_workload=True`` to suppress silent fallback).

    OOD evaluation
    --------------
    Pass ``ood_scenario`` (one of the six OOD scenario names) to shift the
    environment into an out-of-distribution operating regime.
    """

    name = "dc_microgrid"
    description = (
        "DC μGrid: self-contained behind-the-meter microgrid, 1 agent, 288×5min. "
        "5-D action [train, ft, cool, batt, dg]. "
        "reward/cost separated; cost channels: SLA, over-temp, power-deficit."
    )
    difficulty = "simple"
    agent_mode = "single"
    training_contract = "cmdp_env_plus_mdp_fallback"

    _TIGHTNESS_PRESETS: Dict[str, Any] = {
        'loose':    {'cost_threshold': 2.0,  't_critical': 38.0},
        'standard': {'cost_threshold': 0.5,  't_critical': 35.0},
        'strict':   {'cost_threshold': 0.2,  't_critical': 32.0},
    }

    eval_protocol: Dict[str, Any] = {
        "n_episodes":           100,
        "seed_start":           42,
        "split":                None,   # no date-split for microgrid
        "constraint_names":     ["sla", "overtemp", "power_deficit"],
        "cost_thresholds":      [0.2, 0.15, 0.15],
        "cost_threshold":       0.5,
        "metrics": [
            "mean_reward", "std_reward", "normalized_score",
            "mean_episode_cost", "cost_violation_rate",
            "mean_sla_violations", "mean_pue",
            "mean_power_deficit", "mean_carbon_kg",
        ],
    }

    def __init__(
        self,
        # Episode
        max_steps: int = 288,
        delta_t_minutes: float = 5.0,
        # Profile source
        workload_source: str = 'synthetic',
        strict_workload: bool = False,
        ood_scenario: Optional[str] = None,
        # DC params (key overrides — remainder use DCMicrogridEnv defaults)
        n_gpus: int = 1000,
        t_critical: Optional[float] = None,
        # Constraint tightness
        constraint_tightness: str = 'standard',
        **kwargs,
    ):
        super().__init__(constraint_tightness=constraint_tightness, **kwargs)

        self._max_steps      = int(max_steps)
        self._dt_minutes     = float(delta_t_minutes)
        self._workload_src   = workload_source
        self._strict_wl      = bool(strict_workload)
        self._ood_scenario   = ood_scenario
        self._n_gpus         = int(n_gpus)
        self._t_critical     = (
            t_critical if t_critical is not None
            else self._tightness_param('t_critical', default=35.0)
        )
        self._env_kwargs = kwargs  # remaining overrides forwarded to env

    def constraint_spec(self) -> ConstraintSpec:
        return ConstraintSpec(
            selected_names=("sla", "overtemp", "power_deficit"),
            thresholds=(0.2, 0.15, 0.15),
            fallback_weights=(1.0, 1.0, 1.0),
        )

    # ------------------------------------------------------------------
    # Task interface (SingleAgentTask)
    # ------------------------------------------------------------------

    def get_scenario_config(self) -> Dict[str, Any]:
        return {
            'name':        f'{self.name}_scenario',
            'description': self.description,
            'topology':    'behind_the_meter_microgrid',
            'episode':     {'max_steps': self._max_steps, 'delta_t_minutes': self._dt_minutes},
            'action_dim':  5,
            'resources':   ['datacenter', 'battery', 'pv', 'diesel_generator'],
            'workload_source': self._workload_src,
            'ood_scenario':    self._ood_scenario,
        }

    def get_agents_config(self) -> Dict[str, Any]:
        return {
            'agent_type': 'single',
            'action_names': ['train_sched', 'ft_sched', 'cooling_setpoint',
                             'battery_power', 'dg_power'],
        }

    def create_single_agent_env(self):
        """Return a :class:`DCMicrogridEnv`, bypassing PowerEnv/GridEnv."""
        from powerzoo.envs.microgrid.dc_microgrid_env import DCMicrogridEnv
        from powerzoo.data.dc_microgrid_profiles import (
            load_workload_profiles,
            apply_ood_transform,
        )

        # Load profiles
        profiles = load_workload_profiles(
            source=self._workload_src,
            n_steps=self._max_steps,
            strict=self._strict_wl,
        )
        if self._ood_scenario is not None:
            profiles = apply_ood_transform(profiles, self._ood_scenario, strict=True)

        ood_params = profiles.get('_ood_params', {})

        env = DCMicrogridEnv(
            n_gpus=self._n_gpus,
            t_critical=self._t_critical,
            max_steps=self._max_steps,
            delta_t_minutes=self._dt_minutes,
            cpu_profile=profiles.get('cpu'),
            solar_profile=profiles.get('solar'),
            outdoor_temp_profile=profiles.get('temp'),
            dg_derating_factor=ood_params.get('dg_derating_factor', 1.0),
            sla_tighten_factor=ood_params.get('sla_tighten_factor', 1.0),
            **{k: v for k, v in self._env_kwargs.items()
               if k not in ('constraint_tightness',)},
        )
        return self._wrap_single_agent_cmdp(env)

    # Override Task.create_env() to return single-agent env directly.
    def create_env(self):
        return self.create_single_agent_env()


class DCMicrogridSafeTask(DCMicrogridTask):
    """DC μGrid CMDP variant with scalar safe-RL compatibility projection.

    Semantics identical to ``dc_microgrid``; the CMDP cost threshold is
    frozen as ``(0.2, 0.15, 0.15)`` for ``sla``, ``overtemp``, and
    ``power_deficit`` with scalar compatibility alias ``0.5``.
    """

    name = "dc_microgrid_safe"
    description = (
        "DC μGrid CMDP: SLA + over-temp + power-deficit vector costs; "
        "scalar compatibility cost_threshold=0.5. "
        "Otherwise identical to dc_microgrid."
    )
    training_contract = "cmdp_env_plus_scalar_safe_projection"

    eval_protocol: Dict[str, Any] = {
        "n_episodes":           100,
        "seed_start":           42,
        "split":                None,
        "constraint_names":     ["sla", "overtemp", "power_deficit"],
        "cost_thresholds":      [0.2, 0.15, 0.15],
        "cost_threshold":       0.5,
        "metrics": [
            "mean_reward", "std_reward", "normalized_score",
            "mean_episode_cost", "cost_violation_rate",
            "mean_sla_violations", "mean_pue",
            "mean_power_deficit", "mean_carbon_kg",
        ],
    }
