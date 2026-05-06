"""Explicit public benchmark task surface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from powerzoo.tasks.registry import get_task_info, make_task


PUBLIC_TASKS = (
    'dso',
    'comparison_tso_centralized',
    'marl_ders_benchmark',
    'gencos_bidding',
    'dc_microgrid',
    'dc_microgrid_safe',
)


_PUBLIC_TASK_CARD_OVERRIDES: Dict[str, Dict[str, Any]] = {
    'dso': {
        'name': 'dso',
        'description': 'DSO flex-load benchmark — Case33bw, 6 FlexLoad devices, 48×30min.',
        'difficulty': 'middle',
        'agent_mode': 'single',
        'route': 'make_dso_env(...)',
        'frameworks': ('gymnasium',),
        'grid_type': 'distribution',
        'grid_case': 'Case33bw',
        'default_episode_horizon_steps': 48,
        'default_observation_mode': 'flattened',
        'supported_observation_modes': ('flattened',),
        'benchmark_family': 'intertemporal_der_control',
        'reward_contract': 'Objective-only network-loss reward; constraint costs stay separate.',
        'cost_contract': (
            'Core env emits full vector in info["constraint_costs"]; '
            'task CMDP selects ["voltage_violation"] into info["selected_constraint_costs"]. '
            'Legacy scalar info["cost"] is compatibility-only via wrappers.'
        ),
        'action_contract': 'Twelve continuous controls: [curtail_i, shift_out_i] for 6 FlexLoad devices.',
        'training_contract': 'cmdp_env_plus_scalar_safe_projection',
        'constraint_names': ['voltage_violation'],
        'cost_thresholds': [5.0],
        'fallback_weights': [1.0],
        'cost_threshold': 5.0,
        'public': True,
        'task_id': 'dso',
    },
    'comparison_tso_centralized': {
        'route': 'CentralizedComparisonTSOEnv(TaskCMDPWrapper)',
        'frameworks': ('gymnasium',),
        'benchmark_family': 'security_dispatch',
        'reward_contract': 'Objective-only operating-cost reward; reserve shortfall stays in the cost channel.',
        'cost_contract': (
            'Core env emits thermal overload + reserve shortfall via '
            'info["constraint_costs"]; scalar info["cost"] is compatibility-only.'
        ),
        'action_contract': 'Continuous Box(108): [commit_intent(54) | dispatch_score(54)].',
    },
    'marl_ders_benchmark': {
        'route': 'TaskResourceMultiAgentEnv',
        'frameworks': ('auto', 'rllib', 'pettingzoo'),
        'benchmark_family': 'intertemporal_der_control',
        'reward_contract': 'Benchmark env exposes CMDP costs explicitly; current MARL training uses MDP fallback.',
        'cost_contract': (
            'Per-agent info["costs"] uses exact names '
            '{"voltage_violation","thermal_overload","resource"} plus '
            'fixed-order info["constraint_costs"].'
        ),
        'action_contract': 'Twelve heterogeneous 2-D continuous actions (Battery / PV / FlexLoad).',
    },
    'gencos_bidding': {
        'route': 'GenCosMARLEnv',
        'frameworks': ('auto', 'pettingzoo'),
        'benchmark_family': 'market_lite',
        'reward_contract': 'Per-agent dispatch profit reward; thermal overload stays in the cost channel.',
        'cost_contract': (
            'Per-agent info exposes thermal overload via '
            'info["constraint_costs"] / info["costs"]["thermal_overload"].'
        ),
        'action_contract': 'Five agents, each emitting a 3-segment monotone markup vector.',
    },
    'dc_microgrid': {
        'route': 'DCMicrogridEnv(TaskCMDPWrapper)',
        'frameworks': ('gymnasium',),
        'benchmark_family': 'intertemporal_der_control',
        'reward_contract': (
            'Scalarized multi-objective env reward; benchmark CMDP costs are '
            'exposed separately and current reward-only training should use MDP fallback.'
        ),
        'cost_contract': (
            'Core env emits cost_sla / cost_overtemp / cost_power_deficit and the '
            'vector info["constraint_costs"] in the order ["sla","overtemp","power_deficit"]; '
            'scalar info["cost"] is compatibility-only.'
        ),
        'action_contract': (
            'Five continuous controls: train_sched [0,1], ft_sched [0,1], '
            'cooling_setpoint [0,1], battery_power [-1,1], dg_power [0,1].'
        ),
    },
    'dc_microgrid_safe': {
        'route': 'DCMicrogridEnv(TaskCMDPWrapper)',
        'frameworks': ('gymnasium',),
        'benchmark_family': 'intertemporal_der_control',
        'reward_contract': (
            'Same env reward as dc_microgrid; safe-RL libraries consume a scalar projection '
            'of the selected CMDP vector through compatibility wrappers.'
        ),
        'cost_contract': (
            'Selected CMDP thresholds are [0.2, 0.15, 0.15] for '
            'cost_sla / cost_overtemp / cost_power_deficit; scalar threshold 0.5 is a legacy alias.'
        ),
        'action_contract': (
            'Five continuous controls: train_sched [0,1], ft_sched [0,1], '
            'cooling_setpoint [0,1], battery_power [-1,1], dg_power [0,1].'
        ),
    },
}


def _card_from_task(name: str) -> Dict[str, Any]:
    task = make_task(name)
    raw = dict(get_task_info(name))
    scenario = task.get_scenario_config()
    observation = task.get_observation_config()
    spec = task.constraint_spec()

    card: Dict[str, Any] = {
        'task_id': name,
        'public': True,
        'grid_type': scenario.get('grid', {}).get('type'),
        'grid_case': scenario.get('grid', {}).get('case'),
        'default_episode_horizon_steps': scenario.get('episode', {}).get('max_steps'),
        'default_observation_mode': (
            observation.get('mode') if isinstance(observation, dict) else 'flattened'
        ),
        'supported_observation_modes': (
            tuple(observation.get('supported_modes', ())) if isinstance(observation, dict) else ('flattened',)
        ),
        'training_contract': getattr(task, 'training_contract', 'legacy'),
    }
    card.update(raw)
    if spec is not None:
        card.update({
            'constraint_names': list(spec.selected_names),
            'cost_thresholds': list(spec.thresholds),
            'fallback_weights': list(spec.fallback_weights),
            'cost_threshold': spec.scalar_threshold,
        })
    card.update(_PUBLIC_TASK_CARD_OVERRIDES.get(name, {}))
    return card


def list_public_tasks(difficulty: Optional[str] = None) -> List[str]:
    """List public benchmark tasks in stable documentation order."""
    if difficulty is None:
        return list(PUBLIC_TASKS)
    result: List[str] = []
    for task_name in PUBLIC_TASKS:
        card = get_public_task_info(task_name)
        if card.get('difficulty') == difficulty:
            result.append(task_name)
    return result


def get_public_task_info(name: str) -> Dict[str, Any]:
    """Return metadata for one public benchmark task."""
    if name not in PUBLIC_TASKS:
        raise ValueError(f"Unknown public task: '{name}'. Available tasks: {list(PUBLIC_TASKS)}")
    if name == 'dso':
        return dict(_PUBLIC_TASK_CARD_OVERRIDES['dso'])
    return _card_from_task(name)


def get_public_task_catalog(difficulty: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return benchmark task cards for the public surface."""
    return [
        get_public_task_info(task_name)
        for task_name in list_public_tasks(difficulty=difficulty)
    ]


__all__ = [
    'PUBLIC_TASKS',
    'list_public_tasks',
    'get_public_task_info',
    'get_public_task_catalog',
]
