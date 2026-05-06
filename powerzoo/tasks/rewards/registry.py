"""Registry and factory helpers for task-level rewards."""

from typing import Any, Dict

from .base import RewardFunction
from .grid import (
    EconomicDispatchReward,
    NetworkLossReward,
    RenewablePenetrationReward,
    SafetyOnlyReward,
    ZeroReward,
    UnitCommitmentReward,
    VoltageControlReward,
    VoltageViolationReward,
)
from .market import (
    BatteryArbitrageReward,
    BatteryLMPArbitrageReward,
    BatteryLMPArbitrageV2Reward,
    DataCenterSchedulingReward,
    EVArbitrageReward,
)

REWARD_FUNCTIONS = {
    'zero': ZeroReward,
    'safety_only': SafetyOnlyReward,
    'economic_dispatch': EconomicDispatchReward,
    'network_loss': NetworkLossReward,
    'renewable_penetration': RenewablePenetrationReward,
    'voltage_control': VoltageControlReward,
    'voltage_violation': VoltageViolationReward,
    'battery_arbitrage': BatteryArbitrageReward,
    'battery_lmp_arbitrage': BatteryLMPArbitrageReward,
    'battery_lmp_arbitrage_v2': BatteryLMPArbitrageV2Reward,
    'ev_arbitrage': EVArbitrageReward,
    'dc_scheduling': DataCenterSchedulingReward,
    'unit_commitment': UnitCommitmentReward,
}

REWARD_CATEGORIES = {
    'neutral': (
        'zero',
    ),
    'safety': (
        'safety_only',
        'voltage_control',
        'voltage_violation',
    ),
    'economic': (
        'economic_dispatch',
        'renewable_penetration',
        'unit_commitment',
    ),
    'market': (
        'battery_arbitrage',
        'battery_lmp_arbitrage',
        'battery_lmp_arbitrage_v2',
        'ev_arbitrage',
        'dc_scheduling',
    ),
}


def list_reward_types(category: str | None = None) -> list[str]:
    if category is None:
        return list(REWARD_FUNCTIONS.keys())
    if category not in REWARD_CATEGORIES:
        raise ValueError(
            f"Unknown reward category: {category}. "
            f"Available: {list(REWARD_CATEGORIES.keys())}"
        )
    return list(REWARD_CATEGORIES[category])


def get_reward_function(config: Dict[str, Any]) -> RewardFunction:
    reward_type = config.get('type', 'zero')
    if reward_type not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward type: {reward_type}. "
            f"Available: {list(REWARD_FUNCTIONS.keys())}"
        )
    reward_class = REWARD_FUNCTIONS[reward_type]
    params = {k: v for k, v in config.items() if k != 'type'}
    return reward_class(**params)
