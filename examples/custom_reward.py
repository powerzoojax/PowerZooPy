"""custom_reward.py — three patterns for overriding the per-step reward.

Pattern 1: Built-in reward type by name (dict)
Pattern 2: Lambda / callable
Pattern 3: RewardFunction subclass

Usage
-----
    python examples/custom_reward.py

Requirements: pip install powerzoo[rl]
"""

from __future__ import annotations

from powerzoo.rl import make_env, RewardWrapper
from powerzoo.tasks.rewards.base import RewardFunction


# ── Pattern 1: built-in reward type override ─────────────────────────────────
# Switch from the default reward to lmp_arbitrage via a config dict.

env1 = make_env(
    'battery_arbitrage',
    reward={'type': 'battery_lmp_arbitrage', 'profit_weight': 2.0},
)
obs, _ = env1.reset(seed=0)
obs, r1, _, _, _ = env1.step(env1.action_space.sample())
print(f"Pattern 1 — lmp_arbitrage reward: {r1:.4f}")


# ── Pattern 2: lambda callable ────────────────────────────────────────────────
# Use any callable that takes (state_dict, info_dict) and returns a float.

def my_reward(state, info):
    """Minimize power loss, penalize constraint violations."""
    loss = state.get('p_loss_MW', 0.0) or 0.0
    cost = info.get('cost_sum', 0.0) or 0.0
    return -loss - 0.1 * cost

env2 = make_env('battery_arbitrage', reward=my_reward)
obs, _ = env2.reset(seed=0)
obs, r2, _, _, _ = env2.step(env2.action_space.sample())
print(f"Pattern 2 — lambda reward:        {r2:.4f}")


# ── Pattern 3: RewardFunction subclass ───────────────────────────────────────
# Full control: access to state dict and the normalize helper.

class EnergyEfficiencyReward(RewardFunction):
    """Reward high battery round-trip efficiency."""

    def __init__(self, efficiency_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.efficiency_weight = efficiency_weight

    def compute(self, state, info):
        soc = state.get('soc', 0.5)
        cost = info.get('cost_sum', 0.0) or 0.0
        return self.efficiency_weight * soc - cost

env3 = make_env('battery_arbitrage')
env3 = RewardWrapper(env3, EnergyEfficiencyReward(efficiency_weight=2.0))
obs, _ = env3.reset(seed=0)
obs, r3, _, _, _ = env3.step(env3.action_space.sample())
print(f"Pattern 3 — subclass reward:      {r3:.4f}")
