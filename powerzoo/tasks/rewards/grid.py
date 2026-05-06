"""Grid-oriented reward functions."""

from typing import Any, Dict

import numpy as np

from .base import RewardFunction


class SafetyOnlyReward(RewardFunction):
    def __init__(self, penalty_per_violation: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.penalty_per_violation = penalty_per_violation

    def compute(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        reward = 0.0
        if not state.get('is_safe', True):
            n_violations = len(state.get('safety_info', {}).get('unsafe_line_ids', []))
            reward -= self.penalty_per_violation * n_violations
        return self._maybe_normalize(reward)


class ZeroReward(RewardFunction):
    """Neutral fallback reward used when no task-level objective is configured."""

    def compute(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        return self._maybe_normalize(0.0)


class EconomicDispatchReward(RewardFunction):
    def __init__(
        self,
        cost_weight: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cost_weight = cost_weight

    def compute(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        reward = 0.0
        if 'opf_cost' in state:
            reward += -self.cost_weight * state['opf_cost']
        return self._maybe_normalize(reward)


class RenewablePenetrationReward(RewardFunction):
    def __init__(
        self,
        renewable_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.renewable_weight = renewable_weight

    def compute(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        reward = 0.0
        total_gen = info.get('total_generation_mw', 1.0)
        if total_gen > 0:
            renewable_gen = sum(
                abs(res['current_p_mw'])
                for res in info.get('resource_status', {}).values()
                if 'current_p_mw' in res and res['current_p_mw'] < 0
            )
            reward += self.renewable_weight * (renewable_gen / total_gen) * 100
        return self._maybe_normalize(reward)


class VoltageControlReward(RewardFunction):
    def __init__(
        self,
        voltage_deviation_weight: float = 1.0,
        target_voltage: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.voltage_deviation_weight = voltage_deviation_weight
        self.target_voltage = target_voltage

    def compute(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        reward = 0.0
        if 'nodes' in state and 'v_mag' in state['nodes'].columns:
            deviations = np.abs(state['nodes']['v_mag'].values - self.target_voltage)
            reward += -self.voltage_deviation_weight * np.mean(deviations) * 100
        return self._maybe_normalize(reward)


class UnitCommitmentReward(EconomicDispatchReward):
    def __init__(self, cost_weight: float = 0.001, **kwargs):
        super().__init__(cost_weight=cost_weight, **kwargs)


class NetworkLossReward(RewardFunction):
    """Reward = -loss_penalty_weight * p_loss_MW.

    Matches PowerZooJax DistGridEnv default reward for the DSO task.
    ``p_loss_MW`` is read from the info dict produced by DistGridEnv.build_info().
    """

    def __init__(self, loss_penalty_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.loss_penalty_weight = loss_penalty_weight

    def compute(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        loss_mw = float(info.get('p_loss_MW', 0.0))
        return self._maybe_normalize(-self.loss_penalty_weight * loss_mw)


class VoltageViolationReward(VoltageControlReward):
    """VoltageControlReward extended with a quadratic per-bus violation penalty.

    reward = -dev_weight * mean(|V - 1|) * 100
             - violation_weight * sum(max(0, v_threshold - V_i)^2)

    The violation term creates a strong incentive to push every bus above
    ``v_threshold`` (default 0.95 p.u., the IEEE lower limit), while the
    inherited mean-deviation term continues to penalise all deviations from
    nominal.  The two terms compose additively so the reward scale grows with
    the number of violating buses.
    """

    def __init__(
        self,
        violation_weight: float = 500.0,
        v_threshold: float = 0.95,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.violation_weight = violation_weight
        self.v_threshold = v_threshold

    def compute(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        reward = super().compute(state, info)
        if 'nodes' in state and 'v_mag' in state['nodes'].columns:
            v = state['nodes']['v_mag'].values
            reward -= self.violation_weight * float(
                np.sum(np.maximum(0.0, self.v_threshold - v) ** 2)
            )
        return self._maybe_normalize(reward)
