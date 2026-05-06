"""Market- and storage-oriented reward functions."""

from typing import Any, Dict

import numpy as np

from .base import RewardFunction


class BatteryArbitrageReward(RewardFunction):
    def __init__(
        self,
        peak_hours: list = None,
        off_peak_hours: list = None,
        arbitrage_weight: float = 1.0,
        soc_penalty_weight: float = 0.1,
        target_soc: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.peak_hours = peak_hours if peak_hours else [10, 11, 12, 13, 14, 15, 16, 17, 18]
        self.off_peak_hours = off_peak_hours if off_peak_hours else [0, 1, 2, 3, 4, 5]
        self.arbitrage_weight = arbitrage_weight
        self.soc_penalty_weight = soc_penalty_weight
        self.target_soc = target_soc

    def compute(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        reward = 0.0
        time_step = info.get('step_within_day', info.get('time_step', 0))
        delta_t_minutes = info.get('delta_t_minutes', 60)
        hour = int((time_step * delta_t_minutes) / 60) % 24

        total_battery_power = 0.0
        if 'battery_power_mw' in info:
            total_battery_power = info['battery_power_mw']
        elif 'resources' in info:
            for res_id, res_info in info.get('resources', {}).items():
                if 'battery' in res_id.lower() or 'bat' in res_id.lower():
                    total_battery_power += res_info.get('current_p_mw', 0.0)

        if hour in self.peak_hours:
            reward += self.arbitrage_weight * total_battery_power
        elif hour in self.off_peak_hours:
            reward += -self.arbitrage_weight * total_battery_power

        if 'resources' in info:
            for res_id, res_info in info.get('resources', {}).items():
                if 'battery' in res_id.lower() and 'soc' in res_info:
                    soc_deviation = abs(res_info['soc'] - self.target_soc)
                    reward += -self.soc_penalty_weight * soc_deviation * 10

        return self._maybe_normalize(reward)


class BatteryLMPArbitrageReward(RewardFunction):
    def __init__(
        self,
        battery_bus_id: int = None,
        profit_weight: float = 0.01,
        soc_penalty_weight: float = 0.1,
        target_soc: float = 0.5,
        delta_t_hours: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.battery_bus_id = battery_bus_id
        self.profit_weight = profit_weight
        self.soc_penalty_weight = soc_penalty_weight
        self.target_soc = target_soc
        self.delta_t_hours = delta_t_hours

    def compute(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        delta_t_minutes = info.get('delta_t_minutes', 60)
        delta_t_hours = delta_t_minutes / 60.0
        lmp = info.get('lmp', None)
        if lmp is None:
            return 0.0

        total_profit = 0.0
        soc_penalty = 0.0
        for res_id, res_info in info.get('resources', {}).items():
            if 'battery' not in res_id.lower() and 'bat' not in res_id.lower():
                continue
            battery_power = res_info.get('current_p_mw', 0.0)
            if self.battery_bus_id is not None:
                bus_id = self.battery_bus_id
            elif 'bus_id' in res_info:
                bus_id = res_info['bus_id']
            else:
                continue
            node_idx = bus_id - 1 if bus_id > 0 else 0
            if node_idx >= len(lmp):
                continue
            nodal_lmp = lmp[node_idx]
            total_profit += nodal_lmp * battery_power * delta_t_hours
            if 'soc' in res_info:
                soc = res_info['soc']
                soc_deviation = abs(soc - self.target_soc)
                soc_penalty += -self.soc_penalty_weight * soc_deviation * 10

        reward = self.profit_weight * total_profit + soc_penalty
        return self._maybe_normalize(reward)


class BatteryLMPArbitrageV2Reward(RewardFunction):
    def __init__(
        self,
        battery_bus_id: int = None,
        profit_weight: float = 1.0,
        soc_penalty_weight: float = 10.0,
        opportunity_weight: float = 0.5,
        target_soc: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.battery_bus_id = battery_bus_id
        self.profit_weight = profit_weight
        self.soc_penalty_weight = soc_penalty_weight
        self.opportunity_weight = opportunity_weight
        self.target_soc = target_soc

    def compute(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        delta_t_minutes = info.get('delta_t_minutes', 60)
        delta_t_hours = delta_t_minutes / 60.0
        lmp = info.get('lmp', None)
        if lmp is None:
            return 0.0

        total_profit = 0.0
        total_soc_penalty = 0.0
        total_opportunity_cost = 0.0
        for res_id, res_info in info.get('resources', {}).items():
            if 'battery' not in res_id.lower() and 'bat' not in res_id.lower():
                continue
            battery_power = res_info.get('current_p_mw', 0.0)
            soc = res_info.get('soc', 0.5)
            if self.battery_bus_id is not None:
                bus_id = self.battery_bus_id
            elif 'bus_id' in res_info:
                bus_id = res_info['bus_id']
            else:
                continue
            node_idx = bus_id - 1 if bus_id > 0 else 0
            if node_idx >= len(lmp):
                continue
            nodal_lmp = lmp[node_idx]
            total_profit += nodal_lmp * battery_power * delta_t_hours
            total_soc_penalty += -self.soc_penalty_weight * abs(soc - self.target_soc) * 10
            lmp_normalized = np.clip((nodal_lmp - 10) / 30, 0, 1)
            high_price_loss = lmp_normalized * (1 - soc) * 20
            low_price_loss = (1 - lmp_normalized) * soc * 20
            total_opportunity_cost += -(high_price_loss + low_price_loss)

        reward = (
            self.profit_weight * total_profit
            + total_soc_penalty
            + self.opportunity_weight * total_opportunity_cost
        )
        return self._maybe_normalize(reward)


class DataCenterSchedulingReward(RewardFunction):
    """Reward for datacenter GPU scheduling tasks.

    Combines objective terms only:
    - **Power cost**: penalise total DC power consumption (lower is better).
    - **SLA compliance**: penalise each new SLA violation (missed deadline).
    - **Efficiency bonus**: reward operation below the target PUE.

    Thermal safety is exposed through the cost channel, not the reward.
    """

    def __init__(
        self,
        power_weight: float = 1.0,
        sla_weight: float = 50.0,
        overtemp_weight: float = 20.0,
        pue_bonus_weight: float = 0.5,
        target_pue: float = 1.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.power_weight = power_weight
        self.sla_weight = sla_weight
        self.overtemp_weight = overtemp_weight  # kept for backward-compatible config parsing
        self.pue_bonus_weight = pue_bonus_weight
        self.target_pue = target_pue
        self._prev_sla: int = 0

    def reset(self):
        super().reset()
        self._prev_sla = 0

    def compute(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        reward = 0.0

        for res_id, res_info in info.get('resources', {}).items():
            if 'datacenter' not in res_id.lower() and 'dc' not in res_id.lower():
                continue

            # Power cost: penalise total DC power (MW, always positive)
            p_dc = abs(res_info.get('current_p_mw', 0.0))
            reward -= self.power_weight * p_dc

            # SLA violations: penalise *new* violations this step
            # FIXED: prefer step_sla_violations (per-step count) over manual diff
            new_violations = res_info.get('step_sla_violations', None)
            if new_violations is None:
                # Fallback for backward compatibility
                sla_total = res_info.get('sla_violations', 0)
                new_violations = max(0, sla_total - self._prev_sla)
                self._prev_sla = sla_total
            reward -= self.sla_weight * new_violations

            # PUE bonus: reward operating below target PUE
            pue = res_info.get('pue', 2.0)
            if pue < self.target_pue:
                reward += self.pue_bonus_weight * (self.target_pue - pue)

        return self._maybe_normalize(reward)


class EVArbitrageReward(RewardFunction):
    def __init__(
        self,
        peak_hours: list | None = None,
        off_peak_hours: list | None = None,
        arbitrage_weight: float = 100.0,
        departure_bonus: float = 20.0,
        healthy_soc_bonus: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.peak_hours = set(peak_hours or [9, 10, 11, 17, 18, 19, 20])
        self.off_peak_hours = set(off_peak_hours or [0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 23])
        self.arbitrage_weight = arbitrage_weight
        self.departure_bonus = departure_bonus
        self.healthy_soc_bonus = healthy_soc_bonus

    def compute(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        time_step = info.get('step_within_day', info.get('time_step', 0))
        delta_t_minutes = info.get('delta_t_minutes', 60)
        hour = int((time_step * delta_t_minutes) / 60) % 24

        if hour in self.peak_hours:
            price = 0.5
        elif hour in self.off_peak_hours:
            price = 0.1
        else:
            price = 0.25

        reward = 0.0
        for res_id, res_info in info.get('resources', {}).items():
            if 'vehicle' not in res_id.lower() and 'ev' not in res_id.lower():
                continue
            power_kw = float(res_info.get('current_p_mw', 0.0)) * 1000.0
            reward += self.arbitrage_weight * price * power_kw
            if bool(res_info.get('departure_ready', False)):
                reward += self.departure_bonus
            soc = res_info.get('soc')
            soc_min = res_info.get('soc_min')
            soc_max = res_info.get('soc_max')
            if soc is not None and soc_min is not None and soc_max is not None and soc_min <= soc <= soc_max:
                reward += self.healthy_soc_bonus

        return self._maybe_normalize(reward)
