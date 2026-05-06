"""Rule-based heuristic policy.

Provides sensible hand-crafted baselines for common PowerZoo tasks:

- **Generator dispatch (TransGridEnv)**: cheapest-first (merit-order) dispatch.
  Units are sorted by their marginal cost ``mc_b`` and loaded proportionally.
- **Battery storage**: charge in the first half of the day, discharge in the
  second half.  Simple time-of-day proxy for price arbitrage.
- **Renewable**: no curtailment (output at maximum available power).
"""

from typing import Any, Dict, List, Optional

import numpy as np

from powerzoo.benchmarks.policies.base import BasePolicy


class RuleBasedPolicy(BasePolicy):
    """Hand-crafted heuristic baseline.

    Args:
        env_or_grid: The wrapped environment **or** the raw ``GridEnv``.
                     Used to read case metadata (unit costs, capacities, …).
        policy_type: Which heuristic to use:
            ``'merit_order'``   — generator dispatch by merit order (default).
            ``'battery_tod'``   — charge morning / discharge afternoon.
            ``'no_curtail'``    — renewables at full output.
        action_space: Gymnasium action space (optional; used for clipping).

    Example::

        from powerzoo.benchmarks.policies import RuleBasedPolicy, evaluate
        from powerzoo.wrappers import GymnasiumWrapper
        from powerzoo.envs.grid.trans import TransGridEnv

        raw = TransGridEnv()
        env = GymnasiumWrapper(raw)
        policy = RuleBasedPolicy(raw, policy_type='merit_order')
        result = evaluate(policy, env, n_episodes=10)
    """

    def __init__(self, env_or_grid=None,
                 policy_type: str = 'merit_order',
                 action_space=None):
        super().__init__(action_space)
        self.policy_type = policy_type

        # Unwrap to grid env if wrapped
        self._grid = env_or_grid
        while hasattr(self._grid, 'env'):
            self._grid = self._grid.env

        self._steps_per_day: int = getattr(self._grid, 'steps_per_day', 48)

        # Pre-compute merit-order for generators
        self._unit_order: Optional[np.ndarray] = None
        if self._grid is not None and hasattr(self._grid, 'case'):
            case = self._grid.case
            if hasattr(case, 'units') and 'mc_b' in case.units.columns:
                mc_b = case.units['mc_b'].values
                self._unit_order = np.argsort(mc_b)  # cheapest first
            self._p_min = case.units['p_min'].values if hasattr(case, 'units') else np.array([])
            self._p_max = case.units['p_max'].values if hasattr(case, 'units') else np.array([])

    # ------------------------------------------------------------------

    def act(self, obs: Any, info: Optional[Dict] = None) -> np.ndarray:
        if self.policy_type == 'merit_order':
            return self._merit_order_dispatch()
        elif self.policy_type == 'battery_tod':
            return self._battery_time_of_day()
        elif self.policy_type == 'no_curtail':
            return np.array([0.0], dtype=np.float32)
        else:
            raise ValueError(f"Unknown policy_type '{self.policy_type}'")

    def _merit_order_dispatch(self) -> np.ndarray:
        """Proportional merit-order dispatch.

        Loads units cheapest-first up to the required demand.  This
        approximates the optimal economic dispatch without running OPF.
        """
        if self._grid is None or not hasattr(self._grid, '_get_node_loads_p_current'):
            # Fallback: mid-point between p_min and p_max
            return ((self._p_min + self._p_max) / 2.0).astype(np.float32)

        total_demand = float(self._grid._get_node_loads_p_current().sum())
        n_units = len(self._p_min)
        power = self._p_min.copy().astype(np.float64)

        # Remaining demand after p_min coverage
        remaining = total_demand - power.sum()

        # Fill units in merit order
        if self._unit_order is not None and remaining > 0:
            for i in self._unit_order:
                headroom = self._p_max[i] - power[i]
                fill = min(headroom, remaining)
                power[i] += fill
                remaining -= fill
                if remaining <= 0:
                    break

        return np.clip(power, self._p_min, self._p_max).astype(np.float32)

    def _battery_time_of_day(self) -> np.ndarray:
        """Simple ToD heuristic: charge first half, discharge second half."""
        time_step = getattr(self._grid, 'time_step', 0) if self._grid else 0
        frac = (time_step % self._steps_per_day) / self._steps_per_day

        if self.action_space is not None:
            power_limit = float(self.action_space.high[0])
        else:
            power_limit = 20.0

        if frac < 0.4:
            return np.array([-power_limit], dtype=np.float32)   # charge
        elif frac > 0.6:
            return np.array([power_limit], dtype=np.float32)    # discharge
        else:
            return np.array([0.0], dtype=np.float32)            # idle
