"""Oracle policy — runs the built-in OPF solver to get the optimal dispatch.

This policy serves as the **upper bound** baseline: no learned agent should
consistently outperform the oracle on the same objective.

The oracle uses ``solve_ed_opf_detailed`` (DC-OPF, same as TransGridEnv's
default solver) to compute the economically optimal unit dispatch for each
time step, given perfect knowledge of the current load.

Note: the oracle does NOT see the future — it only uses the current step's
load, which is the same information available to any online agent.

Args:
    env_or_grid: Wrapped environment or raw ``TransGridEnv``.
    solver_type: OPF solver backend (``'auto'``, ``'scipy'``, …).
    action_space: Optional Gymnasium action space for clipping.
"""

from typing import Any, Dict, Optional

import numpy as np

from powerzoo.benchmarks.policies.base import BasePolicy


class OraclePolicy(BasePolicy):
    """OPF-based oracle — optimal dispatch given current-step information.

    Example::

        from powerzoo.benchmarks.policies import OraclePolicy, evaluate
        from powerzoo.wrappers import GymnasiumWrapper
        from powerzoo.envs.grid.trans import TransGridEnv

        raw = TransGridEnv()
        env = GymnasiumWrapper(raw)
        policy = OraclePolicy(raw)
        result = evaluate(policy, env, n_episodes=5)
        print(f"Oracle mean reward: {result['mean_reward']:.3f}")
    """

    def __init__(self, env_or_grid=None,
                 solver_type: str = 'auto',
                 action_space=None):
        super().__init__(action_space)
        self.solver_type = solver_type

        self._grid = env_or_grid
        while hasattr(self._grid, 'env'):
            self._grid = self._grid.env

    def act(self, obs: Any, info: Optional[Dict] = None) -> np.ndarray:
        if self._grid is None:
            raise ValueError("OraclePolicy requires a grid env reference.")

        try:
            from powerzoo.envs.grid.cal_dcopf_trans import solve_ed_opf_detailed

            node_net_load_mw = self._grid._calculate_node_net_load()
            result = solve_ed_opf_detailed(
                self._grid.case,
                node_net_load_mw,
                commitment=None,
                verbose=False,
                solver_type=self.solver_type,
            )
            unit_power_mw = result['unit_power_mw'].astype(np.float32)

            # Clip to action space if available
            if self.action_space is not None:
                unit_power_mw = np.clip(unit_power_mw,
                                     self.action_space.low,
                                     self.action_space.high)
            return unit_power_mw

        except Exception as e:
            # Fallback: proportional dispatch
            p_min = self._grid.case.units['p_min'].values.astype(np.float32)
            p_max = self._grid.case.units['p_max'].values.astype(np.float32)
            return ((p_min + p_max) / 2.0)
