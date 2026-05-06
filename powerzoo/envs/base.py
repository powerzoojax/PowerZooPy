"""Base environment for PowerZoo.

PowerZoo requires ``gymnasium`` and follows the Gymnasium >= 0.26 reset /
seeding contract throughout the environment stack.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces


class BaseEnv(gym.Env, ABC):
    """Base environment for PowerZoo.

    Provides:
    - ``np_random``: Gymnasium's seeded RNG, initialised through
      ``super().reset(seed=seed, options=options)``.
    - ``observation_space`` / ``action_space`` placeholders (``None`` until
      subclass initialises them in ``__init__``).
    - Abstract ``step`` / ``obs`` methods that subclasses must implement.
    - Default ``reward`` / ``cost`` hooks for reward-shaping and CMDP cost.

    Seed contract
    -------------
    Subclasses should call ``super().reset(seed=seed, options=options)`` as
    the first line of their ``reset()`` implementation. This delegates RNG
    management to Gymnasium and preserves its ``seed=None`` semantics: reuse
    the current generator when one already exists.
    """

    def __init__(self, delta_t_minutes: float = 1.0):
        super().__init__()
        if (delta_t_minutes <= 0
                or delta_t_minutes != int(delta_t_minutes)
                or 1440 % int(delta_t_minutes) != 0):
            raise ValueError(
                f"delta_t_minutes={delta_t_minutes} must be a positive integer "
                f"divisor of 1440 (e.g. 1, 5, 15, 30, 60)."
            )
        self.time_step: int = 0
        self.delta_t_minutes: float = delta_t_minutes
        # Physical time derivation:
        #   elapsed_minutes = time_step * delta_t_minutes
        #   steps_per_day   = 1440 // int(delta_t_minutes)  (e.g. 96 for 15 min)
        #   step_within_day = time_step % steps_per_day
        # Subclasses that index time-series data (wind/solar profiles, price
        # curves) should use step_within_day + day_id * steps_per_day as the
        # lookup key so that episode start is reproducible across days.
        # Placeholders — concrete subclasses MUST set real Space objects
        # in their own __init__ (before any wrapper is attached).
        self.action_space: Optional[spaces.Space] = None
        self.observation_space: Optional[spaces.Space] = None

    # ====== RL Interface Methods ======

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Apply common Gymnasium reset bookkeeping.

        The base implementation seeds ``np_random`` (via ``super().reset``),
        resets ``time_step`` to 0, and returns the **placeholder** pair
        ``(None, {})``.

        Subclass contract
        -----------------
        * Call ``super().reset(seed=seed, options=options)`` as the **first**
          line — this seeds ``self.np_random`` and resets the clock.
        * After super(), build and return a **valid** ``(observation, info)``
          pair that satisfies ``self.observation_space``.
        * Never return the base placeholder ``(None, {})`` from a concrete
          environment — downstream wrappers and RL libraries will fail on
          ``None`` observations.
        """
        super().reset(seed=seed, options=options)
        self.time_step = 0
        # Sentinel — concrete envs must return a valid (obs, info) pair.
        return None, {}

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Apply action and return ``(obs, reward, terminated, truncated, info)``.

        Subclasses are responsible for advancing ``self.time_step`` inside
        ``step()`` and for computing ``truncated`` from their own episode-limit
        policy (for example ``max_episode_steps``).

        Recommended implementation order
        ---------------------------------
        1. **Action decode** — map the RL agent's action (e.g. normalised
           values in ``[-1, 1]``) to physical setpoints (MW, MVar, °C …).
        2. **Physics** — run power flow / OPF / dynamic simulation for the
           current time step.
        3. **State extract** — read voltages, flows, costs, and resource
           state-of-charge from the solver result.
        4. **Observation** — call ``self.obs(state)`` to produce the agent-
           facing observation array.
        5. **Reward / cost** — compute scalar reward and, if applicable, CMDP
           cost terms; populate ``info`` with ``cost_*`` keys.
        6. **Clock advance** — increment ``self.time_step``; set ``truncated``
           when the episode step limit is reached.
        """
        ...

    @abstractmethod
    def obs(self, state: Any) -> Any:
        """Convert internal *state* dict to a flat observation array.

        Subclasses should implement this to transform the internal full
        physical state into the agent-facing observation, which may be local,
        partial, noisy, or otherwise filtered relative to the simulator state.
        """
        ...

    def reward(self, r: float, state: Any, info: Dict[str, Any]) -> float:
        """Return the scalar reward after optional reward-shaping adjustments.

        This is a lightweight base-class hook for reward shaping. Concrete
        environments or wrappers can override it to transform the scalar
        reward while leaving the main transition logic elsewhere.

        For non-trivial reward shaping, prefer using a ``gym.Wrapper``
        instead; this hook is intended as a lightweight escape hatch for
        environment-internal adjustments only.
        """
        return r

    def constraint_names(self) -> Tuple[str, ...]:
        """Return the fixed-order benchmark constraint names for this env.

        The base contract is zero-constraint / MDP by default.
        Constrained envs should override this and emit matching ``cost_<name>``
        scalars in ``info``.  ``info['constraint_costs']`` is then assembled in
        the same order by :meth:`assemble_constraint_costs`.
        """
        return ()

    @staticmethod
    def _constraint_cost_aliases(name: str) -> Tuple[str, ...]:
        """Legacy info-key aliases kept for backward-compatible cost reads."""
        aliases = {
            'resource': ('cost_resource_violation',),
        }
        return aliases.get(name, ())

    def _constraint_cost_value(self, info: Mapping[str, Any], name: str) -> float:
        """Read one named constraint cost from ``info`` with legacy aliases."""
        primary_key = f"cost_{name}"
        if primary_key in info:
            return max(0.0, float(info[primary_key]))
        for alias in self._constraint_cost_aliases(name):
            if alias in info:
                return max(0.0, float(info[alias]))
        return 0.0

    def assemble_constraint_costs(
        self,
        info: Mapping[str, Any],
        names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Assemble the fixed-order constraint-cost vector from ``info``.

        If ``info['constraint_costs']`` already exists and matches the expected
        length, it is trusted after float32 coercion.  Otherwise the vector is
        rebuilt from ``cost_<name>`` scalars in the order given by
        ``constraint_names()`` (or explicit ``names``).
        """
        ordered_names = tuple(self.constraint_names() if names is None else names)
        if not ordered_names:
            return np.zeros((0,), dtype=np.float32)

        if 'constraint_costs' in info:
            arr = np.asarray(info['constraint_costs'], dtype=np.float32).reshape(-1)
            if arr.shape == (len(ordered_names),):
                return np.maximum(arr, 0.0)

        return np.asarray(
            [self._constraint_cost_value(info, name) for name in ordered_names],
            dtype=np.float32,
        )

    def attach_constraint_costs(
        self,
        info: Dict[str, Any],
        names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Populate ``info['constraint_costs']`` and the scalar ``cost_sum`` alias."""
        ordered_names = tuple(self.constraint_names() if names is None else names)
        costs = self.assemble_constraint_costs(info, names=ordered_names)
        info['constraint_costs'] = costs
        if ordered_names:
            info.setdefault('constraint_names', ordered_names)
        info['cost_sum'] = float(costs.sum())
        return info

    def cost(self, state: Any, info: Dict[str, Any]) -> float:
        """Return the scalar CMDP cost for the current transition.

        Default is ``0.0`` (no constraint violation).  Subclasses that
        model safety constraints should override this and write named
        ``cost_<constraint>`` scalars into ``info``.  Compatibility wrappers
        may still project these vector costs into a legacy scalar
        ``info["cost"]`` alias, but that is no longer the core contract.

        Cost is **separate** from reward — safety penalties must not
        be folded back into the reward signal.
        """
        return 0.0
