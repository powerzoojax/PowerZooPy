import logging
from typing import Any, List, Optional

import numpy as np

from powerzoo.envs.base import BaseEnv

logger = logging.getLogger(__name__)


class ResourceEnv(BaseEnv):
    """Physical sub-resource base class (dynamics only, no CMDP).

    A resource is responsible for one thing: given an action, update its
    internal physical state (SOC, power, temperature, etc.).

    Resources are **not** standalone RL environments.  They do not compute
    reward, cost, or termination signals.  The full CMDP interface::

        (obs, reward, cost, terminated, truncated, info)

    is assembled by ``PowerEnv`` + a ``Task``.  Resources expose two
    read-only query methods:

    - ``obs()``    → observation dict for the RL agent (flattened by PowerEnv)
    - ``status()`` → full state dict for diagnostics, info channel, and costs

    **Cost convention**: any field in ``status()`` whose key starts with
    ``cost_`` is automatically collected by ``PowerEnv`` into the unified
    CMDP cost channel.  Subclasses that detect constraint violations should
    expose them as ``cost_<name>`` fields (≥ 0, in physical units).

    **Physical safety contract**: subclass ``step(action)`` implementations
    must clip actions to physically realisable bounds *before* updating
    internal state (e.g. SOC must never go below 0 or above 1).  The
    amount clipped away may be reported as a ``cost_*`` field in
    ``status()`` for the CMDP cost channel, but internal state must
    never diverge or become unrepresentable.

    **status() base fields**:
        ``current_p_mw``, ``current_q_mvar``, ``time_step``,
        ``bus_id`` (int), ``local_v`` (float | None — per-unit bus voltage,
        None when parent grid has not run a power flow).
    """

    # ====== Initialization ======

    # Valid phase connection strings for three-phase grids.
    _VALID_PHASES = frozenset({'A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC'})

    def __init__(self, parent: Any = None, bus_id: int = -1,
                 delta_t_minutes: float = 15.0, phase: str = 'ABC'):
        super().__init__(delta_t_minutes=delta_t_minutes)
        self._parent: Any = None
        self._bus_id: int = bus_id
        self.resource_id: Optional[str] = None
        self.current_p_mw: float = 0.0
        self.current_q_mvar: float = 0.0
        self.day_id: Optional[int] = None
        self.sub_resources: dict[str, "ResourceEnv"] = {}
        self.phase: str = phase.upper()  # Phase connection: 'A','B','C','AB','AC','BC','ABC'
        if self.phase not in self._VALID_PHASES:
            raise ValueError(
                f"Invalid phase '{self.phase}', must be one of {sorted(self._VALID_PHASES)}"
            )

        # Derived time constants — updated in attach() / detach()
        # Assumes delta_t_minutes evenly divides 1440 (e.g. 15, 30, 60).
        self.dt_hours: float = delta_t_minutes / 60.0
        self.steps_per_day: int = 1440 // int(delta_t_minutes)

        # Defer attach until the concrete resource sets ``action_space``.
        # ``ResourceEnv.__init__`` runs before subclass bodies; early ``attach()``
        # would call ``DistGridEnv.update_action_space()`` while
        # ``BaseEnv.action_space`` is still None, breaking registration.
        self._deferred_parent: Any = parent
        self._deferred_bus_id: int = bus_id

    @property
    def parent(self) -> Any:
        return self._parent

    @property
    def bus_id(self) -> int:
        return self._bus_id

    @bus_id.setter
    def bus_id(self, value: int) -> None:
        """Set the bus ID and update parent's nodes_resources_map if attached."""
        self._bus_id = value
        
        # If parent exists and has _update_nodes_resources_map method, update mapping
        if self._parent is not None and hasattr(self._parent, '_update_nodes_resources_map'):
            self._parent._update_nodes_resources_map()

    # ====== RL Interface Methods ======

    def reset(self, *, seed=None, options=None, day_id: int = None):
        """Reset resource to initial state.  Subclasses should call super() then apply
        resource-specific logic.

        ``options`` may carry ``time_step`` / ``time_offset`` to align the resource's
        internal clock to the grid's episode start.
        """
        super().reset(seed=seed, options=options)
        opts = options or {}
        start_step = int(opts.get('time_step', opts.get('time_offset', 0)) or 0)
        self.time_step = max(start_step, 0)
        self.current_p_mw = 0.0
        self.current_q_mvar = 0.0
        self.day_id = day_id
        return {}

    def step(self, action: Any) -> None:
        """Apply control action and update internal physical state.

        This is a **pure state mutator** — it must not return a value.
        The caller (grid / PowerEnv) reads results afterwards via
        ``obs()`` and ``status()``.

        Subclasses must override this method.
        """
        raise NotImplementedError

    def obs(self, state=None) -> dict:
        """Return local observation as an ordered dict.

        Returns:
            dict[str, float]: keys are human-readable feature names.
            PowerEnv's observation flattener walks the dict by **alphabetical
            key order**, so the flattened layout is determined by key names,
            not insertion order. The key set must be fixed across calls.
        """
        raise NotImplementedError

    def grid_obs(self) -> np.ndarray:
        """Normalised feature vector for embedding this resource in a grid env.

        Time encoding is excluded — the grid provides its own.
        Subclasses should override to expose type-specific features.
        Default: ``[current_p_norm]``.
        """
        cap = float(getattr(self, 'capacity_mw', None) or
                    getattr(self, 'power_mw', None) or 1.0)
        return np.array([self.current_p_mw / max(cap, 1e-6)], dtype=np.float32)

    def grid_obs_names(self, rid: str) -> list:
        """Feature names for ``grid_obs()``, prefixed with ``rid``.
        Length must equal ``len(self.grid_obs())``.
        """
        return [f'{rid}_p_norm']

    def grid_action_bounds(self) -> tuple:
        """Physical ``(low, high)`` bounds for this resource's single grid action dimension.

        Used by grid environments to construct the action space and to
        clip/denormalise actions before passing them to ``step()``.
        """
        if hasattr(self, '_action_phys_low') and hasattr(self, '_action_phys_high'):
            return (float(np.asarray(self._action_phys_low).flat[0]),
                    float(np.asarray(self._action_phys_high).flat[0]))
        if hasattr(self, 'action_space') and getattr(self.action_space, 'shape', ()) == (1,):
            return (float(np.asarray(self.action_space.low).flat[0]),
                    float(np.asarray(self.action_space.high).flat[0]))
        return 0.0, 1.0

    def grid_action_from_normalized(self, raw: float) -> float:
        """Map a normalised ``[-1, 1]`` scalar to physical action units.

        Override for resources with non-standard semantics (e.g. curtailment-based
        renewables where ``+1`` means no curtailment and the mapping is inverted).
        ``raw`` is already clipped to ``[-1, 1]`` by the caller.
        """
        low, high = self.grid_action_bounds()
        return float(np.clip((low + high) / 2 + raw * (high - low) / 2, low, high))

    def status(self):
        """Return current resource status dict.

        Any key starting with ``cost_`` is collected by ``PowerEnv`` as a CMDP
        safety cost (≥ 0, physical units, e.g. ``cost_clipped_power``).
        Economic penalties/benefits go through ``econ_components()``, not here.

        Base fields always present:
            ``current_p_mw``, ``current_q_mvar``, ``time_step``,
            ``bus_id``, ``local_v`` (per-unit bus voltage, None if unavailable).
        """
        return {
            'current_p_mw': self.current_p_mw,
            'current_q_mvar': self.current_q_mvar,
            'time_step': self.time_step,
            'bus_id': int(self._bus_id),
            'local_v': self._get_local_voltage(),
        }

    def econ_components(self, dt_hours: float) -> dict:
        """Economic cost/benefit contributions to the reward signal this step.

        Returns ``{name: value}`` in physical units (e.g. $/step = $/MWh × MWh).
        Negative = cost, positive = benefit.  Default: ``{}`` (no contribution).
        Subclasses override when they carry penalty/benefit parameters.
        """
        return {}

    def _get_local_voltage(self):
        """Extract per-unit voltage at self._bus_id from parent's power flow results.

        Returns None if the parent doesn't expose voltage data.
        """
        if self._parent is None:
            return None

        # Distribution grid: nodes DataFrame has 'v_mag' column
        nodes = getattr(self._parent, '_nodes', None)
        if nodes is None:
            return None

        try:
            import pandas as pd
            if isinstance(nodes, pd.DataFrame) and 'v_mag' in nodes.columns:
                bus_id = self._bus_id
                if bus_id in nodes.index:
                    return float(nodes.loc[bus_id, 'v_mag'])
                if '#id' in nodes.columns:
                    row = nodes[nodes['#id'] == bus_id]
                    if not row.empty:
                        return float(row['v_mag'].iloc[0])
        except (KeyError, IndexError, ValueError):
            logger.debug("Failed to read local voltage at bus %s", self._bus_id)
        return None

    # ====== Resource Management ======

    def attach(self, parent: Any, bus_id: int = None, name: str = None) -> Optional[str]:
        """Attach resource to a parent hub or grid and register it.

        Args:
            parent: The parent grid or hub to attach to.
            bus_id: Optional bus ID override.
            name: Optional custom name; auto-generated (e.g. ``'solar_0'``) if None.

        Returns:
            resource_id: The assigned resource ID.
        """
        if self._parent is not None:
            self.detach()

        self._parent = parent

        # Sync time constants from new parent.
        # NOTE: delta_t_minutes is also updated so that subclass code
        # referencing it (e.g. FlexLoad._dt_h) stays correct after attach.
        if parent is not None and hasattr(parent, 'delta_t_minutes'):
            dt = parent.delta_t_minutes
            self.delta_t_minutes = dt
            self.dt_hours = dt / 60.0
            self.steps_per_day = 1440 // int(dt)

        if bus_id is not None:
            self._bus_id = bus_id

        if parent is not None and hasattr(parent, 'register_resource'):
            self.resource_id = parent.register_resource(self, self._bus_id, name=name)

        return self.resource_id

    def _complete_resource_init(self) -> None:
        """Register with a deferred parent after ``action_space`` is fully built.

        Subclasses must call this at the end of ``__init__`` when
        ``parent`` was passed in. If ``parent`` was ``None``, this is a no-op.
        """
        if self._deferred_parent is not None:
            self.attach(self._deferred_parent, self._deferred_bus_id)
            self._deferred_parent = None

    def detach(self) -> None:
        """Detach resource from its parent and unregister."""
        if self._parent is not None and hasattr(self._parent, 'unregister_resource'):
            if self.resource_id is not None:
                self._parent.unregister_resource(self.resource_id)
        self._parent = None
        self.resource_id = None
        self.dt_hours = self.delta_t_minutes / 60.0
        self.steps_per_day = 1440 // int(self.delta_t_minutes)
