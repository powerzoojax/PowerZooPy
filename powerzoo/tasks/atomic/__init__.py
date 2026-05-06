"""Internal atomic validation task presets.

Atomic tasks are intentionally non-public. They support component-level
validation without widening the user-facing benchmark surface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from powerzoo.tasks.atomic.grid import (
    AtomicDistributionGridTask,
    AtomicTransmissionGridTask,
)
from powerzoo.tasks.atomic.resource import (
    AtomicBatteryTask,
    AtomicDataCenterTask,
    AtomicFlexLoadTask,
    AtomicSolarTask,
    AtomicVehicleTask,
    AtomicWindTask,
)


ATOMIC_TASKS: Dict[str, Type] = {
    'atomic_transmission_grid': AtomicTransmissionGridTask,
    'atomic_distribution_grid': AtomicDistributionGridTask,
    'atomic_battery': AtomicBatteryTask,
    'atomic_solar': AtomicSolarTask,
    'atomic_wind': AtomicWindTask,
    'atomic_vehicle': AtomicVehicleTask,
    'atomic_flexload': AtomicFlexLoadTask,
    'atomic_datacenter': AtomicDataCenterTask,
}


def list_atomic_tasks() -> List[str]:
    return list(ATOMIC_TASKS)


def get_atomic_task(name: str, **kwargs):
    try:
        task_class = ATOMIC_TASKS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown atomic task: '{name}'. Available: {list(ATOMIC_TASKS)}"
        ) from exc
    return task_class(**kwargs)


def make_atomic_task_env(name: str, **kwargs) -> Any:
    return get_atomic_task(name, **kwargs).create_env()


__all__ = [
    'ATOMIC_TASKS',
    'AtomicTransmissionGridTask',
    'AtomicDistributionGridTask',
    'AtomicBatteryTask',
    'AtomicSolarTask',
    'AtomicWindTask',
    'AtomicVehicleTask',
    'AtomicFlexLoadTask',
    'AtomicDataCenterTask',
    'list_atomic_tasks',
    'get_atomic_task',
    'make_atomic_task_env',
]
