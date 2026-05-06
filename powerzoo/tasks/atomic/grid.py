"""Atomic grid-only validation tasks."""

from __future__ import annotations

from powerzoo.tasks.atomic.base import BaseAtomicTask


class AtomicTransmissionGridTask(BaseAtomicTask):
    name = 'atomic_transmission_grid'
    description = 'Atomic transmission-grid validation preset'
    GRID_TYPE = 'transmission'
    CASE = 'Case5'
    DEFAULT_DELTA_T_MINUTES = 30
    DEFAULT_MAX_STEPS = 48


class AtomicDistributionGridTask(BaseAtomicTask):
    name = 'atomic_distribution_grid'
    description = 'Atomic distribution-grid validation preset'
    GRID_TYPE = 'distribution'
    CASE = 'Case33bw'
    DEFAULT_DELTA_T_MINUTES = 30
    DEFAULT_MAX_STEPS = 48
