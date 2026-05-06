"""Base helpers shared by task observation builders."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def get_time_context(
    *,
    grid: Any,
    step_count: int,
    max_steps: int,
    delta_t_minutes: int,
) -> Dict[str, float]:
    """Return normalized time features shared across task adapters."""
    time_step = getattr(grid, 'time_step', step_count)
    steps_per_day = max(int(24 * 60 / delta_t_minutes), 1)
    hour = int((time_step * delta_t_minutes) / 60) % 24
    return {
        'time_step': float(time_step),
        'steps_per_day': float(steps_per_day),
        'hour': float(hour),
        'time_of_day': float(time_step % steps_per_day) / steps_per_day,
        'episode_progress': float(step_count) / max(max_steps, 1),
    }


def concat_observation_parts(*parts: Any) -> np.ndarray:
    """Concatenate observation fragments into one float32 vector."""
    arrays = []
    for part in parts:
        arr = np.asarray(part, dtype=np.float32).reshape(-1)
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(arrays).astype(np.float32)
