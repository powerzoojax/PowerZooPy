"""PowerZoo Middle Tasks

Medium-difficulty task collection:
- Task 4: Unit Commitment (MARL, 5-bus, on/off + ramp constraints)
"""

from powerzoo.tasks.registry import register_task
from powerzoo.tasks.middle.marl_uc import MARLUCTask
from powerzoo.tasks.middle.comparison_tso import (
    CentralizedComparisonTSOTask,
    CentralizedComparisonTSOEnv,
)

register_task('marl_uc', MARLUCTask)
register_task('comparison_tso_centralized', CentralizedComparisonTSOTask)

__all__ = [
    'MARLUCTask',
    'CentralizedComparisonTSOTask',
    'CentralizedComparisonTSOEnv',
]
