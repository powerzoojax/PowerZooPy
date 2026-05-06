"""PowerZoo Complex Tasks

Complex task collection:
- Task 5: Joint Transmission-Distribution CMDP (OPF + DER arbitrage)
- Task 6: Multi-Agent OPF on IEEE 118-bus system
"""

from powerzoo.tasks.registry import register_task

from powerzoo.tasks.complex.joint_trans_dist import (
    JointTransDistTask,
    JointTransDistTask7Days,
)
from powerzoo.tasks.complex.opf_118 import OPF118Task, OPF118Task7Days

register_task('joint_trans_dist',    JointTransDistTask)
register_task('joint_trans_dist_7d', JointTransDistTask7Days)
register_task('opf_118',             OPF118Task)
register_task('opf_118_7d',          OPF118Task7Days)

__all__ = [
    'JointTransDistTask',
    'JointTransDistTask7Days',
    'OPF118Task',
    'OPF118Task7Days',
]
