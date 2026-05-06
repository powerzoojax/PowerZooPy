"""PowerZoo Simple Tasks

Simple task collection:
- marl_opf: multi-agent OPF control (IEEE 5-bus)
- marl_der_arbitrage: multi-agent DER storage arbitrage
- marl_ders_benchmark: 12-agent heterogeneous DERs benchmark (Case118zh)
- marl_ev_v2g: multi-agent EV V2G/G2V control (IEEE 33-bus)
- single_opf: single-agent OPF control
- battery_control: battery charge/discharge control
- dc_microgrid: DC μGrid self-contained microgrid (288×5min, 5-D action)
- dc_microgrid_safe: DC μGrid CMDP variant (vector thresholds, scalar alias 0.5)
"""

from powerzoo.tasks.registry import register_task

# Import and register tasks
from .marl_opf import MARLOPFTask
from .marl_der_arbitrage import MARLDERArbitrageTask
from .marl_ders_benchmark import MARLDERBenchmarkTask
from .marl_ev_v2g import MARLEVTask
from .task4_dc_scheduling import DCSchedulingTask
from .battery_arbitrage import BatteryArbitrageTask
from .task_dc_microgrid import DCMicrogridTask, DCMicrogridSafeTask
from .task_gencos import GenCosTask

# Register with the global registry
register_task('battery_arbitrage', BatteryArbitrageTask)
register_task('marl_opf', MARLOPFTask)
register_task('marl_der_arbitrage', MARLDERArbitrageTask)
register_task('marl_ders_benchmark', MARLDERBenchmarkTask)
register_task('marl_ev_v2g', MARLEVTask)
register_task('dc_scheduling', DCSchedulingTask)
register_task('dc_microgrid', DCMicrogridTask)
register_task('dc_microgrid_safe', DCMicrogridSafeTask)
register_task('gencos_bidding', GenCosTask)

# Optional: more tasks
try:
    from .task2_single_opf import SingleOPFTask
    register_task('single_opf', SingleOPFTask)
except ImportError:
    pass

try:
    from .task3_battery_control import BatteryControlTask
    register_task('battery_control', BatteryControlTask)
except ImportError:
    pass

__all__ = [
    'BatteryArbitrageTask',
    'MARLOPFTask',
    'MARLDERArbitrageTask',
    'MARLDERBenchmarkTask',
    'MARLEVTask',
    'DCSchedulingTask',
    'DCMicrogridTask',
    'DCMicrogridSafeTask',
    'GenCosTask',
]
