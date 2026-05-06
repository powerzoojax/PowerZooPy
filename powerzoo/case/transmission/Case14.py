"""IEEE 14-bus system (MATPOWER case14)

Data Source: MATPOWER case file `source_mfile/case14.m`
Base MVA: 100.0 MVA
Base kV: 69.0 kV (nominal; MATPOWER bus table uses 0 for baseKV)

UC / economic columns on ``self.units`` match PowerZooJax ``create_case14()`` and are
regenerated from ``powerzoo/case/raw_cases/case14.json`` via
``python -m powerzoo.case.raw_cases.scuc_json_to_case_units`` (keep that script and JSON
in sync with the sibling ``powerzoojax/case/raw_cases/`` tree).

All values are in standard MATPOWER units (MW, MVAr, p.u.)


Nodes (bus) columns:
    id, type, Pd, Qd, Gs, Bs, Vm, Va, baseKV, Vmax, Vmin
    - type: 1=PQ, 2=PV, 3=Slack
    - Pd/Qd: Active/Reactive power demand (MW/MVAr)
    - Gs/Bs: Shunt conductance/susceptance (MW/MVAr at V=1.0 p.u.)
    - Vm: Voltage magnitude (p.u.)
    - Va: Voltage angle (degrees)
    - baseKV: Base voltage (kV)
    - Vmax/Vmin: Voltage limits (p.u.)

Units (generators) columns:
    id, bus_id, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, mc_a, mc_b, mc_c,
    p_max, p_min, type, ramp_up, ramp_down, init_start_up_cost, keep_time,
    init_power, init_state, min_up_time, min_down_time, init_no_load_cost
    - Pg/Qg: Active/Reactive power output (MW/MVAr)
    - Qmax/Qmin: Reactive power limits (MVAr)
    - Vg: Voltage setpoint (p.u.)
    - mBase: Machine base (MVA)
    - status: >0 in-service, <=0 out-of-service
    - Pmax/Pmin: Active power limits (MW)
    - mc_a/mc_b/mc_c: Marginal cost coefficients (quadratic)

Lines (branches) columns:
    id, from, to, r, x, b, rateA, ratio, angle, status, floor, cap
    - r/x: Resistance/Reactance (p.u.)
    - b: Line charging susceptance (p.u.)
    - rateA: Long-term rating (MVA)
    - ratio: Transformer tap ratio (0 for line)
    - angle: Transformer phase shift (degrees)
    - status: 1=in-service, 0=out-of-service
    - floor/cap: Power flow limits (MW)

Loads columns:
    id, bus_id, Pd, Qd, mc_a, mc_b, mc_c, d_max, d_min
    - Pd/Qd: Active/Reactive power demand (MW/MVAr)
    - mc_a/mc_b/mc_c: Marginal utility coefficients
    - d_max/d_min: Demand limits (MW)
"""
from powerzoo.case.CaseBase import ClearCase, DataFrame


class Case14(ClearCase):
    GRID_TYPE = "transmission"
    BUS_COUNT = 14
    VOLTAGE_LEVEL = "HV"
    SOURCE = "MATPOWER"
    DESCRIPTION = "IEEE 14-bus test system"

    def __init__(self, *args, **kwargs):
        # System base values
        self.baseMVA = 100.0
        # MATPOWER case14 has baseKV=0 on all buses; use a nominal kV for metadata.
        self.baseKV = 69.0
        
        # Node (bus) data
        self.nodes = DataFrame(
            ['id', 'type', 'Pd', 'Qd', 'Gs', 'Bs', 'Vm', 'Va', 'baseKV', 'Vmax', 'Vmin'],
            [[1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.06, 0.0, 0.0, 1.06, 0.94],
             [2.0, 2.0, 21.7, 12.7, 0.0, 0.0, 1.045, -4.98, 0.0, 1.06, 0.94],
             [3.0, 2.0, 94.2, 19.0, 0.0, 0.0, 1.01, -12.72, 0.0, 1.06, 0.94],
             [4.0, 1.0, 47.8, -3.9, 0.0, 0.0, 1.019, -10.33, 0.0, 1.06, 0.94],
             [5.0, 1.0, 7.6, 1.6, 0.0, 0.0, 1.02, -8.78, 0.0, 1.06, 0.94],
             [6.0, 2.0, 11.2, 7.5, 0.0, 0.0, 1.07, -14.22, 0.0, 1.06, 0.94],
             [7.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.062, -13.37, 0.0, 1.06, 0.94],
             [8.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.09, -13.36, 0.0, 1.06, 0.94],
             [9.0, 1.0, 29.5, 16.6, 0.0, 19.0, 1.056, -14.94, 0.0, 1.06, 0.94],
             [10.0, 1.0, 9.0, 5.8, 0.0, 0.0, 1.051, -15.1, 0.0, 1.06, 0.94],
             [11.0, 1.0, 3.5, 1.8, 0.0, 0.0, 1.057, -14.79, 0.0, 1.06, 0.94],
             [12.0, 1.0, 6.1, 1.6, 0.0, 0.0, 1.055, -15.07, 0.0, 1.06, 0.94],
             [13.0, 1.0, 13.5, 5.8, 0.0, 0.0, 1.05, -15.16, 0.0, 1.06, 0.94],
             [14.0, 1.0, 14.9, 5.0, 0.0, 0.0, 1.036, -16.04, 0.0, 1.06, 0.94]])

        # Generator (unit) data — UC columns aligned with Case118 / PowerZooJax case14
        self.units = DataFrame(
            ['id', 'bus_id', 'Pg', 'Qg', 'Qmax', 'Qmin', 'Vg', 'mBase', 'status',
             'Pmax', 'Pmin', 'mc_a', 'mc_b', 'mc_c', 'p_max', 'p_min',
             'type', 'ramp_up', 'ramp_down', 'init_start_up_cost', 'keep_time',
             'init_power', 'init_state', 'min_up_time', 'min_down_time', 'init_no_load_cost'],
            [[1.0, 1.0, 232.4, -16.9, 10.0, 0.0, 1.06, 100.0, 1.0, 332.4, 0.0, 0.0, 0.05541301, 34.216168, 332.4, 0.0, 'gas', 0.6953069, 0.6953069, 34238.45, 48.0, 230.74888, 1.0, 2.0, 2.0, 0.0],
             [2.0, 2.0, 40.0, 42.4, 50.0, -40.0, 1.045, 100.0, 1.0, 140.0, 0.0, 0.00897295, -1.143378, 67.97416, 140.0, 0.0, 'gas', 0.66157144, 0.66157144, 11382.84, 48.0, 0.0, 0.0, 2.0, 2.0, 0.0],
             [3.0, 3.0, 0.0, 23.4, 40.0, 0.0, 1.01, 100.0, 1.0, 100.0, 0.0, 0.0357393, -2.380081, 72.64597, 100.0, 0.0, 'gas', 0.6621, 0.6621, 5235.41, 48.0, 0.0, 0.0, 2.0, 2.0, 0.0],
             [4.0, 6.0, 0.0, 12.2, 24.0, -6.0, 1.07, 100.0, 1.0, 100.0, 0.0, 0.03130176, -2.3915932, 78.5705, 100.0, 0.0, 'gas', 0.6694, 0.6694, 5272.29, 48.0, 0.0, 0.0, 8.0, 8.0, 0.0],
             [5.0, 8.0, 0.0, 17.4, 24.0, -6.0, 1.09, 100.0, 1.0, 100.0, 0.0, 0.03348683, -2.4855613, 76.65385, 100.0, 0.0, 'gas', 0.6504, 0.6504, 7950.56, 48.0, 0.0, 0.0, 2.0, 2.0, 0.0]])

        # Branch (line) data
        self.lines = DataFrame(
            ['id', 'from', 'to', 'r', 'x', 'b', 'rateA', 'ratio', 'angle', 'status', 'floor', 'cap'],
            [[1.0, 1.0, 2.0, 0.01938, 0.05917, 0.0528, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [2.0, 1.0, 5.0, 0.05403, 0.22304, 0.0492, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [3.0, 2.0, 3.0, 0.04699, 0.19797, 0.0438, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [4.0, 2.0, 4.0, 0.05811, 0.17632, 0.034, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [5.0, 2.0, 5.0, 0.05695, 0.17388, 0.0346, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [6.0, 3.0, 4.0, 0.06701, 0.17103, 0.0128, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [7.0, 4.0, 5.0, 0.01335, 0.04211, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [8.0, 4.0, 7.0, 0.0, 0.20912, 0.0, 0.0, 0.978, 0.0, 1.0, 0.0, 0.0],
             [9.0, 4.0, 9.0, 0.0, 0.55618, 0.0, 0.0, 0.969, 0.0, 1.0, 0.0, 0.0],
             [10.0, 5.0, 6.0, 0.0, 0.25202, 0.0, 0.0, 0.932, 0.0, 1.0, 0.0, 0.0],
             [11.0, 6.0, 11.0, 0.09498, 0.1989, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [12.0, 6.0, 12.0, 0.12291, 0.25581, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [13.0, 6.0, 13.0, 0.06615, 0.13027, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [14.0, 7.0, 8.0, 0.0, 0.17615, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [15.0, 7.0, 9.0, 0.0, 0.11001, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [16.0, 9.0, 10.0, 0.03181, 0.0845, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [17.0, 9.0, 14.0, 0.12711, 0.27038, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [18.0, 10.0, 11.0, 0.08205, 0.19207, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [19.0, 12.0, 13.0, 0.22092, 0.19988, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [20.0, 13.0, 14.0, 0.17093, 0.34802, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])

        # Load data
        self.loads = DataFrame(
            ['id', 'bus_id', 'Pd', 'Qd', 'mc_a', 'mc_b', 'mc_c', 'd_max', 'd_min'],
            [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [2.0, 2.0, 21.7, 12.7, 0.0, 0.0, 0.0, 21.7, 21.7],
             [3.0, 3.0, 94.2, 19.0, 0.0, 0.0, 0.0, 94.2, 94.2],
             [4.0, 4.0, 47.8, -3.9, 0.0, 0.0, 0.0, 47.8, 47.8],
             [5.0, 5.0, 7.6, 1.6, 0.0, 0.0, 0.0, 7.6, 7.6],
             [6.0, 6.0, 11.2, 7.5, 0.0, 0.0, 0.0, 11.2, 11.2],
             [7.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [8.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [9.0, 9.0, 29.5, 16.6, 0.0, 0.0, 0.0, 29.5, 29.5],
             [10.0, 10.0, 9.0, 5.8, 0.0, 0.0, 0.0, 9.0, 9.0],
             [11.0, 11.0, 3.5, 1.8, 0.0, 0.0, 0.0, 3.5, 3.5],
             [12.0, 12.0, 6.1, 1.6, 0.0, 0.0, 0.0, 6.1, 6.1],
             [13.0, 13.0, 13.5, 5.8, 0.0, 0.0, 0.0, 13.5, 13.5],
             [14.0, 14.0, 14.9, 5.0, 0.0, 0.0, 0.0, 14.9, 14.9]])
        
        self.real_params = True
        super().__init__(*args, **kwargs)
