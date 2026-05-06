#!/usr/bin/env python3
"""Build Case118.py (ACPF-ready) and Case118_MATPOWER_temp.py from case118.m.

Run from repo root:
  uv run python powerzoo/case/source_mfile/build_case118_ac.py

Requires: powerzoo/case/source_mfile/case118.m
          powerzoo/case/source_mfile/case118_uc_by_bus.json (from prior Case118 UC export)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[3]
MFILE = Path(__file__).resolve().parent / "case118.m"
UC_JSON = Path(__file__).resolve().parent / "case118_uc_by_bus.json"


def _fmt_row(vals, prec=6):
    out = []
    for v in vals:
        if isinstance(v, float):
            s = f"{v:.{prec}g}"
            if "e" in s.lower():
                s = f"{v:.{prec}f}".rstrip("0").rstrip(".")
            out.append(s)
        else:
            out.append(repr(v) if isinstance(v, str) else str(v))
    return "[" + ", ".join(out) + "]"


def main():
    from powerzoo.case.source_mfile.MfileModel import MFile

    text = MFILE.read_text(encoding="utf-8", errors="replace")
    m = MFile()
    m.read(text)
    uc = json.loads(UC_JSON.read_text(encoding="utf-8"))

    base_mva = float(m.baseMVA)

    # --- nodes ---
    node_rows = []
    for i in range(len(m.bus)):
        row = m.bus.iloc[i]
        bid = int(float(row["bus_i"]))
        node_rows.append(
            [
                float(bid),
                float(row["type"]),
                float(row["Pd"]),
                float(row["Qd"]),
                float(row["Gs"]),
                float(row["Bs"]),
                float(row["Vm"]),
                float(row["Va"]),
                float(row["baseKV"]),
                float(row["Vmax"]),
                float(row["Vmin"]),
            ]
        )

    # --- lines ---
    line_rows = []
    for i in range(len(m.branch)):
        br = m.branch.iloc[i]
        rate_a = float(br["rateA"]) if "rateA" in m.branch.columns else 0.0
        line_rows.append(
            [
                float(i + 1),
                float(br["fbus"]),
                float(br["tbus"]),
                float(br["r"]),
                float(br["x"]),
                float(br["b"]),
                rate_a,
                float(br["ratio"]),
                float(br["angle"]),
                float(br["status"]),
                0.0,
                rate_a if rate_a > 0 else 0.0,
            ]
        )

    # --- units: MATPOWER gen order, UC fields from JSON by bus_id ---
    unit_rows = []
    for i in range(len(m.gen)):
        g = m.gen.iloc[i]
        bid = int(float(g["bus"]))
        u = uc[str(bid)]
        unit_rows.append(
            [
                float(i + 1),
                float(bid),
                float(g["Pg"]),
                float(g["Qg"]),
                float(g["Qmax"]),
                float(g["Qmin"]),
                float(g["Vg"]),
                float(g["mBase"]),
                float(g["status"]),
                float(g["Pmax"]),
                float(g["Pmin"]),
                u["mc_a"],
                u["mc_b"],
                u["mc_c"],
                u["p_max"],
                u["p_min"],
                u["type"],
                u["ramp_up"],
                u["ramp_down"],
                u["init_start_up_cost"],
                u["keep_time"],
                u["init_power"],
                u["init_state"],
                u["min_up_time"],
                u["min_down_time"],
                u["init_no_load_cost"],
            ]
        )

    # --- loads: Case14-style, one per bus ---
    load_rows = []
    for i in range(len(m.bus)):
        row = m.bus.iloc[i]
        bid = int(float(row["bus_i"]))
        pd = float(row["Pd"])
        qd = float(row["Qd"])
        dmax = max(pd, 0.0)
        load_rows.append(
            [float(i + 1), float(bid), pd, qd, 0.0, 0.0, 0.0, dmax, 0.0]
        )

    nodes_lit = ",\n            ".join(_fmt_row(r) for r in node_rows)
    lines_lit = ",\n            ".join(_fmt_row(r) for r in line_rows)
    units_lit = ",\n            ".join(_fmt_row(r, prec=8) for r in unit_rows)
    loads_lit = ",\n            ".join(_fmt_row(r, prec=8) for r in load_rows)

    case118_py = dedent(
        f'''\
        """IEEE 118-bus system (MATPOWER case118)

        Topology and power-flow data (nodes / lines / generator Pg–Qg–Vg) follow
        MATPOWER `data/case118.m` (bundled as `source_mfile/case118.m`).

        UC-style economic columns (`type`, `mc_*`, `p_max`/`p_min`, ramps, …) are
        preserved from the previous PowerZoo Case118 SCUC-style table, keyed by
        `bus_id` (54 generators; unique buses).

        Nodes / lines include full MATPOWER fields so `cal_pf_trans.run_acpf` and
        `run_dcpf` work on `Case118().init()`.

        Base MVA: {base_mva:g} MVA
        """
        from powerzoo.case.CaseBase import ClearCase, DataFrame


        class Case118(ClearCase):
            def __init__(self, *args, **kwargs):
                self.baseMVA = {base_mva:g}
                self.baseKV = 138.0

                self.nodes = DataFrame(
                    ['id', 'type', 'Pd', 'Qd', 'Gs', 'Bs', 'Vm', 'Va', 'baseKV', 'Vmax', 'Vmin'],
                    [
            {nodes_lit}
                    ])

                self.units = DataFrame(
                    ['id', 'bus_id', 'Pg', 'Qg', 'Qmax', 'Qmin', 'Vg', 'mBase', 'status',
                     'Pmax', 'Pmin', 'mc_a', 'mc_b', 'mc_c', 'p_max', 'p_min',
                     'type', 'ramp_up', 'ramp_down', 'init_start_up_cost', 'keep_time',
                     'init_power', 'init_state', 'min_up_time', 'min_down_time', 'init_no_load_cost'],
                    [
            {units_lit}
                    ])

                self.lines = DataFrame(
                    ['id', 'from', 'to', 'r', 'x', 'b', 'rateA', 'ratio', 'angle', 'status', 'floor', 'cap'],
                    [
            {lines_lit}
                    ])

                self.loads = DataFrame(
                    ['id', 'bus_id', 'Pd', 'Qd', 'mc_a', 'mc_b', 'mc_c', 'd_max', 'd_min'],
                    [
            {loads_lit}
                    ])

                self.real_params = True
                super().__init__(*args, **kwargs)


        if __name__ == '__main__':
            c = Case118()
            c.check()
            print(c.get_node_ptdf())
        '''
    )

    out_main = ROOT / "powerzoo" / "case" / "Case118.py"
    out_main.write_text(case118_py, encoding="utf-8")
    print("Wrote", out_main)

    # Minimal temp module: MATPOWER-only (no UC merge), for diff / reference
    temp_units = []
    for i in range(len(m.gen)):
        g = m.gen.iloc[i]
        gc = m.gencost.iloc[i] if i < len(m.gencost) else None
        tc_c2 = float(gc["A"]) if gc is not None else 0.0
        tc_c1 = float(gc["B"]) if gc is not None else 0.0
        # TC(p)=c2·p²+c1·p+c0 → MC(p)=2·c2·p+c1 → mc_a=0, mc_b=2·c2, mc_c=c1
        mc_a = 0.0
        mc_b = 2.0 * tc_c2
        mc_c = tc_c1
        temp_units.append(
            [
                float(i + 1),
                float(g["bus"]),
                float(g["Pg"]),
                float(g["Qg"]),
                float(g["Qmax"]),
                float(g["Qmin"]),
                float(g["Vg"]),
                float(g["mBase"]),
                float(g["status"]),
                float(g["Pmax"]),
                float(g["Pmin"]),
                mc_a,
                mc_b,
                mc_c,
            ]
        )
    units_temp_lit = ",\n            ".join(_fmt_row(r, prec=8) for r in temp_units)
    temp_py = dedent(
        f'''\
        # AUTO-GENERATED by build_case118_ac.py — MATPOWER case118 only (no UC columns).
        # Do not import from production code; use Case118.py instead.
        from powerzoo.case.CaseBase import ClearCase, DataFrame


        class Case118_MATPOWER_temp(ClearCase):
            def __init__(self, *args, **kwargs):
                self.baseMVA = {base_mva:g}
                self.baseKV = 138.0
                self.nodes = DataFrame(
                    ['id', 'type', 'Pd', 'Qd', 'Gs', 'Bs', 'Vm', 'Va', 'baseKV', 'Vmax', 'Vmin'],
                    [
            {nodes_lit}
                    ])
                self.units = DataFrame(
                    ['id', 'bus_id', 'Pg', 'Qg', 'Qmax', 'Qmin', 'Vg', 'mBase', 'status', 'Pmax', 'Pmin',
                     'mc_a', 'mc_b', 'mc_c'],
                    [
            {units_temp_lit}
                    ])
                self.lines = DataFrame(
                    ['id', 'from', 'to', 'r', 'x', 'b', 'rateA', 'ratio', 'angle', 'status', 'floor', 'cap'],
                    [
            {lines_lit}
                    ])
                self.loads = DataFrame(
                    ['id', 'bus_id', 'Pd', 'Qd', 'mc_a', 'mc_b', 'mc_c', 'd_max', 'd_min'],
                    [
            {loads_lit}
                    ])
                self.real_params = True
                super().__init__(*args, **kwargs)
        '''
    )
    out_temp = Path(__file__).resolve().parent / "Case118_MATPOWER_temp.py"
    out_temp.write_text(temp_py, encoding="utf-8")
    print("Wrote", out_temp)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
