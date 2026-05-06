"""MATPOWER M-file to Python Case Converter

This script converts MATPOWER m-files to Python case files for PowerZoo.
It handles unit conversions and generates complete power flow data.

@Date: 2020-10-03
@Updated: 2024
"""

import os
from typing import Optional

import numpy as np
from MfileModel import MFile


def wrap_list(lst, items_per_line=1):
    """Format a list for pretty-printing in Python code.
    
    Args:
        lst: List to format
        items_per_line: Number of items per line
        
    Returns:
        Formatted string representation of the list
    """
    lines = []
    for i in range(0, len(lst), items_per_line):
        chunk = lst[i:i + items_per_line]
        if i == 0:
            line = ", ".join("{!r}".format(x) for x in chunk)
        else:
            line = ", ".join("            {!r}".format(x) for x in chunk)
        lines.append(line)
    return "[" + ",\n ".join(lines) + "]"


def generate_case_file(name: str, m: MFile, nodes: list, units: list,
                       lines: list, loads: list, baseMVA: float = 100.0,
                       baseKV: float = 12.66, unit_note: str = "",
                       grid_type: str = "", description: str = "",
                       phase: Optional[str] = None) -> str:
    """Generate Python case file content.
    
    Args:
        name: Case name (without 'Case' prefix)
        m: Parsed MFile object
        nodes: Node data list
        units: Generator data list
        lines: Branch/line data list
        loads: Load data list
        baseMVA: System base MVA
        baseKV: System base kV
        unit_note: Note about unit conversions performed
        phase: If set, adds ``PHASE`` on the case class (e.g. ``\"1\"`` or ``\"3\"``).

    Returns:
        Python file content as string
    """
    phase_line = f'    PHASE = "{phase}"\n' if phase is not None else ""
    docstring = f'''"""Power System Case: {name}

Data Source: MATPOWER case file
Base MVA: {baseMVA} MVA
Base kV: {baseKV} kV

{unit_note}

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
    id, bus_id, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, mc_a, mc_b, mc_c
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
from math import inf

from powerzoo.case.CaseBase import ClearCase, DataFrame


class Case{name}(ClearCase):
    GRID_TYPE = "{grid_type}"
{phase_line}    BUS_COUNT = {len(nodes)}
    VOLTAGE_LEVEL = "{'HV' if baseKV >= 110 else ('MV' if baseKV >= 1 else 'LV')}"
    SOURCE = "MATPOWER"
    DESCRIPTION = "{description}"

    def __init__(self, *args, **kwargs):
        # System base values
        self.baseMVA = {baseMVA}
        self.baseKV = {baseKV}
        
        # Node (bus) data
        self.nodes = DataFrame(
            ['id', 'type', 'Pd', 'Qd', 'Gs', 'Bs', 'Vm', 'Va', 'baseKV', 'Vmax', 'Vmin'],
            {wrap_list(nodes)})

        # Generator (unit) data
        self.units = DataFrame(
            ['id', 'bus_id', 'Pg', 'Qg', 'Qmax', 'Qmin', 'Vg', 'mBase', 'status', 'Pmax', 'Pmin', 'mc_a', 'mc_b', 'mc_c', 'p_max', 'p_min'],
            {wrap_list(units)})

        # Branch (line) data
        self.lines = DataFrame(
            ['id', 'from', 'to', 'r', 'x', 'b', 'rateA', 'ratio', 'angle', 'status', 'floor', 'cap'],
            {wrap_list(lines)})

        # Load data
        self.loads = DataFrame(
            ['id', 'bus_id', 'Pd', 'Qd', 'mc_a', 'mc_b', 'mc_c', 'd_max', 'd_min'],
            {wrap_list(loads)})
        
        self.real_params = True
        super().__init__(*args, **kwargs)
'''
    return docstring


def convert_mfile_to_py(mfile_path: str, output_path: str = None,
                        convert_to_pu: bool = True,
                        grid_type: str = "",
                        description: str = "",
                        case_name: Optional[str] = None,
                        phase: Optional[str] = None) -> str:
    """Convert a MATPOWER m-file to PowerZoo Python format.
    
    Args:
        mfile_path: Path to the m-file
        output_path: Output path for Python file (optional)
        convert_to_pu: Whether to convert units to p.u. system
        grid_type: ``'transmission'`` or ``'distribution'`` (for metadata)
        description: One-line case description (for metadata)
        case_name: Python class suffix after ``Case`` (e.g. ``\"118zh\"`` → ``Case118zh``).
            If omitted, derived from the m-file function name (``case*`` prefix stripped).
        phase: Passed to :func:`generate_case_file` (recommended ``\"1\"`` for single-phase cases).

    Returns:
        Generated Python code as string
    """
    import re as _re

    with open(mfile_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    m = MFile().read(text)
    
    # Check if units need conversion (based on m-file comments).
    # Do not use a bare ``'kW' in text`` — files like case141.m mention kW in unrelated
    # comments while loads are in kVA.
    tl = text.lower()
    # ``kva`` alone matches ``kvar`` in ``kW & kVAr`` (case33bw) — use explicit Pd wording.
    pd_in_kw = "pd and qd are specified in kw" in tl
    pd_in_kva = _re.search(r"pd\s+is\s+specified\s+in\s+kva", tl) is not None
    rx_in_ohms = 'ohms' in text.lower() or 'r and x specified in ohms' in text.lower()

    pf_match = _re.search(r'(\d+\.?\d*)\s*power\s*factor', text)
    power_factor = float(pf_match.group(1)) if pf_match else None
    
    unit_note = "Unit conversions applied:\n"
    if pd_in_kva and convert_to_pu:
        unit_note += "  - Pd converted from kVA to MW/MVAr (divided by 1000"
        if power_factor:
            unit_note += f", power factor={power_factor}"
        unit_note += ")\n"
    elif pd_in_kw and convert_to_pu:
        unit_note += "  - Pd/Qd converted from kW/kVar to MW/MVAr (divided by 1000)\n"
    if rx_in_ohms and convert_to_pu:
        unit_note += "  - r/x converted from Ohms to p.u. (Z_pu = Z_ohm * Sbase / Vbase^2)\n"
    if not pd_in_kw and not pd_in_kva and not rx_in_ohms:
        unit_note = "All values are in standard MATPOWER units (MW, MVAr, p.u.)\n"
    
    # Get base values
    baseMVA = m.baseMVA
    baseKV = float(m.bus.iloc[0]['baseKV']) if 'baseKV' in m.bus.columns else 1.0
    
    # Calculate base impedance for unit conversion
    Vbase = baseKV * 1e3  # V
    Sbase = baseMVA * 1e6  # VA
    Zbase = (Vbase ** 2) / Sbase  # Ohms
    
    # Process bus data -> nodes
    nodes = []
    for _, row in m.bus.iterrows():
        bus_i = float(row['bus_i'])
        bus_type = float(row['type']) if 'type' in row else 1.0
        Pd = float(row['Pd']) if 'Pd' in row else 0.0
        Qd = float(row['Qd']) if 'Qd' in row else 0.0
        Gs = float(row['Gs']) if 'Gs' in row else 0.0
        Bs = float(row['Bs']) if 'Bs' in row else 0.0
        Vm = float(row['Vm']) if 'Vm' in row else 1.0
        Va = float(row['Va']) if 'Va' in row else 0.0
        bus_baseKV = float(row['baseKV']) if 'baseKV' in row else baseKV
        Vmax = float(row['Vmax']) if 'Vmax' in row else 1.1
        Vmin = float(row['Vmin']) if 'Vmin' in row else 0.9
        
        # Unit conversion (prefer kVA when documented)
        if pd_in_kva and convert_to_pu:
            S_kva = Pd  # Pd column holds apparent power in kVA
            S_mva = S_kva / 1000.0
            if power_factor:
                import math
                Pd = S_mva * power_factor
                Qd = S_mva * math.sin(math.acos(power_factor))
            else:
                Pd = S_mva
                Qd = 0.0
        elif pd_in_kw and convert_to_pu:
            Pd = Pd / 1000.0
            Qd = Qd / 1000.0

        nodes.append([bus_i, bus_type, Pd, Qd, Gs, Bs, Vm, Va, bus_baseKV, Vmax, Vmin])
    
    # Process generator data -> units
    units = []
    for i, row in m.gen.iterrows():
        unit_id = float(i + 1)
        bus_id = float(row['bus'])
        Pg = float(row['Pg']) if 'Pg' in row else 0.0
        Qg = float(row['Qg']) if 'Qg' in row else 0.0
        Qmax = float(row['Qmax']) if 'Qmax' in row else 9999.0
        Qmin = float(row['Qmin']) if 'Qmin' in row else -9999.0
        Vg = float(row['Vg']) if 'Vg' in row else 1.0
        mBase = float(row['mBase']) if 'mBase' in row else baseMVA
        status = float(row['status']) if 'status' in row else 1.0
        Pmax = float(row['Pmax']) if 'Pmax' in row else 9999.0
        Pmin = float(row['Pmin']) if 'Pmin' in row else 0.0
        
        # Convert MATPOWER total-cost coefficients to marginal-cost coefficients.
        # MATPOWER gencost type 2: TC(p) = c2·p² + c1·p + c0  (A=c2, B=c1, C=c0)
        # Marginal cost: MC(p) = dTC/dp = 2·c2·p + c1
        # So: mc_a=0, mc_b=2·c2, mc_c=c1
        mc_a, mc_b, mc_c = 0.0, 0.0, 0.0
        if hasattr(m, 'gencost') and len(m.gencost) > i:
            cost_row = m.gencost.iloc[i]
            tc_c2 = float(cost_row['A']) if 'A' in cost_row else 0.0
            tc_c1 = float(cost_row['B']) if 'B' in cost_row else 0.0
            mc_a = 0.0
            mc_b = 2.0 * tc_c2
            mc_c = tc_c1
        
        # p_max and p_min for compatibility with ClearCase
        p_max = Pmax
        p_min = Pmin
        
        units.append([unit_id, bus_id, Pg, Qg, Qmax, Qmin, Vg, mBase, status, 
                      Pmax, Pmin, mc_a, mc_b, mc_c, p_max, p_min])
    
    # Process branch data -> lines
    lines = []
    for i, row in m.branch.iterrows():
        line_id = float(i + 1)
        fbus = float(row['fbus'])
        tbus = float(row['tbus'])
        r = float(row['r']) if 'r' in row else 0.0
        x = float(row['x']) if 'x' in row else 0.0001
        b = float(row['b']) if 'b' in row else 0.0
        rateA = float(row['rateA']) if 'rateA' in row else 0.0
        ratio = float(row['ratio']) if 'ratio' in row else 0.0
        angle = float(row['angle']) if 'angle' in row else 0.0
        status = float(row['status']) if 'status' in row else 1.0
        
        # Unit conversion for impedance
        if rx_in_ohms and convert_to_pu:
            r = r / Zbase
            x = x / Zbase
        
        # Set floor and cap based on rateA
        if rateA > 0:
            floor = -rateA
            cap = rateA
        else:
            floor = 0.0
            cap = 0.0
        
        lines.append([line_id, fbus, tbus, r, x, b, rateA, ratio, angle, status, floor, cap])
    
    # Process bus data -> loads (one load per bus with demand)
    loads = []
    for i, row in m.bus.iterrows():
        load_id = float(i + 1)
        bus_id = float(row['bus_i'])
        Pd = float(row['Pd']) if 'Pd' in row else 0.0
        Qd = float(row['Qd']) if 'Qd' in row else 0.0
        
        # Unit conversion (prefer kVA when documented)
        if pd_in_kva and convert_to_pu:
            S_kva = Pd
            S_mva = S_kva / 1000.0
            if power_factor:
                import math
                Pd = S_mva * power_factor
                Qd = S_mva * math.sin(math.acos(power_factor))
            else:
                Pd = S_mva
                Qd = 0.0
        elif pd_in_kw and convert_to_pu:
            Pd = Pd / 1000.0
            Qd = Qd / 1000.0

        # Default cost coefficients (can be customized)
        mc_a, mc_b, mc_c = 0.0, 0.0, 0.0
        d_max = Pd
        d_min = Pd
        
        loads.append([load_id, bus_id, Pd, Qd, mc_a, mc_b, mc_c, d_max, d_min])
    
    # Extract case name (class Case{name})
    if case_name is not None:
        name = case_name
    else:
        name = m.name.replace('case', '') if m.name.startswith('case') else m.name

    # Generate output
    output = generate_case_file(name, m, nodes, units, lines, loads,
                                baseMVA, baseKV, unit_note,
                                grid_type=grid_type, description=description,
                                phase=phase)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output)
    
    return output


if __name__ == '__main__':
    # Example: Convert case33bw.m
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mfile_path = os.path.join(script_dir, 'case33bw.m')
    output_path = os.path.join(os.path.dirname(script_dir), 'Case33bw.py')
    
    if os.path.exists(mfile_path):
        output = convert_mfile_to_py(mfile_path, output_path, convert_to_pu=True)
        print(f"Generated: {output_path}")
        print("=" * 50)
        print(output[:2000] + "..." if len(output) > 2000 else output)
    else:
        print(f"File not found: {mfile_path}")
