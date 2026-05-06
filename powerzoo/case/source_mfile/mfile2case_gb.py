"""MATPOWER M-file to Case29GB Python Converter

This script converts GBreducednetwork.m to Case29GB.py format.
Combines logic from json2py.py (for simulated data) and mfile_to_py.py (for reading m-files).

Usage: python mfile2case_gb.py

@Date: 2024
"""

import os
import sys
from pathlib import Path
import numpy as np
from MfileModel import MFile


class CostBase():
    """Base cost parameters for different generator types."""
    def __init__(self, pmax, a, b, c):
        self.pmax = pmax
        self.a = a
        self.b = b
        self.c = c


# Cost parameters for different unit types (from json2py.py)
cb100_coal = CostBase(100, 0.025, -2.1015, 2*71.0)
cb100_gas = CostBase(100, 0.039551, -4.3164, 2*141.1)
cb100_nuclear = CostBase(100, 0.008654, -1.0638, 2*34.44)
cb_dict = {'coal': cb100_coal, 'gas': cb100_gas, 'nuclear': cb100_nuclear}


def get_cost_params(pmax, unit_type):
    """Calculate cost parameters based on unit type and max power.
    
    Args:
        pmax: Maximum power output (MW)
        unit_type: Type of unit ('coal', 'gas', 'nuclear')
        
    Returns:
        Tuple of (mc_a, mc_b, mc_c, no_load_cost)
    """
    pmin_power = pmax * 0.2
    cb = cb_dict[unit_type]
    p_k = cb.pmax / pmax
    mc_a = cb.a * p_k * p_k
    mc_b = cb.b * p_k
    mc_c = round(cb.c * 50 / (pmax / cb.pmax + 50), 0)  # Scale for larger units
    ac_a = mc_a / 3
    ac_b = mc_b / 2
    ac_c = mc_c
    no_load_cost = ac_a * pmin_power ** 3 + ac_b * pmin_power ** 2 + ac_c * pmin_power
    no_load_cost = no_load_cost * 1.5
    return mc_a, mc_b, mc_c, no_load_cost


def determine_unit_type(pmax, mc_b_original, mc_c_original):
    """Determine unit type based on cost parameters and size.
    
    Args:
        pmax: Maximum power output (MW)
        mc_b_original: Original marginal cost coefficient b from m-file
        mc_c_original: Original marginal cost coefficient c from m-file
        
    Returns:
        Tuple of (unit_type, min_up_time, min_down_time, startup_cost_multiplier)
    
    Classification logic based on GB reduced network data analysis:
    - mc_b >= 100: Gas peakers (29 units, ~40 GW, 48.6%)
    - mc_b <= 1: Nuclear baseload (14 units, ~4.4 GW, 5.3%) 
    - Otherwise: Coal/CCGT (23 units, ~38 GW, 46.1%)
    
    This better reflects the UK electricity mix structure.
    """
    if mc_b_original >= 100:  # Very expensive - gas peakers
        unit_type = 'gas'
        min_up_time = 1
        min_down_time = 1
        startup_cost_multiplier = 1.0
    elif mc_b_original <= 1:  # Very cheap baseload - nuclear
        unit_type = 'nuclear'
        min_up_time = 96
        min_down_time = 96
        startup_cost_multiplier = 5.0
    else:  # Medium cost - coal/CCGT
        unit_type = 'coal'
        min_up_time = 4
        min_down_time = 4
        startup_cost_multiplier = 2.0
    
    return unit_type, min_up_time, min_down_time, startup_cost_multiplier


def convert_gb_mfile_to_case():
    """Convert GBreducednetwork.m to Case29GB.py format."""
    
    # File paths
    base_dir = Path(__file__).parent
    mfile_path = base_dir / "GBreducednetwork.m"
    output_path = base_dir.parent / "Case29GB.py"
    
    if not mfile_path.exists():
        print(f"Error: M-file not found: {mfile_path}")
        return
    
    # Read m-file
    with open(mfile_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    m = MFile().read(text)
    
    print(f"Case name: {m.name}")
    print(f"Base MVA: {m.baseMVA}")
    print(f"Number of buses: {len(m.bus)}")
    print(f"Number of generators: {len(m.gen)}")
    print(f"Number of branches: {len(m.branch)}")
    
    # 1. Process buses -> nodes (simplified, just IDs)
    num_buses = len(m.bus)
    print(f"\nProcessing {num_buses} buses...")
    
    # 2. Process generators -> units
    units_data = []
    print(f"\nProcessing {len(m.gen)} generators...")
    
    for i, row in m.gen.iterrows():
        bus_id = int(float(row['bus']))
        pmax = float(row['Pmax'])
        pmin = float(row['Pmin']) if 'Pmin' in row and float(row['Pmin']) > 0 else pmax * 0.2
        pg = float(row['Pg']) if 'Pg' in row else 0.0
        status = int(float(row['status'])) if 'status' in row else 1
        
        # Skip generators with zero or very small capacity
        if pmax <= 0:
            continue
        
        # Get original cost coefficients from gencost
        mc_a_orig, mc_b_orig, mc_c_orig = 0.0, 0.0, 0.0
        startup_cost_orig = 1500  # Default from m-file
        
        if hasattr(m, 'gencost') and len(m.gencost) > i:
            cost_row = m.gencost.iloc[i]
            mc_a_orig = float(cost_row['A']) if 'A' in cost_row else 0.0
            mc_b_orig = float(cost_row['B']) if 'B' in cost_row else 0.0
            mc_c_orig = float(cost_row['C']) if 'C' in cost_row else 0.0
            startup_cost_orig = float(cost_row['startup']) if 'startup' in cost_row else 1500
        
        # Determine unit type based on cost and size
        unit_type, min_up_time, min_down_time, startup_mult = determine_unit_type(
            pmax, mc_b_orig, mc_c_orig)
        
        # Calculate simulated cost parameters (like json2py.py)
        pmax_int = int(round(pmax))
        pmin_int = int(round(pmin))
        mc_a, mc_b, mc_c, no_load_cost = get_cost_params(pmax_int, unit_type)
        
        # Startup cost
        real_start_up_cost = int(round(startup_cost_orig * startup_mult * 3))
        
        # Ramp rates (default 70% per hour, similar to Case118)
        ramp_up = 0.7
        ramp_down = 0.7
        
        # Initial state (based on status and Pg)
        if status > 0 and pg > pmin:
            init_state = 1
            init_power = int(round(pg))
            keep_time = 96  # Assume running for a while
        else:
            init_state = 0
            init_power = 0
            keep_time = -96  # Assume off for a while
        
        unit = {
            'id': len(units_data) + 1,
            'bus_id': bus_id,
            'type': unit_type,
            'mc_a': round(mc_a, 6),
            'mc_b': round(mc_b, 6),
            'mc_c': round(mc_c, 0),
            'p_max': pmax_int,
            'p_min': pmin_int,
            'ramp_up': round(ramp_up, 4),
            'ramp_down': round(ramp_down, 4),
            'real_start_up_cost': real_start_up_cost,
            'keep_time': keep_time,
            'init_power': init_power,
            'init_state': init_state,
            'min_up_time': min_up_time,
            'min_down_time': min_down_time,
            'no_load_cost': round(no_load_cost, 2)
        }
        units_data.append(unit)
    
    print(f"Converted {len(units_data)} generators")
    
    # 3. Process branches -> lines
    lines_data = []
    print(f"\nProcessing {len(m.branch)} branches...")
    
    for i, row in m.branch.iterrows():
        fbus = int(float(row['fbus']))
        tbus = int(float(row['tbus']))
        r = float(row['r']) if 'r' in row else 0.0
        x = float(row['x']) if 'x' in row else 0.0001
        b = float(row['b']) if 'b' in row else 0.0
        rateA = float(row['rateA']) if 'rateA' in row else 0.0
        status = int(float(row['status'])) if 'status' in row else 1
        
        # Skip out-of-service lines
        if status <= 0:
            continue
        
        # Calculate susceptance s = 1/x (for DC power flow)
        if x > 0:
            s = round(1.0 / x, 5)
        else:
            s = 100000.0
        
        # Set flow limits based on rateA (MW limits)
        if rateA > 0:
            floor = -rateA
            cap = rateA
        else:
            floor = -100000.0
            cap = 100000.0
        
        line = {
            'id': len(lines_data) + 1,
            'from': fbus,
            'to': tbus,
            'x': x,
            's': s,
            'floor': floor,
            'cap': cap
        }
        lines_data.append(line)
    
    print(f"Converted {len(lines_data)} lines")
    
    # 4. Process bus loads -> loads
    loads_data = []
    total_load_mw = 0.0
    
    print(f"\nProcessing bus loads...")
    
    for i, row in m.bus.iterrows():
        bus_id = int(float(row['bus_i']))
        pd = float(row['Pd']) if 'Pd' in row else 0.0
        total_load_mw += pd
    
    print(f"Total system load: {total_load_mw:.2f} MW")
    
    # Calculate load ratio for each bus
    for i, row in m.bus.iterrows():
        bus_id = int(float(row['bus_i']))
        pd = float(row['Pd']) if 'Pd' in row else 0.0
        
        if total_load_mw > 0:
            load_ratio = round(pd / total_load_mw, 6)
        else:
            load_ratio = 0.0
        
        load = {
            'id': len(loads_data) + 1,
            'bus_id': bus_id,
            'mc_a': 0.0,
            'mc_b': 0.0,
            'mc_c': 0.0,
            'd_max': load_ratio,
            'd_min': load_ratio
        }
        loads_data.append(load)
    
    print(f"Generated {len(loads_data)} load entries")
    
    # 5. Generate Python code
    py_code = generate_python_code(num_buses, units_data, lines_data, loads_data)
    
    # Write output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(py_code)
    
    print(f"\n Successfully generated: {output_path}")


def generate_python_code(num_buses: int, units_data: list, 
                        lines_data: list, loads_data: list) -> str:
    """Generate Python class code for Case29GB.
    
    Args:
        num_buses: Number of buses in the system
        units_data: Generator/unit data list
        lines_data: Branch/line data list
        loads_data: Load data list
        
    Returns:
        Python code as string
    """
    
    # Format nodes
    nodes_range = f"range(1, {num_buses + 1})"
    
    # Format units
    units_columns = ['id', 'bus_id', 'type', 'mc_a', 'mc_b', 'mc_c', 'p_max', 'p_min',
                    'ramp_up', 'ramp_down', 'init_start_up_cost', 'keep_time',
                    'init_power', 'init_state', 'min_up_time', 'min_down_time', 'init_no_load_cost']
    units_rows = []
    for unit in units_data:
        row = [unit['id'], unit['bus_id'], unit['type'], unit['mc_a'], unit['mc_b'], unit['mc_c'],
               unit['p_max'], unit['p_min'], unit['ramp_up'], unit['ramp_down'],
               unit['real_start_up_cost'], unit['keep_time'], unit['init_power'],
               unit['init_state'], unit['min_up_time'], unit['min_down_time'], unit['no_load_cost']]
        units_rows.append(row)
    
    # Format lines
    lines_columns = ['id', 'from', 'to', 'x', 's', 'floor', 'cap']
    lines_rows = []
    for line in lines_data:
        row = [line['id'], line['from'], line['to'], line['x'], line['s'],
               line['floor'], line['cap']]
        lines_rows.append(row)
    
    # Format loads
    loads_columns = ['id', 'bus_id', 'mc_a', 'mc_b', 'mc_c', 'd_max', 'd_min']
    loads_rows = []
    for load in loads_data:
        row = [load['id'], load['bus_id'], load['mc_a'], load['mc_b'], load['mc_c'],
               load['d_max'], load['d_min']]
        loads_rows.append(row)
    
    code = f'''"""Power System Case: GB Reduced Network

Data Source: MATPOWER GBreducednetwork.m
Description: Reduced model of GB (Great Britain) network
            29 buses, {len(units_data)} generators, {len(lines_data)} transmission lines
            Data provided by Manolis Belivanis from Strathclyde

Generator cost parameters are simulated using the same methodology as Case118.
"""
from powerzoo.case.CaseBase import ClearCase, DataFrame


class Case29GB(ClearCase):
    def __init__(self, *args, **kwargs):
        self.nodes = DataFrame(['id'], {nodes_range})
        
        self.units = DataFrame(
            {units_columns},
            {format_data_rows(units_rows)})
        
        self.lines = DataFrame(
            {lines_columns},
            {format_data_rows(lines_rows)})
        
        self.loads = DataFrame(
            {loads_columns},
            {format_data_rows(loads_rows)})
        
        self.real_params = True
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    c = Case29GB()
    c.check()
'''
    
    return code


def format_data_rows(rows: list) -> str:
    """Format data rows for Python list string.
    
    Args:
        rows: List of data rows
        
    Returns:
        Formatted string representation
    """
    if not rows:
        return "[]"
    
    formatted_rows = []
    for row in rows:
        formatted_values = []
        for v in row:
            if isinstance(v, str):
                formatted_values.append(f"'{v}'")
            else:
                formatted_values.append(str(v))
        formatted_row = "[" + ", ".join(formatted_values) + "]"
        formatted_rows.append(formatted_row)
    
    if len(formatted_rows) == 1:
        return "[" + formatted_rows[0] + "]"
    else:
        result = "[\n"
        for i, row_str in enumerate(formatted_rows):
            if i < len(formatted_rows) - 1:
                result += " " * 12 + row_str + ",\n"
            else:
                result += " " * 12 + row_str + "]"
        return result


if __name__ == '__main__':
    convert_gb_mfile_to_case()
