"""MATPOWER M-file Parser

This module parses MATPOWER case files (.m) and extracts power system data.

@Date: 2020-10-03
@Updated: 2024
"""
import math
import pandas as pd
import re


def _eval_matlab_expr(token: str) -> float:
    """Evaluate a simple MATLAB numeric expression to a Python float.

    Handles plain numbers, scientific notation (``1.33E-05``), and
    simple arithmetic with ``sqrt``, ``/``, ``*`` (e.g. ``50/3``,
    ``12/sqrt(3)``).  Falls back to returning the raw string when
    evaluation fails so the caller can decide what to do.
    """
    try:
        return float(token)
    except (ValueError, TypeError):
        pass
    try:
        safe = token.replace("sqrt", "math.sqrt")
        return float(eval(safe, {"__builtins__": {}}, {"math": math}))
    except Exception:
        return token


class MFile:
    """Parser for MATPOWER m-file case format.
    
    Attributes:
        name: Case name extracted from m-file
        baseMVA: System base power (MVA)
        bus: Bus data DataFrame
        gen: Generator data DataFrame  
        branch: Branch data DataFrame
        gencost: Generator cost data DataFrame
        bus_name: List of bus names (if available)
    """
    
    # MATPOWER bus data columns (13 columns standard format)
    BUS_COLUMNS = [
        'bus_i',    # Bus number (1 to N)
        'type',     # Bus type: 1=PQ, 2=PV, 3=Ref/Slack, 4=Isolated
        'Pd',       # Real power demand (MW)
        'Qd',       # Reactive power demand (MVAr)
        'Gs',       # Shunt conductance (MW at V = 1.0 p.u.)
        'Bs',       # Shunt susceptance (MVAr at V = 1.0 p.u.)
        'area',     # Area number
        'Vm',       # Voltage magnitude (p.u.)
        'Va',       # Voltage angle (degrees)
        'baseKV',   # Base voltage (kV)
        'zone',     # Loss zone
        'Vmax',     # Maximum voltage magnitude (p.u.)
        'Vmin'      # Minimum voltage magnitude (p.u.)
    ]
    
    # MATPOWER generator data columns (21 columns standard format)
    GEN_COLUMNS = [
        'bus',      # Bus number
        'Pg',       # Real power output (MW)
        'Qg',       # Reactive power output (MVAr)
        'Qmax',     # Maximum reactive power output (MVAr)
        'Qmin',     # Minimum reactive power output (MVAr)
        'Vg',       # Voltage magnitude setpoint (p.u.)
        'mBase',    # Total MVA base of machine (MVA)
        'status',   # Machine status: >0 in-service, <=0 out-of-service
        'Pmax',     # Maximum real power output (MW)
        'Pmin',     # Minimum real power output (MW)
        'Pc1',      # Lower real power output of PQ capability curve (MW)
        'Pc2',      # Upper real power output of PQ capability curve (MW)
        'Qc1min',   # Minimum reactive power output at Pc1 (MVAr)
        'Qc1max',   # Maximum reactive power output at Pc1 (MVAr)
        'Qc2min',   # Minimum reactive power output at Pc2 (MVAr)
        'Qc2max',   # Maximum reactive power output at Pc2 (MVAr)
        'ramp_agc', # Ramp rate for load following/AGC (MW/min)
        'ramp_10',  # Ramp rate for 10 minute reserves (MW)
        'ramp_30',  # Ramp rate for 30 minute reserves (MW)
        'ramp_q',   # Ramp rate for reactive power (MVAr/min)
        'apf'       # Area participation factor
    ]
    
    # MATPOWER branch data columns (13 columns standard format)
    BRANCH_COLUMNS = [
        'fbus',     # From bus number
        'tbus',     # To bus number
        'r',        # Resistance (p.u. on system baseMVA and target bus baseKV)
        'x',        # Reactance (p.u.)
        'b',        # Total line charging susceptance (p.u.)
        'rateA',    # MVA rating A (long term rating)
        'rateB',    # MVA rating B (short term rating)
        'rateC',    # MVA rating C (emergency rating)
        'ratio',    # Transformer off nominal turns ratio (0 for line, >0 for transformer)
        'angle',    # Transformer phase shift angle (degrees)
        'status',   # Branch status: 1=in-service, 0=out-of-service
        'angmin',   # Minimum angle difference (degrees)
        'angmax'    # Maximum angle difference (degrees)
    ]
    
    # Generator cost data columns (after processing).
    # A, B, C are MATPOWER total-cost polynomial coefficients:
    #   TC(p) = A·p² + B·p + C   (for type 2, n=3)
    # Callers must convert to marginal cost: MC = dTC/dp = 2A·p + B.
    GENCOST_COLUMNS = ['type', 'startup', 'shutdown', 'A', 'B', 'C']
    
    def __init__(self):
        """Initialize MFile parser with empty DataFrames."""
        self.name = ''
        self.baseMVA = 100.0  # Default base MVA
        self.bus = pd.DataFrame(columns=self.BUS_COLUMNS)
        self.gen = pd.DataFrame(columns=self.GEN_COLUMNS)
        self.branch = pd.DataFrame(columns=self.BRANCH_COLUMNS)
        self.gencost = pd.DataFrame(columns=self.GENCOST_COLUMNS)
        self.bus_name = []

    def read(self, text: str) -> 'MFile':
        """Parse MATPOWER m-file text content.
        
        Args:
            text: Raw text content of m-file
            
        Returns:
            Self reference for method chaining
        """
        # Extract case name
        case_name_match = re.search(r'function\s+mpc\s+=\s+(\w+)', text)
        if case_name_match:
            self.name = case_name_match.group(1)
        
        # Extract baseMVA (may be an expression like 50/3)
        basemva_match = re.search(r'mpc\.baseMVA\s*=\s*([^;%\n]+)', text)
        if basemva_match:
            self.baseMVA = float(_eval_matlab_expr(basemva_match.group(1).strip().rstrip(';')))
        
        # Extract bus names if present
        bus_name = re.findall(r'mpc\.bus_name\s*=\s*{(.*)}', text, re.S)
        if len(bus_name) > 0:
            bus_name = bus_name[0]
            bus_name = re.sub(r'\'', '', bus_name)
            bus_name_list = re.findall(r'(.*);', bus_name)
            bus_name_list = [
                '_'.join(dl.strip().split()) 
                for dl in bus_name_list 
                if len(dl.strip()) > 0 and not dl.strip().startswith('%')
            ]
            self.bus_name = bus_name_list
        
        # Extract data tables (bus, gen, branch, gencost)
        data_tables = re.findall(r'[^\n]*\nmpc\.\w+\s*\=\s*\[.*?\];', text, re.S)
        
        for data_table in data_tables:
            header, body = data_table.split('mpc.')
            # Extract column names from comment line
            header_parts = re.split('%', header)
            if len(header_parts) > 1:
                header = header_parts[1].split()
            else:
                header = []
            
            # Extract table name and data
            name_match = re.search(r'(\w+)\s*\=', body)
            if not name_match:
                continue
            name = name_match.group(1).strip()
            
            data_str_match = re.search(r'\[(.*)\];', body, re.S)
            if not data_str_match:
                continue
            data_str = data_str_match.group(1).strip()
            
            # Parse data rows
            if ';' in data_str:
                data_lines = re.findall('(.*);', data_str)
            else:
                data_lines = data_str.splitlines()
            
            data_cells = [
                [_eval_matlab_expr(tok) for tok in dl.strip().split()]
                for dl in data_lines 
                if len(dl.strip()) > 0 and not dl.strip().startswith('%')
            ]
            
            if len(data_cells) == 0:
                continue
            
            # Create DataFrame based on table type
            df = None
            ncols_data = len(data_cells[0])
            if name == 'bus':
                ncols_std = len(self.BUS_COLUMNS)
                if ncols_data > ncols_std:
                    data_cells = [row[:ncols_std] for row in data_cells]
                actual_cols = min(ncols_std, ncols_data)
                df = pd.DataFrame(data_cells, columns=self.BUS_COLUMNS[:actual_cols] if actual_cols <= ncols_std else header)
            elif name == 'gen':
                ncols_std = len(self.GEN_COLUMNS)
                if ncols_data > ncols_std:
                    data_cells = [row[:ncols_std] for row in data_cells]
                actual_cols = min(ncols_std, ncols_data)
                df = pd.DataFrame(data_cells, columns=self.GEN_COLUMNS[:actual_cols] if actual_cols <= ncols_std else header)
            elif name == 'branch':
                ncols_std = len(self.BRANCH_COLUMNS)
                if ncols_data > ncols_std:
                    data_cells = [row[:ncols_std] for row in data_cells]
                actual_cols = min(ncols_std, ncols_data)
                df = pd.DataFrame(data_cells, columns=self.BRANCH_COLUMNS[:actual_cols] if actual_cols <= ncols_std else header)
            elif name == 'gencost':
                # Process generator cost data (polynomial format)
                data_cells_gencost = []
                for dc in data_cells:
                    line_tmp = dc[0:3]  # type, startup, shutdown
                    n_coeffs = int(dc[3]) if len(dc) > 3 else 0
                    # Extract polynomial coefficients (A*x^2 + B*x + C for n=3)
                    if n_coeffs == 1:
                        line_tmp += [0, 0, dc[-1]]
                    elif n_coeffs == 2:
                        line_tmp += [0, dc[-2], dc[-1]]
                    elif n_coeffs >= 3:
                        line_tmp += [dc[-3], dc[-2], dc[-1]]
                    else:
                        line_tmp += [0, 0, 0]
                    data_cells_gencost.append(line_tmp)
                df = pd.DataFrame(data_cells_gencost, columns=self.GENCOST_COLUMNS)
            else:
                # Generic handling for other tables
                if len(header) == len(data_cells[0]):
                    df = pd.DataFrame(data_cells, columns=header)
            
            if df is not None:
                setattr(self, name, df)
        
        return self
    
    def convert_units(self, pd_qd_in_kw: bool = False, r_x_in_ohms: bool = False) -> 'MFile':
        """Convert units for power flow calculation.
        
        Args:
            pd_qd_in_kw: If True, Pd/Qd are in kW/kVar, convert to MW/MVAr
            r_x_in_ohms: If True, r/x are in Ohms, convert to p.u.
            
        Returns:
            Self reference for method chaining
        """
        if pd_qd_in_kw and len(self.bus) > 0:
            # Convert Pd, Qd from kW/kVar to MW/MVAr
            self.bus['Pd'] = self.bus['Pd'].astype(float) / 1000.0
            self.bus['Qd'] = self.bus['Qd'].astype(float) / 1000.0
        
        if r_x_in_ohms and len(self.branch) > 0 and len(self.bus) > 0:
            # Convert r, x from Ohms to p.u.
            # Zbase = Vbase^2 / Sbase
            # where Vbase is in V, Sbase is in VA
            baseKV = float(self.bus.iloc[0]['baseKV'])  # Base voltage in kV
            Vbase = baseKV * 1e3  # Convert to V
            Sbase = self.baseMVA * 1e6  # Convert to VA
            Zbase = (Vbase ** 2) / Sbase  # Base impedance in Ohms
            
            self.branch['r'] = self.branch['r'].astype(float) / Zbase
            self.branch['x'] = self.branch['x'].astype(float) / Zbase
        
        return self
