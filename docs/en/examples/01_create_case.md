# 01 — Load a Case

**Script:** `examples/01_create_case.py`

PowerZoo ships with several standard IEEE test cases. This example shows how to load them and inspect the topology.

## Available Cases

Cases are organised into `transmission/` and `distribution/` sub-packages
under `powerzoo/case/`.  Use `list_cases()` to discover them programmatically:

```python
from powerzoo.case import list_cases
list_cases()                            # all cases
list_cases(grid_type="distribution")    # distribution only
```

### Transmission

| Name | Buses | Voltage | Description |
|---|---|---|---|
| Case5 | 5 | HV | IEEE 5-bus test system |
| Case14 | 14 | HV | IEEE 14-bus test system |
| Case29GB | 29 | HV | GB reduced 29-bus transmission network |
| Case118 | 118 | HV | IEEE 118-bus test system |
| Case300 | 300 | HV | IEEE 300-bus test system |
| Case1354pegase | 1354 | HV | European PEGASE 1354-bus system |
| Case2383wp | 2383 | HV | Polish 2383-bus winter peak system |

### Distribution

| Name | Buses | Phase | Voltage | Description |
|---|---|---|---|---|
| Case33bw | 33 | 1 | MV | IEEE 33-bus Baran & Wu radial distribution |
| Case118zh | 118 | 1 | MV | 118-bus Zhang distribution system |
| Case123 | 123 | 3 | MV | IEEE 123-bus three-phase distribution |
| Case141 | 141 | 1 | MV | 141-bus Caracas distribution system |
| Case533mt_hi | 533 | 1 | MV | 533-bus Swedish distribution (high load) |
| Case533mt_lo | 533 | 1 | MV | 533-bus Swedish distribution (low load) |

## Code

`load_case` accepts an integer, a string (`'Case5'` or `'case5'`, case-insensitive after the prefix) or a path to a MATPOWER `.m` file. The two forms below are equivalent:

```python
from powerzoo.case import load_case

case = load_case(5)            # integer shorthand
case = load_case("Case5")      # explicit name
case.init()                     # normalise indices and compute internal mappings

print("=== Nodes ===")
print(case.nodes.head())

print("\n=== Lines ===")
print(case.lines[["#id", "from_bus", "to_bus", "cap"]].head())

print("\n=== Units ===")
print(case.units[["#id", "node_id", "p_min", "p_max", "cost_a"]].head())

print("\n=== Loads ===")
print(case.loads[["#id", "node_id", "d_max"]].head())
```

## Expected Output

```
=== Nodes ===
   #id  base_kv
0    1    230.0
1    2    230.0
...

=== Lines ===
   #id  from_bus  to_bus    cap
0    1         1       2  400.0
...
```

## Key Methods on `ClearCase`

| Method | Returns | Description |
|---|---|---|
| `get_node_gsdf()` | `DataFrame` | PTDF/GSDF matrix (lines × nodes) |
| `get_nodes_units_map()` | `ndarray` | Unit-to-node incidence matrix |
| `get_nodes_loads_map()` | `ndarray` | Load-to-node incidence matrix |
| `get_nodes_id(bus_ids)` | `ndarray` | Convert external bus IDs to internal indices |

!!! tip
    Call `case.init()` before accessing PTDF or incidence matrices — it normalises
    row indices and caches the computed matrices for reuse.
