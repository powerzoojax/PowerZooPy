# Cases

## Directory layout

Cases live in two sub-packages that mirror the grid type:

```
powerzoo/case/
├── transmission/   # Case5, Case14, Case118, Case300, Case29GB, ...
├── distribution/   # Case33bw, Case123, Case141, Case533mt_*, ...
├── CaseBase.py     # ClearCase base class
└── _registry.py    # list_cases() discovery
```

## Loading a case

::: powerzoo.case.load_case
    options:
      show_source: false

```python
from powerzoo.case import load_case

case = load_case(5)                             # integer shorthand
case = load_case("Case5")                       # explicit name (case-insensitive after the prefix)
case = load_case("case33bw", grid_type="distribution")
case = load_case("path/to/case30.m")            # MATPOWER .m file
```

`load_case` accepts: an integer (`5` → `Case5`); a string `'Case5'`, `'case5'` or `'5'`; or a path to a MATPOWER `.m` file. Pass `grid_type='transmission'` or `grid_type='distribution'` to restrict the search and warn on mismatch.

## Discovering available cases

```python
from powerzoo.case import list_cases

all_cases = list_cases()                         # everything under transmission/ + distribution/
dist_only = list_cases(grid_type="distribution") # distribution-only
big_cases = list_cases(min_buses=100)            # cases with ≥100 buses
```

Each entry is a `CaseMeta` dataclass with fields: `name`, `module_path`,
`grid_type`, `bus_count`, `phase`, `voltage_level`, `source`, `description`.

## Case metadata attributes

Every `ClearCase` subclass declares class-level metadata:

| Attribute | Type | Values | Description |
|---|---|---|---|
| `GRID_TYPE` | str | `"transmission"`, `"distribution"` | Grid category |
| `BUS_COUNT` | int | | Number of buses |
| `PHASE` | str | `"1"`, `"3"` | Single-phase or three-phase |
| `VOLTAGE_LEVEL` | str | `"HV"`, `"MV"`, `"LV"` | Voltage level class |
| `SOURCE` | str | `"MATPOWER"`, `"custom"`, `"CSV"` | Data origin |
| `DESCRIPTION` | str | | One-line description |

## ClearCase base class

::: powerzoo.case.CaseBase.ClearCase
    options:
      show_source: false
      members:
        - __init__
        - init
        - get_node_gsdf
        - get_nodes_units_map
        - get_nodes_loads_map
        - get_nodes_id
