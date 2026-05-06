# 案例数据

## 目录布局

Case 住在两个映射 grid 类型的子包中：

```
powerzoo/case/
├── transmission/   # Case5, Case14, Case118, Case300, Case29GB, ...
├── distribution/   # Case33bw, Case123, Case141, Case533mt_*, ...
├── CaseBase.py     # ClearCase 基类
└── _registry.py    # list_cases() 发现
```

## 加载一个 case

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

`load_case` 接受：一个整数（`5` → `Case5`）；字符串 `'Case5'`、`'case5'` 或 `'5'`；或 MATPOWER `.m` 文件的路径。传 `grid_type='transmission'` 或 `grid_type='distribution'` 限定搜索范围，不匹配时警告。

## 发现可用 case

```python
from powerzoo.case import list_cases

all_cases = list_cases()                         # everything under transmission/ + distribution/
dist_only = list_cases(grid_type="distribution") # distribution-only
big_cases = list_cases(min_buses=100)            # cases with ≥100 buses
```

每条记录是一个 `CaseMeta` dataclass，字段：`name`、`module_path`、`grid_type`、`bus_count`、`phase`、`voltage_level`、`source`、`description`。

## Case 元数据属性

每个 `ClearCase` 子类声明类级元数据：

| 属性 | 类型 | 取值 | 描述 |
|---|---|---|---|
| `GRID_TYPE` | str | `"transmission"`、`"distribution"` | 电网类别 |
| `BUS_COUNT` | int |  | bus 数 |
| `PHASE` | str | `"1"`、`"3"` | 单相或三相 |
| `VOLTAGE_LEVEL` | str | `"HV"`、`"MV"`、`"LV"` | 电压等级 |
| `SOURCE` | str | `"MATPOWER"`、`"custom"`、`"CSV"` | 数据来源 |
| `DESCRIPTION` | str |  | 一句话描述 |

## ClearCase 基类

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
