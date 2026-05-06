# 01 — 加载一个 Case

**脚本**：`examples/01_create_case.py`

PowerZoo 自带若干标准 IEEE 测试 case。本示例展示如何加载并查看拓扑。

## 可用 case

case 组织在 `powerzoo/case/` 下的 `transmission/` 与 `distribution/` 子包中。可用 `list_cases()` 编程式查询：

```python
from powerzoo.case import list_cases
list_cases()                            # all cases
list_cases(grid_type="distribution")    # distribution only
```

### 输电

| 名称 | Bus 数 | 电压等级 | 描述 |
|---|---|---|---|
| Case5 | 5 | HV | IEEE 5-bus 测试系统 |
| Case14 | 14 | HV | IEEE 14-bus 测试系统 |
| Case29GB | 29 | HV | GB 简化 29-bus 输电网 |
| Case118 | 118 | HV | IEEE 118-bus 测试系统 |
| Case300 | 300 | HV | IEEE 300-bus 测试系统 |
| Case1354pegase | 1354 | HV | 欧洲 PEGASE 1354-bus 系统 |
| Case2383wp | 2383 | HV | 波兰 2383-bus 冬季峰值系统 |

### 配电

| 名称 | Bus 数 | 相数 | 电压等级 | 描述 |
|---|---|---|---|---|
| Case33bw | 33 | 1 | MV | IEEE 33-bus Baran & Wu 辐射配电 |
| Case118zh | 118 | 1 | MV | 118-bus Zhang 配电系统 |
| Case123 | 123 | 3 | MV | IEEE 123-bus 三相配电 |
| Case141 | 141 | 1 | MV | 141-bus Caracas 配电系统 |
| Case533mt_hi | 533 | 1 | MV | 533-bus 瑞典配电（高负荷） |
| Case533mt_lo | 533 | 1 | MV | 533-bus 瑞典配电（低负荷） |

## 代码

`load_case` 接受整数、字符串（`'Case5'` 或 `'case5'`，前缀之后大小写不敏感），或 MATPOWER `.m` 文件路径。下面两种写法等价：

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

## 期望输出

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

## `ClearCase` 的关键方法

| 方法 | 返回 | 描述 |
|---|---|---|
| `get_node_gsdf()` | `DataFrame` | PTDF / GSDF 矩阵（lines × nodes） |
| `get_nodes_units_map()` | `ndarray` | 机组到节点的关联矩阵 |
| `get_nodes_loads_map()` | `ndarray` | 负荷到节点的关联矩阵 |
| `get_nodes_id(bus_ids)` | `ndarray` | 把外部 bus ID 转换为内部下标 |

!!! tip
    访问 PTDF 或关联矩阵前请先调用 `case.init()`——它会归一化行下标，并缓存计算结果以便复用。
