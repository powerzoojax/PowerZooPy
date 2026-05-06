# Tasks

PowerZoo ships ready-made benchmark tasks that combine a grid case, data split, agent design, and evaluation protocol into a single object. Use `make_task_env` to instantiate any task by name:

```python
from powerzoo.tasks import make_task_env, list_public_tasks

print(list_public_tasks())
env = make_task_env("marl_opf", split="train")

# Use PettingZoo Parallel API (no RLlib required)
env = make_task_env("marl_opf", split="train", framework="pettingzoo")
```

Multi-agent tasks always use **specialised task adapters** by default (`TaskOPFMultiAgentEnv`, `TaskUCMultiAgentEnv`, `TaskEVMultiAgentEnv`, etc.). These adapters work with or without RLlib; when `ray[rllib]` is installed, the returned object also satisfies the RLlib `MultiAgentEnv` interface. To get a task-aware **PettingZoo Parallel API** wrapper around the same adapter semantics instead, pass `framework='pettingzoo'`.

The explicit public benchmark set is available via `powerzoo.tasks.public` or the re-exported helpers:

```python
from powerzoo.tasks import PUBLIC_TASKS, list_public_tasks, get_public_task_catalog

print(PUBLIC_TASKS)
print(list_public_tasks())
print(get_public_task_catalog()[0]["task_id"])
print(get_public_task_catalog()[0]["default_episode_horizon_steps"])
```

Only tasks that satisfy the benchmark contract stay in `PUBLIC_TASKS`: documented, registered, instantiable, and smoke-tested. Registered-but-incomplete tasks remain accessible through `list_tasks()` / `make_task_env(...)`, but are not part of the public benchmark set.

The `framework` parameter controls which multi-agent interface is used:

| Value | Description |
|---|---|
| `'auto'` (default) | Specialized task adapter (RLlib-compatible when `ray` is installed) |
| `'pettingzoo'` | Task-aware PettingZoo Parallel API via `powerzoo.tasks.interfaces.TaskPettingZooWrapper` (lightweight, no RLlib needed) |
| `'rllib'` | Same as `'auto'` but raises if `ray[rllib]` is missing |

---

## Registered task routing

| Task name | `make_task_env()` / `create_env()` returns |
|---|---|
| `battery_arbitrage` | `FlattenWrapper` around `PowerEnv` (single-agent Gymnasium) |
| `marl_opf` | `TaskOPFMultiAgentEnv` |
| `marl_der_arbitrage`, `marl_ders_benchmark` | `TaskResourceMultiAgentEnv` |
| `marl_ev_v2g` | `TaskEVMultiAgentEnv` |
| `dc_scheduling` | `FlattenWrapper` around `PowerEnv` (single-agent Gymnasium) |
| `dc_microgrid`, `dc_microgrid_safe` | `DCMicrogridEnv` (single-agent Gymnasium, self-contained) |
| `gencos_bidding` | `GenCosMARLEnv` (PettingZoo Parallel API; competitive 5-agent market) |
| `marl_uc` | `TaskUCMultiAgentEnv` |
| `opf_118` / `opf_118_7d` | `TaskOPFMultiAgentEnv` |
| `joint_trans_dist` / `joint_trans_dist_7d` | Experimental only — still registered, but not part of `PUBLIC_TASKS`; instantiation currently fails until the joint adapter/reward path ships |

---

## Public Benchmark Task Cards

`get_public_task_catalog()` returns stable task-card metadata for the current public benchmark set. Use it as the source of truth when building experiment menus, benchmark summaries, or docs that need to stay in sync with the real public tasks.

```python
from powerzoo.tasks import get_public_task_catalog

for card in get_public_task_catalog():
    print(card["task_id"], card["grid_case"], card["default_episode_horizon_steps"])
```

| Task | Grid | Agent mode | Default observation | Reward / cost contract | Horizon | Frameworks |
|---|---|---|---|---|---|---|
| `battery_arbitrage` | distribution / `Case33bw` | single | `flattened` | objective-only peak / off-peak arbitrage profit with SOC target shaping; SOC violations in `info['cost']` | 48 | `gymnasium` |
| `marl_opf` | transmission / `Case5` | multi | `global` | shared economic dispatch reward; physical violations in `info['cost']` | 48 | `auto`, `rllib`, `pettingzoo` |
| `marl_der_arbitrage` | distribution / `Case33bw` | multi | `local_plus_forecast` | shared battery arbitrage reward; voltage / SOC violations in `info['cost']` | 48 | `auto`, `rllib`, `pettingzoo` |
| `marl_ev_v2g` | distribution / `Case33bw` | multi | `local_plus_forecast` | shared EV arbitrage and departure-readiness reward; grid / EV violations in `info['cost']` | 168 | `auto`, `rllib`, `pettingzoo` |
| `dc_scheduling` | distribution / `Case33bw` | single | `flattened` | objective-only single-agent energy-SLA-PUE reward; grid and datacenter thermal violations in `info['cost_sum']` | 48 | `gymnasium` |
| `dc_microgrid` | self-contained DC microgrid | single | `flattened` | scalarised `r_energy + w_cost·r_cost + w_carbon·r_carbon`; vector in `info['reward_vector']`; SLA / overtemp / power-deficit in `info['cost']` | 288 | `gymnasium` |
| `dc_microgrid_safe` | self-contained DC microgrid | single | `flattened` | same as `dc_microgrid` with CMDP `cost_threshold = 0.5` | 288 | `gymnasium` |
| `marl_uc` | transmission / `Case5` | multi | `global` | shared UC economic reward; physical violations in `info['cost']` | 48 | `auto`, `rllib`, `pettingzoo` |
| `opf_118` | transmission / `Case118` | multi | `global` | shared large-scale economic dispatch reward; physical violations in `info['cost']` | 48 | `auto`, `rllib`, `pettingzoo` |
| `opf_118_7d` | transmission / `Case118` | multi | `global` | shared large-scale economic dispatch reward; physical violations in `info['cost']` | 336 | `auto`, `rllib`, `pettingzoo` |

Internal atomic validation presets live under `powerzoo.tasks.atomic`, but they are intentionally not part of the public benchmark set.

---

## Simple Tasks

::: powerzoo.tasks.simple.MARLOPFTask
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - get_scenario_config
        - get_agents_config
        - create_env

**Key parameters**

| Parameter | Default | Description |
|---|---|---|
| `case` | `'Case5'` | Grid case name |
| `split` | `'train'` | Data split: `'train'`, `'val'`, or `'test'` |
| `action_mode` | `'score'` | `'score'` (softmax allocation) or `'direct'` (MW output) |
| `max_load_ratio` | `0.9` | Maximum load as fraction of total generation capacity |
| `max_steps` | `48` | Steps per episode (48 = 1 day at 30-min resolution) |

**Agent design**

- **Action**: score ∈ [0, 1] — softmax allocation of net load across generators
- **Observation**: global features (total load, line flows, time) + local features (unit index, p_min, p_max, cost coefficients)
- **Reward**: −(generation cost) / 1000 (shared, cooperative)

**Data splits** (non-overlapping, fixed for benchmark reproducibility)

| Split | Date range |
|---|---|
| train | 2023-07-05 – 2024-12-31 |
| val | 2025-01-01 – 2025-06-30 |
| test | 2025-07-01 – 2025-12-15 |

---

## Middle Tasks

::: powerzoo.tasks.middle.MARLUCTask
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - get_scenario_config
        - get_agents_config
        - create_env

Extends `MARLOPFTask` with **unit commitment** decisions. Each generator agent must decide both *how much* to generate and *whether* to be online.

**Key parameters**

| Parameter | Default | Description |
|---|---|---|
| `case` | `'Case5'` | Grid case name |
| `split` | `'train'` | Data split |
| `max_load_ratio` | `0.9` | Maximum load ratio |
| `max_steps` | `48` | Steps per episode |

**UC defaults** (used when not present in `case.units` columns)

| Column | Default | Unit |
|---|---|---|
| `startup_cost` | 500 | $/start |
| `shutdown_cost` | 200 | $/stop |
| `ramp_rate` | 999 | MW/step |
| `min_up_time` | 1 | steps |
| `min_down_time` | 1 | steps |

**Agent design**

- **Action**: `[score, on_off]` — 2-element vector; `on_off ≥ 0.5` commits the unit
- **Observation**: global + local + commitment vector (current on/off status of all units)
- **Reward**: −(generation cost + startup cost + shutdown cost) / 1000 (economic only)
- **Cost signal**: physical constraint violations → `info['cost']` (CMDP separation)

---

## Complex Tasks

::: powerzoo.tasks.complex.OPF118Task
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - get_scenario_config
        - get_agents_config
        - create_env

Large-scale cooperative OPF on the IEEE 118-bus system with 54 generators and 186 transmission lines.

**Key parameters**

| Parameter | Default | Description |
|---|---|---|
| `split` | `'train'` | Data split |
| `max_load_ratio` | `0.85` | Maximum load ratio (lower than 5-bus due to larger system) |
| `max_steps` | `48` | Steps per episode |
| `action_mode` | `'score'` | `'score'` or `'direct'` |

**Agent design**

- 54 cooperative agents (one per generator)
- Same score-based action / OPF observation protocol as `MARLOPFTask`
- Shared cooperative reward

### OPF118Task7Days

7-day (336-step) variant of `OPF118Task`. All parameters are inherited; `max_steps` defaults to `336`.

```python
env = make_task_env("opf_118_7d", split="train")
```
