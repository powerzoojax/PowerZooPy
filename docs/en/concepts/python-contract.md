# Python API Contract

PowerZoo follows a small set of stable conventions on top of Gymnasium / PettingZoo / RLlib. This page is the **authoritative description** of those conventions; other pages assume them.

The contract has four layers:

1. The base env interface (`BaseEnv`).
2. Single-agent task envs (Gymnasium 5-tuple).
3. Multi-agent task envs (PettingZoo Parallel or RLlib `MultiAgentEnv`).
4. The framework / observation / cost contract that every task obeys.

The reward / cost split is central enough to have its own page: [Reward and cost split](reward-cost-split.md).

## 1. `BaseEnv` — the abstract parent

`BaseEnv` (`powerzoo/envs/base.py`) inherits from `gymnasium.Env` and adds two PowerZoo-specific attributes:

| Attribute / method | Purpose |
|---|---|
| `time_step` | Step counter inside the current episode. |
| `delta_t_minutes` | Step length in minutes (must divide 1440). Default 30. |
| `action_space` / `observation_space` | Filled in by subclasses. |
| `reset(seed, options)` | Resets `time_step` and returns `(state, info)` (subclass-specific). |
| `step(action)` | Subclass-specific. Returns Gymnasium-style 5-tuples at the task layer. |
| `obs()` / `reward()` / `cost()` | Hooks; `cost()` defaults to 0 (CMDP-friendly). |

Subclasses do not store mutable state in arbitrary attributes. `GridEnv` and `ResourceEnv` keep their state in well-defined fields (case data, `current_p_mw`, `soc`, …) so that resets are reproducible.

`BaseEnv` is used directly by `GridEnv`, `ResourceEnv`, `PowerEnv`, `MarketEnv` and `DCMicrogridEnv`. The first three are described in [Architecture · Environment stack](../architecture/env-stack.md); the last two have dedicated pages under [Physics](../physics/markets.md) and [Physics · Microgrid](../physics/microgrid.md).

## 2. Single-agent task envs (Gymnasium 5-tuple)

Single-agent tasks (`battery_arbitrage`, `dc_scheduling`, `dc_microgrid`, `dc_microgrid_safe`) return a standard `gymnasium.Env`. Use the usual loop:

```python
from powerzoo.tasks import make_task_env

env = make_task_env("battery_arbitrage", split="train")
obs, info = env.reset(seed=0)
terminated = truncated = False
while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

The flat observation is produced by stacking `FlattenWrapper` on top of `PowerEnv` (which itself binds a `GridEnv` and one or more `ResourceEnv` instances). `info` carries the full physical breakdown described in §4.

## 3. Multi-agent task envs

Multi-agent tasks have two compatible interfaces. Both share identical reward, cost and observation semantics — they only differ in what `step()` returns and how `done` is signalled.

| `framework=` | Returned env | When to use |
|---|---|---|
| `'auto'` (default) | Specialised task adapter (RLlib-compatible when `ray` is installed) | Default; works without RLlib too. The episode ends when `terminateds.get("__all__")` or `truncateds.get("__all__")` is true. |
| `'pettingzoo'` | Task-aware PettingZoo Parallel API wrapper around the same adapter | Use the `while env.agents:` idiom; the wrapper clears `env.agents` when the episode ends. |
| `'rllib'` | Same as `'auto'`, but raises if `ray[rllib]` is missing | Make the dependency explicit in production runs. |

PettingZoo path:

```python
env = make_task_env("marl_opf", framework="pettingzoo")
obs, info = env.reset(seed=42)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
```

Auto / RLlib path:

```python
env = make_task_env("marl_opf", framework="auto")
obs, infos = env.reset(seed=42)
terminated = truncated = False
while not (terminated or truncated):
    actions = {a: env.action_space[a].sample() for a in env.possible_agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    terminated = bool(terms.get("__all__", False))
    truncated  = bool(truncs.get("__all__", False))
```

The PettingZoo bridge that powers `framework='pettingzoo'` lives in `powerzoo/tasks/interfaces/pettingzoo.py` and is re-exported as `powerzoo.wrappers.TaskPettingZooWrapper` for backward compatibility. It must preserve every aspect of the underlying adapter's contract.

## 4. The `info` dict

Every grid env populates `info` with at least the following fields. Tasks and `PowerEnv` may add more, but never less.

| Key | Type | Meaning |
|---|---|---|
| `is_safe` | bool | All physical limits satisfied this step. |
| `pf_converged` | bool | Power-flow solver converged. |
| `cost_exception` | bool | An exception was raised inside the PF solve. |
| `cost_thermal_overload` | float (MW) | Sum of line-flow over-limits. |
| `cost_voltage_violation` | float (pu) | Sum of bus-voltage out-of-band magnitudes. |
| `cost_sum` | float | Total physical violation cost (sum of `cost_*` from grid + resources). |
| `p_slack_MW` / `q_slack_MVAr` | float | Slack-bus active / reactive injection (distribution: feeder-head exchange). |
| `is_diverged` | bool | BFS hit `max_iter` before reaching tolerance. |
| `voltage_collapse` | bool | Severe unclamped low-voltage detected (treated as PF failure). |

`PowerEnv` then aggregates resource cost contributions into the same dict — the full data flow is in [Reward and cost split](reward-cost-split.md).

## 5. Observation modes

PowerZoo defines **five canonical observation modes**. The list lives in code at `powerzoo.tasks.observation`:

```python
from powerzoo.tasks.observation import OBSERVATION_MODES
print(OBSERVATION_MODES)
# ('global', 'local', 'local_plus_forecast', 'local_plus_voltage', 'ders_local')
```

Each mode is a tuple of feature names that the task adapter materialises into a per-agent `Box` observation. Tasks declare which modes they support via `make_observation_config(...)`; you can inspect the actual field order via `get_observation_fields()` on the adapter.

| Mode | What the agent sees | Used as |
|---|---|---|
| `global` | Shared grid summary (total load, normalized line flows, time features) plus the agent's immutable parameters. | *Easiest* setting; closest to centralised training (CTDE). |
| `local` | Only the agent's own state and adjacent grid signals. No system-wide summary. | *Hardest* default setting; typically only solvable with learned communication. |
| `local_plus_forecast` | `local` plus the task's declared forecast window (load and / or price and / or availability). | *Medium* setting; lets you study how forecast quality affects performance. |
| `local_plus_voltage` | `local` plus a per-feeder or per-zone voltage summary. | Distribution-side tasks where voltage is the binding safety signal but you do not want to give away the global state. |
| `ders_local` | Compact `local` layout for heterogeneous DER fleets, in a uniform per-agent vector typed by resource role. | `marl_ders_benchmark` and similar mixed-resource tasks. |

The default mode for each public task lives on the task class itself; `get_public_task_info(name)['default_observation_mode']` reports the current value.

| Task | Default mode | Other modes typically supported |
|---|---|---|
| `marl_opf`, `marl_uc`, `opf_118`, `opf_118_7d` | `global` | `local`, `local_plus_forecast` |
| `marl_der_arbitrage` | `local_plus_forecast` | `local`, `local_plus_voltage` |
| `marl_ders_benchmark` | `ders_local` | `local`, `local_plus_voltage` |
| `marl_ev_v2g` | `local_plus_forecast` | `local` |
| `battery_arbitrage`, `dc_scheduling`, `dc_microgrid*` | `flattened` (single-agent) | n/a |

You can convert one task into a difficulty ladder simply by switching modes:

```python
from powerzoo.tasks import make_task_env

easy   = make_task_env("marl_opf", split="train", obs_mode="global")
medium = make_task_env("marl_opf", split="train", obs_mode="local_plus_forecast")
hard   = make_task_env("marl_opf", split="train", obs_mode="local")
```

## 6. The `envs / tasks / wrappers` boundary

PowerZoo keeps three packages strictly separated, each owning a single concern:

```mermaid
flowchart LR
    subgraph envs ["envs/ — physical simulation"]
      E1[GridEnv / ResourceEnv / PowerEnv]
      E2[time progression, PF, cost_*]
    end
    subgraph tasks ["tasks/ — benchmark presets + adapters"]
      T1[Task definitions]
      T2[Adapters\n(OPF / UC / Resource / EV)]
      T3[Public benchmark set + registry]
    end
    subgraph wrappers ["wrappers/ — generic API adaptation"]
      W1[Gymnasium / Flatten / Normalize]
      W2[SafeRL / Forecast / MARL]
    end
    envs --> tasks
    tasks --> wrappers
```

The arrows are one-directional. `envs/` must work standalone (no `tasks/` import). `wrappers/` must not patch task-specific bugs — those go in `tasks/` or `envs/`. Adapters in `powerzoo/tasks/adapters/` parse task-level actions, call the underlying `PowerEnv`, package per-agent `info`, `cost` and `costs`, and expose `get_observation_fields()`. They do not re-implement physics.

## 7. Public benchmark set

The explicit, stable public benchmark set is `powerzoo.tasks.public.PUBLIC_TASKS`:

```python
from powerzoo.tasks import PUBLIC_TASKS, list_public_tasks, get_public_task_catalog

print(PUBLIC_TASKS)
print(list_public_tasks())
catalog = get_public_task_catalog()
print(catalog[0]['task_id'], catalog[0]['default_episode_horizon_steps'])
```

To stay in `PUBLIC_TASKS`, a task must:

1. Be registered in `powerzoo.tasks.registry`.
2. Be documented and instantiable via `make_task_env(name, split=...)`.
3. Be smoke-tested on at least one episode for each of `train` / `val` / `test`.
4. Be consistent with the contract on this page.

Registered-but-incomplete tasks (`joint_trans_dist*`, atomic validation presets, …) remain accessible through `list_tasks()` and `make_task_env(...)` but are **not** part of the public benchmark set.
