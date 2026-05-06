# Wrappers

Wrappers in `powerzoo/wrappers/` adapt the contract from [Python contract](../concepts/python-contract.md) to the API that a specific RL algorithm expects. They never change physics or task semantics; they only translate API shapes.

This page is the practical reference for what each wrapper does. Per-class signatures live in [API · Wrappers](../api/wrappers.md).

## What every wrapper preserves

All PowerZoo wrappers respect the contract documented in [Python contract](../concepts/python-contract.md):

- The reward channel still carries only the economic objective.
- Core CMDP envs expose fixed-order vector costs through `env.constraint_names()` and
  `info['constraint_costs']`; scalar `info['cost']` exists only in compatibility wrappers.
- The 5 observation modes (`global` / `local` / `local_plus_forecast` / `local_plus_voltage` / `ders_local`) are not modified; only the **container** changes (Dict → flat Box, single-agent → PettingZoo Parallel, …).

You can stack wrappers on each other; the canonical orderings are documented below.

## `GymnasiumWrapper` — adapt a raw `GridEnv`

Adapts a raw `GridEnv` (state-dict return) to the standard Gymnasium 5-tuple:

```python
from powerzoo.envs.grid.trans import TransGridEnv
from powerzoo.wrappers import GymnasiumWrapper

env = GymnasiumWrapper(TransGridEnv())
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Use this when you want to drive a `GridEnv` directly (no task wrapping); for benchmark experiments, `make_task_env(...)` already returns a Gymnasium-style env.

## `NormalizationWrapper` — running statistics

Normalises observations (and optionally actions) to `[-1, 1]` using running statistics. Stack on top of `GymnasiumWrapper`:

```python
from powerzoo.wrappers import GymnasiumWrapper, NormalizationWrapper

env = NormalizationWrapper(GymnasiumWrapper(TransGridEnv()))
```

For task envs, `powerzoo.rl.make_env(name, normalize=True)` does the same thing.

## `ForecastWrapper` — append a load-forecast window

Appends a `horizon`-length demand forecast to every observation. Extends `observation_space` automatically (base dim + `horizon`).

| Parameter | Default | Description |
|---|---|---|
| `horizon` | `6` | Number of future steps to append. |
| `mode` | `'perfect'` | `'perfect'` (ground truth), `'noisy'` (Gaussian noise), or `'none'` (zeros). |
| `noise_std` | `0.02` | Fractional noise std for `mode='noisy'` (e.g. 0.02 = 2 %). |
| `normalize` | `True` | Divide forecast values by dataset maximum. |

```python
from powerzoo.wrappers import GymnasiumWrapper, ForecastWrapper

env = ForecastWrapper(GymnasiumWrapper(TransGridEnv()), horizon=6, mode='noisy')
obs, info = env.reset(seed=0)
# obs[-6:] contains the next 6 half-hour demand values
```

The three forecast modes let you measure forecast-quality sensitivity on the same task, without changing the underlying physics.

## `TaskCMDPWrapper` — attach task-selected CMDP metadata

Keeps the standard Gymnasium 5-tuple, but annotates `info` with:

- `constraint_names` / `constraint_costs` — the core env's full vector
- `selected_constraint_names` / `selected_constraint_costs` — the benchmark task's chosen subset
- `selected_cost_sum` — scalar compatibility alias for the selected subset

This is how DSO, TSO and DC microgrid align a benchmark CMDP spec without mutating the core env semantics.

## `CMDPWrapper` — 6-tuple with vector costs

Returns `(obs, reward, costs, terminated, truncated, info)`, where `costs` is the task-selected vector from `info['selected_constraint_costs']` (or the full vector when no task spec is present).

## `SafeRLWrapper` — 6-tuple for OmniSafe / Safety-Gymnasium

Returns a **6-tuple** `(obs, reward, cost, terminated, truncated, info)`. Cost extraction priority:

1. `info['selected_cost_sum']` — task-selected scalar projection of the CMDP vector.
2. `info['cost_sum']` — full-vector sum (legacy scalar alias).
3. `info['cost']` — compatibility-only fallback.

```python
from powerzoo.wrappers import GymnasiumWrapper, SafeRLWrapper, TaskCMDPWrapper

env = TaskCMDPWrapper(GymnasiumWrapper(TransGridEnv()), constraint_spec=...)
env = SafeRLWrapper(env, cost_threshold=25.0)
obs, info = env.reset(seed=0)
obs, reward, cost, terminated, truncated, info = env.step(env.action_space.sample())
```

| Parameter | Default | Description |
|---|---|---|
| `cost_threshold` | `25.0` | Scalar threshold exposed as `env.cost_threshold` for OmniSafe. |

## `GymnasiumSafeWrapper` — 5-tuple, scalar compatibility in `info`

Returns the standard Gymnasium **5-tuple** but injects the selected scalar projection into `info['cost']`. Use this when your algorithm reads cost from `info` rather than as a separate return value.

```python
from powerzoo.wrappers import GymnasiumWrapper, GymnasiumSafeWrapper

env = GymnasiumSafeWrapper(GymnasiumWrapper(TransGridEnv()))
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(info['cost'])  # non-negative scalar
```

## `MARLWrapper` — single-agent → PettingZoo Parallel

Converts a single-agent PowerZoo env to the **PettingZoo Parallel API**. Each resource registered in the underlying `GridEnv` becomes an independent agent:

```python
from powerzoo.wrappers import MARLWrapper

env = MARLWrapper(TransGridEnv(), agent_type='generators')
obs, info = env.reset(seed=0)
actions = {a: env.action_space(a).sample() for a in env.agents}
obs, rewards, terminations, truncations, info = env.step(actions)
```

For task envs, the **task-aware** `TaskPettingZooWrapper` (defined in `powerzoo/tasks/interfaces/pettingzoo.py`, re-exported from `powerzoo.wrappers` for compatibility) is what `make_task_env(..., framework='pettingzoo')` uses under the hood. It preserves the reward / cost / observation / `__all__` semantics of the underlying task adapter.

## `FlattenWrapper` — Dict → flat Box

Flattens dict / nested `observation_space` and `action_space` into 1-D `Box` spaces. Useful for algorithms that require flat vector inputs:

```python
from powerzoo.wrappers import GymnasiumWrapper, FlattenWrapper

env = FlattenWrapper(GymnasiumWrapper(TransGridEnv()))
```

`PowerEnv` returns Dict observations and accepts Dict actions; `FlattenWrapper` is what makes the single-agent `battery_arbitrage`, `dc_scheduling` and `dc_microgrid*` tasks return a flat `Box`.

## Standard stacks

The most common stacking patterns:

```python
from powerzoo.envs.grid.trans import TransGridEnv
from powerzoo.wrappers import (
    GymnasiumWrapper, NormalizationWrapper, ForecastWrapper, SafeRLWrapper,
)

# Single-agent vanilla
env = GymnasiumWrapper(TransGridEnv())

# Single-agent with normalisation and forecast
env = ForecastWrapper(NormalizationWrapper(GymnasiumWrapper(TransGridEnv())))

# Single-agent Safe RL
env = SafeRLWrapper(NormalizationWrapper(GymnasiumWrapper(TransGridEnv())),
                    cost_threshold=25.0)
```

For task envs (the recommended path), use `powerzoo.rl.make_env(...)` and pass `normalize=True`, `forecast_horizon=N`, `safe_rl=True`, `cost_threshold=...`; it builds the appropriate stack for you.

## See also

- [Python contract](../concepts/python-contract.md) — what every wrapper must preserve.
- [Trainers](trainers.md) — `Trainer` and `make_env` build the wrapper stack automatically.
- [API · Wrappers](../api/wrappers.md) — class signatures.
