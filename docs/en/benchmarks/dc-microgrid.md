# DC microgrid

`dc_microgrid` (and `dc_microgrid_safe`) is the **multi-objective robust RL** benchmark in PowerZoo. Both tasks share the same env, `DCMicrogridEnv`: a self-contained behind-the-meter microgrid (no external grid) composed of `DataCenterEnv` + `BatteryEnv` + inline solar PV + diesel generator. The safe task freezes the same vector CMDP contract and exposes a scalar compatibility budget for libraries that still expect one cost.

The physical model — power balance, action / observation / reward vector, cost components — is documented in [Physics · Microgrid](../physics/microgrid.md). This page focuses on the agent-facing benchmark setup: variants, splits, baselines and metrics.

> **Vocabulary check.** *PUE* (Power Usage Effectiveness) — total facility power / IT equipment power, lower is better. *SLA* (Service Level Agreement) — workload completion guarantee; missing a deadline is an SLA violation. *Pareto front* — set of non-dominated `(reward1, reward2, …)` tuples; multi-objective RL approximates it.

## Why this suite

- Only suite with **no grid backstop**: power balance is a hard internal constraint, and an over-committed IT load creates a real `cost_power_deficit`.
- Only suite with a **vector reward** exposed in `info["reward_vector"] = [r_energy, r_cost, r_carbon]`; multi-objective and Pareto methods can use it directly.
- Only suite with **5-min resolution and a 24-hour horizon** (288 steps), exercising long-horizon credit assignment under thermal inertia.
- Only suite that combines **AI workload modelling** (training + finetuning + inference queues with EDF scheduling) with energy management.

## The two registered tasks

| Task | CMDP thresholds | When to use |
|---|---|---|
| `dc_microgrid` | none | Standard reward-only training; use the reward vector for multi-objective work. |
| `dc_microgrid_safe` | `(0.2, 0.15, 0.15)` for `("sla", "overtemp", "power_deficit")`; scalar alias `0.5` | Safe-RL / fallback benchmarking against explicit per-constraint budgets. |

Both are single-agent Gymnasium envs.

## Physical setup

A summary (full details in [Physics · Microgrid](../physics/microgrid.md)):

- **Underlying env.** `DCMicrogridEnv` — self-contained, no `GridEnv` underneath.
- **Resources.** `DataCenterEnv` (1000 GPUs by default), `BatteryEnv` (2 MWh / 0.5 MW), inline PV (0.4 MW), inline diesel generator (0.6 MW).
- **Episode.** 288 steps × 5 min = 24 hours.

## Agent design

| Item | Value |
|---|---|
| Action | `Box(5)` `[r_train, r_finetune, T_cool_norm, P_batt, P_dg]` |
| Observation | `Box(18)` (utilisation, queues, thermal state, generation availability, last action, time) |
| Reward | scalarised `r_energy + w_cost · r_cost + w_carbon · r_carbon`; vector also in `info["reward_vector"]` |
| Core constraints | `constraint_names = ("sla", "overtemp", "power_deficit")`; `info["constraint_costs"]` follows this order. |
| Task constraints | `selected_constraint_costs` uses the full vector with thresholds `(0.2, 0.15, 0.15)` and fallback weights `(1.0, 1.0, 1.0)`. |
| Scalar compatibility | `info["cost"]` is emitted by compatibility wrappers only; `cost_sum` remains an aggregate diagnostic alias. |

## Variants

`dc_microgrid` exposes a single env config; the main variants are:

- **Reward weighting.** Tune `w_cost` and `w_carbon` in the env constructor to walk along the Pareto front.
- **Workload OOD.** Inject Azure or Alibaba DC traces in place of Google's via `set_profiles(...)` to test workload distribution shift.
- **Solar drought.** Multiply the solar profile by a small constant for several days to test the high-carbon regime.
- **Cooling stress.** Add several °C to the outdoor temperature to force the agent into the high-COP regime.

## Splits

All three Google DC traces are pooled as the data source, with rolling windows. The convention is:

| Split | Trace window | Purpose |
|---|---|---|
| `train` | rotating 30-day windows from Google DC 2019-05+ | training |
| `val` | distinct 7-day windows | hyperparameter tuning |
| `test` | distinct 7-day windows | reporting |

OOD splits (workload swap to Azure / Alibaba, solar drought, cooling stress) are configured per experiment via `set_profiles(...)` and the data pipeline's OOD transforms in `powerzoo.data.dc_microgrid_profiles`.

## Code recipe

```python
from powerzoo.tasks import make_task_env

env = make_task_env("dc_microgrid_safe", split="train")
obs, info = env.reset(seed=0)

terminated = truncated = False
while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(info["reward_vector"], info["selected_constraint_costs"])
```

For a scalar Safe-RL-library projection, keep the Gym 5-tuple and let the wrapper write compatibility `info["cost"]`:

```python
from powerzoo.rl import Trainer

t = Trainer("dc_microgrid_safe", algorithm="SAC", total_timesteps=2_000_000,
            safe_rl=True, cost_threshold=0.5)
t.train()
results = t.evaluate(split="test")
```

For multi-objective training, use a custom loop that reads `info["reward_vector"]` (see [Training · Custom loops](../training/custom-loops.md)).

## Baselines

- **Random** — `RandomPolicy`. Usually infeasible (large `cost_power_deficit`).
- **EDF + fixed cooling** — earliest-deadline-first scheduler with a fixed thermostat. The "rule-based" baseline.
- **Solar-aware heuristic** — schedule training when solar is high, conservative otherwise.
- **Perfect-foresight LP / MILP planner** — offline upper bound on the scalar reward.

## Metrics to report

Multi-objective tasks need the full vector. Primary metrics:

- `total_energy_mwh` — facility energy.
- `total_operating_cost_$` — fuel + battery wear.
- `total_carbon_kg` — diesel emissions.
- Per-constraint episode costs for `sla`, `overtemp`, and `power_deficit`.
- `sla_violation_rate`, `thermal_safety_rate`, `power_balance_deficit_mwh` — the three operational constraint axes.
- `renewable_utilization_pct`, `battery_cycling_count`, `diesel_runtime_hours` — diagnostics.
- `pareto_hypervolume` — for multi-objective methods.
- `robustness_gap` — `perf(IID) - perf(workload_swap)`.

## See also

- [Physics · Microgrid](../physics/microgrid.md) — the env-level model and equations.
- [Resources · DataCenterEnv](../physics/resources.md#datacenterenv-ai-data-center-as-a-controllable-load).
- [Architecture · Data pipeline](../architecture/data-pipeline.md) — `dc_microgrid_profiles` loader.
- [Training · Custom loops](../training/custom-loops.md) — multi-objective loop with `info["reward_vector"]`.
