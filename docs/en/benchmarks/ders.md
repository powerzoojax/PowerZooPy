# DERs — Voltage / DER coordination

The DERs suite covers **scalable safe MARL on a distribution feeder**: many controllable assets share one feeder, see only local signals, and have to keep voltages in band collectively.

> **Vocabulary check.** *DER* (Distributed Energy Resource) — battery, PV, EV or controllable load on a distribution network. *Dec-POMDP* (Decentralised, Partially Observable MDP) — the right MARL model for "agents see local state, share a global reward".

## Tasks in this suite

| Task name | Underlying env | Agents | Steps | Notes |
|---|---|---|---|---|
| `marl_der_arbitrage` | `Case33bw` | 3 batteries (buses 6 / 12 / 18) | 48 × 30 min | Battery-only DER fleet, simplest entry point. |
| `marl_ders_benchmark` | `Case118zh` | 12 heterogeneous DERs (4 Battery + 4 PV + 4 FlexLoad) | 48 × 30 min | Larger feeder, mixed resource types, heterogeneous-policy MARL. |
| `marl_ev_v2g` | `Case33bw` | 5 EVs with V2G / G2V | 168 × 60 min (1 week) | Adds availability mask + departure-SOC hard deadline. |

The three tasks together span a 3 → 12 agent-count axis and add the mixed-resource and EV-deadline complications independently.

## Why this suite

- The only suite with **all coupling mediated by voltage** (no congestion price, no shared LMP). Coupling is purely physics-driven.
- The only suite with a clean **scalability axis** (3 → 5 → 12 agents in one suite).
- The only suite with **heterogeneous agent types** (battery / PV / FlexLoad) that still share a global reward.
- The only suite with **hard deadline constraints** (EV departure SOC) layered on top of DER physics.

## Physical setup

| Task | Underlying env | Resources |
|---|---|---|
| `marl_der_arbitrage` | `Case33bw` (33-bus radial) | 3× `BatteryEnv` |
| `marl_ders_benchmark` | `Case118zh` (118-bus radial) | 4× `BatteryEnv`, 4× `SolarEnv` (or PV inverter), 4× `FlexLoad` |
| `marl_ev_v2g` | `Case33bw` | 5× `VehicleEnv` |

Voltage limits are `v_min = 0.94 pu`, `v_max = 1.06 pu` by default. The BFS solver is the single-phase distribution model documented in [Distribution physics](../physics/distribution.md).

## Agent design

| Task | Action per agent | Default observation |
|---|---|---|
| `marl_der_arbitrage` | `Box(1)` battery setpoint `[-P_max, P_max]` | `local_plus_forecast` (SOC + price + load forecast) |
| `marl_ders_benchmark` | `Box(2)` typed by resource role | `ders_local` (per-agent vector keyed to type) |
| `marl_ev_v2g` | `Box(1)` (masked to 0 when EV is away) | `local_plus_forecast` (SOC + departure-readiness + forecast) |

All three reward functions are **shared and cooperative**: each agent receives the team's collective economic value as reward.

- `marl_der_arbitrage`: arbitrage profit on local LMPs.
- `marl_ders_benchmark`: weighted sum of arbitrage profit + curtailment cost.
- `marl_ev_v2g`: arbitrage profit + departure-readiness bonus.

Cost components use the benchmark CMDP vector contract from [Reward and cost split](../concepts/reward-cost-split.md):

- `cost_voltage_violation` (all three).
- `cost_clipped_power` (battery / EV SOC clipping).
- EV-specific: departure-SOC violation and home-availability violation (reported through the task adapter).

For the JAX-aligned flagship DERs benchmark, `marl_ders_benchmark` freezes:

- `selected_constraint_names = ("voltage_violation", "thermal_overload", "resource")`.
- `cost_thresholds = (0.25, 0.125, 0.125)`.
- `fallback_weights = (4.0, 1.0, 1.0)`.

Current Python MARL training consumes these costs through the standardized MDP fallback reward when needed; it is not advertised as a first-party constrained MARL optimizer.

## Variants and difficulty axis

The main difficulty axes are observation mode and resource count:

```python
from powerzoo.tasks import make_task_env
small  = make_task_env("marl_der_arbitrage")
medium = make_task_env("marl_ders_benchmark")
large_horizon = make_task_env("marl_ev_v2g")  # 168 steps
```

For `marl_der_arbitrage`, switching `obs_mode="local"` (no forecast) makes the same physics significantly harder; `obs_mode="local_plus_voltage"` adds a feeder voltage summary.

## Splits

All three DER tasks use the standard splits:

| Split | Date range |
|---|---|
| `train` | 2023-07-05 – 2024-12-31 |
| `val` | 2025-01-01 – 2025-06-30 |
| `test` | 2025-07-01 – 2025-12-15 |

## Baselines

- **Random** — `RandomPolicy(env.action_space)`. Often near-feasible (voltage limits are only mildly binding under random battery setpoints) but yields zero arbitrage profit.
- **Rule-based** — `RuleBasedPolicy` (peak-valley arbitrage on a fixed time-of-day rule).
- **Oracle (where applicable)** — for `marl_der_arbitrage`, a perfect-foresight LP gives an upper bound on arbitrage profit.

## Metrics to report

- Mean episode reward (per agent and team).
- `voltage_violation_rate` and `max_voltage_deviation` — the primary safety metrics.
- `normalized_score` against the random / oracle baselines.
- For `marl_ev_v2g`: `departure_readiness_rate` (fraction of EVs that hit the departure SOC).
- `decentralization_gap` if you also run a centralised policy: `perf(centralised) − perf(MARL)`.

## Code recipe

```python
from powerzoo.rl import Trainer

t = Trainer("marl_der_arbitrage", framework="pettingzoo", algorithm="SAC")
t.train_il(total_timesteps=500_000)
results = t.evaluate(split="test")
```

For the flagship DERs benchmark with the standardized fallback contract:

```python
from powerzoo.rl import make_env
env = make_env("marl_ders_benchmark", framework="pettingzoo")
print(env.unwrapped.task.constraint_spec())
```

## See also

- [Distribution physics](../physics/distribution.md) — the BFS solver that creates the voltage coupling.
- [Resources](../physics/resources.md) — `BatteryEnv`, `VehicleEnv`, `SolarEnv`, `FlexLoad` parameter tables.
- [Python contract · Observation modes](../concepts/python-contract.md#5-observation-modes) — `local`, `local_plus_forecast`, `local_plus_voltage`, `ders_local`.
- [Training · Wrappers](../training/wrappers.md) — `MARLWrapper`, `TaskPettingZooWrapper`, `SafeRLWrapper`.
