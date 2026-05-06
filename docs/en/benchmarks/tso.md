# TSO — Security dispatch

The TSO suite covers transmission-system-operator tasks: economic dispatch (ED) and unit commitment (UC) on the IEEE `Case5` and `Case118` networks. The unifying RL question is **safe RL with mixed discrete-continuous actions** — the agent must minimise generation cost subject to thermal limits and (in UC) min-up / min-down / ramp constraints.

> **Vocabulary check.** *TSO* (Transmission System Operator) — runs the bulk power network. *ED / OPF* (Economic Dispatch / Optimal Power Flow) — pick generator outputs to minimise cost subject to limits. *UC / SCUC* (Unit Commitment / Security-Constrained UC) — same plus on / off decisions, startup costs, ramp limits, min up / down time.

## Tasks in this suite

| Task name | Underlying env | Agents | Steps | Action | Notes |
|---|---|---|---|---|---|
| `marl_uc` | `Case5` | 5 generators | 48 × 30 min | `[score, on_off]` per agent | Smallest UC benchmark. |
| `opf_118` | `Case118` | 54 generators | 48 × 30 min | `[score]` per agent | Large-scale ED. |
| `opf_118_7d` | `Case118` | 54 generators | 336 × 30 min | `[score]` per agent | 7-day variant of `opf_118`. |

`marl_opf` (5-agent ED on `Case5`) is the smaller starter variant; it sits outside the main benchmark set but uses the same adapter (`TaskOPFMultiAgentEnv`).

## Why this suite

- The only suite where a single agent (or small team) must satisfy a **dense, simultaneous-binding constraint set** (thermal + reserve + UC constraints).
- The only suite with a clean **mixed discrete-continuous action space** in `marl_uc` (`on_off ∈ {0, 1}` × `score ∈ [0, 1]`).
- A standard Safe-RL test case: the constraint set is tight enough that random policies are infeasible on most steps of `opf_118`.

## Physical setup

- **Underlying env.** Transmission grid (`TransGridEnv`), DC physics by default. Switch to AC by passing `physics='ac'`.
- **Resources.** Pure generation; no batteries / EVs / DR by default. The agent's job is dispatch, not arbitrage.
- **Constraints.** Line thermal limits (DC: linear; AC: nonlinear). For `marl_uc`, additionally min-up / min-down / ramp / startup-cost constraints from the `units` table.

## Agent design

| Task | Action per agent | Observation per agent |
|---|---|---|
| `marl_uc` | `Box(2)` `[score, on_off]` (`on_off ≥ 0.5` commits the unit) | global summary + local unit params + commitment vector + time |
| `opf_118` / `opf_118_7d` | `Box(1)` `[score]` (softmax-allocated share of net load) | global features (total load, line flows, time) + local generator params (`p_min`, `p_max`, cost coefficients) |

Reward is **shared and cooperative** — the negative of total generation cost (plus startup / shutdown for UC):

\[
r_t \;=\; -\frac{1}{1000}\,\sum_i \bigl[ C_i(P_{g,i,t}) \cdot u_{i,t} + S_i^{\text{up}} z_{i,t} + S_i^{\text{dn}} w_{i,t} \bigr]
\]

Per [Reward and cost split](../concepts/reward-cost-split.md), core envs and task wrappers expose explicit vector costs instead of treating `info["cost"]` as the normative channel:

- `cost_thermal_overload` — sum of `max(|F_k| - F_k^max, 0)` over lines (MW).
- `cost_voltage_violation` — for AC mode (pu).

The authoritative Python CMDP benchmark surface for the JAX-aligned TSO task is `comparison_tso_centralized`, because it exposes both selected constraints:

- `selected_constraint_names = ("thermal_overload", "reserve_shortfall")`.
- `cost_thresholds = (0.0, 5.0)`.
- `fallback_weights = (1.0, 1.0)`.

Legacy MARL OPF/UC tasks remain runnable and expose exact-name per-agent cost dictionaries, but they are not claimed as true multi-constraint optimizer training in this round.

## Variants and difficulty axis

`marl_uc` exposes a built-in difficulty ladder via `obs_mode`:

```python
from powerzoo.tasks import make_task_env
easy   = make_task_env("marl_uc", split="train", obs_mode="global")
medium = make_task_env("marl_uc", split="train", obs_mode="local_plus_forecast")
hard   = make_task_env("marl_uc", split="train", obs_mode="local")
```

The 7-day variant of `opf_118` extends `max_steps` from 48 to 336. The longer horizon significantly increases credit-assignment difficulty, because forecast errors and renewable variability accumulate over the week.

## Splits

All three TSO tasks use the standard GB demand splits:

| Split | Date range | Purpose |
|---|---|---|
| `train` | 2023-07-05 – 2024-12-31 | Algorithm training |
| `val` | 2025-01-01 – 2025-06-30 | Hyperparameter tuning |
| `test` | 2025-07-01 – 2025-12-15 | Official benchmark evaluation |

## Baselines

- **Random** — `RandomPolicy(env.action_space)`. Defines `normalized_score = 0`.
- **Rule-based** — `RuleBasedPolicy` (proportional dispatch + always-on commitment). A reasonable starting heuristic.
- **Oracle (DC-OPF)** — `OraclePolicy` solves the same DCOPF that the env runs internally. Defines `normalized_score = 1`.

## Metrics to report

- Mean episode reward (= negative cost).
- `normalized_score` against `OraclePolicy`.
- Mean episode cost by constraint from `info["constraint_costs"]` / `selected_constraint_costs`.
- Scalar mean episode cost remains available as a backward-compatible aggregate alias.

## Code recipe

```python
from powerzoo.rl import Trainer

t = Trainer("opf_118", framework="pettingzoo", algorithm="SAC", total_timesteps=2_000_000)
t.train_il()
results = t.evaluate(split="test")
print(results)
```

For the JAX-aligned centralized CMDP benchmark:

```python
from powerzoo.tasks import make_task_env
env = make_task_env("comparison_tso_centralized")
print(env.selected_constraint_names, env.cost_thresholds)
```

## See also

- [Transmission physics](../physics/transmission.md) — DC / AC OPF / PF details.
- [Python contract](../concepts/python-contract.md) — observation modes (TSO defaults to `global`).
- [Training · Trainers](../training/trainers.md), [Training · Wrappers](../training/wrappers.md).
- [API · Tasks](../api/tasks.md) — class signatures.
