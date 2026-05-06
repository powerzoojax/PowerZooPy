# GenCos — Market bidding

`gencos_bidding` is the **competitive MARL** benchmark in PowerZoo. Five generator agents on a 5-bus transmission network each submit a 3-segment monotone offer curve every step; the market clears via a network-constrained SCED, and each agent receives its **private** dispatch profit as reward.

The env-level mechanics are documented in [Physics · Markets](../physics/markets.md). This page covers the agent-facing benchmark setup.

> **Vocabulary check.** *SCED* (Security-Constrained Economic Dispatch) — OPF using submitted offers as the LP objective, subject to thermal limits. *LMP* (Locational Marginal Price) — dual variable of the nodal balance constraint; differs across buses when lines are congested. *Markup* — fractional offer above true marginal cost; the agent's strategic lever. *Nash equilibrium* — joint-policy profile in which no agent can improve by unilaterally changing its strategy.

## Why this suite

- The **only competitive** suite in PowerZoo (the other four are cooperative or single-agent).
- The **only suite with private rewards** — each agent's dispatch profit depends on its own offer curve, the market clearing, and the network LMP at its bus.
- The **only suite where reward depends on the offer**, not on true cost — strategic bidding is the central challenge.
- The **only suite with explicit ramp coupling between successive market intervals** — dispatch at step `t` constrains the feasible range at step `t+1`.

Together these make `gencos_bidding` a useful testbed for self-play, population-based training (PSRO-lite), independent learners and game-theoretic MARL methods.

## Physical setup

| Aspect | Value |
|---|---|
| Underlying env | `Case5` (5-bus IEEE) wrapped by `GenCosMARLEnv`. |
| Agents | 5 generators (`genco_0` … `genco_4`). |
| Episode | 48 steps × 30 min = 1 day. |
| Clearing | `solve_piecewise_ed_opf` (network-constrained SCED on 3-segment offers). |
| Coupling between steps | Ramp limits `[p_min_rt, p_max_rt]` derived from the previous dispatch. |

`Case5` was chosen because line 4 → 5 is structurally congested, which creates persistent LMP spread between bus 5 (cheap generator G5) and buses 2 / 3 / 4 (load centres). Locked-in generators on the load side gain real local market power, so pure price-takers cannot win.

## Agent design

| Item | Value |
|---|---|
| Action per agent | `Box(3) ∈ [-1, 1]` markup scalars; sorted to enforce a monotone 3-segment offer curve. |
| Observation per agent | 12-D private vector — own cost + capacity + last dispatch + last profit + ramp headroom + 4-step LMP history + demand forecast + time. |
| Reward per agent | `LMP[node_i] · P_i · Δt - TC_i(P_i) · Δt` (private profit). |
| Constraint cost | `constraint_names = ("thermal_overload",)`; per-agent `info["costs"]["thermal_overload"]` and fixed-order `constraint_costs` expose the same channel. |

The action mapping is:

```
offer_curve[k] = true_mc × (1 + sorted_action[k] × max_markup)   for k in 0..2
```

with `max_markup = 2.0` (so offers can be up to 3× true marginal cost).

## Variants

- **`gencos_bidding`** (main task, 5-agent partial-info competition).
- A 2-agent variant — `G1` and `G5` strategic, others priced at true cost — is available by passing a custom `case` plus a strategic-agent mask to `make_gencos_env(...)`. This variant is useful for sanity-testing self-play convergence.

## Splits

The benchmark uses the standard GB demand splits to drive per-bus loads:

| Split | Date range |
|---|---|
| `train` | 2023-07-05 – 2024-12-31 |
| `val` | 2025-01-01 – 2025-06-30 |
| `test` | 2025-07-01 – 2025-12-15 |

OOD axes that the env supports natively:

- **Demand shift** — `load × constant` to push the system into deeper congestion.
- **Renewable shock** — multiply solar / wind capacity factors to change the net-load shape.
- **Ramp stress** — tighten ramp limits at construction time to make sequential coupling more binding.
- **Opponent shift** — train with one set of opponent seeds, evaluate against another.

## Code recipe

```python
from powerzoo.envs.market import make_gencos_env

env = make_gencos_env()
obs, info = env.reset(seed=0)
while env.agents:
    actions = {ag: env.action_spaces[ag].sample() for ag in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
```

Or via the registry:

```python
from powerzoo.tasks import make_task_env
env = make_task_env("gencos_bidding", framework="pettingzoo")
```

For competitive training:

```python
from powerzoo.rl import Trainer

t = Trainer("gencos_bidding", framework="pettingzoo", algorithm="PPO")
t.train_il(total_timesteps=5_000_000)   # independent PPO baseline
```

Current Python training treats this as a CMDP env with reward-only or fallback-MDP learning. No first-party multi-constraint MARL optimizer is added for GenCos in this round.

For self-play / PSRO-style training, use `t.get_env()` and write the population loop in your own code (see [Training · Custom loops](../training/custom-loops.md)).

## Baselines

- **Truthful bidding** — every agent offers at true marginal cost (`markup = 0`). The "social planner" reference; equals the centralised DCOPF welfare.
- **Fixed markup** — every agent uses `markup = 0.2`. Naive strategic baseline.
- **Myopic best-response** — each step, agent solves a 1-step LP assuming opponents play their last action.
- **Random markup** — `markup ~ U[0, 1]`.
- **Independent PPO** — `Trainer.train_il`.
- **Self-play PPO** — single shared policy trained against itself.

## Metrics to report

- `cumulative_profit_per_agent` — mean ± std over seeds, per agent.
- `social_welfare_ratio` — `(RL welfare) / (planner welfare)` ∈ `[0, 1]`.
- `price_volatility` — `std(LMP) / mean(LMP)`.
- `market_share_dynamics` — per-agent dispatch share trajectory.
- `exploitability_proxy` — `max_i (BR_i(π_{-i}) - π_i)` using a 1-step myopic best-response.
- `HHI` (Herfindahl-Hirschman Index) — market concentration.
- `ramp_binding_rate` — fraction of steps where dispatch hits the ramp boundary; verifies sequential coupling is active.

## See also

- [Physics · Markets](../physics/markets.md) — `CostBasedMarketEnv`, `BidBasedMarketEnv`, `GenCosMARLEnv` env mechanics.
- [Transmission physics](../physics/transmission.md) — DC-OPF and LMP derivation.
- [Training · Custom loops](../training/custom-loops.md) — self-play and population-based loops.
- [API · Markets](../api/market.md).
