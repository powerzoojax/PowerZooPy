# Resources

A **resource** is a controllable physical asset attached to a grid. PowerZoo's resource layer (`powerzoo/envs/resource/`) covers six asset types: battery, electric vehicle (EV), solar, wind, flexible load and data center.

The shared design rule is that **a resource is not a standalone RL env**. Its `step()` updates internal state (SOC, queue, temperature, …) but does *not* return a reward or termination signal. The Gymnasium `(obs, reward, terminated, truncated, info)` contract plus the CMDP cost vector in `info` is assembled by `PowerEnv` plus a `Task`. To train an RL agent on a single resource, use a task (e.g. `battery_arbitrage` for a single battery) or write a custom `PowerEnv` config — never call `BatteryEnv` directly as a `gymnasium.Env`.

> **Vocabulary check.** *SOC* (State Of Charge) — battery fill, 0–1. *G2V / V2G* — Grid-to-Vehicle / Vehicle-to-Grid. *DR* (Demand Response) — curtailing or shifting load in response to grid signals. *PUE* (Power Usage Effectiveness) — total facility power / IT equipment power, lower is better. *COP* (Coefficient of Performance) — heat removed per unit electrical input.

## `BatteryEnv` — energy storage

Battery state of charge evolves as:

\[
\text{SOC}_{t+1} \;=\; \text{SOC}_t + \frac{\Delta t}{E_{\text{cap}}}
\begin{cases}
-P_t \cdot \eta_{\text{charge}} & \text{if charging } (P_t < 0) \\
-P_t / \eta_{\text{discharge}} & \text{if discharging } (P_t > 0)
\end{cases}
\]

subject to `SOC_min ≤ SOC ≤ SOC_max` and `P_min ≤ P_t ≤ P_max`.

Defaults: one-way `eta_charge = eta_discharge = 0.95` (hence achieved round-trip ≈ 0.9025). The optional `eta_roundtrip` parameter is a sqrt shorthand: setting `eta_roundtrip = 0.9` is equivalent to `eta_charge = eta_discharge = sqrt(0.9) ≈ 0.949`. The legacy `efficiency` kwarg is deprecated.

Action space is 1D `[P_norm]` ∈ `[-1, 1]`, mapped to `[-power_mw, +power_mw]`. With `enable_q_control=True` the action becomes 2D `[P_norm, Q_norm]`. Cost component: `cost_clipped_power` = `|desired − feasible|` after SOC / power clipping.

## `VehicleEnv` — electric vehicle (G2V / V2G)

Same SOC dynamics as `BatteryEnv`, plus three EV-specific constraints:

- **Availability**: the EV can charge or discharge only when parked at home. During commute hours the action is masked to zero.
- **Departure SOC**: the EV must reach `SOC ≥ SOC_departure` before leaving. Missing this deadline is a hard violation.
- **Stochastic schedule**: departure / arrival times can vary per episode.

Action: 1D `[P_norm]` ∈ `[-1, 1]`. Observation is 9D and includes home / away flag, departure-readiness, time-to-departure and the underlying SOC. Cost contributions: `cost_clipped_power` (also non-zero when an action is issued while away), plus EV-specific departure and home-availability costs reported through the task adapter.

## `SolarEnv` and `WindEnv` — renewables

Both subclass `RenewableEnv`. Output is profile-driven (`SOLAR_AVAILABLE_MW` / `WIND_AVAILABLE_MW` from the data pipeline) and capped by nameplate `capacity_mw`. The agent's only control is **curtailment**: a 1D action `[curtail_frac]` ∈ `[0, 1]` reduces output below the available level (1.0 = no curtailment, 0.0 = full curtailment). With `enable_q_control=True` the action becomes 2D `[curtail_frac, Q_norm]`.

Observation is 4D (or 5D with Q control) and includes the underlying capacity factor, current power and a curtailment-cost penalty. There is no SOC integrator; the only state is the time-series index.

## `FlexLoad` — demand response

`FlexLoad` is the demand-response resource. Each device has two independent levers:

- **Curtailment** — permanent demand reduction, capped by `curtail_cap_mw`.
- **Demand shifting** — defer consumption over a `shift_horizon` window, capped by `shift_cap_mw`. Shifted energy enters a **buffer** that must be repaid within the horizon.

Action is 2D `[curtail_mw, shift_out_mw]`, with three scaling modes (`physical`, `unit`, `tanh`). Observation is 8D — `[curtail_norm, shift_out_norm, shift_in_norm, buffer_fill_ratio, buffer_energy_norm, time_sin, time_cos, price_norm]`. Cost components:

| Cost field | Unit | Meaning |
|---|---|---|
| `cost_buffer_overflow` | MWh | Deferred demand exceeding the shift horizon. |
| `cost_curtailment` | $ | Discomfort / compensation for curtailed energy. |
| `cost_shift_discomfort` | $ | Holding cost for buffered deferred demand. |
| `cost_simultaneous` | $ | Complementarity violation (curtailing and shifting at the same step). |

`FlexLoad` also exposes an LMP injection interface (`set_lmp`, `get_bid`) that is SCUC / SCED-compatible — see [Markets](markets.md).

## `DataCenterEnv` — AI data center as a controllable load

`DataCenterEnv` is more elaborate than the other resources. It models:

- **GPU-level IT power**: per-GPU idle and active draw, queue of training and finetuning jobs, exogenous inference load.
- **Cooling**: COP-based cooling power that scales with the cooling setpoint and outdoor temperature.
- **Thermal dynamics**: first-order zone-temperature model with a critical threshold.
- **Workload**: an EDF (Earliest-Deadline-First) scheduler that consumes the queue.

Action is 3D `[r_train, r_finetune, T_cool_setpoint_norm]`:

- `r_train`, `r_finetune` ∈ `[0, 1]` — fraction of available GPUs allocated to each workload.
- `T_cool_setpoint_norm` ∈ `[0, 1]` — normalised cooling thermostat.

Observation is 11D (utilisation, queues, temperature, COP, prices, time). Inference workload is exogenous (diurnal). Cost component: `cost_overtemp` = `max(t_zone − t_critical, 0)`.

`DataCenterEnv` is used by two tasks: `dc_scheduling` (where it sits on a distribution grid as a flexible load) and `dc_microgrid` (where it sits inside a self-contained DC microgrid).

## Cost summary

Every cost field exposed by a resource flows automatically into `info['cost_resource']` and then into the fixed-order `info['constraint_costs']` vector. The legacy alias `info['cost_resource_violation']` remains available during the transition. The full table:

| Resource | Cost field | Unit |
|---|---|---|
| `BatteryEnv` | `cost_clipped_power` | MW |
| `VehicleEnv` | `cost_clipped_power` (+ EV-specific via adapter) | MW |
| `DataCenterEnv` | `cost_overtemp` | °C |
| `FlexLoad` | `cost_buffer_overflow` | MWh |
| `FlexLoad` | `cost_curtailment` | $ |
| `FlexLoad` | `cost_shift_discomfort` | $ |
| `FlexLoad` | `cost_simultaneous` | $ |

Any `cost_*` key in `status()` is collected automatically; the full convention is documented in [Reward and cost split](../concepts/reward-cost-split.md).

## Attaching a resource to a grid

```python
from powerzoo.envs.grid import TransGridEnv
from powerzoo.envs.resource import BatteryEnv

grid = TransGridEnv()
battery = BatteryEnv(capacity_mwh=50.0, power_mw=20.0, parent=grid, bus_id=2)
print(battery.resource_id)  # 'battery_0'
```

The resource is registered automatically via `parent` / `bus_id`. Changing `bus_id` mid-episode triggers a map rebuild via the property setter. See [Architecture · Environment stack](../architecture/env-stack.md) §3 for the registration data flow.

## See also

- [Transmission physics](transmission.md), [Distribution physics](distribution.md) — the grid envs a resource attaches to.
- [Markets](markets.md) — `FlexLoad` and `BatteryEnv` in market envs.
- [Microgrid](microgrid.md) — `DataCenterEnv` + `BatteryEnv` in a self-contained DC microgrid.
- [API · Resources](../api/resource.md) — per-class signatures and parameter tables.
