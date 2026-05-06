# Distribution physics

The distribution layer assumes **radial topology** (no loops) and uses **Backward-Forward Sweep (BFS)** rather than the sparse Newton-Raphson used in transmission. PowerZoo ships two envs:

- `DistGridEnv` (`powerzoo/envs/grid/dist.py`) — single-phase balanced BFS.
- `DistGrid3PhaseEnv` (`powerzoo/envs/grid/dist_3phase.py`) — three-phase unbalanced BFS via BIBC / BCBV matrices.

Both extend `GridEnv`; the env-stack contract (resource registration, `info` schema, reset → step flow) is identical to `TransGridEnv`.

> **Vocabulary check.** *Radial* — the network is a tree; every load has exactly one path back to the substation. *Feeder* — one branch of the tree, from the substation to a leaf. *BFS* (Backward-Forward Sweep) — iterative PF that walks the tree backward to sum currents and forward to update voltages. *DistFlow* — the linearised BFS recursion used in this env. *VUF* (Voltage Unbalance Factor) — the magnitude difference between phases on a three-phase line.

## `DistGridEnv` — single-phase BFS

The single-phase env solves a balanced **DistFlow-style** model. Resources are modelled as **PQ injections**: every attached resource exposes `current_p_mw` (and optionally `current_q_mvar`), and the env aggregates them onto the bus before each PF iteration.

Key conventions:

- Net-load convention is **load-positive / injection-negative** (so a discharging battery enters the BFS as a negative load).
- Feeder-head exchange is exposed as `info['p_slack_MW']` and `info['q_slack_MVAr']`.
- Non-radial inputs are auto-pruned to the BFS first-visit spanning tree by default. Pass `allow_mesh_pruning=False` to fail fast instead.
- An optional `load.reactive_mvar` time series overrides inferred Q scaling; otherwise the env preserves the per-node power factor from the case baseline.
- The default scalar reward is **loss-only**: `-loss_penalty_weight * p_loss_MW` (`loss_penalty_weight=0.1`). Voltage and thermal violations stay in `info['cost_voltage_violation']` and `info['cost_thermal_overload']` unless soft-penalty weights are explicitly enabled (not recommended for benchmark runs).

### Convergence vs collapse

`DistGridEnv` distinguishes two kinds of PF failure:

| `info` key | Meaning |
|---|---|
| `pf_converged` | BFS reached the iteration tolerance. |
| `is_diverged` | BFS hit `max_iter` before satisfying tolerance. |
| `voltage_collapse` | The unclamped voltage update entered a severe low-voltage regime, even though BFS may have iterated. |

A numerical voltage clamp keeps the simulator running, but a severe undervoltage is reported as `voltage_collapse=True` and treated as PF failure at the env level (resource `step()` is still called, but the agent should treat the result as infeasible).

### Available cases

| Case | Buses | Notes |
|---|---|---|
| `Case33bw` | 33 | IEEE 33-bus radial (Baran & Wu). Default. |
| `Case118zh` | 118 | 118-bus distribution (Zhang). Default for `marl_ders_benchmark`. |
| `Case141` | 141 | 141-bus Caracas distribution. |
| `Case533mt_lo`, `Case533mt_hi` | 533 | Swedish 533-bus (low / high load variants). |

## `DistGrid3PhaseEnv` — three-phase unbalanced

`DistGrid3PhaseEnv` extends `DistGridEnv` with a per-phase formulation. Internally it solves the BIBC / BCBV matrix recursion on the Kron-expanded `A/B/C` state vector; the per-phase voltages, currents and flows are returned to the agent through `obs()`.

Conventions:

- Core solver vectors use **node-major `ABC` order** (`[node1_A, node1_B, node1_C, node2_A, ...]`).
- `env.topo3ph` exposes the physical-node-to-matrix mapping for inspection.
- Mutual coupling inside the series `3x3` impedance block is fully supported.
- Off-nominal taps, branch shunt `B` and phase shifts (`ratio` / `angle`) are currently **ignored** — the impedance must already encode the desired transformer behaviour.
- True missing-phase laterals must be encoded as zero impedance in the upstream `3x3` block; the env does not synthesize them.
- When BFS does not converge, the returned voltages and flows are **last-iterate diagnostics only**. Always check `info['pf_converged']` before trusting them.
- The `safety_check` extends the single-phase one with per-phase voltage limits, VUF and phase-aware thermal limits.

### Available three-phase cases

| Case | Buses | Notes |
|---|---|---|
| `Case123` | 123 | IEEE 123-bus three-phase distribution. Default. |

Resources attached to a three-phase grid carry an optional `phase` parameter (`A` / `B` / `C` / `ABC`). `ABC` distributes the resource power equally across all three phases.

## DistFlow physics in one paragraph

For a radial branch from bus `i` to bus `j` carrying real power `P` and reactive power `Q`, with line resistance `R` and reactance `X`, the BFS update reads:

\[
V_j^2 \;\approx\; V_i^2 \;-\; 2\,(R \cdot P + X \cdot Q) \;+\; (R^2 + X^2) \cdot \frac{P^2 + Q^2}{V_i^2}
\]

The full nonlinear term is small for distribution voltages near 1 pu but is retained by `DistGridEnv` to avoid systematic underestimation of voltage drops at feeder ends. Because the `R / X` ratio is ≈ 1.0 in distribution (vs ≈ 0.1 in transmission), real-power injections have a measurable effect on voltage. This is why DER coordination is fundamentally about voltage in distribution, but about thermal limits in transmission.

## What goes into `info`

In addition to the standard keys (see [Python contract](../concepts/python-contract.md) §4):

- `voltages` — bus voltage magnitudes (pu).
- `branch_loading` — per-line apparent power / rating ratio.
- `p_loss_MW`, `q_loss_MVAr` — total network losses.
- `voltage_collapse`, `is_diverged` — the two BFS failure flags described above.
- For `DistGrid3PhaseEnv`: per-phase versions of the above plus `vuf_pct`.

## See also

- [Transmission physics](transmission.md) — the DC / AC OPF / PF counterpart.
- [Resources](resources.md) — controllable assets that attach to a distribution feeder.
- [Benchmarks · DSO](../benchmarks/dso.md), [Benchmarks · DERs](../benchmarks/ders.md).
- [API · Grid](../api/grid.md) — per-method signatures.
