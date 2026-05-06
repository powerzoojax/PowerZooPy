# Transmission physics

`TransGridEnv` (`powerzoo/envs/grid/trans.py`) is the transmission-grid env. It supports four solver modes built from two orthogonal switches:

- `physics ∈ {'dc', 'ac'}` — linearised vs full nonlinear AC equations.
- `solver_mode ∈ {'opf', 'pf'}` — does the env *optimise* dispatch (`opf`) or only *evaluate* a given dispatch (`pf`)?

The four combinations cover the standard transmission tasks:

| `physics` | `solver_mode` | Resulting solver | Typical RL use |
|---|---|---|---|
| `'dc'` | `'opf'` | **DCOPF** — linear program (Gurobi / SciPy / CVXPY) | Agent learns bidding or commitment; environment owns dispatch. |
| `'ac'` | `'opf'` | **ACOPF** — nonlinear program (cyipopt / SLSQP) | Agent learns bidding with reactive power and voltage. |
| `'dc'` | `'pf'` | **DCPF** — `line_flow = PTDF · injection` (matrix-vector mul) | Agent learns dispatch directly; environment evaluates feasibility. |
| `'ac'` | `'pf'` | **ACPF** — Newton-Raphson | Agent learns dispatch with full AC physics. |

The OPF LP backend is selected by `solver_type ∈ {'auto', 'gurobi', 'scipy', 'cvxpy'}`. This is orthogonal to `physics` / `solver_mode` and only chooses the LP library used in OPF mode.

> **Vocabulary check.** *PF* (Power Flow) — solve voltages and line flows given fixed injections. *OPF* (Optimal Power Flow) — also dispatch generators to minimise cost. *DC* in this context means *linearised* (constant 1 pu voltage, no reactive power, no losses), not direct-current. *PTDF* (Power Transfer Distribution Factor) — sensitivity matrix linking nodal injections to line flows. *LMP* (Locational Marginal Price) — dual variable of the nodal power balance.

## DC power flow

DC PF assumes voltages are 1 pu, ignores reactive power, and approximates losses as zero:

\[
P_{\text{line}} \;=\; \text{PTDF} \cdot P_{\text{injection}}
\]

The PTDF matrix is precomputed at `case.init()` from the line reactances. A DC PF call is a single matrix-vector multiply: fast, differentiable, and identical to what the DCOPF LP uses for its line-flow constraints.

## AC power flow

AC PF solves the nonlinear nodal power balance at every bus:

\[
P_i = V_i \sum_j V_j (G_{ij} \cos\theta_{ij} + B_{ij} \sin\theta_{ij})
\]

\[
Q_i = V_i \sum_j V_j (G_{ij} \sin\theta_{ij} - B_{ij} \cos\theta_{ij})
\]

where \(V_i\) is voltage magnitude, \(\theta_{ij}\) is the voltage-angle difference and \(G_{ij}\), \(B_{ij}\) come from the admittance matrix. PowerZoo uses Newton-Raphson by default.

AC PF can fail to converge under infeasible dispatch. `info['pf_converged']` reports the actual outcome; PF failure is a real cost (see [Reward and cost split](../concepts/reward-cost-split.md)).

## DCOPF

DCOPF solves a linear program:

\[
\min_{P_g} \; \sum_i C_i(P_{g,i}) \quad \text{s.t.} \quad
\mathbf{1}^\top P_g = \mathbf{1}^\top D, \quad
|\text{PTDF} \cdot (P_g - D)| \leq \overline{S}, \quad
P_g^{\min} \leq P_g \leq P_g^{\max}
\]

with quadratic cost \(C_i(P) = mc\_a_i P^2 + mc\_b_i P + mc\_c_i\). For cases with `mc_a = mc_b = 0` (`Case5` and most IEEE cases), the cost is flat marginal: \(C_i(P) = mc\_c_i \cdot P\). LMPs are recovered from the duals of the nodal balance constraint.

## ACOPF

ACOPF solves the same objective subject to the AC equations and voltage limits. PowerZoo wraps cyipopt (preferred) or SLSQP. ACOPF is non-convex and may return a local optimum.

## `physics` × `solver_mode` decision tree

```mermaid
flowchart TD
    Q1{Does the agent decide\nunit dispatch (P)?}
    Q1 -->|yes| Q2{Need voltage and Q?}
    Q1 -->|no, env optimises| Q3{Need voltage and Q?}
    Q2 -->|yes| ACPF["physics='ac' + solver_mode='pf'\n→ ACPF"]
    Q2 -->|no| DCPF["physics='dc' + solver_mode='pf'\n→ DCPF"]
    Q3 -->|yes| ACOPF["physics='ac' + solver_mode='opf'\n→ ACOPF"]
    Q3 -->|no| DCOPF["physics='dc' + solver_mode='opf'\n→ DCOPF"]
```

## Case sizes available

| Case | Buses | Lines | Generators | Default for |
|---|---|---|---|---|
| `Case5` | 5 | 6 | 5 | `marl_opf`, `marl_uc`, `gencos_bidding` |
| `Case14` | 14 | 20 | 5 | sandbox / OOD scale variant |
| `Case118` | 118 | 186 | 54 | `opf_118`, `opf_118_7d` |
| `Case300`, `Case1354pegase`, `Case2383wp` | 300+ | 411+ | 80+ | scalability stress tests |
| `Case29GB` | 29 | 99 | 66 | GB reduced transmission (MATPOWER) |
| `Case552GB` | 552 | 673 | 2385 | GB large-scale transmission |

`Case5` and `Case118` are the two main public benchmark cases. Larger cases are available for scalability research but are not part of the standard task set.

## What goes into `info`

Every `TransGridEnv.step()` populates the standard keys (see [Python contract](../concepts/python-contract.md) §4) plus, when applicable:

- `lmp` (np.ndarray, $/MWh) — nodal LMPs from the OPF dual.
- `lmp_quality` — `'gurobi_dual'` / `'scipy_recovered'` / `'cvxpy'` — how the LMP was computed.
- `solver_backend` — actual LP solver used (e.g. `'highs'`).
- `opf_cost` ($/h) — total generation cost from the OPF objective.

## Difficulty presets

`TransGridEnv` accepts `difficulty='easy' / 'medium' / 'hard'`, which sets the load ratio and the time-step length:

| Preset | Load ratio | `delta_t_minutes` | Effect |
|---|---|---|---|
| `easy` | 0.7 | 60 | Loose constraints, fewer steps. |
| `medium` | 0.85 | 30 | Standard benchmark setting. |
| `hard` | 0.95 | 30 | Many lines near their limits. |

The presets are convenient for sanity testing; for benchmark experiments, prefer the explicit task names (`marl_opf` etc.), which fix every hyperparameter via `SPLIT_DATES`.

## See also

- [Distribution physics](distribution.md) — the BFS / DistFlow counterpart to this page.
- [Resources](resources.md) — controllable assets that attach to a transmission grid.
- [Markets](markets.md) — LMP-driven settlement on top of `TransGridEnv`.
- [Benchmarks · TSO](../benchmarks/tso.md), [Benchmarks · GenCos](../benchmarks/gencos.md).
- [API · Grid](../api/grid.md) — per-method signatures.
