"""Test 04 — Reward / Cost separation.

Verifies that reward carries only the economic objective and safety
penalties flow exclusively through the cost / info channel.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .conftest import (
    make_trans_env, simple_time_series, save_figure, write_report,
)

CAT = "04_reward_cost_separation"
N_STEPS = 48


def _collect(env, action_fn, n_steps=N_STEPS):
    """Run *n_steps* using *action_fn* and collect reward/cost/info."""
    records = []
    for t in range(n_steps):
        action = action_fn(env, t)
        state, reward, terminated, truncated, info = env.step(action)
        cost = info.get("cost_sum", info.get("cost", 0.0))
        violations = {}
        for k, v in info.items():
            if any(s in k.lower() for s in ("viol", "unsafe", "penalty", "cost_")):
                violations[k] = v
        records.append(dict(step=t, reward=reward, cost=cost,
                            is_safe=env._is_safe,
                            violations=violations, info_keys=list(info.keys())))
        if terminated or truncated:
            break
    return records


@pytest.mark.functional
def test_reward_cost_separation(summary):
    ts = simple_time_series()

    strategies = {}

    # Strategy 1: OPF-optimal (env dispatches automatically)
    env1 = make_trans_env(time_series=ts, physics="dc", solver_mode="opf")
    env1.reset(seed=42)
    strategies["optimal_opf"] = _collect(
        env1, lambda e, t: {}, n_steps=N_STEPS
    )

    # Strategy 2: all units at max
    env2 = make_trans_env(time_series=ts, physics="dc", solver_mode="pf")
    env2.reset(seed=42)
    p_max = env2.case.units["p_max"].values.astype(np.float32)
    strategies["all_max"] = _collect(
        env2, lambda e, t: {"unit_power_mw": p_max.copy()}, n_steps=N_STEPS
    )

    # Strategy 3: all units at min
    env3 = make_trans_env(time_series=ts, physics="dc", solver_mode="pf")
    env3.reset(seed=42)
    p_min = env3.case.units["p_min"].values.astype(np.float32)
    strategies["all_min"] = _collect(
        env3, lambda e, t: {"unit_power_mw": p_min.copy()}, n_steps=N_STEPS
    )

    # ── analysis ─────────────────────────────────────────────────
    lines = [
        "=" * 60,
        "  Test 04: Reward / cost separation",
        "=" * 60, "",
    ]

    all_rewards = []
    all_costs = []
    all_labels = []
    strategy_stats = {}
    has_cost_field = False

    for name, records in strategies.items():
        rewards = np.array([r["reward"] for r in records])
        costs = np.array([r["cost"] for r in records])
        unsafe_n = sum(1 for r in records if not r["is_safe"])
        all_info_keys = set()
        for r in records:
            all_info_keys.update(r["info_keys"])

        if any("cost_sum" in r["info_keys"] or "cost" in r["info_keys"] for r in records):
            has_cost_field = True

        all_rewards.extend(rewards.tolist())
        all_costs.extend(costs.tolist())
        all_labels.extend([name] * len(rewards))

        corr = float(np.corrcoef(rewards, costs)[0, 1]) if len(rewards) > 1 and costs.std() > 0 else 0.0

        strategy_stats[name] = dict(
            n_steps=len(records),
            reward_mean=float(rewards.mean()),
            reward_std=float(rewards.std()),
            cost_mean=float(costs.mean()),
            cost_std=float(costs.std()),
            unsafe_steps=unsafe_n,
            reward_cost_corr=corr,
        )

        lines.append(f"[strategy: {name}]")
        lines.append(f"  steps: {len(records)}")
        lines.append(f"  reward: mean={rewards.mean():.4f}, std={rewards.std():.4f}, "
                     f"min={rewards.min():.4f}, max={rewards.max():.4f}")
        lines.append(f"  cost:   mean={costs.mean():.4f}, std={costs.std():.4f}")
        lines.append(f"  unsafe steps: {unsafe_n}")
        lines.append(f"  reward–cost correlation: {corr:.4f}")
        lines.append(f"  info keys: {sorted(all_info_keys)}")
        lines.append("")

    # separation verdict
    mixed = False
    for name, s in strategy_stats.items():
        if abs(s["reward_cost_corr"]) > 0.8 and s["cost_std"] > 0:
            mixed = True
            lines.append(f"  ⚠ strategy '{name}': reward and cost highly correlated (r={s['reward_cost_corr']:.2f}); "
                         "reward may embed safety penalties")

    if not has_cost_field:
        lines.append("  ⚠ no 'cost' field in info — cannot verify separation (WARN)")

    lines.append("")
    lines.append(f"verdict: {'PASS — reward and cost separated' if not mixed else 'WARN — reward may mix in cost'}")
    write_report(CAT, lines)

    # ── scatter plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"optimal_opf": "#2ecc71", "all_max": "#3498db", "all_min": "#e74c3c"}
    for name, records in strategies.items():
        r = [x["reward"] for x in records]
        c = [x["cost"] for x in records]
        ax.scatter(r, c, label=name, alpha=0.6, s=30,
                   color=colors.get(name, "gray"))
    ax.set_xlabel("Reward")
    ax.set_ylabel("Cost")
    ax.set_title("Reward vs Cost (by strategy)")
    ax.legend()
    save_figure(fig, CAT, "reward_vs_cost_scatter.png")

    # ── time-series ──────────────────────────────────────────────
    fig2, axes = plt.subplots(len(strategies), 1, figsize=(10, 3 * len(strategies)),
                               sharex=True)
    if len(strategies) == 1:
        axes = [axes]
    for ax, (name, records) in zip(axes, strategies.items()):
        steps = np.arange(len(records))
        r = [x["reward"] for x in records]
        c = [x["cost"] for x in records]
        ax.plot(steps, r, label="reward", color="#2ecc71")
        ax2 = ax.twinx()
        ax2.plot(steps, c, label="cost", color="#e74c3c", ls="--")
        ax.set_ylabel("Reward")
        ax2.set_ylabel("Cost")
        ax.set_title(f"Strategy: {name}")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time step")
    fig2.tight_layout()
    save_figure(fig2, CAT, "reward_cost_timeseries.png")

    # ── violation histogram ──────────────────────────────────────
    viol_types: dict[str, int] = {}
    for records in strategies.values():
        for r in records:
            for k, v in r["violations"].items():
                count = int(v) if isinstance(v, (int, float, np.integer, np.floating)) else 1
                viol_types[k] = viol_types.get(k, 0) + count
    if viol_types:
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.barh(list(viol_types.keys()), list(viol_types.values()), color="#e67e22")
        ax3.set_xlabel("Count")
        ax3.set_title("Violation Type Distribution (all strategies)")
        save_figure(fig3, CAT, "violation_histogram.png")

    # ── summary ──────────────────────────────────────────────────
    if mixed:
        summary.add(CAT, "Reward/cost separation", "WARN",
                     "reward and cost highly correlated; may not be fully separated")
    elif not has_cost_field:
        summary.add(CAT, "Reward/cost separation", "WARN",
                     "no cost field in info; cannot verify")
    else:
        summary.add(CAT, "Reward/cost separation", "PASS",
                     "reward carries economic objective only; cost passed separately")
