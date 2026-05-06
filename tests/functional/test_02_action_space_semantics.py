"""Test 02 — Action space semantics.

Verifies that action_space declarations match actual step() behaviour
across all environment types, and checks RL-compatibility.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .conftest import (
    make_trans_env, make_dist_env, make_battery,
    simple_time_series, report_dir, save_figure, write_report,
)

CAT = "02_action_space_semantics"


def _probe_trans_env(ts):
    """Probe TransGridEnv action_space semantics."""
    env = make_trans_env(time_series=ts, solver_mode="pf", physics="dc")
    env.reset(seed=0)
    space = env.action_space
    low, high = space.low, space.high
    mid = (low + high) / 2.0

    records = []
    for label, action in [("low (p_min)", low), ("high (p_max)", high),
                           ("mid", mid), ("sample", space.sample()),
                           ("rl_zero", np.zeros_like(low)),
                           ("rl_one", np.ones_like(low))]:
        env.reset(seed=0)
        state, reward, term, trunc, info = env.step({"unit_power_mw": action.copy()})
        gen_sum = float(env._unit_power_mw.sum()) if env._unit_power_mw is not None else 0
        records.append(dict(label=label, action_min=float(action.min()),
                            action_max=float(action.max()),
                            gen_sum=gen_sum, reward=reward))
    return dict(env_name="TransGridEnv(dc,pf)", low=low, high=high, records=records)


def _probe_battery():
    """Probe BatteryEnv action_space semantics."""
    bat = make_battery(capacity_mwh=50, power_mw=20, initial_soc=0.5)
    bat.reset()
    space = bat.action_space
    low, high = float(space.low[0]), float(space.high[0])

    test_actions = [low, low / 2, 0.0, high / 2, high,
                    -1.0, 0.0, 0.5, 1.0]
    records = []
    for a in test_actions:
        bat.reset()
        soc_before = bat.soc
        bat.step(np.array([a]))
        soc_after = bat.soc
        records.append(dict(action=a, soc_before=soc_before,
                            soc_after=soc_after, power_mw=bat.current_p_mw))
    return dict(env_name="BatteryEnv", low=low, high=high, records=records)


@pytest.mark.functional
def test_action_space_semantics(summary):
    ts = simple_time_series()
    trans_data = _probe_trans_env(ts)
    bat_data = _probe_battery()

    lines = [
        "=" * 60,
        "  Test 02: action-space semantics",
        "=" * 60, "",
    ]

    # ── TransGridEnv ─────────────────────────────────────────────
    lines.append(f"[{trans_data['env_name']}]")
    lines.append(f"  action_space.low  = {trans_data['low']}")
    lines.append(f"  action_space.high = {trans_data['high']}")
    lines.append("")
    lines.append(f"  {'action':<18} {'a_min':>8} {'a_max':>8} {'gen_sum':>10} {'reward':>10}")
    lines.append("  " + "-" * 60)
    rl_zero_gen = rl_one_gen = None
    for r in trans_data["records"]:
        lines.append(f"  {r['label']:<18} {r['action_min']:>8.1f} {r['action_max']:>8.1f} "
                     f"{r['gen_sum']:>10.2f} {r['reward']:>10.4f}")
        if r["label"] == "rl_zero":
            rl_zero_gen = r["gen_sum"]
        if r["label"] == "rl_one":
            rl_one_gen = r["gen_sum"]

    lines.append("")
    mid_gen = [r["gen_sum"] for r in trans_data["records"] if r["label"] == "mid"][0]
    high_gen = [r["gen_sum"] for r in trans_data["records"] if r["label"] == "high (p_max)"][0]
    trans_ok = mid_gen > 0 and high_gen > mid_gen * 0.5
    trans_rl_warn = False
    if rl_zero_gen is not None and rl_one_gen is not None:
        p_min_sum = float(trans_data["low"].sum())
        p_max_sum = float(trans_data["high"].sum())
        mid_expected = (p_min_sum + p_max_sum) / 2
        if rl_one_gen < p_min_sum * 0.5 or rl_zero_gen < 1e-3:
            trans_rl_warn = True
            lines.append(f"  ⚠ RL-normalized action [0,1] yields gen_sum={rl_one_gen:.1f} MW, "
                         f"far below p_min_sum={p_min_sum:.1f} MW — policy ineffective")
            lines.append(f"    action=[0,...,0] → gen={rl_zero_gen:.1f} MW")
            lines.append(f"    action=[1,...,1] → gen={rl_one_gen:.1f} MW")
            lines.append(f"    whereas action_space span is [{p_min_sum:.0f}, {p_max_sum:.0f}] MW")

    lines.append(f"  physical-range actions behave as expected: {'PASS' if trans_ok else 'FAIL'}")
    lines.append("")

    # ── BatteryEnv ───────────────────────────────────────────────
    lines.append(f"[{bat_data['env_name']}]")
    lines.append(f"  action_space: [{bat_data['low']}, {bat_data['high']}] MW")
    lines.append("")
    lines.append(f"  {'action':>8} {'SOC_0':>8} {'SOC_1':>8} {'power_mw':>10} {'dSOC':>10}")
    lines.append("  " + "-" * 50)
    bat_ok = True
    for r in bat_data["records"]:
        delta = r["soc_after"] - r["soc_before"]
        lines.append(f"  {r['action']:>8.1f} {r['soc_before']:>8.4f} "
                     f"{r['soc_after']:>8.4f} {r['power_mw']:>10.4f} {delta:>10.6f}")
        if r["action"] > 1 and delta >= 0:
            bat_ok = False
        if r["action"] < -1 and delta <= 0:
            bat_ok = False

    lines.append("")
    bat_rl_warn = False
    # With normalize_actions=True (default), action_space should be [0,1].
    # Check that boundary actions (0 and 1) produce meaningful, distinct effects.
    a0 = [r for r in bat_data["records"] if r["action"] == 0.0]
    a1 = [r for r in bat_data["records"] if r["action"] == 1.0]
    if a0 and a1:
        p0 = a0[0]["power_mw"]
        p1 = a1[0]["power_mw"]
        if abs(p0 - p1) < 1e-6:
            bat_rl_warn = True
            lines.append("  ⚠ action=0 and action=1 yield the same power — normalization may be ineffective")
    if not bat_rl_warn:
        lines.append(f"  ✓ action=0 → {a0[0]['power_mw']:.1f} MW (charge), "
                     f"action=1 → {a1[0]['power_mw']:.1f} MW (discharge)"
                     if a0 and a1 else "")
    lines.append(f"  SOC responds with correct sign: {'PASS' if bat_ok else 'FAIL'}")

    write_report(CAT, lines)

    # ── bar chart: action_space range vs RL typical range ────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # TransGridEnv
    ax = axes[0]
    x = np.arange(len(trans_data["low"]))
    ax.bar(x - 0.15, trans_data["low"], 0.3, label="p_min (MW)", color="#3498db")
    ax.bar(x + 0.15, trans_data["high"], 0.3, label="p_max (MW)", color="#e67e22")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axhline(1, color="red", ls="--", lw=1, label="RL output=1.0")
    ax.set_xlabel("Unit index")
    ax.set_ylabel("MW")
    ax.set_title("TransGridEnv action_space\nvs RL [0,1] output")
    ax.legend(fontsize=8)

    # BatteryEnv SOC response
    ax = axes[1]
    phys_actions = [r["action"] for r in bat_data["records"] if abs(r["action"]) <= bat_data["high"]]
    soc_deltas = [r["soc_after"] - r["soc_before"]
                  for r in bat_data["records"] if abs(r["action"]) <= bat_data["high"]]
    ax.plot(phys_actions, soc_deltas, "o-", color="#2ecc71")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Action (MW)")
    ax.set_ylabel("ΔSOC")
    ax.set_title("BatteryEnv SOC response curve")

    fig.tight_layout()
    save_figure(fig, CAT, "action_effect_by_env.png")

    # ── SOC response curve (detailed) ────────────────────────────
    powers = np.linspace(-bat_data["high"], bat_data["high"], 21)
    socs = []
    for p in powers:
        bat = make_battery(capacity_mwh=50, power_mw=20, initial_soc=0.5)
        bat.reset()
        bat.step(np.array([p]))
        socs.append(bat.soc)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(powers, socs, "o-", color="#9b59b6", markersize=4)
    ax2.axhline(0.5, color="gray", ls="--", lw=0.5, label="initial SOC=0.5")
    ax2.set_xlabel("Action power (MW)")
    ax2.set_ylabel("SOC after 1 step")
    ax2.set_title("Battery SOC response to action sweep [-P_max, P_max]")
    ax2.legend()
    save_figure(fig2, CAT, "soc_response_curve.png")

    # ── summary ──────────────────────────────────────────────────
    if trans_rl_warn or bat_rl_warn:
        summary.add(CAT, "Action-space semantics", "WARN",
                     "action_space is physical MW; RL-normalized [0,1]/[-1,1] maps weakly or clips")
    elif not trans_ok or not bat_ok:
        summary.add(CAT, "Action-space semantics", "FAIL",
                     "action_space semantics mismatch observed behavior")
    else:
        summary.add(CAT, "Action-space semantics", "PASS",
                     "action_space matches observed behavior")
