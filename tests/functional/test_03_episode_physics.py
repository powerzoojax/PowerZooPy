"""Test 03 — End-to-end episode physics validation.

Runs full episodes on TransGridEnv and DistGridEnv, checks power balance,
SOC conservation, line overloads, and voltage violations.
"""

import json
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .conftest import (
    make_trans_env, make_dist_env, make_battery,
    simple_time_series, save_figure, write_report, write_json,
)

CAT = "03_episode_physics"
N_STEPS = 48


def _run_trans_episode(ts, n_steps=N_STEPS):
    """Run a full episode on TransGridEnv(dc, opf) and collect traces."""
    env = make_trans_env(time_series=ts, physics="dc", solver_mode="opf")
    bat = make_battery(parent=env, bus_id=2, capacity_mwh=50, power_mw=20)
    state, _ = env.reset(seed=42)

    traces = dict(
        gen_total=[], load_total=[], mismatch_pct=[],
        reward=[], cost=[], is_safe=[],
        soc=[], bat_power=[],
        line_flow_max_ratio=[],
    )

    for t in range(n_steps):
        load = float(env._get_default_node_load().sum())
        action = {}
        state, reward, terminated, truncated, info = env.step(action)

        gen = float(env._unit_power_mw.sum()) if env._unit_power_mw is not None else 0
        mismatch = abs(gen - load) / max(abs(load), 1e-6) * 100

        traces["gen_total"].append(gen)
        traces["load_total"].append(load)
        traces["mismatch_pct"].append(mismatch)
        traces["reward"].append(reward)
        traces["cost"].append(info.get("cost", 0.0))
        traces["is_safe"].append(env._is_safe)
        traces["soc"].append(bat.soc)
        traces["bat_power"].append(bat.current_p_mw)

        if env._lines is not None and "line_flow_mw" in env._lines.columns:
            caps = env.case.lines["cap"].values.astype(float)
            caps = np.where(caps > 0, caps, 1.0)
            ratio = np.abs(env._lines["line_flow_mw"].values) / caps
            traces["line_flow_max_ratio"].append(float(ratio.max()))
        else:
            traces["line_flow_max_ratio"].append(0.0)

        if terminated or truncated:
            break

    return traces


def _run_dist_episode(ts, n_steps=N_STEPS):
    """Run a full episode on DistGridEnv(case33)."""
    env = make_dist_env(time_series=ts)
    bat = make_battery(parent=env, bus_id=10, capacity_mwh=20, power_mw=5)
    state, _ = env.reset(seed=42)

    traces = dict(
        load_total=[], reward=[], cost=[], is_safe=[],
        soc=[], bat_power=[],
        v_min=[], v_max=[], v_mean=[],
        voltage_violations=[], loss_mw=[],
    )

    for t in range(n_steps):
        state, reward, terminated, truncated, info = env.step({})

        if hasattr(env, '_get_node_loads_p'):
            load = float(env._get_node_loads_p().sum())
        else:
            load = 0.0
        traces["load_total"].append(load)
        traces["reward"].append(reward)
        traces["cost"].append(info.get("cost", 0.0))
        traces["is_safe"].append(env._is_safe)
        traces["soc"].append(bat.soc)
        traces["bat_power"].append(bat.current_p_mw)

        if env._nodes is not None and "v_mag" in env._nodes.columns:
            v = env._nodes["v_mag"].values
            traces["v_min"].append(float(v.min()))
            traces["v_max"].append(float(v.max()))
            traces["v_mean"].append(float(v.mean()))
            n_viol = int(((v < env.v_min) | (v > env.v_max)).sum())
            traces["voltage_violations"].append(n_viol)
        else:
            traces["v_min"].append(np.nan)
            traces["v_max"].append(np.nan)
            traces["v_mean"].append(np.nan)
            traces["voltage_violations"].append(0)

        traces["loss_mw"].append(getattr(env, "_p_loss", 0.0))

        if terminated or truncated:
            break

    return traces


@pytest.mark.functional
def test_episode_physics(summary):
    ts = simple_time_series()
    trans = _run_trans_episode(ts)
    dist = _run_dist_episode(ts)

    # ── statistics ───────────────────────────────────────────────
    t_steps = len(trans["gen_total"])
    d_steps = len(dist["load_total"])

    t_mismatch = np.array(trans["mismatch_pct"])
    t_overload_count = sum(1 for r in trans["line_flow_max_ratio"] if r > 1.0)

    d_vviol = np.array(dist["voltage_violations"])
    d_loss = np.array(dist["loss_mw"])

    stats = {
        "trans": {
            "n_steps": t_steps,
            "power_balance_mismatch_pct": {
                "max": float(t_mismatch.max()),
                "mean": float(t_mismatch.mean()),
                "std": float(t_mismatch.std()),
            },
            "line_overload_steps": t_overload_count,
            "unsafe_steps": sum(1 for s in trans["is_safe"] if not s),
            "soc_range": [float(min(trans["soc"])), float(max(trans["soc"]))],
        },
        "dist": {
            "n_steps": d_steps,
            "voltage": {
                "global_min": float(np.nanmin(dist["v_min"])) if dist["v_min"] else None,
                "global_max": float(np.nanmax(dist["v_max"])) if dist["v_max"] else None,
            },
            "voltage_violation_steps": int(d_vviol.sum()),
            "loss_mw_mean": float(d_loss.mean()),
            "unsafe_steps": sum(1 for s in dist["is_safe"] if not s),
        },
    }
    write_json(CAT, "statistics.json", stats)

    # ── report ───────────────────────────────────────────────────
    lines = [
        "=" * 60,
        "  Test 03: end-to-end episode physics",
        "=" * 60, "",
        "[TransGridEnv — DC OPF, Case5]",
        f"  episode steps: {t_steps}",
        f"  power-balance mismatch: max={t_mismatch.max():.3f}%, mean={t_mismatch.mean():.4f}%, std={t_mismatch.std():.4f}%",
        f"  line overload steps: {t_overload_count}/{t_steps}",
        f"  unsafe steps: {stats['trans']['unsafe_steps']}/{t_steps}",
        f"  battery SOC range: {stats['trans']['soc_range']}",
        f"  power balance < 1%: {'PASS' if t_mismatch.max() < 1.0 else 'FAIL'}",
        "",
        "[DistGridEnv — BFS, Case33bw]",
        f"  episode steps: {d_steps}",
        f"  voltage range: [{stats['dist']['voltage']['global_min']}, {stats['dist']['voltage']['global_max']}] p.u.",
        f"  voltage violation steps: {stats['dist']['voltage_violation_steps']}/{d_steps}",
        f"  mean loss: {stats['dist']['loss_mw_mean']:.4f} MW",
        f"  unsafe steps: {stats['dist']['unsafe_steps']}/{d_steps}",
    ]
    write_report(CAT, lines)

    # ── power balance time-series ────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    steps = np.arange(t_steps)
    ax1.fill_between(steps, trans["gen_total"], alpha=0.4, label="Generation", color="#2ecc71")
    ax1.plot(steps, trans["load_total"], "r-", lw=1.5, label="Load")
    ax1.set_ylabel("MW")
    ax1.set_title("TransGridEnv: Generation vs Load")
    ax1.legend()
    ax2.bar(steps, trans["mismatch_pct"], color="#e74c3c", alpha=0.7)
    ax2.set_ylabel("Mismatch %")
    ax2.set_xlabel("Time step")
    ax2.axhline(1.0, color="red", ls="--", lw=0.8, label="1% threshold")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, CAT, "power_balance_timeseries.png")

    # ── SOC trajectory ───────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, trans["soc"][:t_steps], "b-", label="SOC")
    ax2t = ax.twinx()
    ax2t.bar(steps, trans["bat_power"][:t_steps], alpha=0.3, color="orange", label="Battery P (MW)")
    ax.set_ylabel("SOC")
    ax2t.set_ylabel("Power (MW)")
    ax.set_xlabel("Time step")
    ax.set_title("Battery SOC trajectory & power (TransGridEnv)")
    ax.legend(loc="upper left")
    ax2t.legend(loc="upper right")
    save_figure(fig2, CAT, "soc_trajectory.png")

    # ── voltage profile (dist) ───────────────────────────────────
    if not all(np.isnan(dist["v_min"])):
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        d_steps_arr = np.arange(d_steps)
        ax3.fill_between(d_steps_arr, dist["v_min"], dist["v_max"],
                         alpha=0.3, color="#3498db", label="V range")
        ax3.plot(d_steps_arr, dist["v_mean"], "b-", lw=1, label="V mean")
        ax3.axhline(0.95, color="red", ls="--", lw=0.8, label="V limits")
        ax3.axhline(1.05, color="red", ls="--", lw=0.8)
        ax3.set_ylabel("Voltage (p.u.)")
        ax3.set_xlabel("Time step")
        ax3.set_title("DistGridEnv: Node Voltage Profile")
        ax3.legend()
        save_figure(fig3, CAT, "voltage_profile.png")

    # ── load profile ─────────────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(np.arange(t_steps), trans["load_total"], label="Trans load", color="#e67e22")
    ax4.plot(np.arange(d_steps), dist["load_total"], label="Dist load", color="#3498db")
    ax4.set_ylabel("MW")
    ax4.set_xlabel("Time step")
    ax4.set_title("Load Profiles")
    ax4.legend()
    save_figure(fig4, CAT, "load_profile.png")

    # ── summary ──────────────────────────────────────────────────
    physics_ok = t_mismatch.max() < 1.0
    if not physics_ok:
        summary.add(CAT, "End-to-end episode physics", "FAIL",
                     f"power-balance mismatch max={t_mismatch.max():.2f}% exceeds 1%")
    else:
        summary.add(CAT, "End-to-end episode physics", "PASS",
                     f"power-balance mismatch max={t_mismatch.max():.3f}%, "
                     f"voltage violations={stats['dist']['voltage_violation_steps']} steps")
