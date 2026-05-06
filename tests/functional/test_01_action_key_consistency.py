"""Test 01 — Action key consistency across all action-passing paths.

Verifies that dict keys used by wrappers/adapters/market envs match
what the resource environments actually read.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .conftest import (
    make_trans_env, make_battery, make_cost_based_market,
    simple_time_series, report_dir, save_figure, write_report,
)

CAT = "01_action_key_consistency"


# ── helpers ──────────────────────────────────────────────────────────

def _battery_on_grid(ts):
    """Return (grid, battery) with battery attached to bus 2."""
    grid = make_trans_env(time_series=ts)
    bat = make_battery(parent=grid, bus_id=2, capacity_mwh=50, power_mw=20)
    return grid, bat


# ── individual path tests ────────────────────────────────────────────

class _Result:
    """Stores a single path-test outcome."""
    def __init__(self, path, resource, key_sent, key_expected,
                 action_val, received_power, soc_before, soc_after, passed):
        self.path = path
        self.resource = resource
        self.key_sent = key_sent
        self.key_expected = key_expected
        self.action_val = action_val
        self.received_power = received_power
        self.soc_before = soc_before
        self.soc_after = soc_after
        self.passed = passed


def _test_direct_key(ts, key: str, power: float) -> _Result:
    """Send action via dict with given *key* directly to BatteryEnv."""
    grid, bat = _battery_on_grid(ts)
    grid.reset(seed=0)
    soc_before = bat.soc
    bat.step({key: power})
    soc_after = bat.soc
    received = bat.current_p_mw
    ok = abs(received) > 1e-6 if abs(power) > 1e-6 else True
    return _Result(
        path=f"direct dict {{'{key}': {power}}}",
        resource="BatteryEnv",
        key_sent=key,
        key_expected="p_mw",
        action_val=power,
        received_power=received,
        soc_before=soc_before,
        soc_after=soc_after,
        passed=ok,
    )


def _test_direct_float(ts, power: float) -> _Result:
    """Send a raw float to BatteryEnv (should use scalar path)."""
    grid, bat = _battery_on_grid(ts)
    grid.reset(seed=0)
    soc_before = bat.soc
    bat.step(power)
    soc_after = bat.soc
    received = bat.current_p_mw
    ok = abs(received) > 1e-6 if abs(power) > 1e-6 else True
    return _Result(
        path=f"direct float ({power})",
        resource="BatteryEnv",
        key_sent="(scalar)",
        key_expected="(scalar)",
        action_val=power,
        received_power=received,
        soc_before=soc_before,
        soc_after=soc_after,
        passed=ok,
    )


def _test_direct_ndarray(ts, power: float) -> _Result:
    """Send a numpy array to BatteryEnv (should use ndarray path)."""
    grid, bat = _battery_on_grid(ts)
    grid.reset(seed=0)
    soc_before = bat.soc
    bat.step(np.array([power]))
    soc_after = bat.soc
    received = bat.current_p_mw
    ok = abs(received) > 1e-6 if abs(power) > 1e-6 else True
    return _Result(
        path=f"direct ndarray ([{power}])",
        resource="BatteryEnv",
        key_sent="(ndarray)",
        key_expected="(ndarray)",
        action_val=power,
        received_power=received,
        soc_before=soc_before,
        soc_after=soc_after,
        passed=ok,
    )


def _test_cost_based_market(ts) -> _Result:
    """CostBasedMarketEnv → BatteryEnv path."""
    try:
        env = make_cost_based_market(
            battery_capacity_mwh=50, battery_power_mw=20,
        )
        env.reset(seed=0)
        bat = env._battery
        soc_before = bat.soc if bat else None
        action = np.array([10.0])
        env.step(action)
        soc_after = bat.soc if bat else None
        received = bat.current_p_mw if bat else 0.0
        ok = abs(received) > 1e-6
        return _Result(
            path="CostBasedMarketEnv → BatteryEnv",
            resource="BatteryEnv",
            key_sent="p",
            key_expected="p_mw",
            action_val=10.0,
            received_power=received,
            soc_before=soc_before,
            soc_after=soc_after,
            passed=ok,
        )
    except Exception as exc:
        return _Result(
            path="CostBasedMarketEnv → BatteryEnv",
            resource="BatteryEnv",
            key_sent="p",
            key_expected="p_mw",
            action_val=10.0,
            received_power=0.0,
            soc_before=None,
            soc_after=None,
            passed=False,
        )


def _test_flatten_wrapper(ts) -> _Result:
    """FlattenWrapper → BatteryEnv path (checks key in _unflatten_action)."""
    try:
        from powerzoo.wrappers.flatten import FlattenWrapper
        from powerzoo.envs.power_env import PowerEnv

        config = {
            "name": "test_flatten",
            "grid": {"grid_type": "transmission", "delta_t_minutes": 30,
                     "time_series": ts},
            "resources": [
                {"type": "battery", "bus_id": 2,
                 "capacity_mwh": 50, "power_mw": 20}
            ],
        }
        base = PowerEnv(config)
        env = FlattenWrapper(base)
        env.reset(seed=0)

        bat_id = None
        for rid, res in base.resources.items():
            if "battery" in type(res).__name__.lower():
                bat_id = rid
                break

        bat = base.resources[bat_id] if bat_id else None
        soc_before = bat.soc if bat else None

        action = env.action_space.sample()
        action[:] = 10.0
        env.step(action)

        soc_after = bat.soc if bat else None
        received = bat.current_p_mw if bat else 0.0
        ok = abs(received) > 1e-6
        return _Result(
            path="FlattenWrapper → BatteryEnv",
            resource="BatteryEnv",
            key_sent="p (hardcoded in flatten.py)",
            key_expected="p_mw",
            action_val=10.0,
            received_power=received,
            soc_before=soc_before,
            soc_after=soc_after,
            passed=ok,
        )
    except Exception as exc:
        return _Result(
            path="FlattenWrapper → BatteryEnv",
            resource="BatteryEnv",
            key_sent="p",
            key_expected="p_mw",
            action_val=10.0,
            received_power=0.0,
            soc_before=None,
            soc_after=None,
            passed=False,
        )


def _test_marl_wrapper(ts) -> _Result:
    """MARLWrapper (resources mode) → BatteryEnv path."""
    try:
        from powerzoo.wrappers.marl_wrapper import MARLWrapper
        grid = make_trans_env(time_series=ts)
        bat = make_battery(parent=grid, bus_id=2, capacity_mwh=50, power_mw=20)
        env = MARLWrapper(grid, agent_type="resources")
        env.reset(seed=0)

        soc_before = bat.soc
        actions = {a: np.array([10.0]) for a in env.agents}
        env.step(actions)
        soc_after = bat.soc
        received = bat.current_p_mw
        ok = abs(received) > 1e-6
        return _Result(
            path="MARLWrapper(resources) → BatteryEnv",
            resource="BatteryEnv",
            key_sent="p",
            key_expected="p_mw",
            action_val=10.0,
            received_power=received,
            soc_before=soc_before,
            soc_after=soc_after,
            passed=ok,
        )
    except Exception as exc:
        return _Result(
            path="MARLWrapper(resources) → BatteryEnv",
            resource="BatteryEnv",
            key_sent="p",
            key_expected="p_mw",
            action_val=10.0,
            received_power=0.0,
            soc_before=None,
            soc_after=None,
            passed=False,
        )


def _test_grid_step_dict(ts) -> _Result:
    """GridEnv.step({res_id: {'p_mw': val}}) — correct key through grid."""
    grid, bat = _battery_on_grid(ts)
    grid.reset(seed=0)
    soc_before = bat.soc
    grid.step({bat.resource_id: {"p_mw": 10.0}})
    soc_after = bat.soc
    received = bat.current_p_mw
    ok = abs(received) > 1e-6
    return _Result(
        path="GridEnv.step({res_id: {'p_mw': 10}})",
        resource="BatteryEnv",
        key_sent="p_mw",
        key_expected="p_mw",
        action_val=10.0,
        received_power=received,
        soc_before=soc_before,
        soc_after=soc_after,
        passed=ok,
    )


def _test_grid_step_wrong_key(ts) -> _Result:
    """GridEnv.step({res_id: {'p': val}}) — wrong key, expect silent failure."""
    grid, bat = _battery_on_grid(ts)
    grid.reset(seed=0)
    soc_before = bat.soc
    grid.step({bat.resource_id: {"p": 10.0}})
    soc_after = bat.soc
    received = bat.current_p_mw
    silently_ignored = abs(received) < 1e-6
    return _Result(
        path="GridEnv.step({res_id: {'p': 10}}) [BUG PATH]",
        resource="BatteryEnv",
        key_sent="p",
        key_expected="p_mw",
        action_val=10.0,
        received_power=received,
        soc_before=soc_before,
        soc_after=soc_after,
        passed=not silently_ignored,  # FAIL expected: action should NOT be ignored
    )


# ── main test ────────────────────────────────────────────────────────

@pytest.mark.functional
def test_action_key_consistency(summary):
    ts = simple_time_series()

    results: list[_Result] = [
        _test_direct_key(ts, "p_mw", 10.0),
        _test_direct_float(ts, 10.0),
        _test_direct_ndarray(ts, 10.0),
        _test_cost_based_market(ts),
        _test_flatten_wrapper(ts),
        _test_marl_wrapper(ts),
        _test_grid_step_dict(ts),
    ]

    # Defensive tests: wrong key is expected to be silently ignored
    wrong_key_results: list[_Result] = [
        _test_direct_key(ts, "p", 10.0),
        _test_grid_step_wrong_key(ts),
    ]

    # ── build report ─────────────────────────────────────────────
    lines = [
        "=" * 60,
        "  Test 01: action key consistency",
        "=" * 60,
        f"Date: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}",
        "",
        "[Formal paths]",
        f"{'path':<50} {'sent_key':<12} {'expect_key':<10} {'action':<8} "
        f"{'received':<10} {'SOC_0':<8} {'SOC_1':<8} {'result':<6}",
        "-" * 120,
    ]
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        soc_b = f"{r.soc_before:.4f}" if r.soc_before is not None else "N/A"
        soc_a = f"{r.soc_after:.4f}" if r.soc_after is not None else "N/A"
        lines.append(
            f"{r.path:<50} {r.key_sent:<12} {r.key_expected:<10} "
            f"{r.action_val:<8.1f} {r.received_power:<10.4f} "
            f"{soc_b:<8} {soc_a:<8} {tag:<6}"
        )

    fails = [r for r in results if not r.passed]
    lines.append("")
    lines.append(f"Formal paths passed: {len(results) - len(fails)}/{len(results)}")

    if fails:
        lines.append("")
        lines.append("[Failed paths]")
        for r in fails:
            lines.append(f"  ✗ {r.path}")
            lines.append(f"    sent key='{r.key_sent}' but resource expects key='{r.key_expected}'")
            lines.append(f"    received_power={r.received_power:.4f} "
                         f"(SOC: {r.soc_before} → {r.soc_after})")

    # wrong-key defensive checks
    lines.append("")
    lines.append("[Defense: wrong keys should be silently ignored]")
    for r in wrong_key_results:
        ignored = abs(r.received_power) < 1e-6
        tag = "OK (ignored)" if ignored else "UNEXPECTED (accepted)"
        lines.append(f"  {r.path}: {tag}")

    write_report(CAT, lines)

    # ── heatmap (only formal paths) ──────────────────────────────
    all_r = results + wrong_key_results
    paths = [r.path for r in all_r]
    status = []
    for r in all_r:
        if r in wrong_key_results:
            status.append(0.5)  # gray for defensive
        elif r.passed:
            status.append(1.0)
        else:
            status.append(0.0)

    fig, ax = plt.subplots(figsize=(8, max(3, len(paths) * 0.45)))
    colors_arr = np.array(status).reshape(-1, 1)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#e74c3c", "#95a5a6", "#2ecc71"])
    ax.imshow(colors_arr, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(paths)))
    ax.set_yticklabels(paths, fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels(["BatteryEnv"])
    ax.set_title("Action Key Consistency (green=PASS, red=FAIL, gray=defensive)")
    for i, s in enumerate(status):
        label = "PASS" if s == 1.0 else ("FAIL" if s == 0.0 else "N/A")
        ax.text(0, i, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
    save_figure(fig, CAT, "action_key_matrix.png")

    # ── summary ──────────────────────────────────────────────────
    if fails:
        summary.add(CAT, "Action key consistency",
                     "FAIL",
                     f"{len(fails)} formal path(s): action not delivered correctly")
    else:
        summary.add(CAT, "Action key consistency", "PASS",
                     "all wrapper/adapter paths share keys; actions applied correctly")
