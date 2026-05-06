"""Test 07 — Generation cost correctness.

Verifies that mc_a/mc_b/mc_c data and cost calculations are consistent
with MATPOWER gencost across all transmission cases, and that the PF-mode
reward reflects the correct total cost.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .conftest import (
    make_trans_env, simple_time_series,
    report_dir, save_figure, write_report, write_json,
)

CAT = "07_cost_correctness"

# MATPOWER reference: TC(p) = c2·p² + c1·p + c0
# PowerZoo MC convention: MC(p) = mc_a·p² + mc_b·p + mc_c
# Mapping: mc_a=0(for quad TC), mc_b=2·c2, mc_c=c1
# TC(p) = ∫MC = mc_a/3·p³ + mc_b/2·p² + mc_c·p

MATPOWER_GENCOST = {
    5: [
        # (unit_idx, c2, c1, c0)
        (0, 0.0, 14.0, 0.0),
        (1, 0.0, 15.0, 0.0),
        (2, 0.0, 30.0, 0.0),
        (3, 0.0, 40.0, 0.0),
        (4, 0.0, 10.0, 0.0),
    ],
    14: [
        (0, 0.0430293, 20.0, 0.0),
        (1, 0.25,      20.0, 0.0),
        (2, 0.01,      40.0, 0.0),
        (3, 0.01,      40.0, 0.0),
        (4, 0.01,      40.0, 0.0),
    ],
}


def _tc_matpower(c2, c1, c0, p):
    """MATPOWER total cost: TC = c2·p² + c1·p + c0."""
    return c2 * p**2 + c1 * p + c0


def _tc_powerzoo(mc_a, mc_b, mc_c, p):
    """PowerZoo total cost (integral of MC): mc_a/3·p³ + mc_b/2·p² + mc_c·p."""
    return (mc_a / 3) * p**3 + (mc_b / 2) * p**2 + mc_c * p


def _mc_from_matpower(c2, c1):
    """Convert MATPOWER TC coefficients to PowerZoo MC convention."""
    return 0.0, 2.0 * c2, c1  # mc_a, mc_b, mc_c


@pytest.mark.functional
def test_cost_correctness(summary):
    from powerzoo.case import load_case

    lines = [
        "=" * 60,
        "  Test 07: generation cost correctness",
        "=" * 60, "",
    ]

    all_ok = True
    case_results = {}

    # ── Part 1: mc data vs MATPOWER gencost ──────────────────────
    lines.append("[1] mc_a/mc_b/mc_c vs MATPOWER gencost")
    lines.append("")

    for case_id, ref_units in MATPOWER_GENCOST.items():
        c = load_case(case_id)
        lines.append(f"  [Case{case_id}]")
        case_ok = True
        data_rows = []

        for unit_idx, c2, c1, c0 in ref_units:
            u = c.units.iloc[unit_idx]
            mc_a_expected, mc_b_expected, mc_c_expected = _mc_from_matpower(c2, c1)

            mc_a_actual = u['mc_a']
            mc_b_actual = u['mc_b']
            mc_c_actual = u['mc_c']

            a_ok = abs(mc_a_actual - mc_a_expected) < 1e-6
            b_ok = abs(mc_b_actual - mc_b_expected) < 1e-4
            c_ok = abs(mc_c_actual - mc_c_expected) < 1e-4

            tag = "PASS" if (a_ok and b_ok and c_ok) else "FAIL"
            if tag == "FAIL":
                case_ok = False
                all_ok = False

            data_rows.append({
                "unit": int(u['#id']),
                "c2": c2, "c1": c1, "c0": c0,
                "mc_a_exp": mc_a_expected, "mc_b_exp": mc_b_expected, "mc_c_exp": mc_c_expected,
                "mc_a_act": mc_a_actual, "mc_b_act": mc_b_actual, "mc_c_act": mc_c_actual,
                "ok": tag,
            })

            lines.append(
                f"    Gen{int(u['#id'])}: MATPOWER(c2={c2}, c1={c1}) "
                f"→ expect mc_b={mc_b_expected:.4f}, mc_c={mc_c_expected} "
                f"| actual mc_b={mc_b_actual:.4f}, mc_c={mc_c_actual} [{tag}]"
            )

        case_results[case_id] = {"data_ok": case_ok, "rows": data_rows}
        lines.append(f"    → {'PASS' if case_ok else 'FAIL'}")
        lines.append("")

    # ── Part 2: TC computation at test points ────────────────────
    lines.append("[2] Total cost TC vs MATPOWER reference")
    lines.append("")

    tc_errors = []
    for case_id, ref_units in MATPOWER_GENCOST.items():
        c = load_case(case_id)
        lines.append(f"  [Case{case_id}]")

        for unit_idx, c2, c1, c0 in ref_units:
            u = c.units.iloc[unit_idx]
            p_test = u['p_max'] / 2

            tc_mp = _tc_matpower(c2, c1, c0, p_test)
            tc_pz = _tc_powerzoo(u['mc_a'], u['mc_b'], u['mc_c'], p_test)

            err = abs(tc_pz - tc_mp)
            rel = err / max(abs(tc_mp), 1e-6) * 100
            tc_errors.append(rel)

            ok = rel < 0.01
            if not ok:
                all_ok = False

            lines.append(
                f"    Gen{int(u['#id'])} @ {p_test:.0f}MW: "
                f"MATPOWER TC={tc_mp:.2f}, PowerZoo TC={tc_pz:.2f}, "
                f"err={rel:.4f}% [{'PASS' if ok else 'FAIL'}]"
            )
        lines.append("")

    # ── Part 3: PF reward reflects cost ──────────────────────────
    lines.append("[3] PF-mode reward reflects generation cost")
    lines.append("")
    ts = simple_time_series()

    for case_id in [5, 14]:
        c = load_case(case_id)
        env = make_trans_env(
            case=c, time_series=ts, physics='dc', solver_mode='pf',
            normalize_actions=False,
        )
        env.reset(seed=42)
        p = (c.units['p_min'].values + c.units['p_max'].values) / 2
        _, reward, _, _, info = env.step({'unit_power_mw': p})

        mc_a = c.units['mc_a'].values
        mc_b = c.units['mc_b'].values
        mc_c = c.units['mc_c'].values
        expected_tc = float(((mc_a / 3) * p**3 + (mc_b / 2) * p**2 + mc_c * p).sum())
        expected_reward = -0.01 * expected_tc

        r_ok = abs(reward - expected_reward) < 1e-4
        if not r_ok:
            all_ok = False

        lines.append(
            f"  Case{case_id}: dispatch={p.sum():.0f}MW, "
            f"TC={expected_tc:.1f}, reward={reward:.4f}, "
            f"expected={expected_reward:.4f} [{'PASS' if r_ok else 'FAIL'}]"
        )

    lines.append("")

    # ── Part 4: OPF cheaper than random PF ───────────────────────
    lines.append("[4] OPF cost ≤ random PF cost (economic dispatch sanity)")
    lines.append("")

    for case_id in [5, 14]:
        c = load_case(case_id)
        env_opf = make_trans_env(
            case=c, time_series=ts, physics='dc', solver_mode='opf',
            normalize_actions=False,
        )
        env_opf.reset(seed=42)
        _, r_opf, _, _, info_opf = env_opf.step({})
        opf_cost = info_opf.get('opf_cost', 0)

        env_pf = make_trans_env(
            case=c, time_series=ts, physics='dc', solver_mode='pf',
            normalize_actions=False,
        )
        env_pf.reset(seed=42)
        rng = np.random.default_rng(123)
        p_rand = c.units['p_min'].values + rng.random(len(c.units)) * (
            c.units['p_max'].values - c.units['p_min'].values
        )
        _, r_rand, _, _, _ = env_pf.step({'unit_power_mw': p_rand})
        mc_a = c.units['mc_a'].values
        mc_b = c.units['mc_b'].values
        mc_c = c.units['mc_c'].values
        rand_cost = float(((mc_a / 3) * p_rand**3 + (mc_b / 2) * p_rand**2 + mc_c * p_rand).sum())

        econ_ok = opf_cost <= rand_cost + 1e-6
        if not econ_ok:
            all_ok = False

        lines.append(
            f"  Case{case_id}: OPF cost={opf_cost:.1f}, "
            f"random PF cost={rand_cost:.1f}, "
            f"OPF≤random? {'PASS' if econ_ok else 'FAIL'}"
        )
    lines.append("")

    write_report(CAT, lines)

    # ── statistics ───────────────────────────────────────────────
    write_json(CAT, "statistics.json", {
        "tc_relative_errors_pct": tc_errors,
        "max_tc_error_pct": max(tc_errors) if tc_errors else 0,
        "all_pass": all_ok,
    })

    # ── cost curve plot ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, case_id in zip(axes, [5, 14]):
        c = load_case(case_id)
        for i, u in c.units.iterrows():
            p_range = np.linspace(0, u['p_max'], 100)
            mc_vals = u['mc_a'] * p_range**2 + u['mc_b'] * p_range + u['mc_c']
            ax.plot(p_range, mc_vals, label=f"Gen{int(u['#id'])}")
        ax.set_xlabel("P (MW)")
        ax.set_ylabel("MC ($/MWh)")
        ax.set_title(f"Case{case_id} Marginal Cost Curves")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, CAT, "marginal_cost_curves.png")

    # TC comparison plot
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    for ax, case_id in zip(axes2, [5, 14]):
        ref = MATPOWER_GENCOST[case_id]
        c = load_case(case_id)
        for unit_idx, c2, c1, c0 in ref:
            u = c.units.iloc[unit_idx]
            p_range = np.linspace(0, u['p_max'], 100)
            tc_mp = c2 * p_range**2 + c1 * p_range + c0
            tc_pz = (u['mc_a']/3)*p_range**3 + (u['mc_b']/2)*p_range**2 + u['mc_c']*p_range
            ax.plot(p_range, tc_mp, '--', alpha=0.6, label=f"Gen{int(u['#id'])} MATPOWER")
            ax.plot(p_range, tc_pz, '-', alpha=0.8, label=f"Gen{int(u['#id'])} PowerZoo")
        ax.set_xlabel("P (MW)")
        ax.set_ylabel("TC ($)")
        ax.set_title(f"Case{case_id} Total Cost: MATPOWER vs PowerZoo")
        ax.legend(fontsize=7)
    fig2.tight_layout()
    save_figure(fig2, CAT, "total_cost_comparison.png")

    # ── summary ──────────────────────────────────────────────────
    if all_ok:
        summary.add(CAT, "Generation cost correctness", "PASS",
                     "mc matches MATPOWER; TC error < 0.01%; PF reward consistent")
    else:
        summary.add(CAT, "Generation cost correctness", "FAIL",
                     "mc data or TC inconsistent with MATPOWER")
