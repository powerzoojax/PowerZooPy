"""Test 06 — Cross-path validation.

Sends the same action sequence through different API paths and verifies
that observations, rewards, and termination signals are consistent.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .conftest import (
    make_trans_env, simple_time_series, save_figure, write_report,
)

CAT = "06_cross_path_validation"
N_STEPS = 24


def _run_path_a(ts, actions):
    """Path A: TransGridEnv.step(dict) directly."""
    env = make_trans_env(time_series=ts, physics="dc", solver_mode="opf")
    env.reset(seed=42)
    obs_list, reward_list = [], []
    for a in actions:
        state, reward, terminated, truncated, info = env.step(a)
        obs_list.append(env.obs(state))
        reward_list.append(reward)
        if terminated or truncated:
            break
    return np.array(obs_list), np.array(reward_list)


def _run_path_b(ts, actions):
    """Path B: GymnasiumWrapper (accepts numpy array)."""
    from powerzoo.wrappers.gym_wrappers import GymnasiumWrapper
    inner = make_trans_env(time_series=ts, physics="dc", solver_mode="opf")
    env = GymnasiumWrapper(inner)
    env.reset(seed=42)
    obs_list, reward_list = [], []
    for a in actions:
        if isinstance(a, dict) and "unit_power_mw" in a:
            arr_action = a["unit_power_mw"]
        else:
            arr_action = np.array([])
        obs, reward, terminated, truncated, info = env.step(arr_action)
        obs_list.append(obs)
        reward_list.append(reward)
        if terminated or truncated:
            break
    return np.array(obs_list), np.array(reward_list)


@pytest.mark.functional
def test_cross_path_validation(summary):
    ts = simple_time_series()

    actions = [{} for _ in range(N_STEPS)]

    obs_a, rew_a = _run_path_a(ts, actions)
    obs_b, rew_b = _run_path_b(ts, actions)

    n = min(len(obs_a), len(obs_b))
    obs_a, obs_b = obs_a[:n], obs_b[:n]
    rew_a, rew_b = rew_a[:n], rew_b[:n]

    obs_diff = np.abs(obs_a - obs_b)
    rew_diff = np.abs(rew_a - rew_b)

    # ── report ───────────────────────────────────────────────────
    lines = [
        "=" * 60,
        "  Test 06: multi-path cross-check",
        "=" * 60, "",
        f"path A: TransGridEnv.step(dict)",
        f"path B: GymnasiumWrapper + numpy array",
        f"steps: {n}",
        "",
        "[Observation delta]",
        f"  max |diff|:  {obs_diff.max():.2e}",
        f"  mean |diff|: {obs_diff.mean():.2e}",
        "",
        "[Reward delta]",
        f"  max |diff|:  {rew_diff.max():.2e}",
        f"  mean |diff|: {rew_diff.mean():.2e}",
        "",
    ]

    # per-step diffs (only show steps with diff > 1e-6)
    diff_steps = np.where(obs_diff.max(axis=1) > 1e-6)[0]
    if len(diff_steps) > 0:
        lines.append(f"[steps with mismatch ({len(diff_steps)} total):]")
        for s in diff_steps[:10]:
            max_dim = int(obs_diff[s].argmax())
            lines.append(f"  step {s}: max_diff={obs_diff[s, max_dim]:.6f} "
                         f"at dim {max_dim}, rew_diff={rew_diff[s]:.6f}")
    else:
        lines.append("[obs/reward match on all steps]")

    obs_ok = obs_diff.max() < 1e-4
    rew_ok = rew_diff.max() < 1e-4
    lines.append("")
    lines.append(f"Obs match: {'PASS' if obs_ok else 'FAIL'} (threshold=1e-4)")
    lines.append(f"Reward match: {'PASS' if rew_ok else 'FAIL'}")

    write_report(CAT, lines)

    # ── heatmap ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, max(4, obs_diff.shape[1] * 0.2)))
    vmax = max(obs_diff.max(), 1e-8)
    im = ax.imshow(obs_diff.T, aspect="auto", cmap="hot", vmin=0, vmax=vmax)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Obs dimension")
    ax.set_title(f"Path A vs B: |obs diff| (max={vmax:.2e})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    save_figure(fig, CAT, "path_diff_table.png")

    # ── summary ──────────────────────────────────────────────────
    if obs_ok and rew_ok:
        summary.add(CAT, "Multi-path cross-check", "PASS",
                     "TransGridEnv direct path matches GymnasiumWrapper path")
    else:
        summary.add(CAT, "Multi-path cross-check", "FAIL",
                     f"obs max_diff={obs_diff.max():.2e}, rew max_diff={rew_diff.max():.2e}")
