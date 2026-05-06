"""Test 05 — Observation normalization consistency.

Checks that NormalizationWrapper produces obs in [-1, 1], that raw obs
are in reasonable physical ranges, and that observations are reproducible
given the same seed.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .conftest import (
    make_trans_env, simple_time_series, save_figure, write_report,
)

CAT = "05_normalization"
N_STEPS = 48


def _collect_obs(env, n_steps=N_STEPS):
    """Run *n_steps* and return (n_steps, obs_dim) array."""
    obs_list = []
    env.reset(seed=42)
    obs_list.append(env.obs(None))
    for _ in range(n_steps):
        state, *_ = env.step({})
        obs_list.append(env.obs(state))
    return np.array(obs_list)


@pytest.mark.functional
def test_normalization(summary):
    ts = simple_time_series()

    # ── raw observations ─────────────────────────────────────────
    raw_env = make_trans_env(time_series=ts, physics="dc", solver_mode="opf")
    raw_obs = _collect_obs(raw_env)
    obs_names = getattr(raw_env, "obs_names", [f"dim_{i}" for i in range(raw_obs.shape[1])])

    # ── normalized observations ──────────────────────────────────
    from powerzoo.wrappers.gym_wrappers import GymnasiumWrapper, NormalizationWrapper
    norm_inner = GymnasiumWrapper(
        make_trans_env(time_series=ts, physics="dc", solver_mode="opf")
    )
    norm_env = NormalizationWrapper(norm_inner)
    norm_env.reset(seed=42)
    norm_list = [norm_env.observation(raw_obs[0])]
    for i in range(1, len(raw_obs)):
        norm_list.append(norm_env.observation(raw_obs[i]))
    norm_obs = np.array(norm_list)

    # ── reproducibility ──────────────────────────────────────────
    raw_env2 = make_trans_env(time_series=ts, physics="dc", solver_mode="opf")
    raw_obs2 = _collect_obs(raw_env2)
    repro_diff = np.abs(raw_obs - raw_obs2[:len(raw_obs)])

    # ── statistics ───────────────────────────────────────────────
    lines = [
        "=" * 60,
        "  Test 05: observation normalization consistency",
        "=" * 60, "",
        f"obs dim: {raw_obs.shape[1]}",
        f"episode steps: {raw_obs.shape[0]}",
        "",
        "[Raw observation stats]",
        f"  {'dim':<30} {'min':>10} {'max':>10} {'mean':>10} {'std':>10}",
        "  " + "-" * 70,
    ]
    for i in range(raw_obs.shape[1]):
        name = obs_names[i] if i < len(obs_names) else f"dim_{i}"
        col = raw_obs[:, i]
        lines.append(f"  {name:<30} {col.min():>10.4f} {col.max():>10.4f} "
                     f"{col.mean():>10.4f} {col.std():>10.4f}")

    lines.append("")
    lines.append("[Normalized observation stats]")
    in_range = (norm_obs >= -1.0 - 1e-6) & (norm_obs <= 1.0 + 1e-6)
    pct_in = float(in_range.mean()) * 100
    lines.append(f"  fraction of values in [-1, 1]: {pct_in:.2f}%")
    out_dims = np.where(~in_range.all(axis=0))[0]
    if len(out_dims) > 0:
        lines.append(f"  dims outside [-1,1]: {out_dims.tolist()}")
        for d in out_dims[:5]:
            name = obs_names[d] if d < len(obs_names) else f"dim_{d}"
            lines.append(f"    {name}: [{norm_obs[:, d].min():.4f}, {norm_obs[:, d].max():.4f}]")
    lines.append(f"  normalized in [-1,1]: {'PASS' if pct_in > 99 else 'WARN'}")

    lines.append("")
    lines.append("[Reproducibility]")
    lines.append(f"  max|diff| across two runs (same seed): {repro_diff.max():.2e}")
    repro_ok = repro_diff.max() < 1e-6
    lines.append(f"  reproducible: {'PASS' if repro_ok else 'FAIL'}")

    write_report(CAT, lines)

    # ── boxplot: obs distribution per dimension ──────────────────
    n_dims = raw_obs.shape[1]
    fig, ax = plt.subplots(figsize=(max(10, n_dims * 0.5), 5))
    short_names = [n[:15] for n in obs_names[:n_dims]]
    bp = ax.boxplot(norm_obs, vert=True, patch_artist=True, tick_labels=short_names,
                    showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.5)
    ax.axhline(-1, color="red", ls="--", lw=0.8)
    ax.axhline(1, color="red", ls="--", lw=0.8)
    ax.set_ylabel("Normalized value")
    ax.set_title("Normalized Observation Distribution (per dimension)")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    fig.tight_layout()
    save_figure(fig, CAT, "obs_distribution.png")

    # ── heatmap: obs value over time ─────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, max(4, n_dims * 0.3)))
    im = ax2.imshow(norm_obs.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Obs dimension")
    ax2.set_yticks(range(n_dims))
    ax2.set_yticklabels(short_names, fontsize=7)
    ax2.set_title("Normalized Observation Heatmap (dim × time)")
    fig2.colorbar(im, ax=ax2)
    fig2.tight_layout()
    save_figure(fig2, CAT, "obs_range_heatmap.png")

    # ── reproducibility diff ─────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.imshow(repro_diff.T, aspect="auto", cmap="hot", vmin=0,
               vmax=max(repro_diff.max(), 1e-8))
    ax3.set_xlabel("Time step")
    ax3.set_ylabel("Obs dimension")
    ax3.set_title(f"Reproducibility |diff| (max={repro_diff.max():.2e})")
    fig3.colorbar(ax3.images[0], ax=ax3)
    fig3.tight_layout()
    save_figure(fig3, CAT, "reproducibility_diff.png")

    # ── summary ──────────────────────────────────────────────────
    if pct_in < 95:
        summary.add(CAT, "Normalization consistency", "FAIL",
                     f"only {pct_in:.1f}% of normalized obs in [-1,1]")
    elif pct_in < 99:
        summary.add(CAT, "Normalization consistency", "WARN",
                     f"{pct_in:.1f}% obs in [-1,1], some dims out of range")
    elif not repro_ok:
        summary.add(CAT, "Normalization consistency", "WARN",
                     "obs not exactly reproducible (same seed)")
    else:
        summary.add(CAT, "Normalization consistency", "PASS",
                     "obs 100% in [-1,1], reproducible")
