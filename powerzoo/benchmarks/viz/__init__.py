"""PowerZoo — lightweight plotting utilities for notebooks.

All functions produce matplotlib figures and return the Figure object
so callers can further customise or save them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

__all__ = ["plot_episode", "plot_dispatch", "plot_eval_comparison"]

_STYLE = "seaborn-v0_8-whitegrid"


def plot_episode(
    rewards: Sequence[float],
    costs: Optional[Sequence[float]] = None,
    safety: Optional[Sequence[bool]] = None,
    *,
    title: str = "Episode trajectory",
    figsize: tuple = (8, 3),
) -> plt.Figure:
    """Two-panel plot: per-step reward curve and (optionally) cost / safety."""
    with plt.style.context(_STYLE):
        n_panels = 1 + int(costs is not None or safety is not None)
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, constrained_layout=True)
        if n_panels == 1:
            axes = [axes]
        axes[0].plot(rewards, color="steelblue", linewidth=1.2)
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Reward")
        axes[0].set_title(title)
        if n_panels > 1:
            ax = axes[1]
            if costs is not None:
                ax.plot(costs, color="coral", linewidth=1.2, label="cost")
                ax.set_ylabel("Cost")
            if safety is not None:
                ax2 = ax.twinx()
                ax2.fill_between(range(len(safety)), safety, alpha=0.15, color="green", label="safe")
                ax2.set_ylabel("Safe")
                ax2.set_ylim(-0.1, 1.1)
            ax.set_xlabel("Step")
            ax.set_title("Cost / Safety")
            ax.legend(loc="upper right", fontsize=8)
    return fig


def plot_dispatch(
    unit_power_mws: np.ndarray,
    unit_labels: Optional[List[str]] = None,
    load: Optional[Sequence[float]] = None,
    *,
    title: str = "Generator dispatch",
    figsize: tuple = (8, 3),
) -> plt.Figure:
    """Stacked area chart of generator dispatch vs total load."""
    with plt.style.context(_STYLE):
        powers = np.asarray(unit_power_mws)  # (T, n_units)
        T, n = powers.shape
        labels = unit_labels or [f"unit_{i}" for i in range(n)]
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.stackplot(range(T), powers.T, labels=labels, alpha=0.75)
        if load is not None:
            ax.plot(load, color="black", linewidth=1.5, linestyle="--", label="Load")
        ax.set_xlabel("Step")
        ax.set_ylabel("Power (MW)")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7, ncol=min(n + 1, 4))
    return fig


def plot_eval_comparison(
    results: Dict[str, Dict[str, Any]],
    *,
    metric: str = "mean_reward",
    title: str = "Policy comparison",
    figsize: tuple = (6, 3.5),
) -> plt.Figure:
    """Bar chart comparing evaluation results across policies."""
    with plt.style.context(_STYLE):
        names = list(results.keys())
        values = [results[n][metric] for n in names]
        stds = [results[n].get("std_reward", 0) for n in names]
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        bars = ax.bar(names, values, yerr=stds, capsize=4, color=colors, edgecolor="grey")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    return fig
