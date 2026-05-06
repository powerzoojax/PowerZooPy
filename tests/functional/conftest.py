"""Shared fixtures and reporting utilities for functional tests."""

import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

REPORTS_ROOT = Path(__file__).resolve().parent.parent / "reports"

# ── matplotlib defaults ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
})


# ── directory helpers ────────────────────────────────────────────────

def report_dir(category: str) -> Path:
    """Return (and create) the report directory for *category*."""
    d = REPORTS_ROOT / category
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_figure(fig: plt.Figure, category: str, filename: str) -> Path:
    """Save a figure into ``reports/<category>/<filename>`` and close it."""
    p = report_dir(category) / filename
    fig.savefig(p)
    plt.close(fig)
    return p


def write_report(category: str, lines: List[str]) -> Path:
    """Write ``report.txt`` for *category*."""
    p = report_dir(category) / "report.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def write_json(category: str, filename: str, data: Any) -> Path:
    p = report_dir(category) / filename
    p.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    return p


# ── environment construction helpers ─────────────────────────────────

def make_trans_env(physics="dc", solver_mode="opf", **kw):
    from powerzoo.envs.grid.trans import TransGridEnv
    return TransGridEnv(physics=physics, solver_mode=solver_mode, **kw)


def make_dist_env(**kw):
    from powerzoo.envs.grid.dist import DistGridEnv
    return DistGridEnv(**kw)


def make_battery(**kw):
    from powerzoo.envs.resource.battery import BatteryEnv
    return BatteryEnv(**kw)


def make_cost_based_market(**kw):
    from powerzoo.envs.market.cost_based_market import CostBasedMarketEnv
    return CostBasedMarketEnv(**kw)


def simple_time_series(n_steps: int = 96):
    """2-day, 30-min resolution demand/solar/wind time-series."""
    import pandas as pd
    from powerzoo.data import signals as S

    idx = pd.date_range("2024-01-01", periods=n_steps, freq="30min", tz="UTC")
    hours = idx.hour + idx.minute / 60.0
    demand = 400 + 200 * np.sin(2 * np.pi * (hours - 6) / 24.0)
    solar = np.maximum(0, np.sin(2 * np.pi * (hours - 6) / 24.0)) * 150
    wind = 50 + 30 * np.sin(2 * np.pi * hours / 12.0)
    return pd.DataFrame(
        {S.LOAD_ACTUAL_MW: demand, S.SOLAR_AVAILABLE_MW: solar, S.WIND_AVAILABLE_MW: wind},
        index=idx,
    )


# ── summary collector (session-scoped) ───────────────────────────────

class SummaryCollector:
    """Collects per-category verdicts and writes ``summary.txt`` at teardown."""

    def __init__(self):
        self.entries: List[Dict[str, str]] = []

    def add(self, category: str, title: str, verdict: str, detail: str = ""):
        self.entries.append(dict(category=category, title=title,
                                 verdict=verdict, detail=detail))

    def write(self):
        REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
        lines = [
            "=" * 60,
            "  PowerZoo functional test summary",
            "=" * 60,
            f"Date: {datetime.now():%Y-%m-%d %H:%M}",
            "",
        ]
        pass_n = sum(1 for e in self.entries if e["verdict"] == "PASS")
        fail_n = sum(1 for e in self.entries if e["verdict"] == "FAIL")
        warn_n = sum(1 for e in self.entries if e["verdict"] == "WARN")
        lines.append(f"[Overview]  PASS: {pass_n}  |  FAIL: {fail_n}  |  WARN: {warn_n}")
        lines.append("")

        for e in self.entries:
            tag = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}.get(e["verdict"], "?")
            lines.append(f"[{e['category']}] {e['title']}")
            lines.append(f"  {tag} {e['verdict']} — {e['detail']}")
            lines.append(f"  → See reports/{e['category']}/")
            lines.append("")

        p_issues = [e for e in self.entries if e["verdict"] == "FAIL"]
        w_issues = [e for e in self.entries if e["verdict"] == "WARN"]
        if p_issues or w_issues:
            lines.append("[Issue priority]")
            for i, e in enumerate(p_issues):
                lines.append(f"  P{i}: {e['detail']}")
            for i, e in enumerate(w_issues, start=len(p_issues)):
                lines.append(f"  P{i}: {e['detail']}")
        lines.append("")
        (REPORTS_ROOT / "summary.txt").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")


@pytest.fixture(scope="session")
def summary():
    """Session-wide summary collector — call ``summary.add(...)`` from tests."""
    c = SummaryCollector()
    yield c
    c.write()
