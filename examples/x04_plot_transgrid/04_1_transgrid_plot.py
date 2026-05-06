"""Minimal example: transmission grid topology (IEEE 14-bus).

Uses :class:`powerzoo.case.CasePlotter` on a built-in **transmission** case.
No power-flow colormap — topology only.

Note: PowerZoo ships Case5 / Case14 / … / Case300 (IEEE 300-bus), not an
IEEE-30 case. This script picks Case14 as a small HV benchmark.
"""
import os
import sys

sys.path.insert(0, ".")

import matplotlib.pyplot as plt

from powerzoo.case import load_case

OUTPUT_DIR = os.path.dirname(__file__)


def main():
    case = load_case("Case14", grid_type="transmission")
    assert getattr(case, "GRID_TYPE", "") == "transmission"

    plotter = case.plotter
    fig, ax = plotter.plot_topology(
        figsize=(9, 7),
        layout="kamada_kawai",
        title="IEEE 14-bus transmission — topology (no PF colormap)",
        highlight_slack=False,
        highlight_loads=False,
        show_annotations=False,
    )
    out_path = os.path.join(OUTPUT_DIR, "04_1_topology.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
