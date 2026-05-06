"""GB demand data cleaning — domain-aware outlier detection and repair.

Pipeline position:  source CSV  ->  **clean**  ->  parquet

Typical usage
-------------
    # CLI
    python -m powerzoo.data.utils.clean_gb_demand

    # Programmatic
    from powerzoo.data.utils.clean_gb_demand import clean_demand_dataframe
    df_clean = clean_demand_dataframe(df_raw)

Design rationale
----------------
GB system demand never drops below ~12 GW (summer night) and the biggest
legitimate half-hour swing is ~4 GW.  Values outside these bounds are
artefacts of NESO API outages that were back-filled with naive linear
interpolation in the source CSV.

We apply three complementary passes, each targeting a different failure mode:

1. **Hard floor** — any reading < FLOOR_MW is flagged (catches the worst
   artefacts at 750-5000 MW).
2. **Gradient cap** — any half-hour |change| > MAX_DELTA_MW is flagged
   (catches the "ramp into/out of missing block" pattern where the value is
   still above the floor but the slope is physically impossible).
3. **Short-window median** — any point deviating from a 2.5-hour centred
   rolling median by more than SHORT_MEDIAN_BAND_MW is flagged (catches
   moderate anomalies that pass the first two checks without being confused
   by normal daily demand cycles).

Flagged values are repaired by **linear interpolation** between the last
trustworthy neighbour and the next trustworthy neighbour.

All thresholds are conservative — they will **not** remove genuine demand
troughs (Christmas night, summer Sunday at 04:00, etc.).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Domain thresholds (GB-specific, conservative)
# ---------------------------------------------------------------------------
FLOOR_MW: float = 12_000.0
"""Absolute minimum credible GB demand (MW).

Historical record: ~14.5 GW on 2020-05-24 (COVID lockdown sunny Sunday).
Using 12 GW gives ample headroom for future records.
"""

MAX_DELTA_MW: float = 6_000.0
"""Maximum credible half-hour demand change (MW).

Largest observed legitimate swing is ~4 GW (evening ramp in winter).
6 GW provides a comfortable safety margin.
"""

SHORT_MEDIAN_WINDOW: int = 5
"""Short rolling-median window (number of half-hour periods).

5 periods = 2.5 hours.  Short enough to track the intra-day demand curve,
long enough to not be swayed by a single outlier.
"""

SHORT_MEDIAN_BAND_MW: float = 5_000.0
"""Maximum allowed deviation from the short-window rolling median (MW).

Normal demand changes within a 2.5-hour window are at most ~3 GW (steep
evening ramp).  5 GW catches anomalous points without false positives.
"""


# ---------------------------------------------------------------------------
# Core cleaning function
# ---------------------------------------------------------------------------
def clean_demand_dataframe(
    df: pd.DataFrame,
    demand_col: str = "ActualDemand",
    *,
    floor_mw: float = FLOOR_MW,
    max_delta_mw: float = MAX_DELTA_MW,
    short_median_window: int = SHORT_MEDIAN_WINDOW,
    short_median_band_mw: float = SHORT_MEDIAN_BAND_MW,
    verbose: bool = True,
) -> pd.DataFrame:
    """Clean a demand DataFrame in-place and return it.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *demand_col*.  A datetime column or DatetimeIndex is
        recommended but not required.
    demand_col : str
        Column name containing the demand values.
    floor_mw, max_delta_mw, short_median_window, short_median_band_mw
        Tuning knobs; see module-level docstrings.
    verbose : bool
        Print a summary of what was fixed.

    Returns
    -------
    pd.DataFrame
        The same object, modified in-place (also returned for chaining).
    """
    if demand_col not in df.columns:
        raise KeyError(f"Column '{demand_col}' not found in DataFrame")

    demand = df[demand_col].copy()
    n_total = len(demand)
    flag = np.zeros(n_total, dtype=bool)

    # --- Pass 1: hard floor ---
    floor_mask = demand < floor_mw
    flag |= floor_mask.values
    n_floor = int(floor_mask.sum())

    # --- Pass 2: gradient cap ---
    diff = demand.diff().abs()
    grad_mask = diff > max_delta_mw
    # Flag both the step-to point AND the step-from point (the ramp edges)
    grad_idx = np.where(grad_mask.values)[0]
    for i in grad_idx:
        lo = max(0, i - 1)
        hi = min(n_total, i + 2)
        flag[lo:hi] = True
    n_grad = int(grad_mask.sum())

    # --- Pass 3: short-window rolling-median proximity ---
    short_med = demand.rolling(
        short_median_window, center=True, min_periods=1
    ).median()
    median_mask = (demand - short_med).abs() > short_median_band_mw
    flag |= median_mask.values
    n_median = int(median_mask.sum())

    # Total unique flagged
    n_flagged = int(flag.sum())

    # --- Repair: interpolate flagged values ---
    if n_flagged > 0:
        df.loc[flag, demand_col] = np.nan
        df[demand_col] = (
            df[demand_col]
            .interpolate(method="linear")
            .ffill()
            .bfill()
        )

    if verbose:
        print(f"Demand cleaning summary ({demand_col}):")
        print(f"  Total rows:               {n_total:>8,}")
        print(f"  Pass 1 - floor < {floor_mw/1000:.0f} GW:     {n_floor:>8,} flagged")
        print(f"  Pass 2 - |delta| > {max_delta_mw/1000:.0f} GW:   {n_grad:>8,} flagged")
        print(f"  Pass 3 - short median:    {n_median:>8,} flagged")
        print(f"  Total repaired (unique):  {n_flagged:>8,} ({n_flagged / n_total:.3%})")
        new_min = df[demand_col].min()
        new_max = df[demand_col].max()
        print(f"  Clean range: {new_min:,.0f} - {new_max:,.0f} MW")

    return df


# ---------------------------------------------------------------------------
# File-level convenience
# ---------------------------------------------------------------------------
def clean_demand_parquet(
    parquet_path: Path,
    output_path: Optional[Path] = None,
    *,
    verbose: bool = True,
) -> Path:
    """Load a parquet, clean the demand column, overwrite (or write new).

    Parameters
    ----------
    parquet_path : Path
        Input parquet file.
    output_path : Path or None
        If None, overwrites *parquet_path* in place.

    Returns
    -------
    Path
        The written file path.
    """
    parquet_path = Path(parquet_path)
    if output_path is None:
        output_path = parquet_path

    df = pd.read_parquet(parquet_path)
    clean_demand_dataframe(df, verbose=verbose)
    df.to_parquet(output_path, index=False)

    # Update companion JSON metadata
    json_path = output_path.with_suffix(".json")
    if json_path.exists():
        with open(json_path, "r") as f:
            meta = json.load(f)
        # Refresh numeric_statistics for ActualDemand
        if "numeric_statistics" in meta and "ActualDemand" in meta["numeric_statistics"]:
            stats = df["ActualDemand"].describe()
            meta["numeric_statistics"]["ActualDemand"] = {
                "count": int(stats["count"]),
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "min": float(stats["min"]),
                "max": float(stats["max"]),
                "missing": int(df["ActualDemand"].isna().sum()),
            }
        if "cleaning" not in meta:
            meta["cleaning"] = {}
        meta["cleaning"]["ActualDemand"] = (
            "domain-aware outlier repair (floor + gradient + short-median)"
        )
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"  Updated metadata: {json_path.name}")

    if verbose:
        print(f"  Written: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Clean GB demand data (outlier detection + interpolation repair)"
    )
    parser.add_argument(
        "parquet",
        nargs="?",
        default=None,
        help="Path to parquet file. Default: bundled GB_Forecast_Actual_Demand parquet.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyse but do not write.",
    )
    args = parser.parse_args()

    if args.parquet is None:
        # Default: bundled demand parquet
        data_dir = Path(__file__).resolve().parent.parent / "parquet"
        candidates = list(data_dir.glob("*Demand*2023*.parquet"))
        if not candidates:
            print("No default demand parquet found. Pass a path explicitly.")
            return
        parquet_path = candidates[0]
    else:
        parquet_path = Path(args.parquet)

    print(f"Input: {parquet_path}")
    if args.dry_run:
        df = pd.read_parquet(parquet_path)
        clean_demand_dataframe(df, verbose=True)
        print("(dry run - file not modified)")
    else:
        clean_demand_parquet(parquet_path, verbose=True)


if __name__ == "__main__":
    main()
