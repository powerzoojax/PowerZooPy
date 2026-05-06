"""Time alignment engine for heterogeneous data sources.

Two modes:
* **calendar** – data has real-world timestamps.  An optional *offset*
  shifts the data timeline onto the simulation timeline.
* **profile** – data is a short, repeatable pattern (e.g. 8-day DC
  trace).  It is tiled cyclically to cover the simulation window.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class TimeAligner:
    """Map raw-data timestamps onto a unified simulation timeline."""

    @staticmethod
    def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
        """Return *ts* as a UTC-aware Timestamp."""
        ts = pd.Timestamp(ts)
        return ts if ts.tzinfo is not None else ts.tz_localize("UTC")

    # ------------------------------------------------------------------
    # Calendar mode
    # ------------------------------------------------------------------

    @staticmethod
    def align_calendar(
        df: pd.DataFrame,
        sim_start: pd.Timestamp,
        sim_end: pd.Timestamp,
        align_from: Optional[pd.Timestamp] = None,
        time_col: str = "datetime",
    ) -> pd.DataFrame:
        """Shift a calendar-mode dataset onto ``[sim_start, sim_end]``.

        Parameters
        ----------
        df : DataFrame with a *time_col* column (or DatetimeIndex).
        sim_start, sim_end : target simulation window.
        align_from : the real-data timestamp that should correspond to
            *sim_start*.  ``None`` means "no shift needed" (data dates
            already match the simulation dates).
        time_col : name of the datetime column.
        """
        out = df.copy()

        has_col = time_col in out.columns
        if not has_col and isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            if time_col not in out.columns and "index" in out.columns:
                out = out.rename(columns={"index": time_col})

        if time_col not in out.columns:
            return out

        out[time_col] = pd.to_datetime(out[time_col], utc=True)

        if align_from is not None:
            offset = TimeAligner._ensure_utc(sim_start) - TimeAligner._ensure_utc(align_from)
            out[time_col] = out[time_col] + offset

        _sim_start_utc = TimeAligner._ensure_utc(sim_start)
        _sim_end_utc = TimeAligner._ensure_utc(sim_end)
        # Make end inclusive for the whole last day
        _sim_end_inclusive = _sim_end_utc.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        mask = (out[time_col] >= _sim_start_utc) & (out[time_col] <= _sim_end_inclusive)
        out = out.loc[mask].reset_index(drop=True)
        return out

    # ------------------------------------------------------------------
    # Profile mode
    # ------------------------------------------------------------------

    @staticmethod
    def align_profile(
        df: pd.DataFrame,
        sim_start: pd.Timestamp,
        sim_end: pd.Timestamp,
        resolution: str,
        time_col: str = "datetime",
    ) -> pd.DataFrame:
        """Tile a profile cyclically to cover ``[sim_start, sim_end]``.

        The profile's internal ordering is preserved; timestamps are
        replaced by a regular grid on the simulation timeline.

        Parameters
        ----------
        resolution : pandas frequency string, e.g. ``"5min"``, ``"300s"``.
        """
        sim_start = TimeAligner._ensure_utc(sim_start)
        sim_end = TimeAligner._ensure_utc(sim_end)

        sim_end_inclusive = sim_end.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        sim_index = pd.date_range(start=sim_start, end=sim_end_inclusive, freq=resolution)
        n_needed = len(sim_index)

        value_cols = [c for c in df.columns if c != time_col]
        values = df[value_cols].values
        n_profile = len(values)

        if n_profile == 0:
            return pd.DataFrame({time_col: sim_index})

        n_full_tiles = n_needed // n_profile
        remainder = n_needed % n_profile
        parts: list[np.ndarray] = []
        if n_full_tiles > 0:
            parts.append(np.tile(values, (n_full_tiles, 1)))
        if remainder > 0:
            parts.append(values[:remainder])

        tiled = np.concatenate(parts, axis=0) if parts else values[:0]

        result = pd.DataFrame(tiled, columns=value_cols)
        result[time_col] = sim_index[:len(result)]
        return result

    # ------------------------------------------------------------------
    # Convenience dispatcher
    # ------------------------------------------------------------------

    @classmethod
    def align(
        cls,
        df: pd.DataFrame,
        *,
        time_mode: str,
        sim_start: pd.Timestamp,
        sim_end: pd.Timestamp,
        align_from: Optional[pd.Timestamp] = None,
        resolution: str = "30min",
        time_col: str = "datetime",
    ) -> pd.DataFrame:
        """Dispatch to the correct alignment strategy."""
        if time_mode == "profile":
            return cls.align_profile(df, sim_start, sim_end, resolution, time_col)
        return cls.align_calendar(df, sim_start, sim_end, align_from, time_col)
