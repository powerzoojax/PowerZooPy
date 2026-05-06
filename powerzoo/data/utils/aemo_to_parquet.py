"""Convert AEMO CSV files to parquet with companion JSON metadata.

Usage::

    python -m powerzoo.data.utils.aemo_to_parquet

Reads from ``powerzoo/data/source/AEMO_*.csv`` and writes to
``powerzoo/data/parquet/``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

_SOURCE_DIR = Path(__file__).resolve().parent.parent / "source"
_PARQUET_DIR = Path(__file__).resolve().parent.parent / "parquet"


def convert_aemo_5min_demand(
    csv_path: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Convert the AEMO 5-min operational demand CSV to parquet."""
    if csv_path is None:
        candidates = sorted(_SOURCE_DIR.glob("AEMO_5min_Demand*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No AEMO 5-min demand CSV in {_SOURCE_DIR}")
        csv_path = candidates[-1]
    if out_dir is None:
        out_dir = _PARQUET_DIR

    df = pd.read_csv(csv_path)
    df["INTERVAL_DATETIME"] = pd.to_datetime(df["INTERVAL_DATETIME"], utc=True)
    if "LASTCHANGED" in df.columns:
        df = df.drop(columns=["LASTCHANGED"])

    stem = csv_path.stem
    pq_path = out_dir / f"{stem}.parquet"
    df.to_parquet(pq_path, index=False)

    meta = {
        "source_file": csv_path.name,
        "parquet_file": pq_path.name,
        "generated_at": datetime.now().isoformat(),
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "regions": sorted(df["REGIONID"].unique().tolist()),
    }
    json_path = out_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    print(f"Wrote {pq_path} ({len(df):,} rows) + {json_path}")
    return pq_path


def convert_aemo_forecast(
    csv_path: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Convert the AEMO forecast-vs-actual CSV to parquet."""
    if csv_path is None:
        candidates = sorted(_SOURCE_DIR.glob("AEMO_Forecast*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No AEMO forecast CSV in {_SOURCE_DIR}")
        csv_path = candidates[-1]
    if out_dir is None:
        out_dir = _PARQUET_DIR

    df = pd.read_csv(csv_path)
    for col in ("FORECAST_DATETIME", "INTERVAL_DATETIME", "LOAD_DATE"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)

    stem = csv_path.stem
    pq_path = out_dir / f"{stem}.parquet"
    df.to_parquet(pq_path, index=False)

    meta = {
        "source_file": csv_path.name,
        "parquet_file": pq_path.name,
        "generated_at": datetime.now().isoformat(),
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "regions": sorted(df["REGIONID"].unique().tolist()),
    }
    json_path = out_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    print(f"Wrote {pq_path} ({len(df):,} rows) + {json_path}")
    return pq_path


if __name__ == "__main__":
    _PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    convert_aemo_5min_demand()
    convert_aemo_forecast()
