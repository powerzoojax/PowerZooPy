"""Convert data-center trace CSVs to parquet.

Supports:
* Alibaba 2018 machine usage (10s / 300s aggregated)
* Google 2019 instance usage (300s aggregated)
* Azure v2 VM workload (300s aggregated)

Usage::

    python -m powerzoo.data.utils.dc_to_parquet --source alibaba --input FILE

Each converter produces a parquet file plus companion JSON metadata in
``powerzoo/data/parquet/``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

_PARQUET_DIR = Path(__file__).resolve().parent.parent / "parquet"


def convert_alibaba_2018(
    csv_path: Path,
    out_dir: Path | None = None,
    resolution: str = "300s",
) -> Path:
    """Alibaba cluster-trace-v2018 aggregated machine usage.

    Expected columns: ``cpu_util_percent, mem_util_percent, net_in, net_out,
    disk_io_percent`` (plus an optional ``time_stamp`` index column).
    """
    out_dir = out_dir or _PARQUET_DIR
    df = pd.read_csv(csv_path)

    if "time_stamp" not in df.columns and df.index.name != "time_stamp":
        df["time_stamp"] = range(len(df))

    stem = f"alibaba_dc_2018_{resolution}"
    pq_path = out_dir / f"{stem}.parquet"
    df.to_parquet(pq_path, index=False)

    _write_meta(csv_path, pq_path, df)
    return pq_path


def convert_google_2019(
    csv_path: Path,
    out_dir: Path | None = None,
) -> Path:
    """Google cluster-data 2019 aggregated instance usage."""
    out_dir = out_dir or _PARQUET_DIR
    df = pd.read_csv(csv_path)

    if "time_stamp" not in df.columns and df.index.name != "time_stamp":
        df["time_stamp"] = range(len(df))

    stem = "google_dc_2019_300s"
    pq_path = out_dir / f"{stem}.parquet"
    df.to_parquet(pq_path, index=False)

    _write_meta(csv_path, pq_path, df)
    return pq_path


def convert_azure_v2(
    csv_path: Path,
    out_dir: Path | None = None,
) -> Path:
    """Azure Public Dataset v2 aggregated VM workload."""
    out_dir = out_dir or _PARQUET_DIR
    df = pd.read_csv(csv_path)

    if "time_stamp" not in df.columns and df.index.name != "time_stamp":
        df["time_stamp"] = range(len(df))

    stem = "azure_dc_v2_300s"
    pq_path = out_dir / f"{stem}.parquet"
    df.to_parquet(pq_path, index=False)

    _write_meta(csv_path, pq_path, df)
    return pq_path


def _write_meta(csv_path: Path, pq_path: Path, df: pd.DataFrame) -> None:
    meta = {
        "source_file": csv_path.name,
        "parquet_file": pq_path.name,
        "generated_at": datetime.now().isoformat(),
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }
    json_path = pq_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"Wrote {pq_path} ({len(df):,} rows) + {json_path}")


_CONVERTERS = {
    "alibaba": convert_alibaba_2018,
    "google": convert_google_2019,
    "azure": convert_azure_v2,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DC trace CSV to parquet")
    parser.add_argument("--source", choices=list(_CONVERTERS), required=True)
    parser.add_argument("--input", type=Path, required=True, help="Path to CSV")
    parser.add_argument("--outdir", type=Path, default=_PARQUET_DIR)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    _CONVERTERS[args.source](args.input, args.outdir)
