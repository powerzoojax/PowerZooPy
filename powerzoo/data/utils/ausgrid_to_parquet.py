"""Convert Ausgrid zone substation FY CSVs (wide daily rows) to long Parquet.

Each source CSV has one row per calendar day and one column per 15-minute
interval (labels ``HH:MM`` … ``24:00``). This script melts to long form::

    interval_start (UTC), zone_substation, load_mw, fiscal_year

Usage::

    uv run python -m powerzoo.data.utils.ausgrid_to_parquet

Defaults read from
``powerzoo/data/source/Ausgrid Distribution Zone Substation Data FY25_imputed/``
and write to ``powerzoo/data/parquet/`` plus a companion JSON and manifest.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

_SOURCE_DIR = Path(__file__).resolve().parent.parent / "source"
_PARQUET_DIR = Path(__file__).resolve().parent.parent / "parquet"
_MANIFEST_DIR = Path(__file__).resolve().parent.parent / "manifests"

_DEFAULT_SUBDIR = "Ausgrid Distribution Zone Substation Data FY25_imputed"
_TZ = "Australia/Sydney"
_TIME_LABEL_RE = re.compile(r"^\d{2}:\d{2}$")


def _time_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if _TIME_LABEL_RE.match(str(c).strip())]
    return sorted(cols, key=_label_sort_key)


def _label_sort_key(label: str) -> tuple[int, int]:
    if label == "24:00":
        return (24, 0)
    h, m = map(int, str(label).split(":"))
    return (h, m)


def _label_to_start_minutes(label: str) -> int:
    """Interval end is *label* (e.g. 00:15 = first quarter-hour); return start offset in minutes."""
    label = str(label).strip()
    if label == "24:00":
        return 23 * 60 + 45
    h, m = map(int, label.split(":"))
    end_m = h * 60 + m
    start_m = end_m - 15
    if start_m < 0:
        raise ValueError(f"Invalid interval label {label!r}")
    return start_m


def _naive_local_to_utc(naive: pd.Series) -> pd.Series:
    """Local civil time (Australia/Sydney) → UTC. DST edge cases are approximate (benchmark data)."""
    return (
        naive.dt.tz_localize(
            _TZ,
            ambiguous=False,
            nonexistent="shift_forward",
        )
        .dt.tz_convert("UTC")
    )


def _read_one_csv(path: Path, time_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if not time_cols:
        raise ValueError(f"No HH:MM interval columns in {path}")
    id_vars = [c for c in ("year", "Zone Substation", "Date", "Unit") if c in df.columns]
    missing = [c for c in ("year", "Zone Substation", "Date") if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=time_cols,
        var_name="interval_label",
        value_name="load_mw",
    )
    long_df["load_mw"] = pd.to_numeric(long_df["load_mw"], errors="coerce")
    day = pd.to_datetime(long_df["Date"], errors="coerce")
    naive_local = day + pd.to_timedelta(long_df["interval_label"].map(_label_to_start_minutes), unit="m")
    long_df["interval_start"] = _naive_local_to_utc(naive_local)
    long_df = long_df.rename(columns={"Zone Substation": "zone_substation"})
    if "year" in long_df.columns:
        long_df["fiscal_year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
        long_df = long_df.drop(columns=["year"])
    out = long_df[
        ["interval_start", "zone_substation", "load_mw", "fiscal_year"]
    ].copy()
    return out


def convert_ausgrid_zone_substations(
    source_dir: Path | None = None,
    out_dir: Path | None = None,
    manifest_dir: Path | None = None,
    *,
    glob_pattern: str = "*.csv",
    max_files: int | None = None,
    stem: str = "Ausgrid_Zone_Substation_FY25_imputed_15min",
) -> Path:
    """Read all Ausgrid zone-substation CSVs in *source_dir*, write one Parquet + JSON + manifest."""
    if source_dir is None:
        source_dir = _SOURCE_DIR / _DEFAULT_SUBDIR
    if out_dir is None:
        out_dir = _PARQUET_DIR
    if manifest_dir is None:
        manifest_dir = _MANIFEST_DIR

    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    csv_paths = sorted(source_dir.glob(glob_pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No files matching {glob_pattern!r} in {source_dir}")
    if max_files is not None:
        csv_paths = csv_paths[: max(0, max_files)]

    # Lock interval columns from first file so all files share the same schema
    head = pd.read_csv(csv_paths[0], nrows=1)
    time_cols = _time_columns(head)

    frames: list[pd.DataFrame] = []
    for p in csv_paths:
        frames.append(_read_one_csv(p, time_cols))

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["zone_substation", "interval_start"], kind="mergesort").reset_index(
        drop=True
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq_path = out_dir / f"{stem}.parquet"
    out.to_parquet(pq_path, index=False)

    stations = sorted(out["zone_substation"].astype(str).unique().tolist())
    t_min = pd.Timestamp(out["interval_start"].min()).tz_convert("UTC")
    t_max = pd.Timestamp(out["interval_start"].max()).tz_convert("UTC")
    date_range = [t_min.date().isoformat(), t_max.date().isoformat()]

    meta = {
        "source_dir": source_dir.name,
        "source_files": len(csv_paths),
        "parquet_file": pq_path.name,
        "generated_at": datetime.now().isoformat(),
        "timezone_local": _TZ,
        "timezone_stored": "UTC",
        "interval_convention": "interval_start is UTC start of 15-minute block; "
        "CSV column HH:MM is interval end in local civil time. "
        "Local→UTC uses pandas tz_localize (ambiguous=False, nonexistent=shift_forward).",
        "shape": {"rows": len(out), "columns": len(out.columns)},
        "columns": list(out.columns),
        "dtypes": {c: str(out[c].dtype) for c in out.columns},
        "zone_substation_values": stations,
        "date_range": date_range,
    }
    json_path = out_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    manifest_path = Path(manifest_dir) / "ausgrid_zone_substation_fy25_imputed.json"
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": "ausgrid_zone_substation_fy25_imputed",
        "source": "ausgrid",
        "data_type": "actual_series",
        "time_mode": "calendar",
        "resolution": "15min",
        "parquet_file": pq_path.name,
        "column_map": {
            "load_mw": "load.actual_mw",
        },
        "index_map": {
            "interval_start": "datetime",
            "zone_substation": "region",
        },
        "derived": {},
        "normalize": {},
        "data_epoch": None,
        "cyclical": False,
        "region_values": stations,
        "date_range": date_range,
        "metadata_json": json_path.name,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(
        f"Wrote {pq_path} ({len(out):,} rows, {len(stations)} stations) + "
        f"{json_path.name} + {manifest_path.name}"
    )
    return pq_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ausgrid zone substation CSVs → Parquet + manifest")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=f"Directory of CSVs (default: source/{_DEFAULT_SUBDIR})",
    )
    parser.add_argument("--out-dir", type=Path, default=_PARQUET_DIR, help="Parquet output directory")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=_MANIFEST_DIR,
        help="Manifest JSON directory",
    )
    parser.add_argument("--stem", type=str, default="Ausgrid_Zone_Substation_FY25_imputed_15min")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of CSVs (debug)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    convert_ausgrid_zone_substations(
        source_dir=args.source_dir,
        out_dir=args.out_dir,
        manifest_dir=args.manifest_dir,
        max_files=args.max_files,
        stem=args.stem,
    )


if __name__ == "__main__":
    main()
