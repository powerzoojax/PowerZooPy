"""Build a Case5-like case file from GB CSV datasets.

Usage example:
    python powerzoo/case/GB_Data/build_case_from_csv.py

The script reads:
    - Buses.csv
    - Loads.csv
    - Network.csv
    - Units.csv

and generates a Python case file with four DataFrame blocks:
    nodes, units, lines, loads
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _to_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def build_nodes(buses: pd.DataFrame) -> pd.DataFrame:
    nodes = pd.DataFrame(
        {
            "id": _to_numeric(buses["BusID"]).astype(int),
            "x": _to_numeric(buses["x"]),
            "y": _to_numeric(buses["y"]),
        }
    )
    return nodes.sort_values("id").reset_index(drop=True)


def build_loads(
    buses: pd.DataFrame,
    loads_raw: pd.DataFrame,
    total_load_mw: float | None,
    normalize_loads: bool,
) -> pd.DataFrame:
    loads = pd.DataFrame({"bus_id": _to_numeric(buses["BusID"]).astype(int)})
    merged = loads.merge(loads_raw[["BusID", "weight"]], left_on="bus_id", right_on="BusID", how="left")
    weights = _to_numeric(merged["weight"])

    if normalize_loads:
        weight_sum = float(weights.sum())
        if weight_sum > 0:
            weights = weights / weight_sum

    d_max = weights if total_load_mw is None else weights * float(total_load_mw)

    loads_case = pd.DataFrame(
        {
            "id": np.arange(1, len(merged) + 1, dtype=int),
            "bus_id": merged["bus_id"].astype(int),
            "mc_a": 0.0,
            "mc_b": 0.0,
            "mc_c": 0.0,
            "d_max": d_max,
            "d_min": d_max,
        }
    )
    return loads_case


def build_lines(buses: pd.DataFrame, network: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bus_name_to_id: Dict[str, int] = {
        _clean_text(name): int(bus_id)
        for name, bus_id in zip(buses["bus_name"], buses["BusID"])
    }

    from_id = network["Bus0"].map(lambda x: bus_name_to_id.get(_clean_text(x), np.nan))
    to_id = network["Bus1"].map(lambda x: bus_name_to_id.get(_clean_text(x), np.nan))
    missing_mask = from_id.isna() | to_id.isna()
    missing_rows = network.loc[missing_mask, ["AssetID", "Bus0", "Bus1"]].copy()

    valid = network.loc[~missing_mask].copy()
    from_id = from_id.loc[~missing_mask].astype(int)
    to_id = to_id.loc[~missing_mask].astype(int)

    x = _to_numeric(valid["x"], fill_value=0.01)
    x[x <= 0] = 0.01
    cap = _to_numeric(valid["Capacity"])

    lines_case = pd.DataFrame(
        {
            "id": _to_numeric(valid["AssetID"]).astype(int),
            "from": from_id.values,
            "to": to_id.values,
            "x": x.values,
            "floor": (-cap).values,
            "cap": cap.values,
        }
    )
    lines_case = lines_case.sort_values("id").reset_index(drop=True)
    return lines_case, missing_rows


def _fallback_probs(buses: pd.DataFrame, loads_raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    bus_ids = _to_numeric(buses["BusID"]).astype(int).values
    lw = pd.DataFrame({"bus_id": bus_ids}).merge(
        loads_raw[["BusID", "weight"]], left_on="bus_id", right_on="BusID", how="left"
    )
    w = _to_numeric(lw["weight"]).values
    if float(w.sum()) <= 0:
        w = np.ones_like(w, dtype=float)
    p = w / w.sum()
    return bus_ids, p


def build_units(
    buses: pd.DataFrame,
    loads_raw: pd.DataFrame,
    units_raw: pd.DataFrame,
    p_min_ratio: float,
    clip_negative_srmc: bool,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bus_name_to_id: Dict[str, int] = {
        _clean_text(name): int(bus_id)
        for name, bus_id in zip(buses["bus_name"], buses["BusID"])
    }

    region_to_bus_ids: Dict[str, List[int]] = {}
    for _, row in buses.iterrows():
        region = _clean_text(row.get("RegionName", ""))
        if region:
            region_to_bus_ids.setdefault(region, []).append(int(row["BusID"]))

    load_weight_by_bus: Dict[int, float] = {
        int(bid): float(w)
        for bid, w in zip(
            _to_numeric(loads_raw["BusID"]).astype(int).values,
            _to_numeric(loads_raw["weight"]).values,
        )
    }

    fallback_bus_ids, fallback_p = _fallback_probs(buses, loads_raw)
    rng = np.random.default_rng(random_seed)

    valid_bus_ids = set(_to_numeric(buses["BusID"]).astype(int).tolist())

    mapped_bus_ids: List[int] = []
    mapping_sources: List[str] = []

    for _, row in units_raw.iterrows():
        # New-format Units.csv: use BusID directly if valid.
        unit_bus_id = pd.to_numeric(row.get("BusID"), errors="coerce")
        if not pd.isna(unit_bus_id):
            unit_bus_id_int = int(unit_bus_id)
            if unit_bus_id_int in valid_bus_ids:
                mapped_bus_ids.append(unit_bus_id_int)
                mapping_sources.append("bus_id_direct")
                continue

        region = _clean_text(row.get("Region", ""))

        if region in bus_name_to_id:
            mapped_bus_ids.append(bus_name_to_id[region])
            mapping_sources.append("region_as_bus_name_fallback")
            continue

        if region in region_to_bus_ids:
            candidates = region_to_bus_ids[region]
            chosen = max(candidates, key=lambda b: (load_weight_by_bus.get(b, 0.0), -b))
            mapped_bus_ids.append(chosen)
            mapping_sources.append("region_to_region_name_fallback")
            continue

        fallback = int(rng.choice(fallback_bus_ids, p=fallback_p))
        mapped_bus_ids.append(fallback)
        mapping_sources.append("global_load_weighted_fallback")

    p_max = _to_numeric(units_raw["Capacity (MW)"])
    p_min = p_max * float(p_min_ratio)
    mc_c = _to_numeric(units_raw["SRMC_i"])
    if clip_negative_srmc:
        mc_c = mc_c.clip(lower=0.0)

    units_case = pd.DataFrame(
        {
            "id": _to_numeric(units_raw["UnitID"]).astype(int),
            "bus_id": np.array(mapped_bus_ids, dtype=int),
            "mc_a": 0.0,
            "mc_b": 0.0,
            "mc_c": mc_c.values,
            "p_max": p_max.values,
            "p_min": p_min.values,
        }
    ).sort_values("id").reset_index(drop=True)

    mapping_report = units_raw[["UnitID", "Region", "Technology", "Type"]].copy()
    mapping_report["bus_id"] = mapped_bus_ids
    mapping_report["mapping_source"] = mapping_sources
    return units_case, mapping_report


def _rows_as_python_list(df: pd.DataFrame) -> str:
    rows = df.values.tolist()
    return repr(rows)


def write_case_file(
    output_file: Path,
    class_name: str,
    nodes: pd.DataFrame,
    units: pd.DataFrame,
    lines: pd.DataFrame,
    loads: pd.DataFrame,
) -> None:
    content = f"""from powerzoo.case.CaseBase import ClearCase, DataFrame


class {class_name}(ClearCase):
    def __init__(self, *args, **kwargs):
        self.nodes = DataFrame(
            ['id', 'x', 'y'],
            {_rows_as_python_list(nodes[['id', 'x', 'y']])})

        self.units = DataFrame(
            ['id', 'bus_id', 'mc_a', 'mc_b', 'mc_c', 'p_max', 'p_min'],
            {_rows_as_python_list(units[['id', 'bus_id', 'mc_a', 'mc_b', 'mc_c', 'p_max', 'p_min']])})

        self.lines = DataFrame(
            ['id', 'from', 'to', 'x', 'floor', 'cap'],
            {_rows_as_python_list(lines[['id', 'from', 'to', 'x', 'floor', 'cap']])})

        self.loads = DataFrame(
            ['id', 'bus_id', 'mc_a', 'mc_b', 'mc_c', 'd_max', 'd_min'],
            {_rows_as_python_list(loads[['id', 'bus_id', 'mc_a', 'mc_b', 'mc_c', 'd_max', 'd_min']])})

        self.real_params = True
        super().__init__(*args, **kwargs)
"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_data_dir = Path(__file__).resolve().parent
    default_output_file = default_data_dir.parent / "Case552GB.py"
    default_report_dir = default_data_dir / "build_reports"

    parser = argparse.ArgumentParser(description="Build Case file from GB CSV data.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir, help="Directory containing CSV files.")
    parser.add_argument(
        "--output-case-file",
        type=Path,
        default=default_output_file,
        help="Path of generated case Python file.",
    )
    parser.add_argument("--class-name", type=str, default="Case552GB", help="Generated case class name.")
    parser.add_argument(
        "--total-load-mw",
        type=float,
        default=None,
        help="If set, load weights are scaled to this total MW.",
    )
    parser.add_argument(
        "--normalize-loads",
        action="store_true",
        help="Normalize load weights to sum to 1 before optional scaling.",
    )
    parser.add_argument("--p-min-ratio", type=float, default=0.0, help="Set p_min = p_max * ratio.")
    parser.add_argument(
        "--clip-negative-srmc",
        action="store_true",
        help="Clip negative SRMC values to zero.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fallback unit-to-bus mapping.")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=default_report_dir,
        help="Directory for mapping and summary reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    buses = pd.read_csv(data_dir / "Buses.csv")
    loads_raw = pd.read_csv(data_dir / "Loads.csv")
    network = pd.read_csv(data_dir / "Network.csv")
    units_raw = pd.read_csv(data_dir / "Units.csv")

    nodes = build_nodes(buses)
    loads = build_loads(
        buses=buses,
        loads_raw=loads_raw,
        total_load_mw=args.total_load_mw,
        normalize_loads=args.normalize_loads,
    )
    lines, missing_lines = build_lines(buses=buses, network=network)
    units, mapping_report = build_units(
        buses=buses,
        loads_raw=loads_raw,
        units_raw=units_raw,
        p_min_ratio=args.p_min_ratio,
        clip_negative_srmc=args.clip_negative_srmc,
        random_seed=args.seed,
    )

    write_case_file(
        output_file=args.output_case_file,
        class_name=args.class_name,
        nodes=nodes,
        units=units,
        lines=lines,
        loads=loads,
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    mapping_report_file = args.report_dir / "units_mapping_report.csv"
    missing_lines_file = args.report_dir / "missing_line_bus_mapping.csv"
    summary_file = args.report_dir / "build_summary.json"

    mapping_report.to_csv(mapping_report_file, index=False, encoding="utf-8")
    missing_lines.to_csv(missing_lines_file, index=False, encoding="utf-8")

    source_counts = mapping_report["mapping_source"].value_counts().to_dict()
    summary = {
        "nodes": int(len(nodes)),
        "units": int(len(units)),
        "lines": int(len(lines)),
        "loads": int(len(loads)),
        "dropped_lines_due_to_missing_bus_mapping": int(len(missing_lines)),
        "unit_mapping_source_counts": source_counts,
        "output_case_file": str(args.output_case_file),
        "mapping_report_file": str(mapping_report_file),
        "missing_lines_file": str(missing_lines_file),
    }
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
