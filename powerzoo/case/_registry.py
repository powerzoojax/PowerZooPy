"""Case registry for discovering and filtering available case data."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CaseMeta:
    """Lightweight metadata extracted from a Case class without instantiation."""

    name: str
    module_path: str
    grid_type: str = ""
    bus_count: int = 0
    phase: str = "1"
    voltage_level: str = ""
    source: str = ""
    description: str = ""


_CASE_DIR = Path(__file__).resolve().parent
_SUBDIRS = ["transmission", "distribution"]
_cache: Dict[str, CaseMeta] = {}


def _discover() -> Dict[str, CaseMeta]:
    """Scan sub-packages for Case*.py and extract class-level metadata."""
    if _cache:
        return _cache

    for subdir in _SUBDIRS:
        pkg_dir = _CASE_DIR / subdir
        if not pkg_dir.is_dir():
            continue
        for py_file in sorted(pkg_dir.glob("Case*.py")):
            class_name = py_file.stem
            module_name = f"powerzoo.case.{subdir}.{class_name}"
            try:
                mod = importlib.import_module(module_name)
                cls = getattr(mod, class_name, None)
                if cls is None:
                    continue
                meta = CaseMeta(
                    name=class_name,
                    module_path=module_name,
                    grid_type=getattr(cls, "GRID_TYPE", ""),
                    bus_count=getattr(cls, "BUS_COUNT", 0),
                    phase=getattr(cls, "PHASE", "1"),
                    voltage_level=getattr(cls, "VOLTAGE_LEVEL", ""),
                    source=getattr(cls, "SOURCE", ""),
                    description=getattr(cls, "DESCRIPTION", ""),
                )
                _cache[class_name] = meta
            except Exception:
                continue
    return _cache


def list_cases(
    *,
    grid_type: Optional[str] = None,
    min_buses: Optional[int] = None,
    max_buses: Optional[int] = None,
    phase: Optional[str] = None,
    voltage_level: Optional[str] = None,
) -> List[CaseMeta]:
    """Return metadata for all discovered cases, optionally filtered.

    Args:
        grid_type: ``"transmission"`` or ``"distribution"``.
        min_buses: Minimum bus count (inclusive).
        max_buses: Maximum bus count (inclusive).
        phase: ``"1"`` or ``"3"``.
        voltage_level: ``"HV"``, ``"MV"``, or ``"LV"``.

    Returns:
        List of :class:`CaseMeta` sorted by bus count.
    """
    registry = _discover()
    results = list(registry.values())

    if grid_type is not None:
        results = [m for m in results if m.grid_type == grid_type]
    if min_buses is not None:
        results = [m for m in results if m.bus_count >= min_buses]
    if max_buses is not None:
        results = [m for m in results if m.bus_count <= max_buses]
    if phase is not None:
        results = [m for m in results if m.phase == phase]
    if voltage_level is not None:
        results = [m for m in results if m.voltage_level == voltage_level]

    results.sort(key=lambda m: m.bus_count)
    return results
