"""DC Microgrid profile loading and OOD transforms (Python / NumPy version).

Mirrors the semantics of ``powerzoojax.data.dc_microgrid_profiles`` so that
the Python benchmark surface can share the same exogenous task-setting
surface as the JAX side.

Profile format
--------------
All profiles are 1-D NumPy arrays of length ``DC_STEPS_PER_DAY`` (288) or a
multiple thereof.  Values are cyclically indexed: ``arr[t % len(arr)]``.

``load_workload_profiles`` returns a dict with keys::

    'cpu'   : float32 array ∈ [0, 1]  — normalised CPU utilisation proxy
    'solar' : float32 array ∈ [0, 1]  — solar capacity factor
    'temp'  : float32 array [°C]       — outdoor temperature

OOD semantics
-------------
``apply_ood_transform`` accepts the profiles dict and a scenario name.

Workload-based scenarios (``workload_swap``, ``workload_shock``) require that
the *input* profiles dict was loaded from **real** data (``real_data=True``
key present).  When ``strict=True`` (the default) and this key is absent,
the function raises ``ValueError`` — it must NOT silently fall back to
synthetic data.

The remaining scenarios (``renewable_drought``, ``cooling_stress``,
``dg_derating``, ``sla_tighten``) apply parametric transformations and do
not require real data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DC_STEPS_PER_DAY: int = 288          # 288 × 5 min = 24 h
DC_EPISODE_LEN: int = DC_STEPS_PER_DAY

VALID_SOURCES = ('google', 'azure', 'alibaba', 'synthetic')

VALID_OOD_SCENARIOS = (
    'workload_swap',
    'workload_shock',
    'renewable_drought',
    'cooling_stress',
    'dg_derating',
    'sla_tighten',
)

# Manifest name → parquet column for CPU utilisation
_SOURCE_CPU_COLUMN = {
    'google':  'datacenter.cpu_util',
    'azure':   'cpu_util',
    'alibaba': 'machine_cpu_util',
}

# ---------------------------------------------------------------------------
# Synthetic profile generators
# ---------------------------------------------------------------------------

def make_synthetic_cpu_profile(n_steps: int = DC_STEPS_PER_DAY) -> np.ndarray:
    """Synthetic diurnal CPU utilisation profile.

    Mirrors the JAX implementation: sinusoidal with a peak ~14:00
    and trough ~04:00, range [0.30, 0.85].
    """
    t = np.arange(n_steps, dtype=np.float32)
    hour = t / n_steps * 24.0
    cf = 0.575 + 0.275 * np.sin(2.0 * np.pi * (hour - 8.0) / 24.0)
    return np.clip(cf, 0.30, 0.85).astype(np.float32)


def make_synthetic_solar_profile(n_steps: int = DC_STEPS_PER_DAY) -> np.ndarray:
    """Synthetic diurnal solar capacity-factor profile.

    Sinusoidal arch between 06:00–18:00, zero outside.
    Peak at noon ≈ 1.0, no cloud noise.
    """
    t = np.arange(n_steps, dtype=np.float32)
    hour = t / n_steps * 24.0
    cf = np.where(
        (hour >= 6.0) & (hour <= 18.0),
        np.sin(np.pi * (hour - 6.0) / 12.0),
        0.0,
    )
    return np.clip(cf, 0.0, 1.0).astype(np.float32)


def make_synthetic_outdoor_temp_profile(n_steps: int = DC_STEPS_PER_DAY) -> np.ndarray:
    """Synthetic diurnal outdoor temperature profile [°C].

    Mean 20 °C, amplitude 8 °C, trough at 02:00, peak at 14:00.

    Phase ``(hour - 8)`` puts the maximum at h=14 (afternoon), aligned with
    DataCenterEnv._get_outdoor_temp() and the PowerZooJax JAX reference.
    """
    t = np.arange(n_steps, dtype=np.float32)
    hour = t / n_steps * 24.0
    temp = 20.0 + 8.0 * np.sin(2.0 * np.pi * (hour - 8.0) / 24.0)
    return temp.astype(np.float32)


def make_all_synthetic_profiles(n_steps: int = DC_STEPS_PER_DAY) -> Dict[str, np.ndarray]:
    """Return all three synthetic profiles in a single dict."""
    return {
        'cpu':   make_synthetic_cpu_profile(n_steps),
        'solar': make_synthetic_solar_profile(n_steps),
        'temp':  make_synthetic_outdoor_temp_profile(n_steps),
        'real_data': False,  # sentinel used by OOD strict checks
    }


# ---------------------------------------------------------------------------
# Cyclic tiling helper
# ---------------------------------------------------------------------------

def cycle_profile(arr: np.ndarray, n_steps: int, offset: int = 0) -> np.ndarray:
    """Tile *arr* to length *n_steps* starting at *offset*.

    Args:
        arr:     Source profile of length L.
        n_steps: Target length.
        offset:  Starting index within the source (modulo L).

    Returns:
        1-D float32 array of length n_steps.
    """
    if len(arr) == 0:
        raise ValueError("cycle_profile: source array is empty")
    n = len(arr)
    indices = np.arange(offset, offset + n_steps) % n
    return arr[indices].astype(np.float32)


# ---------------------------------------------------------------------------
# Real data loader
# ---------------------------------------------------------------------------

def _find_data_dir() -> Optional[Path]:
    """Locate the PowerZoo parquet data directory by searching upward."""
    # Try common relative paths from this file's location
    here = Path(__file__).resolve().parent
    candidates = [
        here / 'parquet',
        here.parent / 'data' / 'parquet',
        here.parent.parent / 'data' / 'parquet',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_workload_profiles(
    source: str = 'google',
    n_steps: int = DC_STEPS_PER_DAY,
    strict: bool = False,
    data_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Load workload profiles for a given data-center source.

    Args:
        source:   One of ``'google'``, ``'azure'``, ``'alibaba'``, or
                  ``'synthetic'``.  Case-insensitive.
        n_steps:  Target episode length (default 288).
        strict:   When ``True``, raise ``FileNotFoundError`` if the
                  parquet file cannot be found.  When ``False``, fall
                  back to synthetic profiles and emit a warning.
        data_dir: Override the parquet data directory search.

    Returns:
        Dict with keys ``'cpu'``, ``'solar'``, ``'temp'`` (all float32
        ndarrays of length *n_steps*) and ``'real_data'`` (bool).
    """
    source = source.lower()
    if source not in VALID_SOURCES:
        raise ValueError(
            f"source must be one of {VALID_SOURCES}, got '{source}'"
        )

    if source == 'synthetic':
        return make_all_synthetic_profiles(n_steps)

    # Try to load real data
    try:
        profiles = _load_real_profiles(source, n_steps, data_dir)
        profiles['real_data'] = True
        return profiles
    except (FileNotFoundError, ImportError, KeyError, Exception) as exc:
        if strict:
            raise FileNotFoundError(
                f"load_workload_profiles(source='{source}', strict=True): "
                f"could not load real data — {exc}"
            ) from exc
        logger.warning(
            "load_workload_profiles: could not load '%s' data (%s); "
            "falling back to synthetic profiles.",
            source, exc,
        )
        synth = make_all_synthetic_profiles(n_steps)
        synth['real_data'] = False
        return synth


def _load_real_profiles(
    source: str,
    n_steps: int,
    data_dir: Optional[str],
) -> Dict[str, np.ndarray]:
    """Internal: load parquet → extract cpu/solar/temp columns → cycle to n_steps."""
    import pandas as pd

    # Locate manifest
    manifest_dir = Path(__file__).resolve().parent / 'manifests'
    source_to_manifest = {
        'google':  'google_dc_2019.json',
        'azure':   'azure_dc_v2.json',
        'alibaba': 'alibaba_dc_2018.json',
    }
    manifest_file = manifest_dir / source_to_manifest[source]
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    import json
    with open(manifest_file) as f:
        manifest = json.load(f)

    parquet_name = manifest['parquet_file']

    # Locate parquet
    if data_dir is not None:
        parquet_path = Path(data_dir) / parquet_name
    else:
        found = _find_data_dir()
        parquet_path = found / parquet_name if found is not None else Path(parquet_name)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # CPU column
    col_map = manifest.get('column_map', {})
    cpu_raw_col = None
    for raw, semantic in col_map.items():
        if semantic == _SOURCE_CPU_COLUMN.get(source, 'datacenter.cpu_util'):
            cpu_raw_col = raw
            break
    if cpu_raw_col is None:
        # fallback: first column containing 'cpu'
        cpu_candidates = [c for c in df.columns if 'cpu' in c.lower()]
        if not cpu_candidates:
            raise KeyError(f"Cannot find CPU column in {parquet_path}")
        cpu_raw_col = cpu_candidates[0]

    cpu_arr = df[cpu_raw_col].values.astype(np.float32)
    cpu_arr = np.clip(cpu_arr / max(float(np.nanmax(cpu_arr)), 1e-8), 0.0, 1.0)
    cpu_arr = cycle_profile(cpu_arr, n_steps)

    # Solar and temp: synthetic per-day profiles (real parquet files for DC
    # workload data rarely include collocated irradiance/temp; we use synthetic).
    solar_arr = make_synthetic_solar_profile(n_steps)
    temp_arr  = make_synthetic_outdoor_temp_profile(n_steps)

    return {'cpu': cpu_arr, 'solar': solar_arr, 'temp': temp_arr}


# ---------------------------------------------------------------------------
# OOD transforms
# ---------------------------------------------------------------------------

def apply_ood_transform(
    profiles: Dict[str, np.ndarray],
    scenario: str,
    strict: bool = True,
    # Scenario-specific knobs (use defaults matching JAX implementation)
    drought_scale: float = 0.2,
    cooling_temp_offset: float = 8.0,
    dg_derating_factor: float = 0.5,
    sla_tighten_factor: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Apply an OOD transform to a profiles dict.

    Args:
        profiles:              Dict returned by ``load_workload_profiles``.
        scenario:              One of ``VALID_OOD_SCENARIOS``.
        strict:                When ``True``, workload-based scenarios
                               (``workload_swap``, ``workload_shock``) raise
                               ``ValueError`` if ``profiles['real_data']`` is
                               ``False`` or absent.  Never falls back silently.
        drought_scale:         ``renewable_drought`` — multiply solar by this.
        cooling_temp_offset:   ``cooling_stress`` — add this many °C to temp.
        dg_derating_factor:    ``dg_derating`` — returned in ``_ood_params``.
        sla_tighten_factor:    ``sla_tighten`` — returned in ``_ood_params``.
        workload_swap:         replaces ``cpu`` with Azure DC trace (source swap).
        workload_shock:        replaces ``cpu`` with Alibaba DC trace (source swap).

    Returns:
        New profiles dict (never mutates input).  May include extra keys
        (``'_ood_params'``) that ``DCMicrogridEnv`` can read to adjust
        physical parameters (e.g., DG capacity, SLA tightness).

    Raises:
        ValueError: Unknown scenario, or strict workload-OOD with synthetic data.
    """
    if scenario not in VALID_OOD_SCENARIOS:
        raise ValueError(
            f"Unknown OOD scenario '{scenario}'. "
            f"Valid: {VALID_OOD_SCENARIOS}"
        )

    out = {k: v.copy() if isinstance(v, np.ndarray) else v
           for k, v in profiles.items()}
    out.setdefault('_ood_params', {})

    if scenario == 'renewable_drought':
        out['solar'] = np.clip(
            out['solar'] * drought_scale, 0.0, 1.0
        ).astype(np.float32)

    elif scenario == 'cooling_stress':
        out['temp'] = (out['temp'] + cooling_temp_offset).astype(np.float32)

    elif scenario == 'dg_derating':
        # Cannot change DG params in-place; signal to the env via _ood_params.
        out['_ood_params']['dg_derating_factor'] = float(dg_derating_factor)

    elif scenario == 'sla_tighten':
        # Signal to the env to tighten deadline slack.
        out['_ood_params']['sla_tighten_factor'] = float(sla_tighten_factor)

    elif scenario in ('workload_swap', 'workload_shock'):
        real = bool(out.get('real_data', False))
        if strict and not real:
            raise ValueError(
                f"OOD scenario '{scenario}' requires real workload data "
                f"(profiles['real_data'] must be True). "
                f"Load profiles with a real source (google/azure/alibaba) "
                f"before applying workload-based OOD transforms."
            )
        # Source-based swap matching JAX semantics:
        #   workload_swap  → replace cpu with Azure   (google → azure)
        #   workload_shock → replace cpu with Alibaba (google → alibaba)
        src_map = {'workload_swap': 'azure', 'workload_shock': 'alibaba'}
        target_source = src_map[scenario]
        n_steps = len(out['cpu'])
        try:
            target = _load_real_profiles(target_source, n_steps, data_dir=None)
            out['cpu'] = target['cpu']
        except (FileNotFoundError, Exception) as exc:
            if strict:
                raise FileNotFoundError(
                    f"OOD scenario '{scenario}': target source '{target_source}' "
                    f"data unavailable ({exc}). "
                    f"Ensure parquet files are present or pass strict=False."
                ) from exc
            logger.warning(
                "apply_ood_transform '%s': target source '%s' unavailable (%s); "
                "keeping current CPU profile (non-strict fallback).",
                scenario, target_source, exc,
            )

    return out


__all__ = [
    'DC_STEPS_PER_DAY',
    'DC_EPISODE_LEN',
    'VALID_SOURCES',
    'VALID_OOD_SCENARIOS',
    'make_synthetic_cpu_profile',
    'make_synthetic_solar_profile',
    'make_synthetic_outdoor_temp_profile',
    'make_all_synthetic_profiles',
    'cycle_profile',
    'load_workload_profiles',
    'apply_ood_transform',
]
