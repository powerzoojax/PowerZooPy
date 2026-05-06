"""Data Loader for PowerZoo benchmark data.

This module provides :class:`DataLoader`, the **single public data facade**
for PowerZoo.  External code (envs, tasks, resources) should load data
exclusively through this class, using *semantic signal names* rather than
raw source column names.

Two loading APIs are available:

* **Semantic API** (preferred): :meth:`load_signals`,
  :meth:`load_actual_series`, :meth:`load_forecast_panel`.
* **Legacy API** (kept for backward compatibility): :meth:`load_data` with
  raw column names.  A deprecation path maps old names to signals.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, date

import pandas as pd
import numpy as np

from . import signals as S
from .alignment import TimeAligner
from .manifest import DatasetManifest
from .registry import DatasetRegistry


class DataLoader:
    """Benchmark data facade backed by parquet files and manifest metadata.

    Features:
    - Semantic signal loading via :meth:`load_signals`
    - Calendar / profile time-alignment across heterogeneous sources
    - Forecast-panel loading with ``issue_time`` / ``target_time``
    - Legacy column-based loading via :meth:`load_data`
    - Resampling, date filtering, multi-dataset merge
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        manifest_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize DataLoader.

        Args:
            data_dir: Directory containing parquet files.
                If *None*, defaults to ``powerzoo/data/parquet``.
            manifest_dir: Directory containing manifest JSON files.
                If *None*, defaults to ``powerzoo/data/manifests``.
        """
        script_dir = Path(__file__).resolve().parent
        if data_dir is None:
            self.data_dir = script_dir / "parquet"
        else:
            self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Manifest / registry -------------------------------------------
        if manifest_dir is None:
            manifest_dir = script_dir / "manifests"
        self._registry = DatasetRegistry(Path(manifest_dir))

        # Legacy caches --------------------------------------------------
        self._metadata_cache: Dict[str, Dict] = {}
        self._column_index: Optional[Dict[str, List[str]]] = None

        # Currently loaded data (legacy state)
        self.data: Optional[pd.DataFrame] = None
        self.current_source: Optional[str] = None

    # ==================================================================
    # Semantic API (preferred)
    # ==================================================================

    @property
    def registry(self) -> DatasetRegistry:
        """Expose the dataset registry for advanced querying."""
        return self._registry

    def load_signals(
        self,
        signals: List[str],
        *,
        source: Optional[str] = None,
        region: Optional[str] = None,
        start_date: Optional[Union[str, datetime, date]] = None,
        end_date: Optional[Union[str, datetime, date]] = None,
        resample: Optional[str] = None,
        time_alignment: Optional[Dict[str, str]] = None,
        interpolation: str = "linear",
    ) -> pd.DataFrame:
        """Load one or more semantic signals into a single DataFrame.

        This is the **primary public entry-point** for data access.

        Args:
            signals: Semantic signal names (e.g.
                ``["load.actual_mw", "solar.available_mw"]``).
            source: Restrict lookup to a specific data source
                (``"gb"``, ``"aemo"``, ``"alibaba"``, …).
            region: Filter by region (e.g. ``"NSW1"``).
            start_date, end_date: Simulation time window.
            resample: Target frequency (``"5min"``, ``"30min"``, …).
            time_alignment: Per-signal calendar-shift overrides.
                ``{"solar.available_mw": "2024-01-01"}`` means "read
                solar data starting from 2024-01-01 and map it onto
                *start_date*".
            interpolation: Interpolation method when resampling.

        Returns:
            DataFrame indexed by ``datetime`` (simulation timeline).
        """
        if not signals:
            raise ValueError("At least one signal must be requested")

        if time_alignment is None:
            time_alignment = {}

        sig_to_manifest = self._registry.resolve_signals(signals, source=source)

        manifest_groups: Dict[str, list[str]] = {}
        for sig, m in sig_to_manifest.items():
            manifest_groups.setdefault(m.name, []).append(sig)

        frames: list[pd.DataFrame] = []
        sim_start = pd.Timestamp(start_date) if start_date is not None else None
        sim_end = pd.Timestamp(end_date) if end_date is not None else None

        for mname, sigs in manifest_groups.items():
            manifest = self._registry.get_manifest(mname)
            df = self._load_manifest_signals(
                manifest,
                sigs,
                sim_start=sim_start,
                sim_end=sim_end,
                region=region,
                time_alignment=time_alignment,
            )
            frames.append(df)

        if len(frames) == 1:
            result = frames[0]
        else:
            result = frames[0]
            for df_part in frames[1:]:
                df_part = self._normalize_datetime_for_merge(df_part)
                result = self._normalize_datetime_for_merge(result)
                result = result.merge(df_part, on=S.DATETIME, how="inner")

        if S.DATETIME in result.columns:
            result = result.sort_values(S.DATETIME).reset_index(drop=True)

        if resample is not None:
            result = self._resample_data(result, resample, interpolation)

        self.data = result
        self.current_source = ", ".join(manifest_groups.keys())
        return result

    def load_actual_series(
        self,
        signals: List[str],
        **kwargs,
    ) -> pd.DataFrame:
        """Convenience wrapper: load only ``actual_series`` data."""
        kwargs.setdefault("source", None)
        return self.load_signals(signals, **kwargs)

    def load_forecast_panel(
        self,
        signals: List[str],
        *,
        source: Optional[str] = None,
        region: Optional[str] = None,
        start_date: Optional[Union[str, datetime, date]] = None,
        end_date: Optional[Union[str, datetime, date]] = None,
    ) -> pd.DataFrame:
        """Load forecast-panel data with ``issue_time`` + ``target_time``.

        Returns a DataFrame with columns ``region``, ``issue_time``,
        ``target_time``, and the requested signal columns.  The panel is
        **not** flattened to a single ``datetime`` axis.
        """
        if not signals:
            raise ValueError("At least one signal must be requested")

        # Resolve signals preferring forecast_panel manifests
        result_map: Dict[str, DatasetManifest] = {}
        missing: list[str] = []
        for sig in signals:
            candidates = self._registry.find_by_signal(
                sig, source=source, data_type=S.FORECAST_PANEL,
            )
            if not candidates:
                missing.append(sig)
            else:
                result_map[sig] = candidates[0]
        if missing:
            raise ValueError(
                f"Cannot resolve forecast-panel signals: {missing}. "
                f"Make sure there is a manifest with data_type='forecast_panel' "
                f"providing these signals."
            )

        manifest_groups: Dict[str, list[str]] = {}
        for sig, m in result_map.items():
            manifest_groups.setdefault(m.name, []).append(sig)

        frames: list[pd.DataFrame] = []
        for mname, sigs in manifest_groups.items():
            manifest = self._registry.get_manifest(mname)
            df = self._load_raw_parquet(manifest)
            df = self._apply_index_map(df, manifest)
            df = self._apply_column_map(df, manifest, sigs)
            if region is not None and S.REGION in df.columns:
                df = df[df[S.REGION] == region].reset_index(drop=True)
            frames.append(df)

        if len(frames) == 1:
            return frames[0]
        result = frames[0]
        merge_keys = [c for c in [S.REGION, S.ISSUE_TIME, S.TARGET_TIME] if c in result.columns]
        for df_part in frames[1:]:
            result = result.merge(df_part, on=merge_keys, how="inner")
        return result

    # ------------------------------------------------------------------
    # Internal helpers for the semantic API
    # ------------------------------------------------------------------

    def _load_manifest_signals(
        self,
        manifest: DatasetManifest,
        sigs: List[str],
        *,
        sim_start: Optional[pd.Timestamp],
        sim_end: Optional[pd.Timestamp],
        region: Optional[str],
        time_alignment: Dict[str, str],
    ) -> pd.DataFrame:
        """Load requested signals from one manifest, with alignment."""
        df = self._load_raw_parquet(manifest)
        df = self._apply_index_map(df, manifest)
        df = self._apply_derived(df, manifest, sigs)
        df = self._apply_column_map(df, manifest, sigs)
        df = self._apply_normalization(df, manifest, sigs)

        if region is not None and S.REGION in df.columns:
            df = df[df[S.REGION] == region].reset_index(drop=True)

        if sim_start is not None and sim_end is not None:
            align_from = None
            for sig in sigs:
                if sig in time_alignment:
                    align_from = pd.Timestamp(time_alignment[sig])
                    break
            df = TimeAligner.align(
                df,
                time_mode=manifest.time_mode,
                sim_start=sim_start,
                sim_end=sim_end,
                align_from=align_from,
                resolution=manifest.resolution,
                time_col=S.DATETIME,
            )
        return df

    def _load_raw_parquet(self, manifest: DatasetManifest) -> pd.DataFrame:
        """Read a parquet file referenced by *manifest*."""
        path = self.data_dir / manifest.parquet_file
        if not path.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {path} "
                f"(referenced by manifest '{manifest.name}')"
            )
        cols_needed = manifest.raw_columns_needed
        try:
            df = pd.read_parquet(path, columns=cols_needed)
        except Exception:
            df = pd.read_parquet(path)
        return df

    def _apply_index_map(
        self, df: pd.DataFrame, manifest: DatasetManifest
    ) -> pd.DataFrame:
        """Rename raw index columns to canonical names (datetime, region, …)."""
        rename = {}
        for raw_col, canon in manifest.index_map.items():
            if raw_col in df.columns and canon not in df.columns:
                rename[raw_col] = canon
        if rename:
            df = df.rename(columns=rename)
        if S.DATETIME in df.columns:
            df[S.DATETIME] = pd.to_datetime(df[S.DATETIME], utc=True)
        if S.ISSUE_TIME in df.columns:
            df[S.ISSUE_TIME] = pd.to_datetime(df[S.ISSUE_TIME], utc=True)
        if S.TARGET_TIME in df.columns:
            df[S.TARGET_TIME] = pd.to_datetime(df[S.TARGET_TIME], utc=True)
        return df

    def _apply_derived(
        self,
        df: pd.DataFrame,
        manifest: DatasetManifest,
        requested_sigs: List[str],
    ) -> pd.DataFrame:
        """Compute derived signals (e.g. ``wind = offshore + onshore``)."""
        for sig, expr in manifest.derived.items():
            if sig not in requested_sigs:
                continue
            tokens = [t.strip() for t in expr.split("+")]
            missing = [t for t in tokens if t not in df.columns]
            if missing:
                warnings.warn(
                    f"Cannot derive '{sig}': missing columns {missing}",
                    stacklevel=3,
                )
                continue
            df[sig] = sum(df[t] for t in tokens)
        return df

    def _apply_column_map(
        self,
        df: pd.DataFrame,
        manifest: DatasetManifest,
        requested_sigs: List[str],
    ) -> pd.DataFrame:
        """Rename raw data columns to signal names and drop unrequested cols."""
        rename = {}
        for raw_col, sig in manifest.column_map.items():
            if sig in requested_sigs and raw_col in df.columns:
                rename[raw_col] = sig
        if rename:
            df = df.rename(columns=rename)

        keep = set(requested_sigs) | {S.DATETIME, S.REGION, S.ISSUE_TIME, S.TARGET_TIME}
        drop = [c for c in df.columns if c not in keep]
        if drop:
            df = df.drop(columns=drop)
        return df

    def _apply_normalization(
        self,
        df: pd.DataFrame,
        manifest: DatasetManifest,
        sigs: List[str],
    ) -> pd.DataFrame:
        """Divide by the normalizing factor declared in the manifest."""
        for sig in sigs:
            factor = manifest.normalize.get(sig)
            if factor and factor != 0 and sig in df.columns:
                df[sig] = df[sig] / factor
        return df

    # ==================================================================
    # Legacy API (kept for backward compatibility)
    # ==================================================================

    def list_available_datasets(self) -> List[str]:
        """List all available datasets (parquet files).

        Returns:
            List of dataset names (without .parquet extension)
        """
        parquet_files = list(self.data_dir.glob("*.parquet"))
        return [f.stem for f in parquet_files]

    def get_metadata(self, dataset_name: str) -> Dict:
        """
        Get metadata for a dataset.
        
        Args:
            dataset_name: Name of dataset (without extension)
            
        Returns:
            Dictionary containing metadata
        """
        if dataset_name in self._metadata_cache:
            return self._metadata_cache[dataset_name]

        json_file = self.data_dir / f"{dataset_name}.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        self._metadata_cache[dataset_name] = metadata
        return metadata

    def get_available_columns(self, dataset_name: str) -> List[str]:
        """
        Get available columns for a dataset.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            List of column names
        """
        metadata = self.get_metadata(dataset_name)
        return metadata.get('columns', [])

    def get_date_range(self, dataset_name: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Get date range for a dataset.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            Tuple of (min_date, max_date) or None if no date range found
        """
        metadata = self.get_metadata(dataset_name)
        date_ranges = metadata.get('date_ranges')

        if not date_ranges:
            return None

        # Get the first date range (usually there's only one date column)
        for col, range_info in date_ranges.items():
            min_date = pd.to_datetime(range_info['min'])
            max_date = pd.to_datetime(range_info['max'])
            return (min_date, max_date)

        return None

    def _build_column_index(self) -> Dict[str, List[str]]:
        """
        Build reverse index: column name -> list of datasets containing it.
        
        Returns:
            Dictionary mapping column name to list of dataset names
        """
        if self._column_index is not None:
            return self._column_index
        
        index: Dict[str, List[str]] = {}
        for dataset in self.list_available_datasets():
            cols = self.get_available_columns(dataset)
            for col in cols:
                # Skip date columns as they're handled separately
                col_lower = col.lower()
                if col_lower in {'datetime', 'settlement_datetime', 'interval_datetime',
                                'starttime', 'start_time', 'settlement_date', 'date',
                                'settlementperiod', 'settlement_period'}:
                    continue
                index.setdefault(col, []).append(dataset)
        
        self._column_index = index
        return index
    
    def find_datasets_for_columns(self, columns: List[str]) -> Dict[str, List[str]]:
        """
        Find which datasets contain the requested columns.
        
        Args:
            columns: List of column names
            
        Returns:
            Dictionary mapping dataset_name -> list of columns it contains
        """
        col_index = self._build_column_index()
        dataset_to_cols: Dict[str, List[str]] = {}
        missing_cols = []
        
        for col in columns:
            datasets = col_index.get(col)
            if not datasets:
                missing_cols.append(col)
            else:
                # Use the first dataset containing this column
                # (could be enhanced to prefer datasets with more requested columns)
                ds = datasets[0]
                dataset_to_cols.setdefault(ds, []).append(col)
        
        if missing_cols:
            available = sorted(col_index.keys())[:20]
            raise ValueError(
                f"Columns not found: {missing_cols}. "
                f"Available columns (first 20): {available}"
            )
        
        return dataset_to_cols
    
    def _detect_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect date column in DataFrame.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Name of date column or None if not found
        """
        # Known date column names (priority order)
        date_column_names = [
            'SETTLEMENT_DATETIME', 'INTERVAL_DATETIME',
            'startTime', 'START_TIME', 'STARTTIME',
            'SETTLEMENT_DATE', 'datetime', 'date', 'time'
        ]

        # Check known names
        for col_name in date_column_names:
            if col_name in df.columns:
                return col_name

        # Check datetime types
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if datetime_cols:
            return datetime_cols[0]

        return None

    @staticmethod
    def _normalize_datetime_for_merge(df: pd.DataFrame) -> pd.DataFrame:
        """UTC-normalize ``datetime`` and drop duplicate labels.

        Bundled UK feeds are stored as tz-aware UTC, but merges must not rely on
        mixed tz representations. Duplicate timestamps occasionally appear around
        BST/GMT transitions or between exports; keeping the first row avoids
        exploding merges.
        """
        if df is None or "datetime" not in df.columns:
            return df
        out = df.copy()
        out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
        if out["datetime"].duplicated().any():
            out = out.drop_duplicates(subset=["datetime"], keep="first")
        return out

    @staticmethod
    def _match_tz(ts: pd.Timestamp, ref_tz) -> pd.Timestamp:
        """Return *ts* localized to *ref_tz* only when their tz-awareness differs."""
        if ref_tz is not None and ts.tzinfo is None:
            return ts.tz_localize(ref_tz)
        if ref_tz is None and ts.tzinfo is not None:
            return ts.tz_localize(None)
        return ts

    def load_data(
        self,
        dataset_name: Optional[str] = None,
        columns: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime, date]] = None,
        end_date: Optional[Union[str, datetime, date]] = None,
        validate_dates: bool = True,
        merge_how: str = "inner",
        resample: Optional[str] = None,
        interpolation: str = 'linear'
    ) -> pd.DataFrame:
        """
        Load data from parquet file(s) with optional resampling.
        
        If dataset_name is not provided, automatically finds datasets based on columns.
        If columns span multiple datasets, loads and merges them on datetime.
        
        Args:
            dataset_name: Name of dataset (without extension). If None, auto-detect from columns.
            columns: List of columns to load. Required if dataset_name is None.
            start_date: Start date for filtering (inclusive)
            end_date: End date for filtering (inclusive)
            validate_dates: If True, validate dates against metadata date_ranges
            merge_how: How to merge multiple datasets. Default ``'inner'``: only
                timestamps present in **all** requested datasets (avoids NaNs when
                UK settlement half-hours differ slightly between NGESO exports or
                around DST boundaries). Use ``'outer'`` if you need the union of
                timelines and will handle missing values yourself.
            resample: Resample frequency ('5min', '15min', '30min', '60min'). 
                     If None, uses original data frequency.
            interpolation: Interpolation method for resampling. Options:
                          'linear', 'quadratic', 'cubic', 'spline', 'nearest', 
                          'zero', 'slinear', 'pchip', 'akima'
            
        Returns:
            DataFrame with loaded data
        """
        # Auto-detect dataset from columns if not provided
        if dataset_name is None:
            if not columns:
                raise ValueError("Either dataset_name or columns must be provided")
            
            # Find which datasets contain the requested columns
            dataset_to_cols = self.find_datasets_for_columns(columns)
            
            if len(dataset_to_cols) == 1:
                # All columns in one dataset - simple case
                dataset_name = list(dataset_to_cols.keys())[0]
                df = self._load_single_dataset(
                    dataset_name, columns, start_date, end_date, validate_dates
                )
            else:
                # Columns span multiple datasets - need to merge
                df = self._load_and_merge_datasets(
                    dataset_to_cols, start_date, end_date, validate_dates, merge_how
                )
        else:
            # Dataset name provided - use original logic
            df = self._load_single_dataset(
                dataset_name, columns, start_date, end_date, validate_dates
            )
        
        # Apply resampling if requested
        if resample is not None:
            df = self._resample_data(df, resample, interpolation)
        
        return df

    def _load_single_dataset(
        self,
        dataset_name: str,
        columns: Optional[List[str]],
        start_date: Optional[Union[str, datetime, date]],
        end_date: Optional[Union[str, datetime, date]],
        validate_dates: bool
    ) -> pd.DataFrame:
        """
        Load data from a single parquet file.

        Args:
            dataset_name: Name of dataset
            columns: Columns to load
            start_date: Start date for filtering
            end_date: End date for filtering
            validate_dates: Whether to validate dates

        Returns:
            DataFrame with loaded data
        """
        # Get metadata
        metadata = self.get_metadata(dataset_name)

        # Validate dataset exists
        parquet_file = self.data_dir / f"{dataset_name}.parquet"
        if not parquet_file.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

        # Validate columns
        available_columns = metadata.get('columns', [])
        if columns:
            invalid_columns = [col for col in columns if col not in available_columns]
            if invalid_columns:
                raise ValueError(
                    f"Invalid columns: {invalid_columns}. "
                    f"Available columns: {available_columns}"
                )

        # Validate and convert dates
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
        if end_date is not None:
            end_date = pd.to_datetime(end_date)

        # Validate date range against metadata
        if validate_dates and (start_date or end_date):
            dataset_date_range = self.get_date_range(dataset_name)
            if dataset_date_range is None:
                print(f"Warning: No date range found in metadata for {dataset_name}")
            else:
                min_date, max_date = dataset_date_range

                if start_date:
                    start_cmp = self._match_tz(pd.Timestamp(start_date), min_date.tzinfo)
                    min_cmp = self._match_tz(pd.Timestamp(min_date), start_cmp.tzinfo)
                    if start_cmp < min_cmp:
                        raise ValueError(
                            f"Start date {start_date.date()} is before dataset start date {min_date.date()}"
                        )

                if end_date:
                    end_cmp = self._match_tz(pd.Timestamp(end_date), max_date.tzinfo)
                    max_cmp = self._match_tz(pd.Timestamp(max_date), end_cmp.tzinfo)
                    if end_cmp > max_cmp:
                        raise ValueError(
                            f"End date {end_date.date()} is after dataset end date {max_date.date()}"
                        )

        # Load parquet file with optimized filtering
        # Get date column name from metadata for potential filtering
        date_col_in_metadata = None
        date_ranges = metadata.get('date_ranges', {})
        if date_ranges:
            date_col_in_metadata = list(date_ranges.keys())[0]
        
        # Prepare columns to load
        load_columns = None
        if columns:
            load_columns = columns.copy()
            if date_col_in_metadata and date_col_in_metadata not in load_columns:
                load_columns.insert(0, date_col_in_metadata)
        
        # Use PyArrow for efficient filtering (predicate pushdown)
        try:
            import pyarrow.parquet as pq
            import pyarrow.compute as pc
            
            # Read parquet table with PyArrow for better control
            table = pq.read_table(parquet_file, columns=load_columns)
            
            # Apply date filtering at PyArrow level (before converting to pandas)
            if date_col_in_metadata and (start_date is not None or end_date is not None):
                date_col_data = table.column(date_col_in_metadata)
                
                # Build filter mask
                mask = None
                col_tz = date_col_data.type.tz
                if start_date is not None:
                    start_ts = self._match_tz(pd.Timestamp(start_date), col_tz)
                    start_mask = pc.greater_equal(date_col_data, start_ts)
                    mask = start_mask if mask is None else pc.and_(mask, start_mask)

                if end_date is not None:
                    end_ts = self._match_tz(
                        pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
                        col_tz,
                    )
                    end_mask = pc.less_equal(date_col_data, end_ts)
                    mask = end_mask if mask is None else pc.and_(mask, end_mask)
                
                # Filter table
                if mask is not None:
                    table = table.filter(mask)
            
            # Convert to pandas
            df = table.to_pandas()
            
        except ImportError:
            # Fallback to pandas if PyArrow not available
            print("Warning: PyArrow not available, using pandas (less efficient)")
            if columns:
                try:
                    df = pd.read_parquet(parquet_file, columns=load_columns)
                except Exception as e:
                    print(f"Warning: Column-specific load failed, loading all columns: {e}")
                    df = pd.read_parquet(parquet_file)
                    df = df[load_columns]
            else:
                df = pd.read_parquet(parquet_file)
            
            # Manual date filtering in pandas (less efficient)
            if date_col_in_metadata and date_col_in_metadata in df.columns:
                col_tz = df[date_col_in_metadata].dt.tz
                if start_date is not None:
                    df = df[df[date_col_in_metadata] >= self._match_tz(start_date, col_tz)]
                if end_date is not None:
                    end_ts = self._match_tz(
                        pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
                        col_tz,
                    )
                    df = df[df[date_col_in_metadata] <= end_ts]
        
        except Exception as e:
            # Final fallback
            print(f"Warning: Optimized loading failed ({e}), using basic pandas")
            df = pd.read_parquet(parquet_file, columns=load_columns)

        # Detect and rename date column
        date_col = self._detect_date_column(df)
        if date_col:
            # Ensure it's datetime type
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col])
            
            # Rename to 'datetime' if not already
            if date_col != 'datetime':
                df = df.rename(columns={date_col: 'datetime'})
        else:
            print(f"Warning: No date column detected in {dataset_name}")
        
        # Final safety filter — covers the bare-pandas fallback path where date
        # filtering may not have been applied yet; idempotent on already-filtered data.
        if 'datetime' in df.columns and (start_date is not None or end_date is not None):
            col_tz = df['datetime'].dt.tz
            if start_date is not None:
                df = df[df['datetime'] >= self._match_tz(start_date, col_tz)]
            if end_date is not None:
                end_ts = self._match_tz(
                    pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
                    col_tz,
                )
                df = df[df['datetime'] <= end_ts]

        # Reset index
        df = df.reset_index(drop=True)

        # Store loaded data
        self.data = df
        self.current_source = dataset_name

        return df
    
    def _resample_data(
        self,
        df: pd.DataFrame,
        resample_freq: str,
        interpolation: str
    ) -> pd.DataFrame:
        """
        Resample time series data to a different frequency.
        
        Args:
            df: DataFrame with 'datetime' column
            resample_freq: Target frequency ('5min', '15min', '30min', '60min')
            interpolation: Interpolation method
            
        Returns:
            Resampled DataFrame
        """
        if 'datetime' not in df.columns:
            raise ValueError("DataFrame must have 'datetime' column for resampling")
        
        # Validate resample frequency
        valid_freqs = ['5min', '15min', '30min', '60min']
        if resample_freq not in valid_freqs:
            raise ValueError(f"resample must be one of {valid_freqs}, got: {resample_freq}")
        
        # Set datetime as index for resampling
        df_resampled = df.set_index('datetime')
        
        # Determine if this is upsampling or downsampling
        freq_minutes = {
            '5min': 5,
            '15min': 15,
            '30min': 30,
            '60min': 60
        }
        
        target_minutes = freq_minutes[resample_freq]
        
        # Detect original frequency (assume 30min by default)
        if len(df_resampled) > 1:
            time_diff = df_resampled.index[1] - df_resampled.index[0]
            original_minutes = time_diff.total_seconds() / 60
        else:
            original_minutes = 30  # Default assumption
        
        if target_minutes < original_minutes:
            # Upsampling: need interpolation
            df_resampled = df_resampled.resample(resample_freq).asfreq()
            
            # Apply interpolation to numeric columns
            numeric_cols = df_resampled.select_dtypes(include=[np.number]).columns
            
            if interpolation in ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']:
                # pandas built-in interpolation
                for col in numeric_cols:
                    df_resampled[col] = df_resampled[col].interpolate(
                        method=interpolation, 
                        limit_direction='both'
                    )
            elif interpolation in ['spline', 'pchip', 'akima']:
                # scipy-based interpolation
                from scipy import interpolate as scipy_interp
                
                for col in numeric_cols:
                    # Get non-null values
                    mask = df_resampled[col].notna()
                    if mask.sum() < 2:
                        continue
                    
                    x = np.arange(len(df_resampled))[mask]
                    y = df_resampled[col][mask].values
                    
                    # Create interpolator
                    if interpolation == 'spline':
                        f = scipy_interp.UnivariateSpline(x, y, s=0, k=3)
                    elif interpolation == 'pchip':
                        f = scipy_interp.PchipInterpolator(x, y)
                    elif interpolation == 'akima':
                        f = scipy_interp.Akima1DInterpolator(x, y)
                    
                    # Interpolate all values
                    x_new = np.arange(len(df_resampled))
                    df_resampled[col] = f(x_new)
            else:
                raise ValueError(f"Unknown interpolation method: {interpolation}")
                
        elif target_minutes > original_minutes:
            # Downsampling: aggregate
            # pandas >=2.0 requires `numeric_only=True` when the frame contains
            # any object/string columns (e.g. metadata cols carried alongside
            # the numeric series). Without it, the cython aggregator raises
            # `TypeError: agg function failed [how->mean,dtype->object]`.
            # All real measurement columns are numeric, so silently skipping
            # object cols is the right semantics.
            df_resampled = df_resampled.resample(resample_freq).mean(numeric_only=True)
        else:
            # Same frequency: no resampling needed
            pass
        
        # Reset index to make datetime a column again
        df_resampled = df_resampled.reset_index()
        
        return df_resampled
    
    def _load_and_merge_datasets(
        self,
        dataset_to_cols: Dict[str, List[str]],
        start_date: Optional[Union[str, datetime, date]],
        end_date: Optional[Union[str, datetime, date]],
        validate_dates: bool,
        merge_how: str
    ) -> pd.DataFrame:
        """
        Load data from multiple datasets and merge on datetime.
        
        Args:
            dataset_to_cols: Mapping of dataset_name -> columns to load
            start_date: Start date for filtering
            end_date: End date for filtering
            validate_dates: Whether to validate dates
            merge_how: How to merge datasets ('outer' or 'inner')
            
        Returns:
            Merged DataFrame
        """
        if merge_how not in ("inner", "outer"):
            raise ValueError("merge_how must be 'inner' or 'outer'")

        loaded_frames = []
        
        for dataset_name, cols in dataset_to_cols.items():
            df_part = self._load_single_dataset(
                dataset_name, cols, start_date, end_date, validate_dates
            )
            df_part = self._normalize_datetime_for_merge(df_part)
            loaded_frames.append(df_part)
        
        # Merge all DataFrames on datetime
        result = loaded_frames[0]
        for df_part in loaded_frames[1:]:
            result = result.merge(df_part, on='datetime', how=merge_how)
        
        # Sort by datetime
        if 'datetime' in result.columns:
            result = result.sort_values('datetime').reset_index(drop=True)
        
        # Store loaded data
        self.data = result
        self.current_source = ', '.join(dataset_to_cols.keys())
        
        return result

    def print_dataset_info(self, dataset_name: str):
        """
        Print information about a dataset.
        
        Args:
            dataset_name: Name of dataset
        """
        metadata = self.get_metadata(dataset_name)

        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        print(f"\nSource file: {metadata.get('source_file')}")
        print(f"Shape: {metadata['shape']['rows']} rows × {metadata['shape']['columns']} columns")

        print(f"\nColumns:")
        for col in metadata['columns']:
            dtype = metadata['dtypes'].get(col, 'unknown')
            print(f"  - {col} ({dtype})")

        date_ranges = metadata.get('date_ranges')
        if date_ranges:
            print(f"\nDate Range:")
            for col, range_info in date_ranges.items():
                print(f"  {col}:")
                print(f"    Min: {range_info['min']}")
                print(f"    Max: {range_info['max']}")
                print(f"    Count: {range_info['count']:,}")
                if range_info.get('missing', 0) > 0:
                    print(f"    Missing: {range_info['missing']}")

        numeric_stats = metadata.get('numeric_statistics')
        if numeric_stats:
            print(f"\nNumeric Statistics (first 5 columns):")
            for i, (col, stats) in enumerate(list(numeric_stats.items())[:5]):
                print(f"  {col}:")
                print(f"    Mean: {stats.get('mean', 0):.2f}")
                print(f"    Min: {stats.get('min', 0):.2f}")
                print(f"    Max: {stats.get('max', 0):.2f}")
                if i < 4 and i < len(numeric_stats) - 1:
                    print()

        print(f"\n{'=' * 60}")

    def get_summary(self) -> Optional[pd.DataFrame]:
        """
        Get summary of currently loaded data.
        
        Returns:
            DataFrame with summary statistics or None if no data loaded
        """
        if self.data is None:
            print("No data loaded. Use load_data() first.")
            return None

        print(f"\nLoaded data from: {self.current_source}")
        print(f"Shape: {self.data.shape}")
        print(f"\nColumns: {list(self.data.columns)}")

        if 'datetime' in self.data.columns:
            print(f"\nDate range:")
            print(f"  Min: {self.data['datetime'].min()}")
            print(f"  Max: {self.data['datetime'].max()}")

        return self.data.describe()


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()

    print("Available datasets:")
    for dataset in loader.list_available_datasets():
        print(f"  - {dataset}")

    # Show info for first dataset
    datasets = loader.list_available_datasets()
    if datasets:
        print(f"\n{'=' * 60}")
        loader.print_dataset_info(datasets[0])
