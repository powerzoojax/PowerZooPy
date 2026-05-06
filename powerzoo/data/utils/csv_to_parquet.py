"""Convert CSV files to Parquet format

This script converts all CSV files in the source directory to Parquet format
and saves them to the parquet directory. Parquet format provides better
compression and faster read/write performance for large datasets.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np


def _get_parquet_engine() -> str:
    """
    Get available Parquet engine.
    
    Returns:
        str: 'pyarrow' or 'fastparquet'
        
    Raises:
        ImportError: If neither engine is available
    """
    try:
        import pyarrow
        return 'pyarrow'
    except ImportError:
        try:
            import fastparquet
            return 'fastparquet'
        except ImportError:
            raise ImportError(
                "Neither pyarrow nor fastparquet is installed. "
                "Please install one of them: "
                "conda install -c conda-forge pyarrow "
                "or pip install pyarrow"
            )


def _convert_to_serializable(obj: Any) -> Any:
    """
    Convert pandas/numpy types to JSON-serializable types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if pd.isna(obj):
        return None
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    else:
        return obj


def _generate_metadata(df: pd.DataFrame, csv_file: Path, parquet_file: Path) -> Dict[str, Any]:
    """
    Generate metadata JSON for the converted data.
    
    Args:
        df: DataFrame containing the data
        csv_file: Path to source CSV file
        parquet_file: Path to output Parquet file
        
    Returns:
        dict: Metadata dictionary
    """
    metadata = {
        "source_file": str(csv_file.name),
        "parquet_file": str(parquet_file.name),
        "generated_at": datetime.now().isoformat(),
        "shape": {
            "rows": int(len(df)),
            "columns": int(len(df.columns))
        },
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample_data": {}
    }
    
    # Get first 5 rows as sample
    sample_df = df.head(5)
    for i, (_, row) in enumerate(sample_df.iterrows()):
        row_dict = {}
        for col in df.columns:
            row_dict[col] = _convert_to_serializable(row[col])
        metadata["sample_data"][f"row_{i}"] = row_dict
    
    # Extract date range if date columns exist
    # Known date column names
    known_date_columns = [
        'SETTLEMENT_DATETIME', 'INTERVAL_DATETIME', 'SETTLEMENT_DATE', 
        'datetime', 'startTime', 'START_TIME', 'STARTTIME',
        'date', 'DATE', 'time', 'TIME', 'timestamp', 'TIMESTAMP'
    ]
    
    # Also check columns that are already datetime type
    datetime_type_columns = df.select_dtypes(include=[np.datetime64, 'datetime64[ns]']).columns.tolist()
    
    # Combine known names and datetime type columns
    date_columns_to_check = set(known_date_columns) | set(datetime_type_columns)
    
    date_ranges = {}
    
    for col in df.columns:
        # Check if column name matches known date columns or is datetime type
        if col not in date_columns_to_check:
            # Try to detect by column name pattern (case-insensitive)
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['time', 'date', 'datetime', 'timestamp']):
                date_columns_to_check.add(col)
            else:
                continue
        
        try:
            # Try to convert to datetime if not already
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_series = df[col]
            else:
                # Try to parse as datetime
                date_series = pd.to_datetime(df[col], errors='coerce')
            
            # Remove NaT values
            valid_dates = date_series.dropna()
            if len(valid_dates) > 0:
                # Check if we actually have valid dates (not all NaT)
                min_date = valid_dates.min()
                max_date = valid_dates.max()
                
                # Only add if we have meaningful date range
                if pd.notna(min_date) and pd.notna(max_date):
                    date_ranges[col] = {
                        "min": min_date.isoformat(),
                        "max": max_date.isoformat(),
                        "count": int(len(valid_dates)),
                        "missing": int(len(df) - len(valid_dates))
                    }
        except Exception as e:
            # If conversion fails, skip this column
            pass
    
    if date_ranges:
        metadata["date_ranges"] = date_ranges
    else:
        # Add a note if no date ranges were found
        metadata["date_ranges"] = None
        metadata["date_range_note"] = "No date columns detected. Check column names and types."
    
    # Add basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        stats = {}
        for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
            try:
                col_stats = df[col].describe()
                stats[col] = {
                    "count": int(col_stats.get('count', 0)),
                    "mean": float(col_stats.get('mean', 0)) if not pd.isna(col_stats.get('mean')) else None,
                    "std": float(col_stats.get('std', 0)) if not pd.isna(col_stats.get('std')) else None,
                    "min": float(col_stats.get('min', 0)) if not pd.isna(col_stats.get('min')) else None,
                    "max": float(col_stats.get('max', 0)) if not pd.isna(col_stats.get('max')) else None,
                    "missing": int(df[col].isna().sum())
                }
            except Exception:
                pass
        
        if stats:
            metadata["numeric_statistics"] = stats
    
    return metadata


def get_source_and_output_dirs() -> tuple[Path, Path]:
    """
    Get source and output directories.
    
    Source directory: powerzoo/datas/source (../source relative to utils)
    Output directory: powerzoo/datas/parquet (../parquet relative to utils)
    
    Returns:
        tuple: (source_dir, output_dir)
    """
    # Get the directory where this script is located (powerzoo/datas/utils)
    script_dir = Path(__file__).resolve().parent
    # Source directory is ../source relative to utils
    source_dir = script_dir.parent / "source"
    # Output directory is ../parquet relative to utils
    output_dir = script_dir.parent / "parquet"
    
    return source_dir, output_dir


def convert_csv_to_parquet(
    csv_file: Path,
    output_dir: Path,
    compression: str = "snappy",
    verbose: bool = True
) -> Path:
    """
    Convert a single CSV file to Parquet format.
    
    Args:
        csv_file: Path to input CSV file
        output_dir: Directory to save output Parquet file
        compression: Compression codec ('snappy', 'gzip', 'brotli', 'lz4', 'zstd')
        verbose: If True, print progress messages
        
    Returns:
        Path: Path to output Parquet file
    """
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename (replace .csv with .parquet)
    output_file = output_dir / f"{csv_file.stem}.parquet"
    
    if verbose:
        print(f"Converting: {csv_file.name} -> {output_file.name}")
    
    # Read CSV file
    # Use low_memory=False to avoid mixed type inference issues
    df = pd.read_csv(csv_file, low_memory=False)
    
    if verbose:
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Convert date columns if present (common in GB data)
    # Known date column names
    date_columns = [
        'SETTLEMENT_DATETIME', 'INTERVAL_DATETIME', 'SETTLEMENT_DATE',
        'startTime', 'START_TIME', 'STARTTIME', 'datetime'
    ]
    
    # Also check columns that look like dates by name pattern
    for col in df.columns:
        col_lower = col.lower()
        is_date_column = (
            col in date_columns or
            (any(keyword in col_lower for keyword in ['time', 'date', 'datetime', 'timestamp']) and
             col_lower not in ['settlementperiod', 'period'])  # Exclude period numbers
        )
        
        if is_date_column and not pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if verbose:
                    print(f"  Converted {col} to datetime")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not convert {col} to datetime: {e}")
    
    # Save as Parquet
    parquet_engine = _get_parquet_engine()
    df.to_parquet(
        output_file,
        engine=parquet_engine,
        compression=compression,
        index=False
    )
    
    # Generate and save metadata JSON
    metadata = _generate_metadata(df, csv_file, output_file)
    json_file = output_dir / f"{csv_file.stem}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    if verbose:
        # Get file sizes for comparison
        csv_size = csv_file.stat().st_size / (1024 * 1024)  # MB
        parquet_size = output_file.stat().st_size / (1024 * 1024)  # MB
        compression_ratio = (1 - parquet_size / csv_size) * 100 if csv_size > 0 else 0
        print(f"  Saved: {parquet_size:.2f} MB (compression: {compression_ratio:.1f}%)")
        print(f"  Metadata: {json_file.name}")
    
    return output_file


def convert_all_csvs(
    source_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    compression: str = "snappy",
    verbose: bool = True
) -> list[Path]:
    """
    Convert all CSV files in source directory to Parquet format.
    
    Args:
        source_dir: Source directory containing CSV files (default: ../source)
        output_dir: Output directory for Parquet files (default: ../parquet)
        compression: Compression codec for Parquet files
        verbose: If True, print progress messages
        
    Returns:
        list: List of output Parquet file paths
    """
    if source_dir is None or output_dir is None:
        source_dir, output_dir = get_source_and_output_dirs()
    
    # Find all CSV files
    csv_files = list(source_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {source_dir}")
        return []
    
    if verbose:
        print(f"Found {len(csv_files)} CSV file(s) in {source_dir}")
        print(f"Output directory: {output_dir}")
        print("-" * 60)
    
    output_files = []
    for csv_file in csv_files:
        try:
            output_file = convert_csv_to_parquet(
                csv_file, output_dir, compression=compression, verbose=verbose
            )
            output_files.append(output_file)
        except Exception as e:
            print(f"Error converting {csv_file.name}: {e}")
            continue
    
    if verbose:
        print("-" * 60)
        print(f"Conversion complete! {len(output_files)} file(s) converted.")
    
    return output_files


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert CSV files to Parquet format"
    )
    parser.add_argument(
        "--source-dir", "-s",
        type=str,
        default=None,
        help="Source directory containing CSV files (default: ../source)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for Parquet files (default: ../parquet)"
    )
    parser.add_argument(
        "--compression", "-c",
        type=str,
        default="snappy",
        choices=["snappy", "gzip", "brotli", "lz4", "zstd"],
        help="Compression codec (default: snappy)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir) if args.source_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    convert_all_csvs(
        source_dir=source_dir,
        output_dir=output_dir,
        compression=args.compression,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()

