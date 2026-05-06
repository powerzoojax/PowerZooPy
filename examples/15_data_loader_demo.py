"""Example: Using DataLoader to load parquet data

Demonstrates how to use DataLoader to load and filter data from parquet files.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from powerzoo.data.data_loader import DataLoader
from powerzoo.data import signals as S

# Initialize DataLoader
loader = DataLoader()

print("=" * 60)
print("Example 1: List available datasets")
print("=" * 60)
datasets = loader.list_available_datasets()
print(f"Found {len(datasets)} datasets:")
for dataset in datasets:
    print(f"  - {dataset}")

# Show info for each dataset
print("\n" + "=" * 60)
print("Example 2: Dataset information")
print("=" * 60)
for dataset in datasets:
    loader.print_dataset_info(dataset)

# Load data with specific columns and date range
print("\n" + "=" * 60)
print("Example 3: Load data with filters")
print("=" * 60)

# Example 3.1: Load ActualDemand data
dataset_name = "GB_Forecast_Actual_Demand_2023_2025_30min"
print(f"\nLoading {dataset_name}...")

# Get available columns
columns = loader.get_available_columns(dataset_name)
print(f"Available columns: {columns}")

# Get date range
date_range = loader.get_date_range(dataset_name)
if date_range:
    print(f"Date range: {date_range[0].date()} to {date_range[1].date()}")

# Load data for specific date range and signals
df = loader.load_signals(
    [S.LOAD_ACTUAL_MW, S.LOAD_FORECAST_DA_MW],
    source="gb",
    start_date='2024-01-01',
    end_date='2024-01-07',
    resample="30min",
)

print(f"\nLoaded data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nDate range in loaded data:")
print(f"  Min: {df['datetime'].min()}")
print(f"  Max: {df['datetime'].max()}")

# Example 3.2: Load generation data
print("\n" + "=" * 60)
print("Example 4: Load generation by type")
print("=" * 60)

dataset_name = "GB_Gen_by_Type_2016_2025_30min"
print(f"\nLoading {dataset_name}...")

# Load specific generation types
df_gen = loader.load_data(
    dataset_name=dataset_name,
    columns=['Solar', 'Wind Offshore', 'Wind Onshore', 'Nuclear'],
    start_date='2024-06-01',
    end_date='2024-06-03'
)

print(f"\nLoaded data shape: {df_gen.shape}")
print(f"Columns: {list(df_gen.columns)}")
print(f"\nSummary statistics:")
print(df_gen.describe())

# Example 3.3: Load GB market index data through the semantic API
print("\n" + "=" * 60)
print("Example 5: Load GB market MID data")
print("=" * 60)

df_market = loader.load_signals(
    [S.MARKET_MID_PRICE_APX, S.MARKET_MID_VOLUME_APX],
    source="gb",
    start_date="2025-12-14",
    end_date="2025-12-14",
    resample="30min",
)

print(f"\nLoaded data shape: {df_market.shape}")
print(f"Columns: {list(df_market.columns)}")
print(f"\nFirst 5 rows:")
print(df_market.head())

print("\n" + "=" * 60)
print("All examples completed!")
print("=" * 60)
