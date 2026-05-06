"""Example: Auto-load data by column names

Demonstrates how to load data without specifying dataset names.
DataLoader automatically finds the correct datasets based on column names.
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from powerzoo.data import DataLoader

# Initialize DataLoader
loader = DataLoader()

print("=" * 60)
print("Example 1: Load single column (auto-detect dataset)")
print("=" * 60)

# Load Actual without specifying dataset
df1 = loader.load_data(
    columns=['Actual'],
    start_date='2024-01-01',
    end_date='2024-01-07'
)

print(f"\nLoaded from: {loader.current_source}")
print(f"Shape: {df1.shape}")
print(f"Columns: {list(df1.columns)}")
print(f"\nFirst 5 rows:")
print(df1.head())

print("\n" + "=" * 60)
print("Example 2: Load multiple columns from same dataset")
print("=" * 60)

# Load multiple columns from same dataset
df2 = loader.load_data(
    columns=['Actual', 'DAForecast'],
    start_date='2024-06-01',
    end_date='2024-06-03'
)

print(f"\nLoaded from: {loader.current_source}")
print(f"Shape: {df2.shape}")
print(f"Columns: {list(df2.columns)}")
print(f"\nSummary:")
print(df2.describe())

print("\n" + "=" * 60)
print("Example 3: Load columns from multiple datasets (auto-merge)")
print("=" * 60)

# Load columns from different datasets - will auto-merge on datetime
df3 = loader.load_data(
    columns=['Actual', 'Biomass', 'Solar'],
    start_date='2024-06-01',
    end_date='2024-06-03'
)

print(f"\nLoaded from: {loader.current_source}")
print(f"Shape: {df3.shape}")
print(f"Columns: {list(df3.columns)}")
print(f"\nFirst 5 rows:")
print(df3.head())

print("\n" + "=" * 60)
print("Example 4: Find which datasets contain specific columns")
print("=" * 60)

# Check which datasets contain specific columns
columns_to_check = ['Actual', 'Biomass', 'Solar', 'Nuclear']
dataset_map = loader.find_datasets_for_columns(columns_to_check)

print(f"\nColumn distribution:")
for dataset, cols in dataset_map.items():
    print(f"  {dataset}:")
    for col in cols:
        print(f"    - {col}")

print("\n" + "=" * 60)
print("Example 5: Load renewable generation data")
print("=" * 60)

# Load multiple renewable sources
df_renewable = loader.load_data(
    columns=['Solar', 'Wind Offshore', 'Wind Onshore'],
    start_date='2024-06-01',
    end_date='2024-06-07'
)

print(f"\nLoaded from: {loader.current_source}")
print(f"Shape: {df_renewable.shape}")

# Calculate total renewable generation
df_renewable['Total_Renewable'] = (
    df_renewable['Solar'] + 
    df_renewable['Wind Offshore'] + 
    df_renewable['Wind Onshore']
)

print(f"\nRenewable generation statistics:")
print(df_renewable[['Solar', 'Wind Offshore', 'Wind Onshore', 'Total_Renewable']].describe())

print("\n" + "=" * 60)
print("All examples completed!")
print("=" * 60)


import matplotlib.pyplot as plt
# Load multiple renewable sources
df_renewable = loader.load_data(
    columns=['Solar', 'Wind Offshore', 'Wind Onshore'],
    start_date='2024-06-01',
    end_date='2024-06-01'
)
df_renewable[['Solar', 'Wind Offshore', 'Wind Onshore']].plot()
plt.show()
