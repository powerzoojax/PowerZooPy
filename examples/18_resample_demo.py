"""Example: Data Resampling

Demonstrates how to resample time series data to different frequencies.
"""
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from powerzoo.data import DataLoader

# Initialize DataLoader
loader = DataLoader()

print("=" * 60)
print("Example 1: Original data (30min)")
print("=" * 60)

# Load original data
df_30min = loader.load_data(
    columns=['Actual'],
    start_date='2024-06-01',
    end_date='2024-06-02'
)

print(f"Shape: {df_30min.shape}")
print(f"Time resolution: ~{(df_30min['datetime'][1] - df_30min['datetime'][0]).total_seconds() / 60:.0f} minutes")
print(f"\nFirst 5 rows:")
print(df_30min.head())

print("\n" + "=" * 60)
print("Example 2: Upsample to 5min (linear interpolation)")
print("=" * 60)

df_5min = loader.load_data(
    columns=['Actual'],
    start_date='2024-06-01',
    end_date='2024-06-02',
    resample='5min',
    interpolation='linear'
)

print(f"Shape: {df_5min.shape}")
print(f"Time resolution: ~{(df_5min['datetime'][1] - df_5min['datetime'][0]).total_seconds() / 60:.0f} minutes")
print(f"\nFirst 10 rows:")
print(df_5min.head(10))

print("\n" + "=" * 60)
print("Example 3: Upsample to 15min (cubic interpolation)")
print("=" * 60)

df_15min = loader.load_data(
    columns=['Actual'],
    start_date='2024-06-01',
    end_date='2024-06-02',
    resample='15min',
    interpolation='cubic'
)

print(f"Shape: {df_15min.shape}")
print(f"Time resolution: ~{(df_15min['datetime'][1] - df_15min['datetime'][0]).total_seconds() / 60:.0f} minutes")

print("\n" + "=" * 60)
print("Example 4: Downsample to 60min (aggregation)")
print("=" * 60)

df_60min = loader.load_data(
    columns=['Actual'],
    start_date='2024-06-01',
    end_date='2024-06-02',
    resample='60min',
    interpolation='linear'  # Not used for downsampling
)

print(f"Shape: {df_60min.shape}")
print(f"Time resolution: ~{(df_60min['datetime'][1] - df_60min['datetime'][0]).total_seconds() / 60:.0f} minutes")
print(f"\nFirst 5 rows:")
print(df_60min.head())

print("\n" + "=" * 60)
print("Example 5: Compare different interpolation methods")
print("=" * 60)

# Load with different interpolation methods
methods = ['linear', 'quadratic', 'cubic', 'pchip']
results = {}

for method in methods:
    df = loader.load_data(
        columns=['Actual'],
        start_date='2024-06-01 10:00:00',
        end_date='2024-06-01 14:00:00',
        resample='5min',
        interpolation=method
    )
    results[method] = df

print(f"Loaded data with {len(methods)} interpolation methods")
print(f"Time points per method: {len(results['linear'])}")

# Plot comparison
fig, ax = plt.subplots(figsize=(12, 6))

for method, df in results.items():
    ax.plot(df['datetime'], df['Actual'], label=method, marker='o' if method == 'linear' else None, markersize=3)

# Plot original 30min data
ax.plot(df_30min[(df_30min['datetime'] >= '2024-06-01 10:00:00') &
                 (df_30min['datetime'] <= '2024-06-01 14:00:00')]['datetime'],
        df_30min[(df_30min['datetime'] >= '2024-06-01 10:00:00') &
                 (df_30min['datetime'] <= '2024-06-01 14:00:00')]['Actual'],
        'ko', markersize=8, label='Original (30min)', zorder=10)

ax.set_xlabel('Time')
ax.set_ylabel('Actual Demand (MW)')
ax.set_title('Comparison of Interpolation Methods (30min -> 5min)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Save plot
plot_path = os.path.join(os.path.dirname(__file__), 'x18_resample_demo', 'interpolation_comparison')
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path + '.png', dpi=150)
plt.savefig(plot_path + '.pdf', dpi=150)
print(f"\nPlot saved to: {plot_path}")

print("\n" + "=" * 60)
print("All examples completed!")
print("=" * 60)
