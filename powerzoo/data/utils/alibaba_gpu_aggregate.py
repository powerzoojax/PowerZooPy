"""Download Alibaba GPU 2020 trace and aggregate to cluster-level time series.

Downloads ``pai_machine_metric.tar.gz`` (~198 MB) and
``pai_machine_spec.tar.gz`` (~32 KB), aggregates per-instance machine-level
GPU / CPU metrics into 5-min cluster-average time series, then saves a
tiny CSV + parquet (~20 KB).

Usage::

    python -m powerzoo.data.utils.alibaba_gpu_aggregate [--bin-seconds 300]
"""

from __future__ import annotations

import argparse
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

_ALIYUN_BASE = "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces"

_SOURCE_DIR = Path(__file__).resolve().parent.parent / "source"
_PARQUET_DIR = Path(__file__).resolve().parent.parent / "parquet"


def _download(url: str, dest: Path) -> Path:
    print(f"  Downloading {url} ...")
    urlretrieve(url, dest)
    print(f"  → {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    return dest


def _extract_csv(tar_path: Path, tmpdir: Path) -> Path:
    """Extract CSV from tar.gz, return path to CSV."""
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        csv_member = [m for m in members if m.name.endswith(".csv")]
        if not csv_member:
            raise FileNotFoundError(f"No CSV in {tar_path}")
        tf.extract(csv_member[0], tmpdir, filter="data")
        return tmpdir / csv_member[0].name


def aggregate_gpu_trace(bin_seconds: int = 300) -> Path:
    """Download, aggregate, and save the GPU cluster time series."""
    _SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="alibaba_gpu_") as tmpdir:
        tmpdir = Path(tmpdir)

        # --- Download & extract machine_metric ---
        print("Step 1/3: Downloading pai_machine_metric (~198 MB)...")
        tar_path = tmpdir / "pai_machine_metric.tar.gz"
        _download(f"{_ALIYUN_BASE}/pai_machine_metric.tar.gz", tar_path)

        print("  Extracting...")
        csv_path = _extract_csv(tar_path, tmpdir)
        del tar_path  # allow GC

        print(f"  Reading CSV ({csv_path.stat().st_size / 1024 / 1024:.1f} MB)...")
        # CSV has no header; positional columns per README:
        # 0:worker_name  1:machine  2:start_time  3:end_time
        # 4:cpu_iowait  5:cpu_kernel  6:cpu_usr  7:machine_gpu
        # 8:machine_load_1  9:net_receive  10:num_worker  11:machine_cpu
        POSITIONAL_NAMES = [
            "worker_name", "machine", "start_time", "end_time",
            "machine_cpu_iowait", "machine_cpu_kernel", "machine_cpu_usr",
            "machine_gpu", "machine_load_1", "machine_net_receive",
            "machine_num_worker", "machine_cpu",
        ]
        use_idx = [1, 2, 3, 7, 11]  # machine, start, end, gpu, cpu
        df = pd.read_csv(
            csv_path, header=None, names=POSITIONAL_NAMES, usecols=use_idx,
        )
        print(f"  Loaded {len(df):,} rows")

        # --- Download machine_spec (tiny, for GPU count normalization) ---
        print("\nStep 2/4: Downloading pai_machine_spec (~32 KB)...")
        spec_tar = tmpdir / "pai_machine_spec.tar.gz"
        _download(f"{_ALIYUN_BASE}/pai_machine_spec.tar.gz", spec_tar)
        spec_csv = _extract_csv(spec_tar, tmpdir)
        SPEC_NAMES = ["machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"]
        spec = pd.read_csv(spec_csv, header=None, names=SPEC_NAMES)
        gpu_per_machine = spec.set_index("machine")["cap_gpu"].to_dict()
        print(f"  {len(spec)} machines, GPU counts: {sorted(set(gpu_per_machine.values()))}")

        # --- Aggregate to time bins ---
        # machine_gpu is the SUM of GPU utilizations across all GPUs on a machine
        # (e.g. 8-GPU machine at 100% each → machine_gpu=800).
        # Normalize to per-GPU average by dividing by cap_gpu.
        print(f"\nStep 3/4: Aggregating to {bin_seconds}s bins (this may take a minute)...")
        df = df.dropna(subset=["start_time", "end_time", "machine_gpu"])

        # Per-machine average: each machine's GPU metric is the mean of
        # its instance-level observations, normalized by GPU count.
        machine_agg = (
            df.groupby("machine")
            .agg(
                start_time=("start_time", "min"),
                end_time=("end_time", "max"),
                machine_gpu=("machine_gpu", "mean"),
                machine_cpu=("machine_cpu", "mean"),
            )
            .reset_index()
        )
        # Normalize machine_gpu by number of GPUs → per-GPU utilization %
        machine_agg["machine_gpu"] = machine_agg.apply(
            lambda r: r["machine_gpu"] / max(gpu_per_machine.get(r["machine"], 1), 1),
            axis=1,
        )
        print(f"  Unique machines with data: {len(machine_agg)}")

        t_min = machine_agg["start_time"].min()
        t_max = machine_agg["end_time"].max()
        bins = np.arange(t_min, t_max + bin_seconds, bin_seconds)
        n_bins = len(bins) - 1

        gpu_sum = np.zeros(n_bins, dtype=np.float64)
        cpu_sum = np.zeros(n_bins, dtype=np.float64)
        count = np.zeros(n_bins, dtype=np.float64)

        start_arr = machine_agg["start_time"].values.astype(np.float64)
        end_arr = machine_agg["end_time"].values.astype(np.float64)
        gpu_arr = machine_agg["machine_gpu"].values.astype(np.float64)
        cpu_arr = machine_agg["machine_cpu"].values.astype(np.float64)

        for i in range(len(machine_agg)):
            s, e = start_arr[i], end_arr[i]
            if np.isnan(s) or np.isnan(e) or e <= s:
                continue
            g, c = gpu_arr[i], cpu_arr[i]
            if np.isnan(g):
                continue

            bin_s = max(0, int((s - t_min) // bin_seconds))
            bin_e = min(n_bins - 1, int((e - t_min) // bin_seconds))

            for b in range(bin_s, bin_e + 1):
                gpu_sum[b] += g
                cpu_sum[b] += c
                count[b] += 1

        mask = count > 0
        gpu_avg = np.where(mask, gpu_sum / count, np.nan)
        cpu_avg = np.where(mask, cpu_sum / count, np.nan)

        result = pd.DataFrame({
            "time_stamp": np.arange(n_bins),
            "gpu_util_percent": gpu_avg,
            "cpu_util_percent": cpu_avg,
        })

        result = result.dropna().reset_index(drop=True)
        result["time_stamp"] = np.arange(len(result))

        print(f"  Result: {len(result)} time steps "
              f"({len(result) * bin_seconds / 86400:.1f} days)")
        print(f"  GPU util: [{result['gpu_util_percent'].min():.1f}%, "
              f"{result['gpu_util_percent'].max():.1f}%]")
        print(f"  CPU util: [{result['cpu_util_percent'].min():.1f}%, "
              f"{result['cpu_util_percent'].max():.1f}%]")

    # --- Save ---
    csv_out = _SOURCE_DIR / "alibaba2020_gpu_cluster_300s.csv"
    result.to_csv(csv_out, index=False)
    print(f"\nSaved CSV: {csv_out.name} ({csv_out.stat().st_size / 1024:.1f} KB)")

    pq_out = _PARQUET_DIR / "alibaba_gpu_2020_300s.parquet"
    result.to_parquet(pq_out, index=False)
    print(f"Saved parquet: {pq_out.name} ({pq_out.stat().st_size / 1024:.1f} KB)")

    return pq_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate Alibaba GPU 2020 trace to cluster-level time series"
    )
    parser.add_argument(
        "--bin-seconds", type=int, default=300,
        help="Time bin width in seconds (default: 300 = 5min)"
    )
    args = parser.parse_args()
    aggregate_gpu_trace(bin_seconds=args.bin_seconds)
