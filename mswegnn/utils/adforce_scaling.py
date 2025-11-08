# mswegnn/utils/adforce_scaling.py

"""
Computes normalization statistics (mean, std) for the Adforce dataset.

This script iterates through all training NetCDF files, computes the
global mean and standard deviation for all input and output features,
and saves them to a YAML file.

This updated version computes stats for:
1. x_static: (DEM, slopex, slopey, area)
2. x_dynamic: (WX, WY, P)
3. y: (WD, VX, VY)
4. y_delta: (WD(t+1)-WD(t), VX(t+1)-VX(t), VY(t+1)-VY(t))

This script is intended to be run *once* on the *training* dataset,
and its output ('scaling_stats.yaml') is then used by
AdforceLazyDataset during training and validation.
"""

import os
from typing import List, Dict
import yaml
import numpy as np
import xarray as xr
from tqdm import tqdm
import warnings
import argparse
import glob
import sys


# These lists define the variables we will calculate stats for.
VARS_STATIC = ["DEM", "slopex", "slopey", "area"]
VARS_DYNAMIC = ["WX", "WY", "P"]
VARS_TARGET = ["WD", "VX", "VY"]


class StatsAggregator:
    """
    Helper class to compute mean/std in a single pass using the
    numerically stable 'sum' and 'sum-of-squares' method.

    This is fast because it relies on vectorized numpy operations.
    (This class is from your original script)
    """

    def __init__(self, var_names: list[str]):
        self.var_names = var_names
        self.stats = {
            var: {"n": 0, "sum": 0.0, "sum_sq": 0.0} for var in self.var_names
        }

    def update(self, var_name: str, data_array: np.ndarray):
        """
        Updates the running sums for a single variable from a new batch of data.
        """
        if var_name not in self.stats:
            warnings.warn(f"'{var_name}' not in tracked variables. Skipping.")
            return

        data = data_array.astype(np.float64)
        n = np.isfinite(data).sum()
        if n == 0:
            return

        self.stats[var_name]["n"] += n
        self.stats[var_name]["sum"] += np.nansum(data)
        self.stats[var_name]["sum_sq"] += np.nansum(data**2)

    def compute(self) -> tuple[dict, dict]:
        """
        Computes the final mean and std from the aggregated stats.
        """
        means = {}
        stds = {}

        for var in self.var_names:
            stats = self.stats[var]
            n = stats["n"]
            if n == 0:
                means[var] = 0.0
                stds[var] = 1.0
                warnings.warn(f"No valid data found for '{var}'. Using mean=0, std=1.")
                continue

            sum_ = stats["sum"]
            sum_sq = stats["sum_sq"]
            mean = sum_ / n
            variance = (sum_sq / n) - (mean**2)
            std = np.sqrt(max(0.0, variance))
            means[var] = float(mean)
            stds[var] = float(std)

        return means, stds


def compute_and_save_adforce_stats(nc_files: list[str], output_path: str):
    """
    Calculates mean/std stats for Adforce data across multiple files
    and saves them to a YAML file.
    """
    static_agg = StatsAggregator(VARS_STATIC)
    dynamic_agg = StatsAggregator(VARS_DYNAMIC)
    target_agg = StatsAggregator(VARS_TARGET)
    target_delta_agg = StatsAggregator(VARS_TARGET)

    for nc_path in tqdm(nc_files, desc="Calculating Scaling Stats"):
        try:
            with xr.open_dataset(nc_path) as ds:

                # --- Check for variable existence ---
                missing_vars = []
                # Check static/dynamic first
                for v in VARS_STATIC + VARS_DYNAMIC:
                    if v not in ds.data_vars:
                        missing_vars.append(v)

                # Special check for targets, as they are crucial
                missing_targets = []
                for v in VARS_TARGET:
                    if v not in ds.data_vars:
                        missing_targets.append(v)

                if missing_targets:
                    warnings.warn(
                        f"Skipping file {nc_path}: Missing CRITICAL target variables: {missing_targets}"
                    )
                    continue

                if missing_vars:
                    warnings.warn(
                        f"WARNING: File {nc_path} is missing non-critical variables: {missing_vars}"
                    )
                # --- End Check ---

                # Static variables
                for var in VARS_STATIC:
                    if var in ds:  # Only update if it exists
                        static_agg.update(var, ds[var].values)

                # Dynamic variables
                for var in VARS_DYNAMIC:
                    if var in ds:  # Only update if it exists
                        dynamic_agg.update(var, ds[var].values)

                # --- MODIFIED: Process targets and deltas WITH MASKING ---

                # 1. Load all target data
                #    .to_array() creates shape (variable, time, nodes)
                y_data = ds[VARS_TARGET].to_array().load().values

                # 2. **CRITICAL FIX 1:** Mask fill values
                #    Identify non-physical Water Depth (e.g., WD < -1m)
                #    VARS_TARGET = ["WD", "VX", "VY"], so WD is at index 0.
                wd_data = y_data[0]  # Shape (time, nodes)

                # Create a mask where water depth is non-physical
                fill_mask = wd_data < -1.0

                # Apply this mask to all target variables
                y_data_cleaned = y_data.copy()
                y_data_cleaned[0, fill_mask] = np.nan  # Mask WD
                y_data_cleaned[1, fill_mask] = np.nan  # Mask VX
                y_data_cleaned[2, fill_mask] = np.nan  # Mask VY

                # 3. Update stats for the raw targets (y)
                for i, var in enumerate(VARS_TARGET):
                    target_agg.update(var, y_data_cleaned[i])

                # 4. **CRITICAL FIX 2:** Calculate deltas along the TIME axis
                #    y_data_cleaned shape is (variable, time, nodes)
                #    We must diff along axis=1 (the time dimension)
                deltas = np.diff(
                    y_data_cleaned, axis=1
                )  # Shape (variable, time-1, nodes)

                # 5. Update stats for the deltas (y_delta)
                for i, var in enumerate(VARS_TARGET):
                    target_delta_agg.update(
                        var, deltas[i]
                    )  # deltas[i] is (time-1, nodes)

                # --- END MODIFIED ---

        except Exception as e:
            warnings.warn(f"Failed to process {nc_path}: {e}. Skipping file.")
            continue

    # Compute the final statistics
    static_means, static_stds = static_agg.compute()
    dynamic_means, dynamic_stds = dynamic_agg.compute()
    target_means, target_stds = target_agg.compute()
    target_delta_means, target_delta_stds = target_delta_agg.compute()

    # Format the data for YAML
    output_data = {
        "x_static_mean": [static_means[v] for v in VARS_STATIC],
        "x_static_std": [static_stds[v] for v in VARS_STATIC],
        "x_dynamic_mean": [dynamic_means[v] for v in VARS_DYNAMIC],
        "x_dynamic_std": [dynamic_stds[v] for v in VARS_DYNAMIC],
        "y_mean": [target_means[v] for v in VARS_TARGET],
        "y_std": [target_stds[v] for v in VARS_TARGET],
        "y_delta_mean": [target_delta_means[v] for v in VARS_TARGET],
        "y_delta_std": [target_delta_stds[v] for v in VARS_TARGET],
    }

    # Save to the YAML file
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(output_data, f, sort_keys=False)
    except Exception as e:
        raise IOError(f"Failed to write scaling stats to {output_path}: {e}")

    print("\n" + "=" * 30)
    print(f"Stats calculation complete. Saved to {output_path}.")
    print(yaml.dump(output_data))
    print("=" * 30)


if __name__ == "__main__":
    """
    Run the main script.
    
    To run the main script:
    python -m mswegnn.utils.adforce_scaling \
        -o data_processed/train/scaling_stats.yaml \
        --files /path/to/data/*.nc
    """

    parser = argparse.ArgumentParser(
        description="Compute normalization stats for Adforce dataset."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to save the output scaling_stats.yaml. If specified, runs the main script.",
        required=True,
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="List of training .nc files (can use glob patterns like 'data/*.nc'). Required.",
        required=True,
    )

    args = parser.parse_args()

    train_files = []
    for pattern in args.files:
        train_files.extend(glob.glob(pattern))

    if not train_files:
        print("Error: No files found matching the patterns in --files.")
        sys.exit(1)

    compute_and_save_adforce_stats(
        train_files=sorted(list(set(train_files))),
        output_path=args.output,
    )
