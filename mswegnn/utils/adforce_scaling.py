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
# Using lists here (like your old script) makes the code
# more maintainable and ensures the order is correct.
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
        # We store stats in a dict, e.g.,
        # self.stats['DEM'] = {'n': 0, 'sum': 0.0, 'sum_sq': 0.0}
        self.stats = {
            var: {"n": 0, "sum": 0.0, "sum_sq": 0.0} for var in self.var_names
        }

    def update(self, var_name: str, data_array: np.ndarray):
        """
        Updates the running sums for a single variable from a new batch of data.

        Args:
            var_name (str): The name of the variable (e.g., 'WD').
            data_array (np.ndarray): The numpy array of new data points.
        """
        if var_name not in self.stats:
            warnings.warn(f"'{var_name}' not in tracked variables. Skipping.")
            return

        # Use float64 for high precision in sum/sum_sq
        data = data_array.astype(np.float64)

        # Count non-NaN values
        n = np.isfinite(data).sum()
        if n == 0:
            return  # Skip if data is all NaN

        self.stats[var_name]["n"] += n
        self.stats[var_name]["sum"] += np.nansum(data)
        self.stats[var_name]["sum_sq"] += np.nansum(data**2)

    def compute(self) -> tuple[dict, dict]:
        """
        Computes the final mean and std from the aggregated stats.

        Returns:
            tuple[dict, dict]: (dict_of_means, dict_of_stds)
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

            # mean = E[X] = sum(X) / n
            mean = sum_ / n

            # variance = E[X^2] - (E[X])^2
            variance = (sum_sq / n) - (mean**2)

            # Clamp variance at 0 to avoid numerical issues (e.g., -1e-10)
            std = np.sqrt(max(0.0, variance))

            # Convert from numpy float to standard python float for YAML
            means[var] = float(mean)
            stds[var] = float(std)

        return means, stds


def compute_and_save_adforce_stats(nc_files: list[str], output_path: str):
    """
    Calculates mean/std stats for Adforce data across multiple files
    and saves them to a YAML file.

    This function processes files in a single pass to save memory.

    Args:
        nc_files (list[str]): A list of file paths to the training .nc files.
        output_path (str): The file path to save the 'scaling_stats.yaml'.
    """
    # Initialize aggregators for our variable groups
    static_agg = StatsAggregator(VARS_STATIC)
    dynamic_agg = StatsAggregator(VARS_DYNAMIC)
    target_agg = StatsAggregator(VARS_TARGET)
    
    # --- NEW: Add an aggregator for the target *deltas* ---
    # We re-use the VARS_TARGET keys (WD, VX, VY)
    # to store the stats for (delta_WD, delta_VX, delta_VY)
    target_delta_agg = StatsAggregator(VARS_TARGET)
    # --- END NEW ---

    # Iterate over all training files
    for nc_path in tqdm(nc_files, desc="Calculating Scaling Stats"):
        try:
            with xr.open_dataset(nc_path) as ds:
                
                # --- MODIFIED: Check for variable existence ---
                # Check for all required variables first
                missing_vars = []
                all_vars = VARS_STATIC + VARS_DYNAMIC + VARS_TARGET
                for v in all_vars:
                    if v not in ds.data_vars:
                        missing_vars.append(v)
                
                if missing_vars:
                    # This handles the errors you saw in your log
                    warnings.warn(
                        f"Skipping file {nc_path}: Missing variables: {missing_vars}"
                    )
                    continue
                # --- END MODIFIED ---

                # Static variables are 1D (shape [nodes])
                for var in VARS_STATIC:
                    static_agg.update(var, ds[var].values)

                # Dynamic variables are 2D (shape [time, nodes])
                for var in VARS_DYNAMIC:
                    dynamic_agg.update(var, ds[var].values)

                # --- MODIFIED: Process targets and deltas ---
                
                # 1. Load all target data first
                #    .to_array() stacks them in the order of VARS_TARGET
                #    Use .load() to pull data into memory for np.diff
                y_data = ds[VARS_TARGET].to_array().load().values # Shape [vars, nodes, time]
                
                # 2. Update stats for the raw targets (y)
                for i, var in enumerate(VARS_TARGET):
                    target_agg.update(var, y_data[i]) # y_data[i] is [nodes, time]
                    
                # 3. Calculate deltas (increments)
                #    np.diff computes y(t+1) - y(t) along the time axis (axis=2)
                deltas = np.diff(y_data, axis=2) # Shape [vars, nodes, time-1]
                
                # 4. Update stats for the deltas (y_delta)
                for i, var in enumerate(VARS_TARGET):
                    target_delta_agg.update(var, deltas[i]) # deltas[i] is [nodes, time-1]
                
                # --- END MODIFIED ---

        except Exception as e:
            warnings.warn(f"Failed to process {nc_path}: {e}. Skipping file.")
            continue

    # Compute the final statistics
    static_means, static_stds = static_agg.compute()
    dynamic_means, dynamic_stds = dynamic_agg.compute()
    target_means, target_stds = target_agg.compute()
    
    # --- NEW: Compute delta stats ---
    target_delta_means, target_delta_stds = target_delta_agg.compute()
    # --- END NEW ---

    # Format the data for YAML, preserving the specific list order
    output_data = {
        "x_static_mean": [static_means[v] for v in VARS_STATIC],
        "x_static_std": [static_stds[v] for v in VARS_STATIC],
        "x_dynamic_mean": [dynamic_means[v] for v in VARS_DYNAMIC],
        "x_dynamic_std": [dynamic_stds[v] for v in VARS_DYNAMIC],
        "y_mean": [target_means[v] for v in VARS_TARGET],
        "y_std": [target_stds[v] for v in VARS_TARGET],
        
        # --- NEW: Add delta stats to the output file ---
        "y_delta_mean": [target_delta_means[v] for v in VARS_TARGET],
        "y_delta_std": [target_delta_stds[v] for v in VARS_TARGET],
        # --- END NEW ---
    }

    # Save to the YAML file
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            # sort_keys=False preserves the order defined above
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
        required=True # Made this required
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="List of training .nc files (can use glob patterns like 'data/*.nc'). Required.",
        required=True # Made this required
    )

    args = parser.parse_args()

    # Run the main script
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