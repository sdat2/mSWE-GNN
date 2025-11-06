"""
# mswegnn/utils/adforce_scaling.py

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
import argparse
import warnings
import doctest # Import for doctests
import sys     # Import for sys.exit


# --- NEW: Define variable names as constants ---
VARS_STATIC = ["DEM", "slopex", "slopey", "area"]
VARS_DYNAMIC = ["WX", "WY", "P"]
VARS_TARGET = ["WD", "VX", "VY"]
# --- END NEW ---


class WelfordAggregator:
    """
    Implements Welford's online algorithm for calculating mean and variance.
    Numerically stable for large datasets.

    Doctest:
    >>> import numpy as np
    >>> # 1. Test with 2 features
    >>> agg = WelfordAggregator(num_features=2)
    >>> data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    >>> agg.update(data)
    >>> stats = agg.finalize()
    >>> # Check mean (should be [2.0, 20.0])
    >>> np.allclose(stats['mean'], [2.0, 20.0])
    True
    >>> # Check std (should be [sqrt( (1^2 + 0^2 + 1^2) / 3 ), ...])
    >>> # Population std, not sample. (1, 0, 1) -> M2 = 2. Var = 2/3. Std = sqrt(2/3)
    >>> # (10, 0, 10) -> M2 = 200. Var = 200/3. Std = sqrt(200/3)
    >>> expected_std = np.sqrt([2.0/3.0, 200.0/3.0])
    >>> np.allclose(stats['std'], expected_std)
    True

    >>> # 2. Test with 1 feature
    >>> agg_1d = WelfordAggregator(num_features=1)
    >>> data_1d = np.array([[1.], [2.], [3.], [4.], [5.]])
    >>> agg_1d.update(data_1d)
    >>> stats_1d = agg_1d.finalize()
    >>> # Mean should be 3.0
    >>> np.allclose(stats_1d['mean'], [3.0])
    True
    >>> # M2 = (2^2 + 1^2 + 0^2 + 1^2 + 2^2) = 10. Var = 10/5 = 2. Std = sqrt(2)
    >>> np.allclose(stats_1d['std'], [np.sqrt(2.0)])
    True
    """

    def __init__(self, num_features: int):
        self.num_features = num_features
        self.count = 0
        self.mean = np.zeros(num_features, dtype=np.float64)
        self.M2 = np.zeros(num_features, dtype=np.float64)

    def update(self, data: np.ndarray):
        """
        Update the aggregator with a batch of data.
        Assumes data is shape [num_samples, num_features] or [*, num_features].
        """
        # --- Handle NaNs and Infs ---
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        # --- End Handle NaNs ---
        
        if data.ndim > 2:
            # Flatten all dimensions except the last one (features)
            data = data.reshape(-1, self.num_features)

        if data.size == 0:
            return

        for x in data:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.M2 += delta * delta2

    def finalize(self) -> Dict[str, List[float]]:
        """
        Returns the final mean and standard deviation.
        """
        if self.count < 2:
            warnings.warn("WelfordAggregator has less than 2 samples.")
            return {"mean": [0.0] * self.num_features, "std": [1.0] * self.num_features}

        variance = self.M2 / self.count
        std = np.sqrt(variance)

        # Convert to native Python floats for YAML serialization
        return {"mean": self.mean.tolist(), "std": std.tolist()}


class StatsAggregator:
    """
    Main class to coordinate statistics computation for all variable types.
    """

    def __init__(self):
        # --- NEW: Initialize aggregators based on list lengths ---
        self.x_static_agg = WelfordAggregator(num_features=len(VARS_STATIC))
        self.x_dynamic_agg = WelfordAggregator(num_features=len(VARS_DYNAMIC))
        self.y_agg = WelfordAggregator(num_features=len(VARS_TARGET))
        self.y_delta_agg = WelfordAggregator(num_features=len(VARS_TARGET))
        # --- END NEW ---

    def update_stats(self, nc_path: str):
        """
        Loads a single NetCDF file and updates all aggregators.
        """
        try:
            with xr.open_dataset(nc_path, cache=False) as ds:
                # 1. --- Update x_static using VARS_STATIC ---
                # We stack the variables in the order defined by the list
                static_data = np.stack(
                    [ds[var].values for var in VARS_STATIC],
                    axis=1,
                )
                self.x_static_agg.update(static_data)

                # 2. --- Update x_dynamic using VARS_DYNAMIC ---
                # .to_array() preserves the list order
                dyn_array = ds[VARS_DYNAMIC].to_array().values
                # Transpose to [nodes, time, vars] then update
                self.x_dynamic_agg.update(dyn_array.transpose(1, 2, 0))

                # 3. --- Update y using VARS_TARGET ---
                y_data_array = ds[VARS_TARGET].to_array().values
                # Transpose to [nodes, time, vars] then update
                self.y_agg.update(y_data_array.transpose(1, 2, 0))

                # 4. --- Update y_delta (based on y_data_array) ---
                #    y_data_array is [vars, nodes, time]
                deltas = np.diff(y_data_array, axis=2)
                # Transpose to [nodes, time_deltas, vars] then update
                self.y_delta_agg.update(deltas.transpose(1, 2, 0))

        except KeyError as e:
            warnings.warn(f"File {nc_path} is missing a required variable: {e}. Skipping file.")
        except Exception as e:
            warnings.warn(f"Failed to process file {nc_path}: {e}. Skipping.")

    def finalize(self) -> Dict[str, List[float]]:
        """
        Finalizes all aggregators and returns a combined dictionary.
        The lists in the output YAML will match the order of the
        VARS_* lists defined at the top of this file.
        """
        stats = {}
        stats.update(
            {f"x_static_{k}": v for k, v in self.x_static_agg.finalize().items()}
        )
        stats.update(
            {f"x_dynamic_{k}": v for k, v in self.x_dynamic_agg.finalize().items()}
        )
        stats.update({f"y_{k}": v for k, v in self.y_agg.finalize().items()})
        stats.update(
            {f"y_delta_{k}": v for k, v in self.y_delta_agg.finalize().items()}
        )

        return stats


def compute_and_save_adforce_stats(train_files: List[str], output_path: str):
    """
    Main execution function.
    """
    if not train_files:
        raise ValueError("No training files provided.")

    print(f"Calculating stats from {len(train_files)} training files...")

    # 2. Initialize aggregator
    aggregator = StatsAggregator()

    # 3. Iterate and update
    for nc_file in tqdm(train_files, desc="Processing files"):
        aggregator.update_stats(nc_file) # No longer needs previous_t

    # 4. Finalize
    final_stats = aggregator.finalize()

    # 5. Save to YAML
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(final_stats, f, default_flow_style=False, sort_keys=False)

    print("\n--- Final Statistics ---")
    print(yaml.dump(final_stats, default_flow_style=False))
    print(f"\nStats saved to {output_path}")


if __name__ == "__main__":
    """
    Run doctests or the main script.
    
    To run doctests:
    python -m mswegnn.utils.adforce_scaling
    
    To run the main script:
    python -m mswegnn.utils.adforce_scaling \
        -o data_processed/train/scaling_stats.yaml \
        --files /path/to/data/*.nc
    """
    import glob

    parser = argparse.ArgumentParser(
        description="Compute normalization stats for Adforce dataset."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to save the output scaling_stats.yaml. If specified, runs the main script.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="List of training .nc files (can use glob patterns like 'data/*.nc'). Required if -o is set.",
    )

    args = parser.parse_args()

    # --- NEW: If no args given, run doctests. Otherwise, run main. ---
    if args.output is None and args.files is None:
        # Run doctests
        print("Running doctests...")
        result = doctest.testmod(verbose=True)
        if result.failed == 0:
            print(f"All {result.attempted} doctests passed.")
        else:
            print(f"!!! {result.failed} doctests FAILED out of {result.attempted} !!!")
            sys.exit(1) # Exit with error code if tests fail
    
    elif args.output and args.files:
        # Run the main script
        train_files = []
        for pattern in args.files:
            train_files.extend(glob.glob(pattern))
        
        if not train_files:
            print("Error: No files found matching the patterns in --files.")
        else:
            compute_and_save_adforce_stats(
                train_files=sorted(list(set(train_files))),
                output_path=args.output,
            )
    else:
        print("Error: You must provide *both* --output and --files to run the script,")
        print("or *neither* to run the doctests.")
        parser.print_help()
        sys.exit(1)