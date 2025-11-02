"""mswegnn/utils/scaling.py"""

import xarray as xr
import numpy as np
import yaml
from tqdm import tqdm
import warnings


# These lists define the 10 variables we will calculate stats for,
# matching the variables loaded in adforce_dataset.py
# (We exclude 'node_type' as it's a categorical flag)
VAR_STATIC = ["DEM", "slopex", "slopey", "area"]
VAR_DYNAMIC = ["WX", "WY", "P"]
VAR_TARGET = ["WD", "VX", "VY"]


class StatsAggregator:
    """
    Helper class to compute mean/std in a single pass.

    This uses the 'sum' and 'sum-of-squares' method, which is
    numerically stable when using float64 and easily parallelizable.
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
    # Initialize aggregators for our three variable groups
    static_agg = StatsAggregator(VAR_STATIC)
    dynamic_agg = StatsAggregator(VAR_DYNAMIC)
    target_agg = StatsAggregator(VAR_TARGET)

    # Iterate over all training files
    for nc_path in tqdm(nc_files, desc="Calculating Scaling Stats"):
        try:
            with xr.open_dataset(nc_path) as ds:
                # Static variables are 1D (shape [nodes])
                for var in VAR_STATIC:
                    static_agg.update(var, ds[var].values)

                # Dynamic/Target variables are 2D (shape [time, nodes])
                for var in VAR_DYNAMIC:
                    dynamic_agg.update(var, ds[var].values)

                for var in VAR_TARGET:
                    target_agg.update(var, ds[var].values)

        except Exception as e:
            warnings.warn(f"Failed to process {nc_path}: {e}. Skipping file.")
            continue

    # Compute the final statistics
    static_means, static_stds = static_agg.compute()
    dynamic_means, dynamic_stds = dynamic_agg.compute()
    target_means, target_stds = target_agg.compute()

    # Format the data for YAML, preserving the specific list order
    output_data = {
        "x_static_mean": [static_means[v] for v in VAR_STATIC],
        "x_static_std": [static_stds[v] for v in VAR_STATIC],
        "x_dynamic_mean": [dynamic_means[v] for v in VAR_DYNAMIC],
        "x_dynamic_std": [dynamic_stds[v] for v in VAR_DYNAMIC],
        "y_mean": [target_means[v] for v in VAR_TARGET],
        "y_std": [target_stds[v] for v in VAR_TARGET],
    }

    # Save to the YAML file
    try:
        with open(output_path, "w") as f:
            # sort_keys=False preserves the order defined above
            yaml.dump(output_data, f, sort_keys=False)
    except Exception as e:
        raise IOError(f"Failed to write scaling stats to {output_path}: {e}")

    print("\n" + "=" * 30)
    print(f"Stats calculation complete. Saved to {output_path}.")
    print(yaml.dump(output_data))
    print("=" * 30)
